import gymnasium as gym
import random
import re
import numpy as np
import os
import time
import gc
from jinja2 import Template
from dotenv import load_dotenv
from google import genai
# import anthropic
import time
import torch

try:
    from openai import OpenAI
    OPENAI_CLIENT_AVAILABLE = True
except ImportError:
    OPENAI_CLIENT_AVAILABLE = False
    print("Warning: openai package not installed. OpenAI-compatible models (e.g., Groq) won't be available.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: transformers package not installed. HuggingFace models won't be available.")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not installed. Ollama models won't be available.")

load_dotenv()

class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        self.use_cuda = False  # Track if we're using CUDA
        self.hf_input_device = "cpu"
        
        llm_model_name_lower = llm_model_name.lower()

        # Detect explicit provider prefixes before generic slash-based routing.
        is_ollama_prefixed = llm_model_name_lower.startswith("ollama/")

        # Detect Groq models first so provider-prefixed model names are never
        # routed to HuggingFace/local execution paths.
        known_groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            # Groq-hosted OpenAI OSS model ids
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]
        is_groq_model = llm_model_name_lower.startswith("groq/") or llm_model_name_lower in known_groq_models

        # Detect if this is a HuggingFace model (contains / indicating org/model format)
        # HuggingFace paths: "Qwen/Qwen3.5-72B-Instruct", "meta-llama/Llama-3.1-70B", etc.
        # Exclude Groq and explicit Ollama provider-prefixed models.
        is_hf_model = (not is_groq_model) and (not is_ollama_prefixed) and ("/" in llm_model_name)
        
        # Detect if this is an Ollama model (model:version format or common local model names)
        # Ollama examples: "llama3:70b", "qwen:14b", "mistral:latest"
        is_ollama_model = is_ollama_prefixed or ((not is_hf_model and not is_groq_model) and (
            ":" in llm_model_name
            or any(
                llm_model_name_lower.startswith(prefix)
                for prefix in ["llama", "qwen", "mistral", "phi", "gemma", "codellama", "gpt-oss"]
            )
        ))
        
        known_models = [
            "o1-preview",
            "gpt-4o",
            "gemini-2.0-flash-exp",
            "gpt-4o-mini",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "o3-mini-2025-01-31",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "claude-3-7-sonnet-20250219",
        ]
        
        if not is_hf_model and not is_ollama_model and not is_groq_model:
            assert llm_model_name in known_models, (
                f"Unknown model: {llm_model_name}. Use a known model, an Ollama model, a Groq model, or a HuggingFace model."
            )
        
        self.llm_model_name = llm_model_name
        
        # Setup model group and client
        if is_hf_model:
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("HuggingFace transformers not installed. Install with: pip install transformers bitsandbytes accelerate")
            self.model_group = "huggingface"
            
            print(f"Loading HuggingFace model: {llm_model_name}")

            cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if cuda_count > 0:
                gpu_mem_budget = os.environ.get("HF_GPU_MAX_MEMORY_GIB", "78GiB")
                cpu_mem_budget = os.environ.get("HF_CPU_MAX_MEMORY_GIB", "120GiB")
                max_memory = {gpu_idx: gpu_mem_budget for gpu_idx in range(cuda_count)}
                max_memory["cpu"] = cpu_mem_budget
                device_map = "auto"
                print(
                    f"CUDA detected ({cuda_count} GPU(s)); using device_map='auto' with max_memory={max_memory}"
                )
            else:
                max_memory = None
                device_map = "cpu"
                print("Warning: CUDA not detected. Large models may not be runnable on CPU.")
            
            # Load tokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            
            # Inspect the model config to determine whether the model is already
            # quantized.  This is more reliable than guessing from the model name.
            print("Inspecting model config for existing quantization...")
            model_config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
            
            # A quantized model will have a populated quantization_config in its
            # config.  GPTQ / AWQ / FP8 / Mxfp4 models all set this field.
            config_quant = getattr(model_config, "quantization_config", None)
            is_prequantized = config_quant is not None

            # Also catch models whose names signal quantization but whose config
            # may not yet be saved correctly (older checkpoints).
            name_quant_signals = ['fp8', 'gptq', 'awq', 'int4', 'int8', 'gpt-oss', '-oss', 'mxfp4']
            if not is_prequantized and any(x in llm_model_name.lower() for x in name_quant_signals):
                print("Warning: quantization not detected in config but model name suggests "
                      "it is pre-quantized. Loading without additional quantization.")
                is_prequantized = True

            if is_prequantized:
                quant_type = getattr(config_quant, "quant_type", None) or \
                             getattr(config_quant, "bits", None) or "unknown"
                print(f"Pre-quantized model detected (type: {quant_type}). "
                      "Loading without additional quantization.")
                load_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": "auto",
                    "device_map": device_map,
                    "low_cpu_mem_usage": True,
                }
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
                self.hf_model = AutoModelForCausalLM.from_pretrained(llm_model_name, **load_kwargs)
                print("Successfully loaded pre-quantized model")
            else:
                # Apply NF4 4-bit quantization — best quality/VRAM tradeoff for
                # inference.  Double-quant further reduces the quantization
                # constants' footprint (~0.4 bits/param saved) without a
                # measurable quality drop for this text-generation task.
                print("No existing quantization detected. Applying NF4 4-bit quantization.")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": device_map,
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if max_memory is not None:
                    load_kwargs["max_memory"] = max_memory
                self.hf_model = AutoModelForCausalLM.from_pretrained(llm_model_name, **load_kwargs)
                print("Successfully loaded with NF4 4-bit quantization")

            self.hf_model.eval()
            self.hf_input_device = self._detect_hf_input_device()
            print(f"HuggingFace input device: {self.hf_input_device}")
            
            # Remove unsupported generation parameters to avoid warnings
            if hasattr(self.hf_model, 'generation_config'):
                self.hf_model.generation_config.top_p = None
                self.hf_model.generation_config.top_k = None
            
            # Set pad token if not present
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            
            quant_status = "pre-quantized (passthrough)" if is_prequantized else "NF4 4-bit quantization applied"
            print(f"HuggingFace model ready — {quant_status}")
            
            # Track if model is on CUDA
            self.use_cuda = torch.cuda.is_available() and "cuda" in str(self.hf_input_device)
        elif is_ollama_model:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama package not installed. Install with: pip install ollama")
            self.model_group = "ollama"
            self.ollama_model_name = llm_model_name.split("/", 1)[1] if is_ollama_prefixed else llm_model_name
            ollama_host = os.environ.get("OLLAMA_HOST")
            self.ollama_client = ollama.Client(host=ollama_host) if ollama_host else ollama.Client()
            print(f"Using Ollama model: {self.ollama_model_name}")
        elif is_groq_model:
            if not OPENAI_CLIENT_AVAILABLE:
                raise ImportError("openai package not installed. Install with: pip install openai")

            groq_api_key = os.environ.get("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("Missing GROQ_API_KEY in environment.")

            self.model_group = "groq"
            self.groq_model_name = llm_model_name.split("/", 1)[1] if llm_model_name_lower.startswith("groq/") else llm_model_name
            self.groq_client = OpenAI(
                api_key=groq_api_key,
                base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            )
            print(f"Using Groq model: {self.groq_model_name}")
        elif "gemini" in llm_model_name:
            self.model_group = "gemini"
            # get env gemini keys into list
            self.gemini_api_keys = []
            key_index = 1
            while True:
                key_name = f"GEMINI_API_KEY_{key_index}" if key_index > 1 else "GEMINI_API_KEY"
                if key_name in os.environ:
                    self.gemini_api_keys.append(os.environ[key_name])
                    key_index += 1
                else:
                    break
            
            if not self.gemini_api_keys:
                raise ValueError(".env file has no keys.")
            
            self.current_gemini_key_index = 0
            self.gemini_client = genai.Client(api_key=self.gemini_api_keys[self.current_gemini_key_index])
            print(f"Loaded {len(self.gemini_api_keys)} keys.")
        # elif "claude" in llm_model_name:
        #     self.model_group = "anthropic"
        #     self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        # else:
        #     self.model_group = "openai"
        #     self.client = OpenAI()

    def _rotate_gemini_key(self):
        """Rotate to the next Gemini API key"""
        if self.model_group == "gemini" and len(self.gemini_api_keys) > 1:
            self.current_gemini_key_index = (self.current_gemini_key_index + 1) % len(self.gemini_api_keys)
            self.gemini_client = genai.Client(api_key=self.gemini_api_keys[self.current_gemini_key_index])
            print(f"Switched to Gemini API key #{self.current_gemini_key_index + 1}")

    def _assistant_role_name(self):
        """Return the provider-specific assistant role label."""
        return "model" if self.model_group == "gemini" else "assistant"
    
    def _format_hf_chat(self):
        """Format conversation history for HuggingFace chat models"""
        # Try to use the model's chat template if available
        if hasattr(self.hf_tokenizer, 'apply_chat_template') and self.hf_tokenizer.chat_template is not None:
            try:
                return self.hf_tokenizer.apply_chat_template(
                    self.llm_conversation,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                pass
        
        # Fallback to manual formatting
        prompt = ""
        for msg in self.llm_conversation:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
            elif role == "system":
                prompt += f"<|system|>\n{content}\n"
        
        # Add generation prompt
        prompt += "<|assistant|>\n"
        return prompt

    def _detect_hf_input_device(self):
        """Choose an input device compatible with sharded HF models."""
        hf_device_map = getattr(self.hf_model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            unique_devices = []
            for mapped_device in hf_device_map.values():
                if isinstance(mapped_device, int):
                    normalized = f"cuda:{mapped_device}"
                else:
                    normalized = str(mapped_device)
                if normalized not in unique_devices:
                    unique_devices.append(normalized)
            for preferred_prefix in ("cuda", "mps", "xpu"):
                for mapped_device in unique_devices:
                    if mapped_device.startswith(preferred_prefix):
                        return mapped_device
            if unique_devices:
                return unique_devices[0]
        return str(getattr(self.hf_model, "device", "cpu"))

    def _clear_cuda_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        gc.collect()

    def _normalize_local_response(self, response: str) -> str:
        """Post-process raw output from local models (HuggingFace / Ollama).
        """
        # 1. Strip <tool_call>...<tool_call> blocks
        response = re.sub(r"<tool_call>.*?<tool_call>", "", response, flags=re.DOTALL).strip()

        # 2. Strip markdown code fences (```python\n...``` or ```\n...```)
        response = re.sub(r"```[^\n]*\n?", "", response).strip()

        # 3. For the flat params format, pull the `params[0]:` line to the top
        #    so parse_parameters (which only reads split('\n')[0]) finds it.
        lines = response.split("\n")
        params_line_idx = next(
            (i for i, ln in enumerate(lines)
             if re.search(r"params\s*\[\s*0\s*\]\s*:", ln, re.IGNORECASE)),
            None,
        )
        if params_line_idx is not None and params_line_idx > 0:
            lines = (
                [lines[params_line_idx]]
                + lines[:params_line_idx]
                + lines[params_line_idx + 1:]
            )

        # 4. For the L/U matrix format, parse_factor_matrices scans all lines
        #    for 'L matrix:' / 'U matrix:' headers, so no reordering is needed.

        return "\n".join(lines)

    def _generate_with_hf(self, max_new_tokens=384, temperature=0.7, do_sample=True):
        """Generate a response from the loaded HuggingFace model.

        Automatically recovers from VRAM OOM by halving max_new_tokens and
        retrying until the budget drops below 64 tokens, at which point a
        RuntimeError is raised so callers can surface the problem clearly.
        """
        prompt = self._format_hf_chat()
        current_max_tokens = max_new_tokens
        min_tokens = 64
        use_kv_cache = True

        while current_max_tokens >= min_tokens:
            inputs = None
            try:
                inputs = self.hf_tokenizer(
                    prompt, return_tensors="pt", padding=True
                ).to(self.hf_input_device)

                gen_kwargs = dict(
                    max_new_tokens=current_max_tokens,
                    do_sample=do_sample,
                    pad_token_id=self.hf_tokenizer.pad_token_id,
                    use_cache=use_kv_cache,
                )
                if do_sample:
                    gen_kwargs["temperature"] = temperature

                with torch.no_grad():
                    outputs = self.hf_model.generate(**inputs, **gen_kwargs)

                response = self.hf_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                del inputs, outputs
                if self.use_cuda:
                    self._clear_cuda_cache()
                return response

            except Exception as e:
                if inputs is not None:
                    del inputs
                if self.use_cuda:
                    self._clear_cuda_cache()

                is_oom = "out of memory" in str(e).lower() or (
                    hasattr(torch.cuda, "OutOfMemoryError")
                    and isinstance(e, torch.cuda.OutOfMemoryError)
                )
                if is_oom:
                    if use_kv_cache:
                        use_kv_cache = False
                        print("[OOM] Retrying with use_cache=False to lower KV memory.")
                        continue
                    current_max_tokens //= 2
                    if current_max_tokens < min_tokens:
                        raise RuntimeError(
                            f"[OOM] VRAM exhausted even at {min_tokens} max_new_tokens. "
                            "Free up VRAM, reduce input length, or use a smaller model."
                        ) from e
                    print(
                        f"[OOM] VRAM exhausted. Retrying with "
                        f"max_new_tokens={current_max_tokens}..."
                    )
                else:
                    raise

        raise RuntimeError("[OOM] Could not generate a response within VRAM constraints.")

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def clear_cache(self):
        """Clear GPU cache to prevent OOM between episodes."""
        if self.use_cuda:
            self._clear_cuda_cache()
            print("[Memory] GPU cache cleared")

    def add_llm_conversation(self, text, role):
        # if self.model_group == "openai":
        #     self.llm_conversation.append({"role": role, "content": text})
        # elif self.model_group == "anthropic":
        #     self.llm_conversation.append({"role": role, "content": text})
        # else:
        if self.model_group == "gemini":
            self.llm_conversation.append({"role": role, "parts": text})
        elif self.model_group == "huggingface":
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "ollama":
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "groq":
            self.llm_conversation.append({"role": role, "content": text})

    def query_llm(self):
        for attempt in range(10):
            try:
                # if self.model_group == "openai":
                #     completion = self.client.chat.completions.create(
                #         model=self.llm_model_name,
                #         messages=self.llm_conversation,
                #     )
                #     response = completion.choices[0].message.content
                # elif self.model_group == "anthropic":
                #     message = self.client.messages.create(
                #         model=self.llm_model_name,
                #         messages=self.llm_conversation,
                #         max_tokens=1024,
                #     )
                #     response = message.content[0].text
                # else:
                if self.model_group == "gemini":
                    # Convert conversation history for new API
                    contents = []
                    for msg in self.llm_conversation:
                        contents.append({"role": msg["role"], "parts": [{"text": msg["parts"]}]})
                    
                    response = self.gemini_client.models.generate_content(
                        model=self.llm_model_name,
                        contents=contents
                    )
                    response = response.text
                elif self.model_group == "huggingface":
                    response = self._normalize_local_response(self._generate_with_hf())
                elif self.model_group == "ollama":
                    # Use Ollama for local models
                    response = self.ollama_client.chat(
                        model=self.ollama_model_name,
                        messages=self.llm_conversation
                    )
                    response = self._normalize_local_response(response['message']['content'])
                elif self.model_group == "groq":
                    completion = self.groq_client.chat.completions.create(
                        model=self.groq_model_name,
                        messages=self.llm_conversation,
                    )
                    response = completion.choices[0].message.content or ""
            except Exception as e:
                print(f"Error: {e}")
                
                # Check if it's a rate limit error and rotate Gemini key
                if self.model_group == "gemini" and ("429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower() or "resource_exhausted" in str(e).lower()):
                    self._rotate_gemini_key()
                    print("Retrying with new API key...")
                    continue
                
                print("Retrying...")
                if attempt == 9:
                    raise Exception("Failed to get response from LLM after 10 attempts")
                else:
                    print("LLM provider charging up...")
                    time.sleep(60)
                    continue

            # if self.model_group == "openai":
            #     # add the response to self.llm_conversation
            #     self.add_llm_conversation(response, "assistant")
            # else:
            self.add_llm_conversation(response, self._assistant_role_name())

            return response
        
        raise Exception("Failed to get response from LLM after all retry attempts")

    def query_llm_multiple_response(self, num_responses, temperature):
        for attempt in range(5):
            try:
                # if self.model_group == "openai":
                #     completion = self.client.chat.completions.create(
                #         model=self.llm_model_name,
                #         messages=self.llm_conversation,
                #         n=num_responses,
                #         temperature=temperature,
                #     )
                #     responses = [
                #         completion.choices[i].message.content
                #         for i in range(num_responses)
                #     ]
                # else:
                if self.model_group == "gemini":
                    # Convert conversation for new API
                    contents = []
                    for msg in self.llm_conversation:
                        contents.append({"role": msg["role"], "parts": [{"text": msg["parts"]}]})
                    
                    # Note: New API may not support multiple candidates in the same way
                    # Generate multiple responses sequentially
                    responses = []
                    for _ in range(num_responses):
                        response = self.gemini_client.models.generate_content(
                            model=self.llm_model_name,
                            contents=contents,
                            config={"temperature": temperature}
                        )
                        responses.append(response.text)
                elif self.model_group == "huggingface":
                    # Generate multiple responses sequentially with OOM recovery
                    responses = []
                    for _ in range(num_responses):
                        responses.append(
                            self._normalize_local_response(
                                self._generate_with_hf(
                                    temperature=temperature, do_sample=True
                                )
                            )
                        )
                elif self.model_group == "ollama":
                    # Generate multiple responses with Ollama
                    responses = []
                    for _ in range(num_responses):
                        response = self.ollama_client.chat(
                            model=self.ollama_model_name,
                            messages=self.llm_conversation,
                            options={"temperature": temperature}
                        )
                        responses.append(
                            self._normalize_local_response(response['message']['content'])
                        )
                elif self.model_group == "groq":
                    responses = []
                    for _ in range(num_responses):
                        completion = self.groq_client.chat.completions.create(
                            model=self.groq_model_name,
                            messages=self.llm_conversation,
                            temperature=temperature,
                        )
                        responses.append(completion.choices[0].message.content or "")

            except Exception as e:
                print(f"Error: {e}")
                
                # If rate limit error, attempt to use new API key
                if self.model_group == "gemini" and ("429" in str(e) or "quota" in str(e).lower() or "rate" in str(e).lower() or "resource_exhausted" in str(e).lower()):
                    self._rotate_gemini_key()
                    print("Retrying with new API key...")
                    continue
                
                print("Retrying...")
                if attempt == 4:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)

            return responses

    def parse_parameters(self, parameters_string):
        new_parameters_list = []

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    parameters_row = [
                        float(x.strip().strip(",")) for x in row.split(",")
                    ]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(e)

        return new_parameters_list

    def llm_update_parameters(self, parameters, replay_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "replay_buffer_string": str(replay_buffer),
                "parameters_string": str(parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        self.add_llm_conversation(new_parameters_with_reasoning, self._assistant_role_name())
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [new_parameters_with_reasoning, new_parameters]

    def llm_update_parameters_sas(self, episode_reward_buffer, parse_parameters=None):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {"episode_reward_buffer_string": str(episode_reward_buffer)}
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        self.add_llm_conversation(new_parameters_with_reasoning, self._assistant_role_name())
        self.add_llm_conversation(
            self.llm_output_conversion_template.render(),
            "user",
        )
        new_parameters = self.query_llm()

        if parse_parameters is None:
            new_parameters_list = self.parse_parameters(new_parameters)
        else:
            new_parameters_list = parse_parameters(new_parameters)

        return new_parameters_list, [
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            new_parameters,
        ]

    def llm_update_parameters_num_optim(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
        dim_state=None,
        dim_action=None,
        factor_rank=None,
        use_factorized=False,
        frozen_factor=None,
        schedule_context=None,
    ):
        self.reset_llm_conversation()

        schedule_context = schedule_context or {}

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
                "dim_state": dim_state,
                "dim_action": dim_action,
                "factor_rank": factor_rank,
                "frozen_factor": frozen_factor,
                "lu_schedule_enabled": bool(schedule_context.get("enabled", False)),
                "lu_schedule_phase": schedule_context.get("phase"),
                "lu_schedule_l_episodes": schedule_context.get("l_episodes"),
                "lu_schedule_u_iterations": schedule_context.get("u_iterations"),
                "lu_schedule_cycle_step": schedule_context.get("cycle_step"),
                "lu_schedule_cycle_length": schedule_context.get("cycle_length"),
            }
        )

        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )

    def llm_update_parameters_num_optim_q_table(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        actions,
        num_states,
        optimum,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "actions": actions,
                "rank": num_states,
                "optimum": str(optimum),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_imitation(
        self,
        demonstrations_str,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "expert_demonstration_string": demonstrations_str,
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_propose_multiple_parameters_num_optim_based_on_anchor(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
        num_candidates,
        temperature,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        # print(system_prompt)
        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning_list = self.query_llm_multiple_response(
            num_candidates, temperature
        )
        # print(new_parameters_with_reasoning_list)

        new_parameters_list = []
        reasonings_list = []
        for new_params in new_parameters_with_reasoning_list:
            new_params_np = parse_parameters(new_params)
            new_parameters_list.append(new_params_np)
            reasonings_list.append(new_params)

        return (
            system_prompt,
            new_parameters_list,
            reasonings_list,
        )

    def llm_propose_parameters_num_optim_based_on_anchor_thread(
        self,
        new_candidates,
        new_idx,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        search_std,
        anchor_parameters,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "search_std": str(search_std),
                "anchor_parameters": str(anchor_parameters),
            }
        )

        self.add_llm_conversation(system_prompt, "user")
        new_parameters_with_reasoning = self.query_llm()

        print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)
        new_candidates[new_idx] = new_parameters_list

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
        )

    def llm_update_parameters_num_optim_semantics(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        env_desc_file,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "env_description": env_desc_file,
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
            }
        )


        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time

        # print(system_prompt)

        # self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        # self.add_llm_conversation(
        #     self.llm_output_conversion_template.render(),
        #     "user",
        # )
        # new_parameters = self.query_llm()
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )
