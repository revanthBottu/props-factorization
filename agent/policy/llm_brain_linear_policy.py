import gymnasium as gym
import random
import re
import math
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


def _is_oom_exception(err: Exception) -> bool:
    err_text = str(err).lower()
    return "out of memory" in err_text or "cuda out of memory" in err_text

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
                # Keep substantial VRAM headroom for prefill/decoding activations,
                # KV cache, and allocator fragmentation at generation time.
                gpu_reserve_gib = float(os.environ.get("HF_GPU_MEMORY_RESERVE_GIB", "10"))
                requested_gpu_budget = os.environ.get("HF_GPU_MAX_MEMORY_GIB")
                cpu_mem_budget = os.environ.get("HF_CPU_MAX_MEMORY_GIB", "120GiB")
                max_memory = {}
                for gpu_idx in range(cuda_count):
                    total_gib = torch.cuda.get_device_properties(gpu_idx).total_memory / (1024**3)
                    detected_budget_gib = max(8, int(total_gib - gpu_reserve_gib))
                    if requested_gpu_budget is not None:
                        try:
                            requested_budget_gib = float(requested_gpu_budget.replace("GiB", "").strip())
                        except ValueError:
                            requested_budget_gib = detected_budget_gib
                            print(
                                f"Warning: HF_GPU_MAX_MEMORY_GIB='{requested_gpu_budget}' is invalid. "
                                f"Using detected budget {detected_budget_gib}GiB for GPU {gpu_idx}."
                            )
                        final_budget_gib = int(min(requested_budget_gib, detected_budget_gib))
                    else:
                        final_budget_gib = detected_budget_gib
                    max_memory[gpu_idx] = f"{final_budget_gib}GiB"
                max_memory["cpu"] = cpu_mem_budget
                device_map = "auto"
                print(
                    f"CUDA detected ({cuda_count} GPU(s)); using device_map='auto' with max_memory={max_memory}"
                )
            else:
                max_memory = None
                device_map = "cpu"
                print("Warning: CUDA not detected. Large models may not be runnable on CPU.")

            # Optional disk offload for large models to reduce VRAM pressure.
            # Enable with HF_ENABLE_DISK_OFFLOAD=1.
            offload_kwargs = {}
            if os.environ.get("HF_ENABLE_DISK_OFFLOAD", "0") == "1":
                offload_folder = os.environ.get("HF_DISK_OFFLOAD_DIR", ".hf_offload")
                os.makedirs(offload_folder, exist_ok=True)
                offload_kwargs = {
                    "offload_folder": offload_folder,
                    "offload_state_dict": True,
                }
                print(f"HF disk offload enabled at: {offload_folder}")
            
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
            name_quant_signals = ['fp8', 'gptq', 'awq', 'int4', 'int8', 'mxfp4']
            if not is_prequantized and any(x in llm_model_name.lower() for x in name_quant_signals):
                print("Warning: quantization not detected in config but model name suggests "
                      "it is pre-quantized. Loading without additional quantization.")
                is_prequantized = True

            if is_prequantized:
                quant_type = getattr(config_quant, "quant_type", None) or \
                             getattr(config_quant, "bits", None) or "unknown"
                print(f"Pre-quantized model detected (type: {quant_type}). "
                      "Loading without additional quantization.")
                try:
                    load_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": "auto",
                        "device_map": device_map,
                        "low_cpu_mem_usage": True,
                    }
                    if max_memory is not None:
                        load_kwargs["max_memory"] = max_memory
                    load_kwargs.update(offload_kwargs)
                    self.hf_model = AutoModelForCausalLM.from_pretrained(llm_model_name, **load_kwargs)
                    print("Successfully loaded pre-quantized model")
                except Exception as e:
                    if not _is_oom_exception(e):
                        raise
                    print(
                        "[OOM] Pre-quantized load failed. Retrying with NF4 4-bit quantization "
                        "to reduce memory pressure."
                    )
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
                    load_kwargs.update(offload_kwargs)
                    self.hf_model = AutoModelForCausalLM.from_pretrained(llm_model_name, **load_kwargs)
                    is_prequantized = False
                    print("Successfully loaded with NF4 4-bit quantization after OOM fallback")
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
                load_kwargs.update(offload_kwargs)
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

    def _generate_with_hf(self, max_new_tokens=None, temperature=0.7, do_sample=True):
        """Generate a response from the loaded HuggingFace model.

        OOM recovery prioritizes preserving full output length. By default,
        max_new_tokens is NOT reduced on OOM to avoid clipping structured
        matrix outputs (L/U rows). Prompt-window reduction remains available
        as a last-resort fallback and can be disabled via env.
        """
        prompt = self._format_hf_chat()
        if max_new_tokens is None:
            # Larger default budget to fit full matrix outputs without clipping.
            max_new_tokens = int(os.environ.get("HF_MAX_NEW_TOKENS", "1024"))

        # Keep output token budget fixed by default to avoid truncated matrices.
        allow_output_shrink = os.environ.get("HF_ALLOW_OUTPUT_TOKEN_SHRINK", "0") == "1"
        current_max_tokens = max_new_tokens
        min_tokens = int(os.environ.get("HF_MIN_NEW_TOKENS", "256"))

        # If disabled, we can still shrink prompt window for OOM, but never output length.
        allow_prompt_shrink = os.environ.get("HF_ALLOW_PROMPT_WINDOW_SHRINK", "1") == "1"
        truncate_prompt = os.environ.get("HF_TRUNCATE_PROMPT", "0") == "1"
        min_input_tokens = 512
        use_kv_cache = True

        tokenizer_hard_cap = getattr(self.hf_tokenizer, "model_max_length", 4096)
        if tokenizer_hard_cap is None or tokenizer_hard_cap <= 0 or tokenizer_hard_cap > 1_000_000:
            tokenizer_hard_cap = 4096
        env_prompt_cap = int(os.environ.get("HF_MAX_INPUT_TOKENS", "4096"))
        current_prompt_cap = max(min_input_tokens, min(env_prompt_cap, int(tokenizer_hard_cap)))

        while current_max_tokens >= min_tokens:
            inputs = None
            try:
                tokenizer_kwargs = {
                    "return_tensors": "pt",
                    "padding": True,
                }
                if truncate_prompt:
                    tokenizer_kwargs["truncation"] = True
                    tokenizer_kwargs["max_length"] = current_prompt_cap
                else:
                    tokenizer_kwargs["truncation"] = False

                inputs = self.hf_tokenizer(
                    prompt,
                    **tokenizer_kwargs,
                ).to(self.hf_input_device)

                input_token_count = int(inputs["input_ids"].shape[1])
                if truncate_prompt and input_token_count >= current_prompt_cap:
                    print(
                        f"[HF] Prompt truncated to {current_prompt_cap} tokens "
                        "to stay within VRAM budget."
                    )

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
                    if truncate_prompt and allow_prompt_shrink and current_prompt_cap > min_input_tokens:
                        current_prompt_cap = max(min_input_tokens, current_prompt_cap // 2)
                        print(
                            f"[OOM] Retrying with smaller prompt window: "
                            f"max_input_tokens={current_prompt_cap}..."
                        )
                        continue

                    if allow_output_shrink:
                        current_max_tokens //= 2
                        if current_max_tokens < min_tokens:
                            raise RuntimeError(
                                f"[OOM] VRAM exhausted even at {min_tokens} max_new_tokens. "
                                "Free up VRAM, reduce input length, or use a smaller model."
                            ) from e
                        print(
                            f"[OOM] Retrying with smaller output budget: "
                            f"max_new_tokens={current_max_tokens}..."
                        )
                        continue

                    raise RuntimeError(
                        "[OOM] Generation failed without shrinking output tokens. "
                        "To permit output-token shrinking, set HF_ALLOW_OUTPUT_TOKEN_SHRINK=1. "
                        "Other options: raise HF_GPU_MEMORY_RESERVE_GIB headroom strategy, "
                        "use a smaller/stronger-quantized model, or enable prompt truncation via "
                        "HF_TRUNCATE_PROMPT=1 with HF_ALLOW_PROMPT_WINDOW_SHRINK=1 and lower "
                        "HF_MAX_INPUT_TOKENS."
                    ) from e
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

    def _extract_reward_values(self, episode_reward_buffer):
        text = str(episode_reward_buffer) if episode_reward_buffer is not None else ""
        number_pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

        reward_matches = re.findall(
            rf"f\(params\)\s*:\s*({number_pattern})",
            text,
            flags=re.IGNORECASE,
        )
        if not reward_matches:
            reward_matches = re.findall(
                rf"total\s+reward\s*[:=]\s*({number_pattern})",
                text,
                flags=re.IGNORECASE,
            )

        rewards = []
        for reward_str in reward_matches:
            try:
                rewards.append(float(reward_str))
            except (TypeError, ValueError):
                continue
        return rewards

    def _build_reward_delta_context(self, episode_reward_buffer):
        rewards = self._extract_reward_values(episode_reward_buffer)

        last_reward = rewards[-1] if len(rewards) >= 1 else None
        previous_reward = rewards[-2] if len(rewards) >= 2 else None
        best_reward = max(rewards) if rewards else None

        delta_last_vs_previous = (
            None
            if last_reward is None or previous_reward is None
            else last_reward - previous_reward
        )
        delta_last_vs_best = (
            None
            if last_reward is None or best_reward is None
            else last_reward - best_reward
        )
        delta_previous_vs_zero = None if previous_reward is None else previous_reward

        far_below_zero_threshold = -10.0
        delta_last_vs_previous_far_below_zero = (
            delta_last_vs_previous is not None
            and delta_last_vs_previous <= far_below_zero_threshold
        )

        def _format_delta(value):
            if value is None or not math.isfinite(value):
                return "N/A"
            return f"{value:.2f}"

        return {
            "reward_delta_last_vs_previous": _format_delta(delta_last_vs_previous),
            "reward_delta_last_vs_best": _format_delta(delta_last_vs_best),
            "reward_delta_previous_vs_zero": _format_delta(delta_previous_vs_zero),
            "reward_delta_last_vs_previous_far_below_zero": delta_last_vs_previous_far_below_zero,
            "reward_delta_far_below_zero_threshold": f"{far_below_zero_threshold:.2f}",
        }

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
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                **self._build_reward_delta_context(episode_reward_buffer),
            }
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
        decomposition_type="lu",
        factor_names=None,
        frozen_factor=None,
        schedule_context=None,
        reward_context=None,
    ):
        self.reset_llm_conversation()

        schedule_context = schedule_context or {}
        reward_context = reward_context or {}
        force_new_matrix_exploration = bool(
            reward_context.get("force_new_matrix_exploration", False)
        )
        force_new_matrix_threshold = reward_context.get("force_new_matrix_threshold", -100.0)
        force_new_matrix_index_delta = reward_context.get("force_new_matrix_index_delta", 0.35)
        force_new_matrix_reference_count = reward_context.get("force_new_matrix_reference_count", 5)
        reward_dip_reset_to_best_active = bool(
            reward_context.get("reward_dip_reset_to_best_active", False)
        )
        reward_dip_reset_threshold = reward_context.get("reward_dip_reset_threshold", -200.0)
        reward_dip_latest_reward = reward_context.get("reward_dip_latest_reward")
        reward_dip_best_reward = reward_context.get("reward_dip_best_reward")
        matrix_delta_soft_signal_enabled = bool(
            reward_context.get("matrix_delta_soft_signal_enabled", False)
        )
        matrix_delta_soft_signal_active = bool(
            reward_context.get("matrix_delta_soft_signal_active", False)
        )
        matrix_delta_soft_limit = reward_context.get("matrix_delta_soft_limit")
        matrix_delta_soft_exceed_count = reward_context.get("matrix_delta_soft_exceed_count")
        matrix_delta_soft_max_abs_delta = reward_context.get("matrix_delta_soft_max_abs_delta")
        matrix_warning_signal_active = bool(
            reward_context.get("matrix_warning_signal_active", False)
        )
        matrix_warning_signal_notes = reward_context.get("matrix_warning_signal_notes") or []
        matrix_invalid_reset_active = bool(
            reward_context.get("matrix_invalid_reset_active", False)
        )
        matrix_invalid_reset_reason = reward_context.get("matrix_invalid_reset_reason")
        svd_reset_active = bool(reward_context.get("svd_reset_active", False))
        svd_reset_latest_reward = reward_context.get("svd_reset_latest_reward")
        svd_reset_sigma = bool(reward_context.get("svd_reset_sigma", False))
        svd_reset_uv = bool(reward_context.get("svd_reset_uv", False))
        svd_sigma_reset_threshold = reward_context.get("svd_sigma_reset_threshold", -100.0)
        svd_uv_reset_threshold = reward_context.get("svd_uv_reset_threshold", -1000.0)
        factor_value_bound = 6.0
        if decomposition_type != "svd" and factor_rank is not None and factor_rank > 0:
            factor_value_bound = math.sqrt(6.0 / float(factor_rank))

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                **self._build_reward_delta_context(episode_reward_buffer),
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
                "dim_state": dim_state,
                "dim_action": dim_action,
                "factor_rank": factor_rank,
                "factor_value_bound": factor_value_bound,
                "decomposition_type": decomposition_type,
                "factor_names": factor_names or [],
                "frozen_factor": frozen_factor,
                "lu_schedule_enabled": bool(schedule_context.get("enabled", False)),
                "lu_schedule_phase": schedule_context.get("phase"),
                "lu_schedule_l_episodes": schedule_context.get("l_episodes"),
                "lu_schedule_u_iterations": schedule_context.get("u_iterations"),
                "lu_schedule_cycle_step": schedule_context.get("cycle_step"),
                "lu_schedule_cycle_length": schedule_context.get("cycle_length"),
                "last_reward": reward_context.get("last_reward"),
                "delta_from_prev_reward": reward_context.get("delta_from_prev_reward"),
                "delta_from_best_reward": reward_context.get("delta_from_best_reward"),
                "delta_from_zero_reward": reward_context.get("delta_from_zero_reward"),
                "distance_below_zero": reward_context.get("distance_below_zero"),
                "delta_toward_zero_from_prev": reward_context.get("delta_toward_zero_from_prev"),
                "force_new_matrix_exploration": force_new_matrix_exploration,
                "force_new_matrix_threshold": force_new_matrix_threshold,
                "force_new_matrix_index_delta": force_new_matrix_index_delta,
                "force_new_matrix_reference_count": force_new_matrix_reference_count,
                "reward_dip_reset_to_best_active": reward_dip_reset_to_best_active,
                "reward_dip_reset_threshold": reward_dip_reset_threshold,
                "reward_dip_latest_reward": reward_dip_latest_reward,
                "reward_dip_best_reward": reward_dip_best_reward,
                "matrix_delta_soft_signal_enabled": matrix_delta_soft_signal_enabled,
                "matrix_delta_soft_signal_active": matrix_delta_soft_signal_active,
                "matrix_delta_soft_limit": matrix_delta_soft_limit,
                "matrix_delta_soft_exceed_count": matrix_delta_soft_exceed_count,
                "matrix_delta_soft_max_abs_delta": matrix_delta_soft_max_abs_delta,
                "matrix_warning_signal_active": matrix_warning_signal_active,
                "matrix_warning_signal_notes": matrix_warning_signal_notes,
                "matrix_invalid_reset_active": matrix_invalid_reset_active,
                "matrix_invalid_reset_reason": matrix_invalid_reset_reason,
                "svd_reset_active": svd_reset_active,
                "svd_reset_latest_reward": svd_reset_latest_reward,
                "svd_reset_sigma": svd_reset_sigma,
                "svd_reset_uv": svd_reset_uv,
                "svd_sigma_reset_threshold": svd_sigma_reset_threshold,
                "svd_uv_reset_threshold": svd_uv_reset_threshold,
            }
        )

        if reward_dip_reset_to_best_active:
            try:
                dip_threshold_text = f"{float(reward_dip_reset_threshold):.2f}"
            except (TypeError, ValueError):
                dip_threshold_text = "-200.00"
            try:
                latest_reward_text = f"{float(reward_dip_latest_reward):.2f}"
            except (TypeError, ValueError):
                latest_reward_text = "N/A"
            try:
                best_reward_text = f"{float(reward_dip_best_reward):.2f}"
            except (TypeError, ValueError):
                best_reward_text = "N/A"

            system_prompt += (
                "\n\n[REWARD DIP RESET BASELINE]\n"
                f"Latest reward ({latest_reward_text}) dropped below {dip_threshold_text}.\n"
                f"Policy was reset to best-so-far baseline (reward={best_reward_text}).\n"
                "Build off this baseline: preserve core structure that likely helped, then make measured exploratory edits.\n"
                "Do not jump to unrelated random structures unless repeatedly failing."
            )

        if svd_reset_active:
            try:
                svd_latest_reward_text = f"{float(svd_reset_latest_reward):.2f}"
            except (TypeError, ValueError):
                svd_latest_reward_text = "N/A"

            try:
                sigma_threshold_text = f"{float(svd_sigma_reset_threshold):.2f}"
            except (TypeError, ValueError):
                sigma_threshold_text = "-100.00"

            try:
                uv_threshold_text = f"{float(svd_uv_reset_threshold):.2f}"
            except (TypeError, ValueError):
                uv_threshold_text = "-1000.00"

            reset_parts = []
            if svd_reset_sigma:
                reset_parts.append("S was reset to near-zero")
            if svd_reset_uv:
                reset_parts.append("U and Vt were reset to near-zero")
            reset_text = "; ".join(reset_parts) if reset_parts else "No factor reset details available"

            system_prompt += (
                "\n\n[SVD LOW-REWARD RESET]\n"
                f"Latest reward was {svd_latest_reward_text}. Thresholds: sigma<= {sigma_threshold_text}, U/Vt<= {uv_threshold_text}.\n"
                f"Applied reset: {reset_text}.\n"
                "The baseline factors shown in the examples are already reset. Use them as the starting point for this proposal."
            )

        if matrix_delta_soft_signal_enabled and matrix_delta_soft_signal_active:
            try:
                limit_text = f"{float(matrix_delta_soft_limit):.4f}"
            except (TypeError, ValueError):
                limit_text = "N/A"
            try:
                exceed_text = str(int(matrix_delta_soft_exceed_count))
            except (TypeError, ValueError):
                exceed_text = "N/A"
            try:
                max_delta_text = f"{float(matrix_delta_soft_max_abs_delta):.4f}"
            except (TypeError, ValueError):
                max_delta_text = "N/A"

            system_prompt += (
                "\n\n[SOFT DELTA-LIMIT SIGNAL]\n"
                "The last proposal exceeded preferred per-entry delta bounds.\n"
                f"Preferred |delta| <= {limit_text}; exceeded entries={exceed_text}; max observed |delta|={max_delta_text}.\n"
                "No hard clipping is applied. Use this as guidance to smooth and coordinate updates while still exploring."
            )

        if matrix_warning_signal_active and len(matrix_warning_signal_notes) > 0:
            warning_lines = []
            for note in matrix_warning_signal_notes[:10]:
                warning_lines.append(f"- {note}")
            warning_block = "\n".join(warning_lines)
            system_prompt += (
                "\n\n[MATRIX WARNING SIGNALS]\n"
                "Recent matrix diagnostics detected potentially problematic structure patterns.\n"
                "Treat these as warnings and adjust your next proposal accordingly (no hard invalidation from these alone).\n"
                f"{warning_block}"
            )

        if matrix_invalid_reset_active:
            reason_text = str(matrix_invalid_reset_reason or "Invalid matrix proposal.")
            system_prompt += (
                "\n\n[INVALID MATRIX RESET]\n"
                "The previous proposal was invalid and the baseline was reset to near-zero values before this retry.\n"
                f"Invalidation reason: {reason_text}\n"
                "Only bad-shape proposals and exact full-matrix duplicates are invalidated."
            )

        if force_new_matrix_exploration:
            try:
                threshold_text = f"{float(force_new_matrix_threshold):.2f}"
            except (TypeError, ValueError):
                threshold_text = "-100.00"

            try:
                index_delta_text = f"{float(force_new_matrix_index_delta):.2f}"
            except (TypeError, ValueError):
                index_delta_text = "0.35"

            try:
                reference_count_text = str(int(force_new_matrix_reference_count))
            except (TypeError, ValueError):
                reference_count_text = "5"

            system_prompt += (
                "\n\n[MANDATORY EXPLORATION RESET]\n"
                f"Latest reward is below {threshold_text}.\n"
                "Do not exploit prior matrices/factors and do not make incremental edits.\n"
                "You must produce a completely new proposal with a different value/sign structure.\n"
                f"Strict uniqueness rule: compared to each of the last {reference_count_text} attempts, "
                f"every editable index must change by at least {index_delta_text} in absolute value."
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
                **self._build_reward_delta_context(episode_reward_buffer),
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
                **self._build_reward_delta_context(episode_reward_buffer),
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
                **self._build_reward_delta_context(episode_reward_buffer),
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
                **self._build_reward_delta_context(episode_reward_buffer),
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
                **self._build_reward_delta_context(episode_reward_buffer),
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
                **self._build_reward_delta_context(episode_reward_buffer),
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
