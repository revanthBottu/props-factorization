import gymnasium as gym
import random
import numpy as np
import os
import time
from jinja2 import Template
# from openai import OpenAI
from dotenv import load_dotenv
from google import genai
# import anthropic
import time
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
        
        # Detect if this is a HuggingFace model (contains / indicating org/model format)
        # HuggingFace paths: "Qwen/Qwen3.5-72B-Instruct", "meta-llama/Llama-3.1-70B", etc.
        is_hf_model = "/" in llm_model_name
        
        # Detect if this is an Ollama model (model:version format or common local model names)
        # Ollama examples: "llama3:70b", "qwen:14b", "mistral:latest"
        is_ollama_model = (not is_hf_model) and (":" in llm_model_name or any(llm_model_name.lower().startswith(prefix) for prefix in 
                                                       ["llama", "qwen", "mistral", "phi", "gemma", "codellama"]))
        
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
            "gemini-2.5-flash-lite",
            "o3-mini-2025-01-31",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "claude-3-7-sonnet-20250219",
        ]
        
        if not is_hf_model and not is_ollama_model:
            assert llm_model_name in known_models, f"Unknown model: {llm_model_name}. Use a known model, an Ollama model, or a HuggingFace model."
        
        self.llm_model_name = llm_model_name
        
        # Setup model group and client
        if is_hf_model:
            if not HUGGINGFACE_AVAILABLE:
                raise ImportError("HuggingFace transformers not installed. Install with: pip install transformers bitsandbytes accelerate")
            self.model_group = "huggingface"
            
            print(f"Loading HuggingFace model: {llm_model_name}")
            
            # Load tokenizer
            self.hf_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
            
            # Detect if model is pre-quantized (FP8, GPTQ, AWQ, Mxfp4, or gpt-oss)
            # gpt-oss uses Mxfp4 quantization
            is_prequantized = any(x in llm_model_name.lower() for x in ['fp8', 'gptq', 'awq', 'int4', 'int8', 'gpt-oss', '-oss'])
            
            if is_prequantized:
                print(f"Detected pre-quantized model (Mxfp4/other), loading without additional quantization")
                print(f"Enabling CPU offloading: 60GB GPU + 350GB RAM available")
                # Load pre-quantized model with CPU offloading
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    dtype="auto",
                    low_cpu_mem_usage=True,
                    max_memory={0: "60GiB", "cpu": "350GiB"},  # Reduced GPU to leave room for generation
                    offload_folder="offload",
                    offload_state_dict=True
                )
                print("Successfully loaded pre-quantized model with CPU offloading enabled")
            else:
                print(f"Loading model with 4-bit quantization (required for 120B+ models)")
                print(f"Enabling CPU offloading: 60GB GPU + 350GB RAM available")
                # Configure 4-bit quantization for large models
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load model with quantization and CPU offloading
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    llm_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "60GiB", "cpu": "350GiB"},
                    offload_folder="offload"
                )
                print("Successfully loaded with 4-bit quantization and CPU offloading")
            
            # Remove unsupported generation parameters to avoid warnings
            if hasattr(self.hf_model, 'generation_config'):
                self.hf_model.generation_config.top_p = None
                self.hf_model.generation_config.top_k = None
            
            # Set pad token if not present
            if self.hf_tokenizer.pad_token is None:
                self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token
            
            quant_status = "pre-quantized" if is_prequantized else "4-bit quantization"
            print(f"Successfully loaded HuggingFace model with {quant_status}")
            
            # Track if model is on CUDA
            self.use_cuda = torch.cuda.is_available() and next(self.hf_model.parameters()).is_cuda
        elif is_ollama_model:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama package not installed. Install with: pip install ollama")
            self.model_group = "ollama"
            self.ollama_client = ollama.Client()
            print(f"Using Ollama model: {llm_model_name}")
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

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def clear_cache(self):
        """Clear GPU cache to prevent OOM between episodes."""
        if self.use_cuda:
            torch.cuda.empty_cache()
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
                    # Use HuggingFace transformers
                    # Format conversation for chat template
                    prompt = self._format_hf_chat()
                    
                    # Tokenize and generate
                    inputs = self.hf_tokenizer(prompt, return_tensors="pt", padding=True).to(self.hf_model.device)
                    
                    with torch.no_grad():
                        outputs = self.hf_model.generate(
                            **inputs,
                            max_new_tokens=384,  # Reduced for memory - still enough for LU matrices
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=self.hf_tokenizer.pad_token_id,
                            use_cache=True
                        )
                    
                    # Decode only the new tokens
                    response = self.hf_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    
                    # Immediate memory cleanup to prevent OOM
                    del inputs, outputs
                    if self.use_cuda:
                        torch.cuda.empty_cache()
                elif self.model_group == "ollama":
                    # Use Ollama for local models
                    response = self.ollama_client.chat(
                        model=self.llm_model_name,
                        messages=self.llm_conversation
                    )
                    response = response['message']['content']
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
                    print("Gemini charging up...")
                    time.sleep(60)
                    continue

            # if self.model_group == "openai":
            #     # add the response to self.llm_conversation
            #     self.add_llm_conversation(response, "assistant")
            # else:
            if self.model_group == "gemini":
                self.add_llm_conversation(response, "model")
            elif self.model_group == "huggingface":
                self.add_llm_conversation(response, "assistant")
            elif self.model_group == "ollama":
                self.add_llm_conversation(response, "assistant")

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
                    # Generate multiple responses with HuggingFace
                    responses = []
                    prompt = self._format_hf_chat()
                    inputs = self.hf_tokenizer(prompt, return_tensors="pt", padding=True).to(self.hf_model.device)
                    
                    for _ in range(num_responses):
                        with torch.no_grad():
                            outputs = self.hf_model.generate(
                                **inputs,
                                max_new_tokens=384,  # Reduced for memory
                                do_sample=True,
                                temperature=temperature,
                                pad_token_id=self.hf_tokenizer.pad_token_id,
                                use_cache=True
                            )
                        response = self.hf_tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                        responses.append(response)
                        
                        # Clean up after each response
                        del outputs
                        if self.use_cuda:
                            torch.cuda.empty_cache()
                elif self.model_group == "ollama":
                    # Generate multiple responses with Ollama
                    responses = []
                    for _ in range(num_responses):
                        response = self.ollama_client.chat(
                            model=self.llm_model_name,
                            messages=self.llm_conversation,
                            options={"temperature": temperature}
                        )
                        responses.append(response['message']['content'])

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

        if self.model_group == "openai":
            self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
        else:
            self.add_llm_conversation(new_parameters_with_reasoning, "model")
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

        self.add_llm_conversation(new_parameters_with_reasoning, "assistant")
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
        lu_rank=None,
        use_lu=False,
    ):
        self.reset_llm_conversation()

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
                "lu_rank": lu_rank,
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
