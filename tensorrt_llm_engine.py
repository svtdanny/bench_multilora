from tensorrt_llm import LLM, BuildConfig
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams
from tensorrt_llm.executor import LoRARequest
from tensorrt_llm.lora_manager import LoraConfig
import time
import torch


class TensorRTLLMEngineBenchmark:
    def __init__(self, loras: list[str], model_name: str, max_lora_rank: int = 8, offload_mem: int = 0):
        self.loras = loras
        self.model_name = model_name
        self.max_lora_rank = max_lora_rank
        self.offload_mem = offload_mem
        
        # Initialize TensorRT-LLM engine with LoRA support
        # We need at least one lora_dir to build the engine with LoRA support
        build_config = BuildConfig(max_seq_len=35000, max_input_len=35000, max_num_tokens=35000)
        build_config.lora_config = LoraConfig(lora_dir=[loras[0]] if loras else [])
        kv_cache_config = KvCacheConfig(host_cache_size=1024*1024*offload_mem) # need to convert to bytes
        
        self.llm = LLM(
            model=model_name,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            build_config=build_config,
            kv_cache_config=kv_cache_config,
        )

    def process_requests(self, requests: list):
        """
        Process a list of requests where each request contains:
        - request[0]: prompt text
        - request[1]: object with max_tokens and lora_id attributes
        """
        prompts = []
        lora_requests = []
        sampling_params = []
        
        # Prepare prompts and LoRA requests
        for request in requests:
            prompt = request[0]
            params = request[1]
            
            prompts.append(prompt)
            sampling_params.append(SamplingParams(max_tokens=params.max_new_tokens))
            
            if hasattr(params, 'lora_id') and params.lora_id is not None:
                if params.lora_id < len(self.loras):
                    lora_path = self.loras[params.lora_id]
                    # Create LoRARequest with unique name, id, and path
                    lora_req = LoRARequest(f"lora_{params.lora_id}", params.lora_id + 1, lora_path)
                    lora_requests.append(lora_req)
                else:
                    lora_requests.append(None)
            else:
                lora_requests.append(None)
        
        start_time = time.time()
        
        # Generate responses using TensorRT-LLM
        outputs = list(self.llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_requests
        ))
        
        torch.cuda.synchronize()
        
        end_time = time.time()
        return end_time - start_time 