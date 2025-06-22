from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest
import time
import torch


class VLLMEngineBenchmark:
    def __init__(self, loras: list[str], engine_args: EngineArgs):
        self.engine_args = engine_args
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.loras = loras

    def process_requests(self, requests: list):
        requests_impl = [
            (
                request[0], 
                SamplingParams(
                    max_tokens=request[1].max_new_tokens), 
                    LoRARequest(self.loras[request[1].lora_id], 1, self.loras[request[1].lora_id]) if request[1].lora_id is not None else None
            )
            for request in requests
        ]
        request_id = 0
        
        start_time = time.time()
        
        while request_id < len(requests_impl) or self.engine.has_unfinished_requests():
            if request_id < len(requests_impl):
                prompt, sampling_params, lora_request = requests_impl[request_id]
                self.engine.add_request(str(request_id),
                                prompt,
                                sampling_params,
                                lora_request=lora_request)
                request_id += 1
                
            else:
                request_outputs = self.engine.step()
        
        torch.cuda.synchronize()

        end_time = time.time()
        return end_time - start_time
