from huggingface_hub import snapshot_download
from dataclasses import dataclass
import random
import argparse
from transformers import AutoTokenizer
import pandas as pd
from tqdm import tqdm
import gc
import torch
import os

def create_engine(engine_name: str, model_name: str, loras: list[str], offload_mem: int = 0):
    if engine_name == "vllm":
        from vllm_engine import VLLMEngineBenchmark
        from vllm import EngineArgs
        engine_args = EngineArgs(model=model_name,
                                enable_lora=True,
                                max_loras=8,
                                max_lora_rank=8,
                                cpu_offload_gb=offload_mem,
                                swap_space=offload_mem,
                                )
        return VLLMEngineBenchmark(loras, engine_args)
    elif engine_name == "trt_llm":
        from tensorrt_llm_engine import TensorRTLLMEngineBenchmark
        return TensorRTLLMEngineBenchmark(loras, model_name, max_lora_rank=8, offload_mem=offload_mem)
    elif engine_name == "peft":
        from peft_engine import PEFTEngineBenchmark
        return PEFTEngineBenchmark(loras, model_name)
    else:
        raise ValueError(f"Engine {engine_name} not supported")


def generate_requests_fixed_context(tokenizer, num_loras: int, num_requests: int, context_size: list[int], max_new_tokens: int, seed: int = 42):
    random.seed(seed)
    
    requests = []
    for i in range(num_requests):
        # Generate random token ids directly
        sampled_context_size = random.choice(context_size)
        token_ids = [random.randint(0, tokenizer.vocab_size-1) for _ in range(sampled_context_size)]
        # Decode to text
        text = tokenizer.decode(token_ids)
        
        # Alternate between LoRA 0 and 1
        lora_id = random.randint(0, num_loras-1)
        requests.append((text, BenchmarkSamplingParams(max_new_tokens=max_new_tokens, lora_id=lora_id)))
    
    return requests

@dataclass
class BenchmarkSamplingParams:
    max_new_tokens: int
    lora_id: int

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-LoRA inference benchmarking")
    
    parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-3B",
                        help="Model name to use for inference")
    parser.add_argument("--lora_path", type=str, nargs="+", 
                        default=["/home/jovyan/sivtsov/tight_inference/multilora/adapters_3b/seed-42"],
                        help="Path(s) to LoRA adapters")
    parser.add_argument("--engine_name", type=str, default="vllm",
                        choices=["vllm", "trt_llm", "peft"], help="Engine name to use")
    parser.add_argument("--num_requests", type=int, default=10,
                        help="Number of requests to generate")
    parser.add_argument("--context_size", type=int, nargs="+", default=[128],
                        help="Context size for each request")
    parser.add_argument("--max_new_tokens", type=int, default=20,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--warmup_iters", type=int, default=2,
                        help="Number of warmup iterations")
    parser.add_argument("--num_iters", type=int, default=5,
                        help="Number of benchmark iterations")
    
    args = parser.parse_args()
    return args

def measure_experiment(engine, requests, warmup_iters, num_iters):
    for _ in range(warmup_iters):
        _ = engine.process_requests(requests)
        
    times = []
    for _ in range(num_iters):
        iter_timing = engine.process_requests(requests)
        times.append(iter_timing)
    
    time = sum(times) / len(times)

    print("Raw timings: ", times)
    print(f"Time: {time} seconds, {len(requests)/time} requests/s, per request: {time/len(requests)} seconds")
    
    return time
    

def main():
    args = parse_args()

    model_name = args.model_name
    lora_path = args.lora_path
    engine_name = args.engine_name
    num_requests = args.num_requests
    context_size = args.context_size
    max_new_tokens = args.max_new_tokens
    warmup_iters = args.warmup_iters
    num_iters = args.num_iters
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    requests = generate_requests_fixed_context(tokenizer, len(lora_path), num_requests, context_size, max_new_tokens)
    
    # requests = [
    #     ("What is the capital of France?", BenchmarkSamplingParams(max_tokens=32, lora_id=0)),
    #     ("What is the capital of Germany?", BenchmarkSamplingParams(max_tokens=32, lora_id=0))
    # ]
    
    engine = create_engine(engine_name, model_name, lora_path)

    time = measure_experiment(engine, requests, warmup_iters, num_iters)
    return time

def fixed_loras_experiment(engines: list[str], context_size: list[int], max_new_tokens: int, num_loras: int, num_requests: int, warmup_iters: int, num_iters: int, offload_mem: int = 0):
    args = parse_args()

    model_name = args.model_name
    lora_path = args.lora_path
    lora_path = lora_path[:num_loras]
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = pd.DataFrame(index=engines, columns=context_size)
    
    for engine_name in engines:
        print(f"Engine {engine_name}")
        engine = create_engine(engine_name, model_name, lora_path, offload_mem=offload_mem)
        for ctx_size in tqdm(context_size):
            requests = generate_requests_fixed_context(tokenizer, len(lora_path), num_requests, [ctx_size], max_new_tokens)
            
            time = measure_experiment(engine, requests, warmup_iters, num_iters)
            results.loc[engine_name, ctx_size] = time
            print(f"Engine {engine_name}, context size {ctx_size}: {time} seconds")
            
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    
    print("\nResults DataFrame:")
    print(results)
    return results

def variable_loras_experiment(engines: list[str], context_size: list[int], max_new_tokens: int, num_loras: list[int], num_requests, warmup_iters, num_iters, offload_mem: int = 0):    
    args = parse_args()
    
    model_name = args.model_name
    lora_path = args.lora_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = pd.DataFrame(index=engines, columns=num_loras)
    
    for engine_name in engines:
        print(f"Engine {engine_name}")
        for num_lora in tqdm(num_loras):
            engine = create_engine(engine_name, model_name, lora_path[:num_lora], offload_mem)
            requests = generate_requests_fixed_context(tokenizer, num_lora, num_requests, context_size, max_new_tokens)

            time = measure_experiment(engine, requests, warmup_iters, num_iters)
            results.loc[engine_name, num_lora] = time
            print(f"Engine {engine_name}, num loras {num_lora}: {time} seconds")
            
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    
    # print("\nResults variable_loras_experiment DataFrame:")
    # print(results)
    return results

def variable_num_requests_experiment(engines: list[str], context_size: list[int], max_new_tokens: int, num_loras: int, num_requests: list[int], warmup_iters, num_iters, offload_mem: int = 0):    
    args = parse_args()
    
    model_name = args.model_name
    lora_path = args.lora_path
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    results = pd.DataFrame(index=engines, columns=num_requests)
    
    for engine_name in engines:
        print(f"Engine {engine_name}")
        engine = create_engine(engine_name, model_name, lora_path[:num_loras], offload_mem=offload_mem)
        for num_req in tqdm(num_requests):
            requests = generate_requests_fixed_context(tokenizer, num_loras, num_req, context_size, max_new_tokens)
            print([len(req[0]) for req in requests])

            time = measure_experiment(engine, requests, warmup_iters, num_iters)
            results.loc[engine_name, num_req] = time
            print(f"Engine {engine_name}, num requests {num_req}: {time} seconds")
        
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    
    # print("\nResults variable_num_requests_experiment DataFrame:")
    # print(results)
    return results

def exp_1_fixed_context(engines, offload_mem: int = 0):
    context_size = [1000, 2000, 4000, 8000, 16000, 32000][:CUT_CONTEXT_TO]
    max_new_tokens = 500
    num_requests = 20
    warmup_iters = 1
    num_iters = 1
    num_loras = 4
    
    results = fixed_loras_experiment(engines, context_size, max_new_tokens, num_loras, num_requests, warmup_iters, num_iters, offload_mem=offload_mem)
    if offload_mem == 0:
        print("Result exp_1_context:")
        print(results)
        results.to_csv("exp_1_context.csv")
    else:
        print(f"Result exp_1_context_offload_mem_{offload_mem}:")
        print(results)
        results.to_csv(f"{PREFIX}exp_1_context_offload_mem_{offload_mem}.csv")
    
def exp_2_num_requests_random_context(engines, offload_mem: int = 0):
    context_size = [1000, 2000, 4000, 8000, 16000, 32000][:CUT_CONTEXT_TO]
    max_new_tokens = 500
    num_requests = [1, 2, 4, 8, 12, 16, 32][:CUT_NUM_REQUESTS_TO]
    warmup_iters = 1
    num_iters = 1
    num_loras = 4
    
    results = variable_num_requests_experiment(engines, context_size, max_new_tokens, num_loras, num_requests, warmup_iters, num_iters, offload_mem=offload_mem)
    if offload_mem == 0:
        print("Result exp_2_num_requests:")
        print(results)
        results.to_csv("exp_2_num_requests.csv")
    else:
        print(f"Result exp_2_num_requests_offload_mem_{offload_mem}:")
        print(results)
        results.to_csv(f"{PREFIX}exp_2_num_requests_offload_mem_{offload_mem}.csv")
    
def exp_3_num_requests_fixed_context(engines, context_size, offload_mem: int = 0):
    max_new_tokens = 500
    num_requests = [1, 2, 4, 8, 12, 16, 32][:CUT_NUM_REQUESTS_TO]
    warmup_iters = 1
    num_iters = 1
    num_loras = 4
    
    results = variable_num_requests_experiment(engines, [context_size], max_new_tokens, num_loras, num_requests, warmup_iters, num_iters, offload_mem=offload_mem)
    if offload_mem == 0:
        print(f"Result exp_3_num_requests_fixed_context_{context_size}:")
        print(results)
        results.to_csv(f"{PREFIX}exp_3_num_requests_fixed_context_{context_size}.csv")
    else:
        print(f"Result exp_3_num_requests_fixed_context_{context_size}_offload_mem_{offload_mem}:")
        print(results)
        results.to_csv(f"{PREFIX}exp_3_num_requests_fixed_context_{context_size}_offload_mem_{offload_mem}.csv")
    
def exp_4_loras(engines, context_size, offload_mem: int = 0):
    max_new_tokens = 500
    num_loras = [1, 2, 4, 8]
    num_requests = 20
    warmup_iters = 1
    num_iters = 1

    results = variable_loras_experiment(engines, context_size, max_new_tokens, num_loras, num_requests, warmup_iters, num_iters, offload_mem=offload_mem)
    if offload_mem == 0:
        print("Result exp_4_loras:")
        print(results)
        results.to_csv(f"{PREFIX}exp_4_loras.csv")
    else:
        print(f"Result exp_4_loras_offload_mem_{offload_mem}:")
        print(results)
        results.to_csv(f"{PREFIX}exp_4_loras_offload_mem_{offload_mem}.csv")

if __name__ == "__main__":
    # time = main()
    
    GPU_NAME = "a100"
    MODEL = "3B"
    # MODEL = "7B"
    
    PREFIX = ""
    
    CUT_CONTEXT_TO = 6
    CUT_NUM_REQUESTS_TO = 7

    # engines = ["peft", "vllm"][::-1]  
    # engines = ["peft"]
    # engines = ["vllm"]
    engines = ["trt_llm"]
    # engines = ["peft", "vllm", "trt_llm"]
    # FULL TEST
    exp_1_fixed_context(engines)
    exp_2_num_requests_random_context(engines)
    exp_3_num_requests_fixed_context(engines, 8000)
    exp_3_num_requests_fixed_context(engines, 16000)
    exp_4_loras(engines, [16000]) # Run this one time. Its not really relevant, if multilora support is implemented.

    # OFFLOADING
    exp_1_fixed_context(engines, offload_mem=100)
    exp_2_num_requests_random_context(engines, offload_mem=100)
    # exp_3_num_requests_fixed_context(engines, 8000, offload_mem=100)
    # exp_4_loras(engines, [8000], offload_mem=100)
