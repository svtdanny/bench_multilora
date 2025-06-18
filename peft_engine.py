from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import time
import torch


class PEFTEngineBenchmark:
    def __init__(self, loras: list[str], model_name: str, device: str = "cuda"):
        self.loras = loras
        self.model_name = model_name
        self.device = device
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device
        )
        
        # Load the first adapter as the base PEFT model
        if loras:
            self.peft_model = PeftModel.from_pretrained(
                self.base_model, 
                loras[0], 
                adapter_name=f"0"
            )
            
            # Load additional adapters
            for i, lora_path in enumerate(loras[1:], 1):
                self.peft_model.load_adapter(lora_path, adapter_name=f"{i}")
        else:
            self.peft_model = self.base_model
            
        self.current_adapter = None

    def _switch_adapter(self, lora_id: int):
        """Switch to the specified adapter if it's different from current."""
        adapter_name = f"{lora_id}"
        if self.current_adapter != adapter_name:
            self.peft_model.set_adapter(adapter_name)
            self.current_adapter = adapter_name

    def _generate_single(self, prompt: str, max_new_tokens: int = 128):
        """Generate text for a single prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (excluding input)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return generated_text

    def process_requests(self, requests: list):
        """
        Process a list of requests where each request contains:
        - request[0]: prompt text
        - request[1]: object with max_tokens and lora_id attributes
        """
        start_time = time.time()
        
        for request in requests:
            prompt = request[0]
            params = request[1]
            
            # Switch to the appropriate adapter
            lora_id = getattr(params, 'lora_id', None)
            self._switch_adapter(lora_id)
            
            # Generate response
            max_new_tokens = getattr(params, 'max_new_tokens')
            generated_text = self._generate_single(prompt, max_new_tokens)
        
        torch.cuda.synchronize()
        
        end_time = time.time()
        return end_time - start_time
