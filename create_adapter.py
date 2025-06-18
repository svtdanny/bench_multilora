import torch, random, os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "unsloth/Llama-3.2-3B"          # local or HF repo
save_dir = "adapters_3b"
tok  = AutoTokenizer.from_pretrained(model_name)
base = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_4bit=True,                    # remove if you have >20 GB VRAM
        device_map="auto")

def make_adapter(seed:int, steps:int=0, dataset="wikitext", split="train[:1%]"):
    torch.manual_seed(seed)
    cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=8, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                        "gate_proj","up_proj","down_proj"]
    )

    m = get_peft_model(base, cfg)
    m = prepare_model_for_kbit_training(m)     # keeps gradients in fp32 if 4-bit

    if steps:  # quick & dirty training (optional)
        ds = load_dataset(dataset, split=split)
        optim = torch.optim.AdamW(m.parameters(), lr=1e-4)
        for step, ex in zip(range(steps), ds):
            batch = tok(ex["text"], return_tensors="pt").to(m.device)
            loss  = m(**batch, labels=batch["input_ids"]).loss
            loss.backward(); optim.step(); optim.zero_grad()

    out_dir = f"{save_dir}/seed-{seed}"
    os.makedirs(out_dir, exist_ok=True)
    m.save_pretrained(out_dir)
    print("✅ saved", out_dir)

for s in range(42, 42+8):        # as many as you need
    make_adapter(s, steps=0)      # put steps>0 if you want real training
