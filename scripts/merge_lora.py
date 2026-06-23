"""
Merge LoRA adapter into base model and save to disk.
This avoids OOM during inference by loading a single merged model.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = "outputs/feiyue-qwen3-4b-worker/checkpoint-21"
MERGED_PATH = "outputs/feiyue-qwen3-4b-merged"

print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

print(f"Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging LoRA weights...")
model = model.merge_and_unload()

print(f"Saving merged model to: {MERGED_PATH}")
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print("Done! Merged model saved.")
