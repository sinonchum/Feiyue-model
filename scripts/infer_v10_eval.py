"""
Run inference on V10 test set using the merged LoRA model (8-bit).
Saves responses for Gemini scoring.
"""
import json, torch, time, sys
sys.stdout.reconfigure(line_buffering=True)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MERGED_MODEL = r"C:\Users\simon\Feiyue-model\outputs\feiyue-v10-merged"
TEST_FILE = r"C:\Users\simon\Feiyue-model\data\v10_final\test.jsonl"
OUTPUT_FILE = r"C:\Users\simon\Feiyue-model\data\v10_test_responses.jsonl"
MAX_NEW_TOKENS = 1024

def load_model():
    print(f"Loading merged model: {MERGED_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded in 8-bit. VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    return model, tokenizer

def run_inference(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def main():
    model, tokenizer = load_model()

    # Load test items
    test_items = []
    with open(TEST_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                test_items.append(json.loads(line))
    print(f"Loaded {len(test_items)} test items", flush=True)

    results = []
    start_time = time.time()

    for i, item in enumerate(test_items):
        meta = item.get("metadata", {})
        task_id = meta.get("task_id", f"item-{i}")
        domain = meta.get("domain", "unknown")
        difficulty = meta.get("difficulty", "unknown")

        t0 = time.time()
        try:
            response = run_inference(model, tokenizer, item["messages"])
        except Exception as e:
            response = f"[ERROR: {e}]"
        elapsed = time.time() - t0

        results.append({
            "task_id": task_id,
            "domain": domain,
            "difficulty": difficulty,
            "response": response,
            "response_length": len(response),
            "elapsed_s": round(elapsed, 1),
        })

        if (i + 1) % 10 == 0 or i == len(test_items) - 1:
            avg_time = (time.time() - start_time) / (i + 1)
            eta_min = (len(test_items) - i - 1) * avg_time / 60
            print(f"  [{i+1}/{len(test_items)}] {task_id} ({domain}) - {elapsed:.1f}s | ETA: {eta_min:.0f}min", flush=True)

    # Save results
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    total_time = time.time() - start_time
    print(f"\nDone! {len(results)} items in {total_time/60:.1f} min", flush=True)
    print(f"Saved to: {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
