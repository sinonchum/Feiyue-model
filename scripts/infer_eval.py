"""
Run inference on the 8-item coding eval set using the merged LoRA model.
Uses 8-bit quantization (Gemini 3.1 Pro recommended) for fair evaluation.
"""
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MERGED_MODEL = "outputs/feiyue-qwen3-4b-merged"
EVAL_FILE = "data/eval_coding.jsonl"
OUTPUT_FILE = "data/eval_coding_responses.jsonl"
MAX_NEW_TOKENS = 1024

def load_model():
    print(f"Loading merged model: {MERGED_MODEL}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Gemini 3.1 Pro recommended: 8-bit for coding eval (preserves >99% fidelity)
    # llm_int8_threshold=6.0 keeps outlier weights in FP16
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
    print("Model loaded in 8-bit!", flush=True)
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
    eval_items = []
    with open(EVAL_FILE) as f:
        for line in f:
            eval_items.append(json.loads(line))

    print(f"\nRunning inference on {len(eval_items)} eval items...\n", flush=True)
    results = []
    for i, item in enumerate(eval_items):
        messages = item["messages"]
        metadata = item.get("metadata", {})
        task_id = metadata.get("task_id", f"item-{i}")
        difficulty = metadata.get("difficulty", "unknown")

        prompt_messages = []
        for msg in messages:
            prompt_messages.append(msg)
            if msg["role"] == "user":
                break

        print(f"[{i+1}/{len(eval_items)}] {task_id} ({difficulty})...", end=" ", flush=True)
        start = time.time()
        response = run_inference(model, tokenizer, prompt_messages)
        elapsed = time.time() - start
        print(f"done ({elapsed:.1f}s, {len(response)} chars)", flush=True)

        results.append({
            "task_id": task_id,
            "difficulty": difficulty,
            "prompt": prompt_messages[-1]["content"][:300],
            "response": response,
            "response_length": len(response),
            "elapsed_s": round(elapsed, 1),
        })

    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} responses to {OUTPUT_FILE}", flush=True)
    print("\n--- Quick Summary ---", flush=True)
    for r in results:
        preview = r['response'][:300].replace('\n', '\\n')
        print(f"  [{r['difficulty']}] {r['task_id']}: {r['response_length']} chars, {r['elapsed_s']}s", flush=True)
        print(f"    → {preview}", flush=True)
        print(flush=True)

if __name__ == "__main__":
    main()
