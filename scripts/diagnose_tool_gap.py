"""Ask Gemini 3.1 Pro to diagnose why model doesn't use <tool_call> during inference."""
import json
import subprocess
import urllib.request
import urllib.error

result = subprocess.run(
    ["C:/Users/simon/google-cloud-sdk/bin/gcloud.cmd", "auth", "application-default", "print-access-token"],
    capture_output=True, text=True
)
token = result.stdout.strip()

# Read training data samples
with open("data/train.jsonl") as f:
    train_lines = f.readlines()

# Extract 2 full training samples with <tool_call>
samples = []
for line in train_lines:
    item = json.loads(line)
    has_tool = any('<tool_call>' in m.get('content','') for m in item['messages'] if m['role']=='assistant')
    if has_tool and len(samples) < 2:
        # Show system + first 3 turns
        truncated = []
        for m in item['messages'][:6]:
            truncated.append({
                'role': m['role'],
                'content': m['content'][:400]
            })
        samples.append({
            'task_id': item.get('metadata',{}).get('task_id','?'),
            'messages': truncated
        })

# Read 1 eval sample (what model actually outputs)
with open("data/eval_coding_responses.jsonl") as f:
    eval_data = [json.loads(l) for l in f.readlines()]

prompt = f"""I have a fine-tuned Qwen3-4B-Instruct model trained as a Feiyue worker agent.
The training data contains 70% `<tool_call>` XML blocks, yet the model outputs 
direct markdown code blocks during inference. I need to diagnose why.

TRAINING DATA SAMPLES (model learns from these):
{json.dumps(samples, indent=2)}

INFERENCE RESPONSES (model outputs these instead of <tool_call>):
{json.dumps([
    {'task_id': r['task_id'], 'response_first_400': r['response'][:400]}
    for r in eval_data[:3]
], indent=2)}

The formatting function used during SFT training was:
```
def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return text
```

The inference code uses:
```
tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
```

Questions:
1. What is the most likely root cause of the model NOT using <tool_call> during inference?
2. Is it the formatting_func vs inference config mismatch? The enable_thinking flag?
3. How should I fix this - retrain with different config, or fix the inference setup?
4. Give a concrete, actionable step-by-step fix.

Return detailed analysis with evidence from the samples."""

url = "https://aiplatform.googleapis.com/v1/projects/project-7c8d85fa-4f69-41a0-abb/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"

payload = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192}
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("Diagnosing train/inference gap with Gemini 3.1 Pro...", flush=True)
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode())
    text = result["candidates"][0]["content"].get("parts", [{"text": "(no text, model only produced thoughts — increase maxOutputTokens)"}])[0].get("text", "")
    if not text:
        print("WARNING: No text parts in response. Model may have only produced thoughts.")
        # Try to get thinking tokens as fallback
        thoughts_count = result.get("usageMetadata", {}).get("thoughtsTokenCount", 0)
        print(f"Thoughts tokens: {thoughts_count}")
        print(json.dumps(result, indent=2)[:3000])
    print("\n--- Gemini 3.1 Pro Diagnosis ---\n")
    print(text)
    with open("data/gemini_training_gap_diagnosis.txt", "w") as f:
        f.write(text)
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP Error {e.code}: {body[:500]}")
