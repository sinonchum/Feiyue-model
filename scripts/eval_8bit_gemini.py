"""
Evaluate 8-bit inference results with Gemini 3.1 Pro.
"""
import json
import subprocess
import urllib.request
import urllib.error

# Get access token
result = subprocess.run(
    ["C:/Users/simon/google-cloud-sdk/bin/gcloud.cmd", "auth", "application-default", "print-access-token"],
    capture_output=True, text=True
)
token = result.stdout.strip()

# Load responses
responses = []
with open("data/eval_coding_responses.jsonl") as f:
    for line in f:
        responses.append(json.loads(line))

# Load ground truth
ground_truth = []
with open("data/eval_coding.jsonl") as f:
    for line in f:
        gt = json.loads(line)
        meta = gt.get("metadata", {})
        ground_truth.append({
            "task_id": meta.get("task_id", ""),
            "difficulty": meta.get("difficulty", ""),
            "tools_used": meta.get("tools_used", []),
        })

# Build eval items
eval_items = ""
for i, (r, gt) in enumerate(zip(responses, ground_truth)):
    eval_items += f"""
### Item {i+1}: {r['task_id']} (Difficulty: {r['difficulty']})
**Expected tool usage:** {', '.join(gt['tools_used'])}
**Model response ({r['response_length']} chars, {r['elapsed_s']}s):**
```
{r['response'][:2000]}
```

"""

prompt = f"""You are evaluating a fine-tuned Qwen3-4B model (8-bit quantized) trained as a Feiyue worker agent.
The model was trained to use Hermes tools (file_read, file_write, apply_patch, run_tests, search_files) 
to complete coding tasks via <tool_call> XML blocks.

Evaluate each of the 8 coding eval responses. For each item score:
1. **Format Compliance** (0-2): Does it use <tool_call> XML blocks correctly? Or does it output code directly (acceptable)?
2. **Task Understanding** (0-3): Does the response correctly understand and address the task?
3. **Code Quality** (0-3): Is the code correct, idiomatic, and complete?
4. **Tool Usage** (0-2): Does it use the expected tools appropriately?

{eval_items}

Return your evaluation as JSON:
{{
  "evaluations": [
    {{
      "task_id": "...",
      "format_compliance": 0-2,
      "task_understanding": 0-3,
      "code_quality": 0-3,
      "tool_usage": 0-2,
      "total": 0-10,
      "notes": "brief explanation"
    }}
  ],
  "summary": {{
    "easy_avg": "X/10",
    "medium_avg": "X/10",
    "hard_avg": "X/10",
    "overall_avg": "X/10",
    "strengths": ["..."],
    "weaknesses": ["..."],
    "recommendation": "..."
  }}
}}"""

# Call Gemini 3.1 Pro
url = "https://aiplatform.googleapis.com/v1/projects/project-7c8d85fa-4f69-41a0-abb/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"

payload = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("Sending 8-bit eval results to Gemini 3.1 Pro...", flush=True)
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode())
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    with open("data/gemini_8bit_eval.txt", "w") as f:
        f.write(text)
    print("\n--- Gemini 3.1 Pro Evaluation (8-bit) ---\n")
    print(text)
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP Error {e.code}: {body[:500]}")
