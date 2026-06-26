"""
Gemini 3.1 Pro evaluation of V10 inference results.
Stratified sampling across 3 domains with domain-aware scoring.
"""
import json, time, urllib.request, urllib.error, os, random

# Paths
RESPONSES_FILE = r"C:\Users\simon\Feiyue-model\data\v10_test_responses.jsonl"
TEST_FILE = r"C:\Users\simon\Feiyue-model\data\v10_final\test.jsonl"
TOKEN_FILE = r"C:\Users\simon\gcloud_adc_token.txt"
OUTPUT_FILE = r"C:\Users\simon\Feiyue-model\data\v10_gemini_eval.json"

SAMPLE_PER_DOMAIN = 10  # 30 total
MAX_RESPONSE_CHARS = 1500

# --- Load token ---
def get_token():
    with open(TOKEN_FILE) as f:
        return f.read().strip()

# --- Load data ---
responses = []
with open(RESPONSES_FILE, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            responses.append(json.loads(line))
print(f"Loaded {len(responses)} responses")

test_items = []
with open(TEST_FILE, encoding='utf-8') as f:
    for line in f:
        if line.strip():
            test_items.append(json.loads(line))
print(f"Loaded {len(test_items)} test items")

# Build lookup: task_id -> {messages, metadata}
test_lookup = {}
for item in test_items:
    tid = item["metadata"]["task_id"]
    test_lookup[tid] = item

# --- Stratified sampling ---
by_domain = {}
for r in responses:
    d = r.get("domain", "unknown")
    by_domain.setdefault(d, []).append(r)

samples = []
for domain, items in by_domain.items():
    n = min(SAMPLE_PER_DOMAIN, len(items))
    sampled = random.Random(42).sample(items, n)
    samples.extend(sampled)
    print(f"  {domain}: {n}/{len(items)} sampled")

print(f"Total sample: {len(samples)} items")

# --- Build Gemini prompt ---
eval_blocks = []
for i, r in enumerate(samples):
    tid = r["task_id"]
    gt = test_lookup.get(tid, {})
    meta = gt.get("metadata", {})
    domain = r["domain"]

    # Get the original user request from test data
    user_msg = ""
    for msg in gt.get("messages", []):
        if msg.get("role") == "user":
            user_msg = msg["content"][:500]
            break

    eval_blocks.append(f"""
### Item {i+1}: {tid} | Domain: {domain} | Difficulty: {r['difficulty']}

**Task:**
{user_msg}

**Model response ({r['response_length']} chars, {r['elapsed_s']}s):**
```
{r['response'][:MAX_RESPONSE_CHARS]}
```
""")

eval_text = "\n".join(eval_blocks)

prompt = f"""You are evaluating a fine-tuned Qwen3-4B model trained as a multi-capability AI agent. The model was trained on 3 domains: code (write/implement), prd (product requirements), code_review (review/fix).

Training: 8-bit QLoRA on RTX 5060, 3 epochs, eval_loss=0.7688.

Evaluate the {len(samples)} sampled responses below. Score each item on 4 dimensions (domain-aware):

**All domains:**
1. Format Compliance (0-2): Proper <think> blocks, clear structure, no garbled text
2. Task Understanding (0-3): Correctly interprets the task and addresses it

**Code & code_review domains:**
3. Code Quality (0-3): Correctness, idiom, completeness
4. Tool Usage (0-2): Appropriate use of tools if needed (N/A scored as 2)

**PRD domain:**
3. PRD Quality (0-3): Structured, actionable, covers scope/requirements/tradeoffs
4. Clarity & Depth (0-2): Clear writing, sufficient detail, not superficial

{eval_text}

Return valid JSON only (no markdown, no code fences):
{{
  "evaluations": [
    {{
      "item": 1,
      "task_id": "...",
      "domain": "code|prd|code_review",
      "format_compliance": int,
      "task_understanding": int,
      "code_or_prd_quality": int,
      "tool_or_clarity": int,
      "total": int,
      "notes": "1-2 sentence assessment"
    }}
  ],
  "summary": {{
    "code_avg": "X/10",
    "prd_avg": "X/10",
    "code_review_avg": "X/10",
    "overall_avg": "X/10",
    "strengths": ["..."],
    "weaknesses": ["..."],
    "recommendation": "What to improve for next training iteration"
  }}
}}"""

# --- Call Gemini ---
token = get_token()
url = "https://aiplatform.googleapis.com/v1/projects/project-7c8d85fa-4f69-41a0-abb/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"

payload = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192}
}
headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

print(f"Sending {len(samples)} items to Gemini 3.1 Pro... (prompt: {len(prompt)} chars)", flush=True)
t0 = time.time()

req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
try:
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode())
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    elapsed = time.time() - t0

    # Save raw
    with open(OUTPUT_FILE.replace('.json', '_raw.txt'), 'w', encoding='utf-8') as f:
        f.write(text)

    # Try to parse JSON from Gemini response
    # Strip markdown code fences if present
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip()

    eval_data = json.loads(clean)

    # Save structured
    eval_data["_meta"] = {
        "model": "gemini-3.1-pro-preview",
        "samples": len(samples),
        "elapsed_s": round(elapsed, 1),
        "prompt_chars": len(prompt),
    }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False)

    print(f"\n=== Gemini 3.1 Pro Evaluation ({elapsed:.1f}s) ===")
    summary = eval_data.get("summary", {})
    for k, v in summary.items():
        if k not in ("strengths", "weaknesses", "recommendation"):
            print(f"  {k}: {v}")
    print(f"  strengths: {summary.get('strengths', [])}")
    print(f"  weaknesses: {summary.get('weaknesses', [])}")
    print(f"  recommendation: {summary.get('recommendation', 'N/A')}")

    # Per-item scores
    print(f"\n--- Per-Item Scores ---")
    for ev in eval_data.get("evaluations", []):
        print(f"  [{ev.get('domain','?')}] {ev.get('task_id','?')}: {ev.get('total','?')}/10 - {ev.get('notes','')}")

except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP Error {e.code}: {body[:800]}")
except json.JSONDecodeError as e:
    print(f"JSON parse error: {e}")
    print(f"Raw response saved to {OUTPUT_FILE.replace('.json', '_raw.txt')}")
    print(f"First 500 chars: {text[:500]}")
