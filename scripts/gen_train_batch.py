#!/usr/bin/env python3
"""Generate Feiyue coding training data in small batches using Gemini 2.5 Pro."""
import json, urllib.request, os, sys, re
from pathlib import Path

def call_gemini(prompt, max_tokens=8192, timeout=180):
    with open(os.path.expanduser('~/gcloud_token.txt')) as f:
        token = f.read().strip()
    project = 'gen-lang-client-0869773739'
    region = 'us-central1'
    url = f'https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/gemini-2.5-pro:generateContent'
    req = urllib.request.Request(url,
        data=json.dumps({
            'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': 0.5, 'maxOutputTokens': max_tokens}
        }).encode(),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
        return data['candidates'][0]['content']['parts'][0]['text']

def extract_json(text):
    import re
    text = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try: return json.loads(text)
    except: return None

SYSTEM_PROMPT = "You are a Feiyue worker agent operating inside a Hermes runtime. Execute TaskContracts using Hermes tools."

def generate_batch(batch_num, size, difficulty, tools):
    descs = {
        'easy': 'Simple functions, single-file edits, basic bug fixes, assert tests',
        'medium': 'Multi-file refactoring, pattern search across files, linter fixes, patch application',
        'hard': 'Complex multi-tool workflows, error recovery, cross-file dependencies, verification-driven development'
    }
    tool_list = ', '.join(tools)

    prompt = f"""Generate {size} Feiyue worker training trajectories. Difficulty: {difficulty}.

Topics: {descs[difficulty]}
Tools to use: {tool_list}

FORMAT — each trajectory is EXACTLY this JSON structure:
{{
  "messages": [
    {{"role": "system", "content": "{SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "{{ task contract JSON }}"}},
    {{"role": "assistant", "content": "<tool_call>\\n{{\\"name\\": \\"...\\", \\"arguments\\": {{...}}}}\\n</tool_call>"}},
    {{"role": "tool", "content": "<tool_response>\\n{{...}}\\n</tool_response>"}},
    {{"role": "assistant", "content": "Done. Verification passed."}}
  ],
  "metadata": {{"task_id": "train-{difficulty}-{batch_num:02d}-n", "status": "verified", "difficulty": "{difficulty}", "domain": "code", "tools_used": [...], "teacher_used": false, "attempts": 1, "verification_passed": true, "source": "gemini_synthetic"}}
}}

CRITICAL RULES:
- Tool calls ONLY in <tool_call> XML blocks inside "content" string — NEVER use "tool_calls" array
- Tool responses in <tool_response> XML blocks
- Only "role" and "content" keys per message — NO "tool_call_id"
- ALL code must be REAL Python/TS — no placeholder comments
- 5-7 messages per trajectory
- assistant/tool alternate

Output ONLY a JSON array of {size} objects. No explanation."""

    print(f"  Batch {batch_num}: {difficulty} x{size}...")
    response = call_gemini(prompt, max_tokens=size * 1500, timeout=300)
    if not response:
        return []

    data = extract_json(response)
    if not data:
        print(f"    Parse failed: {response[:200]}")
        return []

    if isinstance(data, dict):
        data = [data]
    return data


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_new = []

    # Generate in small batches
    batches = [
        # (batch_num, size, difficulty, tools)
        (1, 8, 'easy', ['file_write', 'run_tests']),
        (2, 8, 'easy', ['file_read', 'file_write']),
        (3, 8, 'medium', ['file_write', 'run_tests', 'apply_patch']),
        (4, 8, 'medium', ['search_files', 'file_read', 'file_write', 'run_linter']),
        (5, 5, 'hard', ['file_read', 'search_files', 'file_write', 'apply_patch', 'run_tests']),
        (6, 5, 'hard', ['file_read', 'search_files', 'file_write', 'apply_patch', 'run_linter', 'shell_exec']),
    ]

    for num, size, diff, tools in batches:
        data = generate_batch(num, size, diff, tools)
        if data:
            # Validate
            valid = 0
            for i, item in enumerate(data):
                msgs = item.get('messages', [])
                has_tc = any('<tool_call>' in m.get('content','') for m in msgs if m.get('role') == 'assistant')
                has_tr = any('<tool_response>' in m.get('content','') for m in msgs if m.get('role') == 'tool')
                has_meta = 'metadata' in item
                has_bad = any('tool_calls' in m for m in msgs)
                if has_tc and has_tr and has_meta and not has_bad:
                    # Fix task_id to be unique
                    item['metadata']['task_id'] = f"train-{diff}-{num:02d}-{i:02d}"
                    all_new.append(item)
                    valid += 1
            print(f"    Valid: {valid}/{len(data)}")
        else:
            print(f"    FAILED")

    # Merge with existing
    train_path = out_dir / "extracted_train.jsonl"
    existing = []
    if train_path.exists():
        with open(train_path, encoding='utf-8') as f:
            existing = [json.loads(line) for line in f if line.strip()]

    all_data = existing + all_new
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Stats
    diffs = {}
    for item in all_data:
        d = item.get('metadata', {}).get('difficulty', '?')
        diffs[d] = diffs.get(d, 0) + 1

    print(f"\nTotal training: {len(existing)} existing + {len(all_new)} new = {len(all_data)}")
    print(f"Difficulty: {diffs}")
    print(f"Output: {train_path}")


if __name__ == "__main__":
    main()
