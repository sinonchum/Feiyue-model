#!/usr/bin/env python3
"""Generate Feiyue coding eval set using Gemini 2.5 Pro."""
import json, urllib.request, os, sys
from pathlib import Path

def call_gemini(prompt, max_tokens=8192, temp=0.3):
    with open(os.path.expanduser('~/gcloud_token.txt')) as f:
        token = f.read().strip()
    project = 'gen-lang-client-0869773739'
    region = 'us-central1'
    url = f'https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/gemini-2.5-pro:generateContent'
    req = urllib.request.Request(url,
        data=json.dumps({
            'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
            'generationConfig': {'temperature': temp, 'maxOutputTokens': max_tokens}
        }).encode(),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data['candidates'][0]['content']['parts'][0]['text']

def extract_json(text):
    """Extract JSON from markdown-fenced response."""
    import re
    text = text.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

SYSTEM_PROMPT = """You are a Feiyue worker agent operating inside a Hermes runtime. You receive TaskContracts from a strong model (teacher) and execute them using the tools available in the Hermes environment.

RULES:
1. Plan before acting: think about what tools you need and in what order
2. Tool calls must be valid JSON in <tool_call> blocks
3. After each tool call, verify the result before proceeding
4. If verification fails, analyze the error and retry with corrections
5. Minimize unnecessary tool calls
6. All file paths must be relative to the project root
7. Never output secrets, API keys, or absolute paths

Available tools: file_read, file_write, list_directory, apply_patch, run_tests, run_linter, search_files, shell_exec, update_plan"""

def main():
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── STEP 1: Generate eval set ──────────────────────────────────
    print("=== Generating 8 coding eval problems ===")
    eval_prompt = f"""Generate 8 diverse coding evaluation problems for a Feiyue worker model test set.

Each problem must be a multi-turn ChatML trajectory in this EXACT format:
{{
  "messages": [
    {{"role": "system", "content": "{SYSTEM_PROMPT[:200]}..."}},
    {{"role": "user", "content": "{{ <TaskContract JSON with task_id, description, difficulty, allowed_files> }}"}},
    {{"role": "assistant", "content": "<tool_call>\\n{{ <JSON with name and arguments> }}\\n</tool_call>"}},
    {{"role": "tool", "content": "<tool_response>\\n{{ <JSON response> }}\\n</tool_response>"}},
    {{"role": "assistant", "content": "<tool_call>\\n{{ ... }}\\n</tool_call>"}},
    {{"role": "tool", "content": "<tool_response>\\n{{ ... }}\\n</tool_response>"}},
    {{"role": "assistant", "content": "Task complete. Verification passed."}}
  ],
  "metadata": {{
    "task_id": "eval-<unique-id>",
    "status": "verified",
    "difficulty": "<easy|medium|hard>",
    "domain": "code",
    "tools_used": ["..."],
    "teacher_used": false,
    "attempts": 1,
    "verification_passed": true,
    "source": "eval_set"
  }}
}}

CRITICAL FORMAT RULES:
- Tool calls MUST use <tool_call> XML block inside the "content" string — NEVER use "tool_calls" array
- Tool responses MUST use <tool_response> XML block
- Every message has ONLY "role" and "content" keys — NO "tool_call_id" or "tool_calls"
- assistant/tool messages alternate: assistant → tool → assistant → tool → assistant
- Trajectory ends with assistant text summary

VARIETY REQUIREMENTS:
- 3 easy (simple functions, bug fixes)
- 3 medium (multi-file refactoring, code search tasks)
- 2 hard (complex features with multiple tools, error recovery)
- Cover tools: file_write, file_read, search_files, run_tests, apply_patch
- Each trajectory must have 5-9 messages total
- Content must be REAL Python/TypeScript code, not placeholder comments

Output ONLY a JSON array of 8 trajectory objects. No explanation before or after."""

    response = call_gemini(eval_prompt, max_tokens=16384)
    eval_data = extract_json(response)

    if eval_data and isinstance(eval_data, list):
        eval_path = out_dir / "eval_coding.jsonl"
        with open(eval_path, 'w', encoding='utf-8') as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  Wrote {len(eval_data)} eval problems → {eval_path}")

        # Validate format
        valid = 0
        for i, item in enumerate(eval_data):
            msgs = item.get('messages', [])
            has_tc = any('<tool_call>' in m.get('content','') for m in msgs if m.get('role') == 'assistant')
            has_tr = any('<tool_response>' in m.get('content','') for m in msgs if m.get('role') == 'tool')
            has_meta = 'metadata' in item
            ok = has_tc and has_tr and has_meta
            print(f"  Eval {i}: tc={has_tc} tr={has_tr} meta={has_meta} → {'OK' if ok else 'FAIL'}")
            if ok:
                valid += 1
        print(f"  Valid eval samples: {valid}/{len(eval_data)}")
    else:
        print(f"  FAILED: {response[:500]}")

    # ── STEP 2: Generate training data ─────────────────────────────
    print("\n=== Generating 50 coding training samples ===")
    train_prompt = f"""Generate 50 Feiyue worker training trajectories for fine-tuning a coding agent.

Each trajectory follows the same format as above — multi-turn ChatML with <tool_call>/<tool_response> XML blocks.

DISTRIBUTION:
- 20 easy: simple functions, single-file edits, assert tests
- 20 medium: multi-file refactoring, pattern search, linter fixes
- 10 hard: complex features, error recovery, multi-tool workflows

TOOL COVERAGE: file_write, file_read, search_files, run_tests, apply_patch, run_linter

CRITICAL FORMAT RULES (SAME AS ABOVE):
- Tool calls MUST use <tool_call> XML block in "content" string
- NEVER use "tool_calls" array field
- Only "role" and "content" keys per message
- assistant/tool alternate, end with assistant summary

CONTENT RULES:
- ALL code must be real, valid Python (occasionally TypeScript) — no placeholder comments like "# TODO" or "# your code here"
- TaskContracts must have realistic task_id, description, verification_command, allowed_files
- Metadata must include task_id, status, difficulty, domain, tools_used, verification_passed

Output ONLY a JSON array of 50 trajectory objects. No explanation before or after."""

    response = call_gemini(train_prompt, max_tokens=65536)
    train_data = extract_json(response)

    if train_data and isinstance(train_data, list):
        train_path = out_dir / "raw" / "extracted_train.jsonl"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        # Read existing data first
        existing = []
        if train_path.exists():
            with open(train_path, encoding='utf-8') as f:
                existing = [json.loads(line) for line in f if line.strip()]

        all_data = existing + train_data
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"  Total training: {len(existing)} existing + {len(train_data)} new = {len(all_data)} → {train_path}")

        # Validate
        valid = 0
        difficulties = {}
        for i, item in enumerate(train_data):
            msgs = item.get('messages', [])
            has_tc = any('<tool_call>' in m.get('content','') for m in msgs if m.get('role') == 'assistant')
            has_tr = any('<tool_response>' in m.get('content','') for m in msgs if m.get('role') == 'tool')
            has_meta = 'metadata' in item
            ok = has_tc and has_tr and has_meta
            if ok:
                valid += 1
                d = item['metadata'].get('difficulty', '?')
                difficulties[d] = difficulties.get(d, 0) + 1
        print(f"  Valid: {valid}/{len(train_data)} | Difficulty: {difficulties}")
    else:
        print(f"  FAILED: {response[:500]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
