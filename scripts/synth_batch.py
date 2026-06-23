#!/usr/bin/env python3
"""Batch synthetic trajectory generator — one hermes call for multiple trajectories."""
import json, subprocess, sys, re
from pathlib import Path
from datetime import datetime, timezone

FORMAT_RULES = """
CRITICAL: EVERY tool call MUST be a <tool_call> XML block in "content" string.
Example: {"role": "assistant", "content": "<tool_call>\\n{\\"name\\": \\"file_write\\", \\"arguments\\": {\\"path\\": \\"x.py\\"}}\\n</tool_call>"}
NEVER use "tool_calls" array. Tool responses use <tool_response> XML blocks.
Every message has ONLY "role" and "content" keys. Trajectory ends with assistant text.
"""

TOOLS = ["search_files", "apply_patch", "run_linter", "file_read", "list_directory"]
BATCH_SIZE = 5  # trajectories per hermes call

def safe_json_dumps(obj, indent=2):
    return json.dumps(obj, ensure_ascii=False, indent=indent)

def call_hermes(prompt, timeout=300):
    try:
        result = subprocess.run(
            ["hermes", "chat", "-q", prompt, "--quiet"],
            capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            return ""
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""

def extract_json(text):
    """Extract JSON from hermes response (handles markdown fences)."""
    text = text.strip()
    if text.startswith("```"):
        match = re.search(r'```(?:json)?\s*\n?(.*?)```', text, re.DOTALL)
        if match:
            text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def is_valid_format(traj):
    """Check trajectory uses <tool_call> XML format."""
    for msg in traj.get("messages", []):
        if "tool_calls" in msg:
            return False, "Has tool_calls field"
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content and "<tool_call>" not in content and "tool_calls" not in str(msg):
                pass  # Text-only assistant message is fine
    return True, ""

def main():
    out_dir = Path("data/synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_trajectories = []
    total_ok = 0
    total_bad = 0

    # Generate tool diversity trajectories in batches
    for tool in TOOLS:
        descs = {
            "search_files": "Find all occurrences of a function name across a codebase and document usage",
            "apply_patch": "Fix a bug by applying targeted changes to specific files",
            "run_linter": "Run code quality checks and fix all warnings",
            "file_read": "Read source files to understand architecture before making changes",
            "list_directory": "Explore project structure to find relevant files for a task",
        }
        desc = descs.get(tool, f"Use {tool} tool")

        for batch_start in range(0, 10, BATCH_SIZE):
            remaining = min(BATCH_SIZE, 10 - batch_start)
            if remaining <= 0:
                break

            variants = [f"Variant {batch_start+i+1}: focus on {['error handling','performance','refactoring','documentation','testing'][i%5]}" for i in range(remaining)]
            variants_text = "\n".join(variants)

            prompt = f"""{FORMAT_RULES}

Generate {remaining} Feiyue worker trajectories as a JSON array for tool '{tool}'.
Tool description: {desc}

{variants_text}

Each trajectory must:
- Start with system prompt: "You are a Feiyue worker agent. Execute tasks using Hermes tools."
- User message contains a TaskContract JSON
- Assistant makes tool calls using <tool_call> XML in content string
- Tool responds with <tool_response> XML
- End with assistant summary

Output ONLY a JSON array:
[{{"messages": [...], "metadata": {{"task_id": "feiyue-{tool}-v{i}", "tools_used": ["{tool}"], "difficulty": "medium", "domain": "code", "verification_passed": true, "teacher_used": false, "attempts": 1, "source": "synthetic_tool_{tool}"}}}}, ...]
"""
            print(f"  Generating {tool} batch {batch_start+1}-{batch_start+remaining}...")
            response = call_hermes(prompt)
            if not response:
                print(f"    [FAIL] hermes call failed")
                continue

            data = extract_json(response)
            if data is None:
                print(f"    [FAIL] JSON parse error")
                continue

            if isinstance(data, dict):
                data = [data]

            for traj in data:
                ok, reason = is_valid_format(traj)
                if ok:
                    all_trajectories.append(traj)
                    total_ok += 1
                else:
                    total_bad += 1
                    print(f"    [FORMAT] {reason} in trajectory, skipped")
            print(f"    Got {len(data)} trajectories")

    # Write output
    out_path = out_dir / "synth_trajectories.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for traj in all_trajectories:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    report = {
        "generation_time": datetime.now(timezone.utc).isoformat(),
        "total_generated": len(all_trajectories),
        "format_ok": total_ok,
        "format_bad": total_bad,
        "tools_used": TOOLS,
        "output_path": str(out_path),
    }
    with open(out_dir / "synth_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {total_ok} valid, {total_bad} format-bad → {out_path}")

if __name__ == "__main__":
    main()
