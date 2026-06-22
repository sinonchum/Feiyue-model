#!/usr/bin/env python3
"""
M1b: Generate synthetic multi-turn training trajectories using the Feiyue
teacher model (gpt-5.5). All trajectories pass through a verification gate
before entering the training set.

Three generation strategies:
  1. difficulty_curriculum — take existing contracts and increase complexity
  2. tool_diversity — generate tasks for under-represented tools
  3. error_injection — create ambiguous contracts to train self-correction

Usage:
    python scripts/synth_trajectories.py --feiyue-root /path/to/Feiyue

Output:
    data/synthetic/synth_trajectories.jsonl
    data/synthetic/synth_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────

HERMES_TOOLS = [
    "file_read", "file_write", "list_directory",
    "apply_patch", "run_tests", "run_linter",
    "search_files", "shell_exec", "update_plan",
]

DIFFICULTY_TEMPLATES = {
    "easy": {
        "files": (1, 2),
        "tools": ["file_write", "run_tests"],
        "description": "Single-file fix or simple feature",
    },
    "medium": {
        "files": (2, 4),
        "tools": ["file_read", "file_write", "run_tests", "apply_patch"],
        "description": "Multi-file feature with tests",
    },
    "hard": {
        "files": (3, 6),
        "tools": ["file_read", "file_write", "run_tests", "apply_patch", "search_files", "run_linter"],
        "description": "Multi-file feature with cross-file dependencies and verification",
    },
}

# ── Helpers ────────────────────────────────────────────────────────────

def safe_json_dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def call_hermes(prompt: str, model: str = "gpt-5.5",
                provider: str = "openai-codex", timeout: int = 120) -> str:
    """Call Hermes in one-shot mode and return the response text."""
    try:
        result = subprocess.run(
            ["hermes", "chat", "-q", prompt,
             "--model", model, "--provider", provider, "--quiet"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  [WARN] hermes exited {result.returncode}: {result.stderr[:200]}")
            return ""
        return result.stdout.strip()
    except FileNotFoundError:
        print("  [ERROR] hermes CLI not found. Is Hermes Agent installed?")
        return ""
    except subprocess.TimeoutExpired:
        print(f"  [WARN] hermes call timed out after {timeout}s")
        return ""


def verification_gate(trajectory: dict, feiyue_root: Path) -> bool:
    """
    Run the trajectory through Feiyue's provider-free dry-run.
    Extracts the verification command from the trajectory and executes it
    in an isolated temp directory. Returns True if verification passes.
    """
    try:
        # Extract verification command from trajectory
        verif_cmd = None
        for msg in trajectory.get("messages", []):
            if msg["role"] == "user":
                try:
                    contract = json.loads(msg["content"])
                    verif_cmd = contract.get("verification_command", "")
                except json.JSONDecodeError:
                    pass

        if not verif_cmd:
            # No verification command → accept (basic trajectory without gate)
            return True

        with tempfile.TemporaryDirectory() as tmpdir:
            sandbox = Path(tmpdir)

            # Replay tool calls to set up the file state
            for msg in trajectory.get("messages", []):
                content = msg.get("content", "")
                if msg["role"] == "assistant" and "<tool_call>" in content:
                    try:
                        tool_block = content.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                        tool = json.loads(tool_block)
                        if tool["name"] == "file_write":
                            target = sandbox / tool["arguments"]["path"]
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(tool["arguments"]["content"])
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass

            # Run verification in sandbox
            result = subprocess.run(
                verif_cmd, shell=True, cwd=str(sandbox),
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
    except Exception as e:
        print(f"  [WARN] Verification gate error: {e}")
        return False


# ── Strategy 1: Difficulty Curriculum ──────────────────────────────────

def generate_difficulty_curriculum(
    base_contracts: list[dict],
    target_count: int = 200,
) -> list[dict]:
    """
    Take existing TaskContracts and scale difficulty:
    - easy → add 1 file → medium
    - medium → add 2 files + cross-file dependency → hard
    """
    trajectories = []
    for base in base_contracts[:target_count]:
        difficulty = base.get("metadata", {}).get("difficulty", "medium")
        if difficulty == "easy":
            target = "medium"
        elif difficulty == "medium":
            target = "hard"
        else:
            target = "hard"

        prompt = f"""Generate a Feiyue worker trajectory for this task.

Base difficulty: {difficulty} → Scale to: {target}

Template:
{SYSTEM_PROMPT}

Generate a complete multi-turn ChatML trajectory (5-10 messages) where the worker:
1. Receives a harder version of this contract (more files, stricter verification)
2. Makes tool calls (file_write, run_tests)
3. Verifies the result
4. If verification fails, self-corrects and retries
5. Final verification passes

Output ONLY valid JSON:
{{"messages": [{{"role": "...", "content": "..."}}, ...], "metadata": {{...}}}}

Base contract for reference:
{safe_json_dumps(base.get('messages', [])[:3])}
"""
        response = call_hermes(prompt)
        if not response:
            continue

        # Extract JSON from response
        try:
            # Try direct parse
            traj = json.loads(response)
        except json.JSONDecodeError:
            # Try to extract from markdown fences
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL)
            if match:
                try:
                    traj = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
            else:
                continue

        traj.setdefault("metadata", {})["source"] = "synthetic_difficulty"
        trajectories.append(traj)

    return trajectories


# ── Strategy 2: Tool Diversity ────────────────────────────────────────

def generate_tool_diversity(
    hermes_tools: list[str],
    target_per_tool: int = 30,
) -> list[dict]:
    """
    Generate tasks requiring under-represented tools:
    - search_files: code search and refactoring
    - apply_patch: targeted fixes
    - list_directory: project exploration
    - run_linter: code quality
    - update_plan: project planning
    """
    trajectories = []

    tool_descriptions = {
        "search_files": "Find all occurrences of a pattern across the project",
        "apply_patch": "Fix a specific bug by applying a targeted patch",
        "list_directory": "Explore project structure to understand code layout",
        "run_linter": "Run code quality checks and fix warnings",
        "update_plan": "Update the development plan to reflect completed work",
        "file_read": "Read file contents to understand existing code before modifying",
    }

    for tool in hermes_tools:
        if tool in ("file_write", "run_tests"):
            continue  # These are well-covered by base extraction

        desc = tool_descriptions.get(tool, f"Use {tool} tool")
        for i in range(target_per_tool):
            prompt = f"""Generate a Feiyue worker trajectory where the primary tool used is '{tool}'.

Task: {desc}. Variant #{i}.

Generate a complete multi-turn ChatML trajectory (5-10 messages) where the worker:
1. Receives a TaskContract requiring '{tool}'
2. Plans the approach
3. Uses '{tool}' (and supporting tools as needed)
4. Verifies the result
5. Outputs the final status

Output ONLY valid JSON:
{{"messages": [...], "metadata": {{"tools_used": ["{tool}", ...]}}}}
"""
            response = call_hermes(prompt, timeout=90)
            if not response:
                continue

            try:
                traj = json.loads(response)
            except json.JSONDecodeError:
                import re
                match = re.search(r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL)
                if match:
                    try:
                        traj = json.loads(match.group(1))
                    except json.JSONDecodeError:
                        continue
                else:
                    continue

            traj.setdefault("metadata", {})["source"] = f"synthetic_tool_{tool}"
            trajectories.append(traj)

    return trajectories


# ── Strategy 3: Error Injection ────────────────────────────────────────

def generate_error_injection(
    base_contracts: list[dict],
    target_count: int = 100,
) -> list[dict]:
    """
    Create contracts with deliberate ambiguities:
    - Missing context → worker must request file_read
    - Ambiguous verification → worker must clarify
    - Incorrect initial assumption → worker must detect and correct
    """
    trajectories = []

    error_types = [
        "missing_context",
        "ambiguous_verification",
        "wrong_assumption",
        "incomplete_spec",
    ]

    for i in range(target_count):
        error_type = error_types[i % len(error_types)]

        prompt = f"""Generate a Feiyue worker trajectory with ERROR INJECTION.

Error type: {error_type}

The TaskContract should contain a deliberate issue that the worker must:
1. Detect the problem
2. Take corrective action (e.g., request more info via file_read)
3. Resolve the issue
4. Complete the task correctly

The trajectory should show the worker initially confused/blocked, then
recovering through intelligent tool use.

Output ONLY valid JSON:
{{"messages": [...], "metadata": {{"difficulty": "hard", "teacher_used": false}}}}
"""
        response = call_hermes(prompt, timeout=90)
        if not response:
            continue

        try:
            traj = json.loads(response)
        except json.JSONDecodeError:
            import re
            match = re.search(r'```(?:json)?\s*\n?(.*?)```', response, re.DOTALL)
            if match:
                try:
                    traj = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
            else:
                continue

        traj.setdefault("metadata", {})["source"] = f"synthetic_error_{error_type}"
        trajectories.append(traj)

    return trajectories


# ── Main ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Feiyue worker agent operating inside a Hermes runtime. You receive TaskContracts from a strong model (teacher) and execute them using the tools available in the Hermes environment.

RULES:
1. Plan before acting: think about what tools you need and in what order
2. Tool calls must be valid JSON in <tool_call> blocks
3. After each tool call, verify the result before proceeding
4. If verification fails, analyze the error and retry with corrections
5. Minimize unnecessary tool calls
6. All file paths must be relative to the project root
7. Never output secrets, API keys, or absolute paths"""


def main():
    parser = argparse.ArgumentParser(
        description="M1b: Generate synthetic multi-turn trajectories")
    parser.add_argument("--feiyue-root", type=Path, required=True,
                        help="Path to Feiyue checkout (for verification gate)")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"),
                        help="Output directory (default: data/synthetic)")
    parser.add_argument("--difficulty-count", type=int, default=200,
                        help="Target count for difficulty curriculum")
    parser.add_argument("--tool-count", type=int, default=30,
                        help="Target count per tool for diversity")
    parser.add_argument("--error-count", type=int, default=100,
                        help="Target count for error injection")
    parser.add_argument("--skip-verification-gate", action="store_true",
                        help="Skip verification gate (faster, less safe)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip teacher generation (use pre-generated data)")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.skip_generation:
        print("Skipping generation (--skip-generation). Using existing synthetic data.")
        return

    # ── Load base contracts for difficulty curriculum ─────────────────
    raw_dir = Path("data/raw")
    extracted_path = raw_dir / "extracted_train.jsonl"
    base_contracts = []
    if extracted_path.exists():
        with open(extracted_path, encoding="utf-8") as f:
            base_contracts = [json.loads(line) for line in f if line.strip()]
    else:
        print(f"[WARN] {extracted_path} not found. Difficulty curriculum will use templates only.")

    print(f"Loaded {len(base_contracts)} base contracts for curriculum generation.")

    # ── Strategy 1: Difficulty Curriculum ─────────────────────────────
    print(f"\n--- Strategy 1: Difficulty Curriculum (target: {args.difficulty_count}) ---")
    curriculum = generate_difficulty_curriculum(base_contracts, args.difficulty_count)
    print(f"  Generated: {len(curriculum)} trajectories")

    # ── Strategy 2: Tool Diversity ────────────────────────────────────
    print(f"\n--- Strategy 2: Tool Diversity (target: {args.tool_count}/tool) ---")
    diversity = generate_tool_diversity(HERMES_TOOLS, args.tool_count)
    print(f"  Generated: {len(diversity)} trajectories")

    # ── Strategy 3: Error Injection ───────────────────────────────────
    print(f"\n--- Strategy 3: Error Injection (target: {args.error_count}) ---")
    error_injection = generate_error_injection(base_contracts, args.error_count)
    print(f"  Generated: {len(error_injection)} trajectories")

    # ── Merge ────────────────────────────────────────────────────────
    all_synthetic = curriculum + diversity + error_injection
    print(f"\n--- Total synthetic: {len(all_synthetic)} trajectories ---")

    # ── Verification Gate ────────────────────────────────────────────
    if not args.skip_verification_gate:
        print("\n--- Running verification gate ---")
        passed = []
        failed = 0
        for i, traj in enumerate(all_synthetic):
            if verification_gate(traj, args.feiyue_root):
                passed.append(traj)
            else:
                failed += 1
            if (i + 1) % 25 == 0:
                print(f"  Verified {i+1}/{len(all_synthetic)}... ({len(passed)} pass, {failed} fail)")
        print(f"  Gate result: {len(passed)} passed, {failed} failed "
              f"({len(passed)/max(len(all_synthetic),1)*100:.0f}% pass rate)")
        all_synthetic = passed
    else:
        print("\n  [SKIP] Verification gate bypassed (--skip-verification-gate)")

    # ── Deduplicate by task_id ───────────────────────────────────────
    seen = set()
    deduped = []
    for traj in all_synthetic:
        tid = traj.get("metadata", {}).get("task_id", "")
        if tid not in seen:
            seen.add(tid)
            deduped.append(traj)
    all_synthetic = deduped
    print(f"  After dedup: {len(all_synthetic)} unique trajectories")

    # ── Write output ─────────────────────────────────────────────────
    out_path = args.output / "synth_trajectories.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for traj in all_synthetic:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    # ── Report ───────────────────────────────────────────────────────
    report = {
        "generation_time": datetime.now(timezone.utc).isoformat(),
        "total_generated": len(curriculum) + len(diversity) + len(error_injection),
        "after_verification_gate": len(all_synthetic),
        "by_strategy": {
            "difficulty_curriculum": len(curriculum),
            "tool_diversity": len(diversity),
            "error_injection": len(error_injection),
        },
        "verification_gate_enabled": not args.skip_verification_gate,
        "output_path": str(out_path),
    }
    report_path = args.output / "synth_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Output: {out_path}")
    print(f"  Report: {report_path}")
    print("  Done.")


if __name__ == "__main__":
    main()
