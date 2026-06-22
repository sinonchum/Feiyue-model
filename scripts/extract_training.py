"""Extract training pairs from Feiyue evidence files for Qwen 3 8B fine-tuning.

Usage:
    python scripts/extract_training.py /path/to/Feiyue --output data/train.jsonl

Output format: ChatML JSONL with system/user/assistant messages + metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a Feiyue worker agent operating inside a Hermes runtime. You receive TaskContracts from a strong model (teacher) and produce CandidateFileWrite outputs.

RULES:
1. Your response must be valid JSON matching the CandidateFileWrite schema
2. Output ONLY the JSON object — no markdown fences, no preamble
3. The 'path' field MUST be relative to the project root
4. The 'content' field contains the complete file contents
5. If you receive teacher guidance about a previous failure, incorporate it exactly
6. Every file write must match the verification criteria in the TaskContract

CandidateFileWrite schema:
{"writes": [{"path": "relative/path/to/file.py", "content": "complete file contents here"}]}"""


def load_evidence(path: Path) -> dict[str, Any]:
    """Load a JSON evidence file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_provider_runs(hermes_dir: Path) -> list[dict]:
    """Extract training pairs from provider-runs/ directory."""
    samples = []
    provider_dir = hermes_dir / "provider-runs"
    if not provider_dir.exists():
        return samples

    for run_dir in sorted(provider_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ev_path = run_dir / "run-evidence.json"
        if not ev_path.exists():
            continue

        evidence = load_evidence(ev_path)

        # Skip teacher-only runs (they produce guidance, not worker output)
        if "teacher" in evidence.get("run_id", ""):
            continue

        command = evidence.get("command", [])
        prompt_text = command[-1] if len(command) > 2 else ""
        provider = evidence.get("provider_or_profile", "unknown")
        exit_code = evidence.get("exit_code")
        artifacts = evidence.get("artifacts", [])
        commit_sha = evidence.get("commit_sha", "")

        # Try to extract writes from artifacts or command output
        writes = []
        for artifact in artifacts:
            if isinstance(artifact, dict) and "writes" in artifact:
                writes = artifact["writes"]
                break

        if not writes and exit_code == 0:
            # Build a minimal sample from the evidence
            writes = [{"path": f"docs/{evidence.get('run_id', 'unknown')}.md",
                        "content": f"# Auto-generated from run {evidence.get('run_id')}"}]

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "task_id": evidence.get("run_id", run_dir.name),
                    "description": prompt_text[:500] if prompt_text else "Execute task from contract",
                    "verification_command": f"exit code check: {exit_code}",
                    "allowed_files": [w.get("path", "") for w in writes],
                })},
                {"role": "assistant", "content": json.dumps({"writes": writes})},
            ],
            "metadata": {
                "task_id": evidence.get("run_id", run_dir.name),
                "status": "verified" if exit_code == 0 else "failed",
                "difficulty": "medium",
                "domain": "code",
                "teacher_used": False,
                "verification_passed": exit_code == 0,
                "provider": provider,
                "commit_sha": commit_sha,
            },
        }
        samples.append(sample)

    return samples


def extract_workflow_smokes(hermes_dir: Path) -> list[dict]:
    """Extract training pairs from workflow-smokes/ directory."""
    samples = []
    smoke_dir = hermes_dir / "workflow-smokes"
    if not smoke_dir.exists():
        return samples

    for run_dir in sorted(smoke_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ev_path = run_dir / "evidence.json"
        if not ev_path.exists():
            continue

        evidence = load_evidence(ev_path)
        status = evidence.get("status", "unknown")
        verification_passed = status == "verified"

        sample = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({
                    "task_id": run_dir.name,
                    "description": f"Workflow smoke: {evidence.get('task_id', run_dir.name)}",
                    "context": json.dumps({k: v for k, v in evidence.items()
                                           if k in ("changed_files", "verification_command",
                                                     "attempt_count", "execution_performed")}),
                })},
                {"role": "assistant", "content": json.dumps({
                    "writes": [{"path": f, "content": "# Written by worker"} for f in evidence.get("changed_files", [])]
                })},
            ],
            "metadata": {
                "task_id": run_dir.name,
                "status": status,
                "difficulty": "easy",
                "domain": "config",
                "teacher_used": evidence.get("teacher_guidance_events", False),
                "verification_passed": verification_passed,
            },
        }
        samples.append(sample)

    return samples


def extract_multi_worker(hermes_dir: Path) -> list[dict]:
    """Extract training pairs from multi-worker-workflows/ directory."""
    samples = []
    mw_dir = hermes_dir / "multi-worker-workflows"
    if not mw_dir.exists():
        return samples

    for run_dir in sorted(mw_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        ev_path = run_dir / "evidence.json"
        if not ev_path.exists():
            continue

        evidence = load_evidence(ev_path)
        assignment_reports = evidence.get("assignment_reports", [])

        for report in assignment_reports:
            candidate_files = report.get("candidate_files", [])
            writes = [{"path": f, "content": f"# Modified by {report.get('profile_id')}"}
                      for f in candidate_files]

            sample = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "task_id": f"{run_dir.name}/{report.get('assignment_id', 'unknown')}",
                        "description": f"Role: {report.get('role')}. Profile: {report.get('profile_id')}",
                        "allowed_files": candidate_files,
                    })},
                    {"role": "assistant", "content": json.dumps({"writes": writes})},
                ],
                "metadata": {
                    "task_id": run_dir.name,
                    "status": report.get("status", "unknown"),
                    "difficulty": "medium",
                    "domain": "code",
                    "teacher_used": False,
                    "verification_passed": report.get("status") == "candidate_ready",
                    "profile": report.get("profile_id"),
                },
            }
            samples.append(sample)

    return samples


def extract_capability_history(hermes_dir: Path) -> list[dict]:
    """Extract high-level capability records as training context samples."""
    samples = []
    hist_path = hermes_dir / "capability-history" / "latest.json"
    if not hist_path.exists():
        return samples

    data = load_evidence(hist_path)
    records = data.get("records", [])

    # Build one summary sample showing capability trends
    summary = {
        "total_records": data.get("total_records", 0),
        "profiles": {},
    }
    for rec in records[:50]:  # Cap at 50 to avoid bloat
        pid = rec.get("profile_id", "unknown")
        if pid not in summary["profiles"]:
            summary["profiles"][pid] = {"verified": 0, "total": 0}
        summary["profiles"][pid]["total"] += 1
        if rec.get("verified"):
            summary["profiles"][pid]["verified"] += 1

    samples.append({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Summarize capability history across profiles"},
            {"role": "assistant", "content": json.dumps(summary)},
        ],
        "metadata": {
            "task_id": "capability-summary",
            "status": "verified",
            "difficulty": "easy",
            "domain": "analytics",
            "teacher_used": False,
            "verification_passed": True,
        },
    })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Extract Feiyue training data")
    parser.add_argument("feiyue_root", type=Path, help="Path to Feiyue checkout")
    parser.add_argument("--output", type=Path, default=Path("data/train.jsonl"),
                        help="Output path for training JSONL")
    parser.add_argument("--format", choices=["chatml"], default="chatml",
                        help="Output format (default: chatml)")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train/val split ratio")

    args = parser.parse_args()
    hermes_dir = args.feiyue_root / ".hermes"

    if not hermes_dir.exists():
        print(f"Error: .hermes directory not found in {args.feiyue_root}", file=sys.stderr)
        sys.exit(1)

    # Extract from all sources
    all_samples = []
    all_samples.extend(extract_provider_runs(hermes_dir))
    all_samples.extend(extract_workflow_smokes(hermes_dir))
    all_samples.extend(extract_multi_worker(hermes_dir))
    all_samples.extend(extract_capability_history(hermes_dir))

    # Split
    split_idx = int(len(all_samples) * args.split)
    train = all_samples[:split_idx]
    val = all_samples[split_idx:]

    # Write
    args.output.parent.mkdir(parents=True, exist_ok=True)

    train_path = args.output.parent / "train.jsonl"
    val_path = args.output.parent / "val.jsonl"

    for path, samples in [(train_path, train), (val_path, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Print stats
    verified = sum(1 for s in all_samples if s["metadata"]["verification_passed"])
    with_teacher = sum(1 for s in all_samples if s["metadata"]["teacher_used"])

    print(f"Extracted {len(all_samples)} training samples")
    print(f"  Verified: {verified} ({verified/max(len(all_samples),1)*100:.0f}%)")
    print(f"  With teacher: {with_teacher}")
    print(f"  Train: {len(train)} → {train_path}")
    print(f"  Val: {len(val)} → {val_path}")


if __name__ == "__main__":
    main()
