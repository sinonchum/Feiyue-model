#!/usr/bin/env python3
"""
M1c: Merge, validate, deduplicate, and split all training data sources.

Inputs:
  - data/raw/extracted_train.jsonl    (M1a: Feiyue evidence)
  - data/raw/extracted_val.jsonl
  - data/raw/extracted_test.jsonl
  - data/synthetic/synth_trajectories.jsonl  (M1b: teacher-generated)
  - data/fixtures/fixture_samples.jsonl      (test fixture extraction)

Outputs:
  - data/train.jsonl    (Final SFT training set)
  - data/val.jsonl      (Final SFT validation set)
  - data/test.jsonl     (Held-out evaluation set)
  - data/prepare_report.json

Usage:
    # Run with all sources
    python scripts/prepare_data.py --all

    # Run with only extracted data
    python scripts/prepare_data.py --extracted data/raw/

    # Preview without writing
    python scripts/prepare_data.py --all --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Constants ──────────────────────────────────────────────────────────

VALID_ROLES = {"system", "user", "assistant", "tool"}
REQUIRED_METADATA_FIELDS = {"task_id", "status", "difficulty", "domain", "tools_used", "verification_passed"}
SECRET_PATTERNS = [
    r'sk-[a-zA-Z0-9]{20,}',                    # OpenAI keys
    r'ghp_[a-zA-Z0-9]{36}',                    # GitHub tokens
    r'AIza[0-9A-Za-z\-_]{35}',                 # Google API keys
    r'[a-zA-Z0-9+/]{40,}={0,2}',               # Base64-ish secrets (high false positive, warn only)
]
PATH_PATTERNS = [
    r'/Users/\S+',       # macOS
    r'C:\\Users\\\S+',    # Windows
    r'/home/\S+',         # Linux
]


# ── Helpers ────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning list of dicts."""
    samples = []
    if not path.exists():
        return samples
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return samples


def sha256(sample: dict) -> str:
    """Deterministic content hash for deduplication."""
    content = json.dumps(sample.get("messages", []), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()


# ── Validation ─────────────────────────────────────────────────────────

def validate_sample(sample: dict, idx: int) -> list[str]:
    """Return list of validation error messages (empty = valid)."""
    errors = []

    # Structural
    if "messages" not in sample:
        errors.append(f"[{idx}] missing 'messages' key")
        return errors
    msgs = sample["messages"]
    if not isinstance(msgs, list) or len(msgs) < 3:
        errors.append(f"[{idx}] messages must be list with ≥3 entries, got {len(msgs)}")
        return errors

    # Role validation
    for i, msg in enumerate(msgs):
        role = msg.get("role", "")
        if role not in VALID_ROLES:
            errors.append(f"[{idx}] msg[{i}]: invalid role '{role}'")
        content = msg.get("content", "")
        if not content or not isinstance(content, str):
            errors.append(f"[{idx}] msg[{i}]: missing or non-string content")

    # Tool call validation: <tool_call> blocks must be valid JSON
    full_text = "\n".join(m.get("content", "") for m in msgs)
    tool_blocks = re.findall(r'<tool_call>\s*\n?(.*?)\n?\s*</tool_call>', full_text, re.DOTALL)
    for block in tool_blocks:
        try:
            parsed = json.loads(block.strip())
            if "name" not in parsed or "arguments" not in parsed:
                errors.append(f"[{idx}] tool_call missing 'name' or 'arguments': {block[:80]}")
        except json.JSONDecodeError:
            errors.append(f"[{idx}] invalid JSON in tool_call: {block[:80]}")

    # Tool response validation
    resp_blocks = re.findall(r'<tool_response>\s*\n?(.*?)\n?\s*</tool_response>', full_text, re.DOTALL)
    for block in resp_blocks:
        try:
            json.loads(block.strip())
        except json.JSONDecodeError:
            errors.append(f"[{idx}] invalid JSON in tool_response: {block[:80]}")

    # No absolute paths
    for pattern in PATH_PATTERNS:
        if re.search(pattern, full_text):
            errors.append(f"[{idx}] absolute path found matching '{pattern}'")

    # No secrets
    for pattern in SECRET_PATTERNS:
        if re.search(pattern, full_text):
            errors.append(f"[{idx}] potential secret found matching '{pattern}'")

    # Metadata
    meta = sample.get("metadata", {})
    if not meta.get("task_id"):
        errors.append(f"[{idx}] missing metadata.task_id")
    for field in REQUIRED_METADATA_FIELDS:
        if field not in meta:
            errors.append(f"[{idx}] missing metadata.{field}")

    # Final message must be assistant role
    if msgs and msgs[-1]["role"] != "assistant":
        errors.append(f"[{idx}] trajectory must end with assistant message")

    return errors


# ── Statistics ─────────────────────────────────────────────────────────

def compute_stats(samples: list[dict]) -> dict:
    """Compute aggregate statistics for a dataset split."""
    if not samples:
        return {"count": 0}

    n = len(samples)
    verified = sum(1 for s in samples if s["metadata"].get("verification_passed", False))
    multi_turn = sum(1 for s in samples if s["metadata"].get("attempts", 1) > 1)
    teacher_used = sum(1 for s in samples if s["metadata"].get("teacher_used", False))
    avg_msgs = sum(len(s["messages"]) for s in samples) / n

    difficulty = {}
    domain = {}
    tools = {}
    source = {}
    for s in samples:
        m = s["metadata"]
        difficulty[m["difficulty"]] = difficulty.get(m["difficulty"], 0) + 1
        domain[m["domain"]] = domain.get(m["domain"], 0) + 1
        for t in m["tools_used"]:
            tools[t] = tools.get(t, 0) + 1
        source[m.get("source", "unknown")] = source.get(m.get("source", "unknown"), 0) + 1

    return {
        "count": n,
        "verified_rate": f"{verified}/{n} ({verified/n*100:.1f}%)",
        "multi_turn_rate": f"{multi_turn}/{n} ({multi_turn/n*100:.1f}%)",
        "teacher_used_rate": f"{teacher_used}/{n} ({teacher_used/n*100:.1f}%)",
        "avg_messages": round(avg_msgs, 1),
        "difficulty_distribution": difficulty,
        "domain_distribution": domain,
        "tool_usage": tools,
        "source_distribution": source,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="M1c: Merge, validate, and split training data")
    parser.add_argument("--all", action="store_true",
                        help="Load from all default sources")
    parser.add_argument("--extracted", type=Path, default=Path("data/raw"),
                        help="Path to M1a extracted data directory")
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic/synth_trajectories.jsonl"),
                        help="Path to M1b synthetic trajectories")
    parser.add_argument("--fixtures", type=Path, default=Path("data/fixtures/fixture_samples.jsonl"),
                        help="Path to test fixture samples")
    parser.add_argument("--output", type=Path, default=Path("data"),
                        help="Output directory for final datasets")
    parser.add_argument("--split", type=float, nargs=3,
                        default=[0.80, 0.10, 0.10],
                        help="Train/val/test split ratios")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling and splitting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate and report without writing output files")

    args = parser.parse_args()

    # ── Load all sources ────────────────────────────────────────────
    all_raw: list[dict] = []

    sources_loaded = False

    if args.all:
        sources_loaded = True
        # M1a: extracted evidence trajectories
        for fname in ["extracted_train.jsonl", "extracted_val.jsonl", "extracted_test.jsonl"]:
            path = args.extracted / fname
            samples = load_jsonl(path)
            all_raw.extend(samples)
            print(f"  Loaded {len(samples):>5} from {path}")

        # M1b: synthetic trajectories
        if args.synthetic.exists():
            samples = load_jsonl(args.synthetic)
            all_raw.extend(samples)
            print(f"  Loaded {len(samples):>5} from {args.synthetic}")
        else:
            print(f"  [SKIP] {args.synthetic} not found")

        # Fixtures
        if args.fixtures.exists():
            samples = load_jsonl(args.fixtures)
            all_raw.extend(samples)
            print(f"  Loaded {len(samples):>5} from {args.fixtures}")
        else:
            print(f"  [SKIP] {args.fixtures} not found")

    # Individual source loading for non --all mode
    if args.extracted.exists() and not args.all:
        for fname in ["extracted_train.jsonl", "extracted_val.jsonl", "extracted_test.jsonl"]:
            path = args.extracted / fname
            samples = load_jsonl(path)
            all_raw.extend(samples)
            sources_loaded = True

    if not sources_loaded and not args.all:
        print("Error: No data sources loaded. Use --all or specify paths.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Total raw samples: {len(all_raw)}")

    # ── Validate ────────────────────────────────────────────────────
    print("\n--- Validation ---")
    valid_samples = []
    error_counts: dict[str, int] = {}
    for i, sample in enumerate(all_raw):
        errs = validate_sample(sample, i)
        if errs:
            for e in errs:
                key = e.split("] ", 1)[-1] if "] " in e else e
                error_counts[key] = error_counts.get(key, 0) + 1
        else:
            valid_samples.append(sample)

    print(f"  Valid:   {len(valid_samples)}")
    print(f"  Invalid: {len(all_raw) - len(valid_samples)}")
    if error_counts:
        print("  Top errors:")
        sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])[:5]
        for err, count in sorted_errors:
            print(f"    [{count:>4}x] {err}")

    if not valid_samples:
        print("\nERROR: No valid samples after validation. Check data extraction.", file=sys.stderr)
        sys.exit(1)

    # ── Deduplicate ─────────────────────────────────────────────────
    print("\n--- Deduplication ---")
    seen_hashes: set[str] = set()
    seen_ids: set[str] = set()
    deduped = []
    dupe_hash = 0
    dupe_id = 0
    for s in valid_samples:
        h = sha256(s)
        tid = s["metadata"].get("task_id", "")
        if h in seen_hashes:
            dupe_hash += 1
            continue
        if tid and tid in seen_ids:
            dupe_id += 1
            continue
        seen_hashes.add(h)
        if tid:
            seen_ids.add(tid)
        deduped.append(s)

    print(f"  Content duplicates removed: {dupe_hash}")
    print(f"  Task ID duplicates removed: {dupe_id}")
    print(f"  After dedup:                {len(deduped)}")

    # ── Shuffle and Split ───────────────────────────────────────────
    print("\n--- Splitting ---")
    rng = random.Random(args.seed)
    rng.shuffle(deduped)

    n = len(deduped)
    n_train = int(n * args.split[0])
    n_val = int(n * args.split[1])
    train = deduped[:n_train]
    val = deduped[n_train:n_train + n_val]
    test = deduped[n_train + n_val:]

    print(f"  Train: {len(train)} ({args.split[0]*100:.0f}%)")
    print(f"  Val:   {len(val)} ({args.split[1]*100:.0f}%)")
    print(f"  Test:  {len(test)} ({args.split[2]*100:.0f}%)")

    # ── Statistics per split ────────────────────────────────────────
    print("\n--- Split Statistics ---")
    train_stats = compute_stats(train)
    val_stats = compute_stats(val)
    test_stats = compute_stats(test)

    for name, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        print(f"\n  {name}:")
        print(f"    Count:        {stats['count']}")
        print(f"    Verified:     {stats['verified_rate']}")
        print(f"    Multi-turn:   {stats['multi_turn_rate']}")
        print(f"    Teacher used: {stats['teacher_used_rate']}")
        print(f"    Avg msgs:     {stats['avg_messages']}")
        print(f"    Difficulty:   {stats['difficulty_distribution']}")

    # ── Write output ────────────────────────────────────────────────
    if args.dry_run:
        print("\n  [DRY RUN] No files written.")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        splits = {
            "train.jsonl": train,
            "val.jsonl": val,
            "test.jsonl": test,
        }
        for fname, data in splits.items():
            out_path = args.output / fname
            with open(out_path, "w", encoding="utf-8") as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            print(f"\n  Wrote {len(data)} samples → {out_path}")

        # Report
        report = {
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "total_raw": len(all_raw),
            "valid_after_validation": len(valid_samples),
            "valid_after_dedup": len(deduped),
            "validation_errors": error_counts,
            "dedup_content_removed": dupe_hash,
            "dedup_id_removed": dupe_id,
            "split_ratios": args.split,
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
        }
        report_path = args.output / "prepare_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n  Report: {report_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
