"""
Clean V10 training data based on Gemini 3.1 Pro audit findings.
Removes:
1. Samples where final assistant response is ONLY "Task complete." (toxic)
2. Samples where final assistant response < 100 chars with no code markers
Produces cleaned train/val/test for V11.
"""
import json, re, os, shutil
from pathlib import Path

SRC_DIR = Path(r"C:\Users\simon\Feiyue-model\data\v10_final")
DST_DIR = Path(r"C:\Users\simon\Feiyue-model\data\v11_clean")

CODE_MARKERS = [r'```', r'\bdef\b', r'\bfunction\b', r'\bclass\b', r'\bimport\b',
                r'\bconst\b', r'\blet\b', r'\bvar\b', r'\bPRD\b', r'\bProduct Requirement',
                r'## ', r'\bSummary\b', r'\bvulnerability\b', r'\bbug\b', r'\bfix\b',
                r'\bissue\b', r'\bsecurity\b']

def has_content(content):
    """Check if response has substantive content beyond just 'Task complete.'"""
    # Strip think blocks
    no_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    if no_think.lower() in ('task complete.', 'task complete', ''):
        return False
    # Check for code/content markers
    if len(no_think) < 100:
        for marker in CODE_MARKERS:
            if re.search(marker, content):
                return True
        return False
    return True

def clean_file(src_path, dst_path):
    items = []
    with open(src_path, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    kept = []
    removed_task_complete = 0
    removed_short = 0

    for item in items:
        messages = item.get("messages", [])
        # Find last assistant message
        removed = False
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                no_think = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

                if no_think.lower() in ('task complete.', 'task complete', ''):
                    removed_task_complete += 1
                    removed = True
                elif len(no_think) < 100:
                    has_marker = any(re.search(m, content) for m in CODE_MARKERS)
                    if not has_marker:
                        removed_short += 1
                        removed = True
                break

        if not removed:
            kept.append(item)

    # Write cleaned file
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, 'w', encoding='utf-8') as f:
        for item in kept:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"  {src_path.name}: {len(items)} → {len(kept)} kept "
          f"(-{removed_task_complete} 'Task complete.', -{removed_short} short)")

    return len(items), len(kept), removed_task_complete, removed_short

# Clean all three splits
total_in, total_out, total_tc, total_short = 0, 0, 0, 0
for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    src = SRC_DIR / split
    dst = DST_DIR / split
    if src.exists():
        in_cnt, out_cnt, tc, short = clean_file(src, dst)
        total_in += in_cnt
        total_out += out_cnt
        total_tc += tc
        total_short += short

print(f"\nTotal: {total_in} → {total_out} (-{total_tc} 'Task complete.', -{total_short} short)")
print(f"Saved to: {DST_DIR}")

# Copy to standard training paths
for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
    src = DST_DIR / split
    dst = Path(r"C:\Users\simon\Feiyue-model\data") / f"v11_{split}"
    if src.exists():
        shutil.copy(src, dst)
        print(f"Copied: {dst}")
