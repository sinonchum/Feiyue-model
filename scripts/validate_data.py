"""Validate and clean training data per Gemini 3.1 Pro diagnosis."""
import json
import re

def validate_record(item):
    """Return (is_valid, reasons)"""
    messages = item.get('messages', [])
    reasons = []
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # 1. Check for truncated content (trailing unfinished text)
        if content.endswith(('"4. If verifi', 'the', 'a', 'an', 'the ', ',', ' in')):
            reasons.append(f'msg[{i}] ({role}): truncated ending: "{content[-40:]}"')
        
        # 2. Check unclosed tool_call
        if role == 'assistant':
            tool_calls = re.findall(r'<tool_call>', content)
            tool_closes = re.findall(r'</tool_call>', content)
            if tool_calls and not tool_closes:
                reasons.append(f'msg[{i}] (assistant): <tool_call> without </tool_call>')
                continue
            if len(tool_calls) != len(tool_closes):
                reasons.append(f'msg[{i}] (assistant): mismatched tool_call tags ({len(tool_calls)} open vs {len(tool_closes)} close)')
                continue
            
            # 3. Check JSON validity inside tool_call blocks
            for tci, (tc_start, tc_end) in enumerate(zip(
                [m.start() for m in re.finditer(r'<tool_call>', content)],
                [m.start() for m in re.finditer(r'</tool_call>', content)]
            )):
                json_str = content[tc_start + len('<tool_call>'):tc_end].strip()
                if not json_str:
                    reasons.append(f'msg[{i}] (assistant): empty tool_call #{tci}')
                    continue
                try:
                    parsed = json.loads(json_str)
                    # Check required fields
                    if 'name' not in parsed:
                        reasons.append(f'msg[{i}] (assistant): tool_call #{tci} missing "name"')
                except json.JSONDecodeError as e:
                    reasons.append(f'msg[{i}] (assistant): tool_call #{tci} invalid JSON: {str(e)[:60]}')
        
        # 4. Check for empty content
        if role in ('assistant',) and not content.strip():
            reasons.append(f'msg[{i}] ({role}): empty content')
    
    return (len(reasons) == 0, reasons)


# Validate all data files
for fpath in ['data/train.jsonl', 'data/val.jsonl', 'data/test.jsonl']:
    print(f"\n{'='*60}")
    print(f"Validating: {fpath}")
    print(f"{'='*60}")
    
    valid = []
    invalid = []
    
    with open(fpath) as f:
        for line_no, line in enumerate(f, 1):
            item = json.loads(line)
            is_valid, reasons = validate_record(item)
            
            if is_valid:
                valid.append(item)
            else:
                task_id = item.get('metadata', {}).get('task_id', f'line-{line_no}')
                invalid.append({
                    'line': line_no,
                    'task_id': task_id,
                    'reasons': reasons
                })
    
    total = len(valid) + len(invalid)
    print(f"Total: {total} | Valid: {len(valid)} | Invalid: {len(invalid)}")
    
    if invalid:
        print(f"\n❌ Invalid records ({len(invalid)}):")
        for inv in invalid[:10]:  # Show first 10
            print(f"  Line {inv['line']}: {inv['task_id']}")
            for r in inv['reasons']:
                print(f"    → {r}")
        if len(invalid) > 10:
            print(f"  ... and {len(invalid)-10} more")
    
    # Save cleaned data
    if fpath.startswith('data/train'):
        clean_path = 'data/train_clean.jsonl'
    elif fpath.startswith('data/val'):
        clean_path = 'data/val_clean.jsonl'
    else:
        clean_path = 'data/test_clean.jsonl'
    
    with open(clean_path, 'w') as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"\nSaved {len(valid)} clean records to {clean_path}")

print("\nDone!")
