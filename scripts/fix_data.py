"""Fix truncated system prompts in training data."""
import json

FULL_SYSTEM_PROMPT = "You are a Feiyue worker agent operating inside a Hermes runtime. Execute TaskContracts using Hermes tools."

for src, dst in [
    ('data/train.jsonl', 'data/train_fixed.jsonl'),
    ('data/val.jsonl', 'data/val_fixed.jsonl'),
    ('data/test.jsonl', 'data/test_fixed.jsonl'),
]:
    fixed = 0
    total = 0
    with open(src) as f:
        lines = f.readlines()
    
    items = []
    for line in lines:
        item = json.loads(line)
        total += 1
        for msg in item['messages']:
            if msg['role'] == 'system' and len(msg['content']) < 50:
                # Likely truncated - check if it starts with our prefix
                if msg['content'].startswith('You are a Feiyue'):
                    msg['content'] = FULL_SYSTEM_PROMPT
                    fixed += 1
        
        # Also fix: ensure assistant turns with <tool_call> have </tool_call>
        for msg in item['messages']:
            if msg['role'] == 'assistant':
                content = msg['content']
                # Add missing </tool_call> if tool_call is present but close tag missing
                if '<tool_call>' in content:
                    # Count open vs close tags
                    opens = content.count('<tool_call>')
                    closes = content.count('</tool_call>')
                    if opens > closes:
                        # Add missing close tags
                        for _ in range(opens - closes):
                            content += '\n</tool_call>'
                        msg['content'] = content
        
        items.append(item)
    
    with open(dst, 'w') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f'{src}: {total} total, {fixed} system prompts fixed → {dst}')

# Now re-validate
print("\nRe-validating fixed data...\n")
from validate_data import validate_record

for fpath in ['data/train_fixed.jsonl', 'data/val_fixed.jsonl', 'data/test_fixed.jsonl']:
    valid = 0
    invalid = 0
    with open(fpath) as f:
        for line in f:
            item = json.loads(line)
            is_valid, reasons = validate_record(item)
            if is_valid:
                valid += 1
            else:
                invalid += 1
                if invalid <= 3:
                    tid = item.get('metadata', {}).get('task_id', '?')
                    print(f'  Still invalid: {tid}: {reasons}')
    
    print(f'{fpath}: {valid} valid, {invalid} invalid ({valid/(valid+invalid)*100:.0f}% clean)')
