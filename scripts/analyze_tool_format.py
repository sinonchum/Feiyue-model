"""Analyze <tool_call> XML format coverage in training data."""
import json

files = ['data/train.jsonl', 'data/val.jsonl', 'data/test.jsonl']
for fpath in files:
    with open(fpath) as f:
        lines = f.readlines()

    tool_call_count = 0
    direct_code_count = 0
    text_only_count = 0

    for line in lines:
        item = json.loads(line)
        for msg in item.get('messages', []):
            if msg.get('role') == 'assistant':
                content = msg.get('content', '')
                if '<tool_call>' in content:
                    tool_call_count += 1
                elif '```' in content:
                    direct_code_count += 1
                else:
                    text_only_count += 1

    total = tool_call_count + direct_code_count + text_only_count
    pct = lambda n: n / max(total, 1) * 100
    print(f'{fpath}: assistant turns={total}')
    print(f'  <tool_call> XML:   {tool_call_count:3d} ({pct(tool_call_count):.0f}%)')
    print(f'  direct code (```): {direct_code_count:3d} ({pct(direct_code_count):.0f}%)')
    print(f'  text/reasoning:    {text_only_count:3d} ({pct(text_only_count):.0f}%)')
    print()
