"""Fix training data: normalize system prompts and validate."""
import json

# The two system prompt variants found:
SHORT_PROMPT = "You are a Feiyue worker agent operating inside a Hermes runtime. Execute TaskContracts using Hermes tools."
LONG_PROMPT = """You are a Feiyue worker agent operating inside a Hermes runtime. You receive TaskContracts from a strong model (teacher) and execute them using the tools available in the Hermes environment.

RULES:
1. Plan before acting: think about what tools you need and in what order
2. Tool calls must be valid JSON in <tool_call> blocks
3. After each tool call, verify the result before proceeding
4. If verification fails, analyze the error and retry with corrections
5. Minimize unnecessary tool calls
6. All file paths must be relative to the project root
7. Never output secrets, API keys, or absolute paths

Available tools: file_read, file_write, list_directory, apply_patch, run_tests, run_linter, search_files, shell_exec, update_plan"""

# Use LONG prompt for all training data (has detailed tool-calling instructions)
TARGET_PROMPT = LONG_PROMPT

def validate_record(item):
    """Return (is_valid, reasons)"""
    messages = item.get('messages', [])
    reasons = []
    
    for i, msg in enumerate(messages):
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        # Check for actual truncation (content ending mid-sentence)
        truncation_indicators = ['"4. If verifi', 'By doing this', 'you can', 'should first']
        for indicator in truncation_indicators:
            if content.rstrip().endswith(indicator):
                reasons.append(f'msg[{i}] ({role}): truncated ending: "{content[-40:]}"')
        
        # Check unclosed tool_call
        if role == 'assistant':
            open_tags = content.count('<tool_call>')
            close_tags = content.count('</tool_call>')
            if open_tags > close_tags:
                reasons.append(f'msg[{i}] (assistant): unclosed tool_call ({open_tags} open vs {close_tags} close)')
                continue
            
            # Check JSON validity inside tool_call blocks
            import re
            blocks = re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL)
            for bi, block in enumerate(blocks):
                block = block.strip()
                if not block:
                    reasons.append(f'msg[{i}] (assistant): empty tool_call block #{bi}')
                    continue
                try:
                    parsed = json.loads(block)
                    if 'name' not in parsed:
                        reasons.append(f'msg[{i}] (assistant): tool_call #{bi} missing "name" field')
                except json.JSONDecodeError as e:
                    reasons.append(f'msg[{i}] (assistant): tool_call #{bi} invalid JSON: {str(e)[:60]}')
        
        # Check for empty assistant content
        if role == 'assistant' and not content.strip():
            reasons.append(f'msg[{i}] (assistant): empty content')
    
    return (len(reasons) == 0, reasons)


for src, dst in [
    ('data/train.jsonl', 'data/train_fixed.jsonl'),
    ('data/val.jsonl', 'data/val_fixed.jsonl'),
    ('data/test.jsonl', 'data/test_fixed.jsonl'),
]:
    normalized = 0
    items = []
    
    with open(src) as f:
        for line in f:
            item = json.loads(line)
            
            # Normalize system prompt
            for msg in item['messages']:
                if msg['role'] == 'system' and msg['content'] != TARGET_PROMPT:
                    msg['content'] = TARGET_PROMPT
                    normalized += 1
            
            items.append(item)
    
    # Validate all items
    for item in items:
        is_valid, reasons = validate_record(item)
        if not is_valid:
            # Fix: add missing </tool_call> tags
            for msg in item['messages']:
                if msg['role'] == 'assistant':
                    content = msg['content']
                    opens = content.count('<tool_call>')
                    closes = content.count('</tool_call>')
                    if opens > closes:
                        for _ in range(opens - closes):
                            content += '\n</tool_call>'
                        msg['content'] = content
                        normalized += 1
    
    # Save
    with open(dst, 'w') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f'{src}: {len(items)} records, {normalized} normalized → {dst}')

# Re-validate
print("\n--- Re-validation ---")
for fpath in ['data/train_fixed.jsonl', 'data/val_fixed.jsonl', 'data/test_fixed.jsonl']:
    valid = 0
    invalid = 0
    invalid_details = []
    with open(fpath) as f:
        for line_no, line in enumerate(f, 1):
            item = json.loads(line)
            is_valid, reasons = validate_record(item)
            if is_valid:
                valid += 1
            else:
                invalid += 1
                tid = item.get('metadata', {}).get('task_id', f'line-{line_no}')
                invalid_details.append((tid, reasons))
    
    pct = valid / max(valid + invalid, 1) * 100
    print(f'{fpath}: {valid}/{valid+invalid} valid ({pct:.0f}%)')
    if invalid_details:
        for tid, reasons in invalid_details[:3]:
            print(f'  {tid}:')
            for r in reasons:
                print(f'    → {r}')

print("\nDone!")
