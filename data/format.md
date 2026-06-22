# Feiyue Training Data Format v2.0

## Overview

Feiyue-Model training data follows **multi-turn ChatML format** with structured tool-call blocks. This format captures the full Feiyue worker trajectory: receive TaskContract → plan → execute tool calls → verify → self-correct if needed.

## ChatML Multi-Turn Message Format

```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "<TaskContract JSON>"},
    {"role": "assistant", "content": "<tool_call>\n{...}\n</tool_call>"},
    {"role": "tool", "content": "{...}"},
    {"role": "assistant", "content": "<tool_call>\n{...}\n</tool_call>"},
    {"role": "tool", "content": "{...}"},
    {"role": "assistant", "content": "Task complete. Verification passed."}
  ],
  "metadata": {
    "task_id": "unique-id",
    "status": "verified|failed|needs_teacher",
    "difficulty": "easy|medium|hard",
    "domain": "code|docs|tests|config|multi-file",
    "tools_used": ["file_write", "run_tests"],
    "teacher_used": false,
    "attempts": 1,
    "verification_passed": true
  }
}
```

## System Prompt

```
You are a Feiyue worker agent operating inside a Hermes runtime. You receive
TaskContracts from a strong model (teacher) and execute them using the tools
available in the Hermes environment.

RULES:
1. Plan before acting: think about what tools you need and in what order
2. Tool calls must be valid JSON in <tool_call> blocks
3. After each tool call, verify the result before proceeding
4. If verification fails, analyze the error and retry with corrections
5. Minimize unnecessary tool calls — each call should have a clear purpose
6. All file paths must be relative to the project root
7. Never output secrets, API keys, or absolute paths

Available tools: file_read, file_write, list_directory, apply_patch,
run_tests, run_linter, search_files, shell_exec, update_plan
```

## User Message (TaskContract) Format

```json
{
  "task_id": "real-repo-3c",
  "description": "Update marker file to pass verification",
  "verification_command": "grep -q EXPECTED_STRING docs/file.md",
  "allowed_files": ["docs/file.md"],
  "context": "The marker file needs updating...",
  "teacher_guidance": null,
  "attempt_index": 0
}
```

## Tool Call Format

Each tool invocation is wrapped in `<tool_call>` tags:

```
<tool_call>
{
  "name": "file_write",
  "arguments": {
    "path": "docs/file.md",
    "content": "# Updated file\nMARKER_STRING"
  }
}
</tool_call>
```

## Tool Response Format

```
<tool_response>
{
  "success": true,
  "path": "docs/file.md",
  "bytes_written": 42
}
</tool_response>
```

## Training Sample Categories

### 1. Positive Trajectories (verification_passed: true)
- Worker correctly understood TaskContract
- Produced valid tool call sequence
- Verification command passed on first attempt
- Source: workflow-smokes, real-multi-worker-runs with exit_code=0

### 2. Self-Correction Trajectories (verification_passed: true, attempts > 1)
- Worker failed initial attempt → analyzed error → retried → passed
- These teach the model self-correction behavior
- Source: teacher-retry pairs from multi-worker-workflows

### 3. Failed Trajectories (verification_passed: false)
- Used for DPO/GRPO contrastive training (rejected samples)
- Source: provider-runs with exit_code≠0
- NOT used in SFT phase

## Data Extraction

```bash
# Extract from Feiyue evidence files
python scripts/extract_training.py /path/to/Feiyue --output data/raw/

# Generate synthetic domain trajectories
python scripts/synth_trajectories.py --feiyue-root /path/to/Feiyue --output data/synthetic/

# Merge all sources, validate, split
python scripts/prepare_data.py --all --output data/
```

## Validation Rules

1. All messages have valid `role` in {system, user, assistant, tool}
2. All assistant tool-call messages parse as valid JSON with `name` + `arguments`
3. All tool messages parse as valid JSON
4. No absolute paths in any message content
5. No API keys or secrets in any message
6. At least one tool call per trajectory (non-trivial samples only)
7. Final message role is `assistant` (trajectory ends with model output)
8. Metadata fields present and valid
