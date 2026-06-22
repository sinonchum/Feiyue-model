# Feiyue Training Data Format

## Schema Specification

Feiyue training data follows ChatML format, suitable for fine-tuning with Unsloth, Axolotl, or any ChatML-compatible framework.

## ChatML Message Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "<system prompt describing worker role>"
    },
    {
      "role": "user",
      "content": "<TaskContract JSON or natural language task description>"
    },
    {
      "role": "assistant",
      "content": "<CandidateFileWrite JSON or structured output>"
    }
  ],
  "metadata": {
    "task_id": "unique-task-identifier",
    "status": "verified|needs_teacher|blocked",
    "difficulty": "easy|medium|hard",
    "domain": "code|docs|tests|config",
    "teacher_used": false,
    "verification_passed": true
  }
}
```

## System Prompt Template

```
You are a Feiyue worker agent operating inside a Hermes runtime. You receive TaskContracts from a strong model (teacher) and produce CandidateFileWrite outputs.

RULES:
1. Your response must be valid JSON matching the CandidateFileWrite schema
2. Output ONLY the JSON object — no markdown fences, no preamble
3. The 'path' field MUST be relative to the project root
4. The 'content' field contains the complete file contents
5. If you receive teacher guidance about a previous failure, incorporate it exactly
6. Every file write must match the verification criteria in the TaskContract

CandidateFileWrite schema:
{
  "writes": [
    {
      "path": "relative/path/to/file.py",
      "content": "complete file contents here"
    }
  ]
}
```

## User Message Format

```json
{
  "task_id": "unique-id",
  "description": "Natural language description of what to do",
  "context": "Relevant project context, file snippets, or constraints",
  "verification_command": "pytest tests/test_x.py -q",
  "allowed_files": ["path/to/edit.py"],
  "teacher_guidance": "Optional: guidance from teacher after failed attempt",
  "attempt_index": 1
}
```

## Assistant Message Format

```json
{
  "writes": [
    {
      "path": "relative/path/to/file.py",
      "content": "# Fixed implementation\n\ndef example():\n    return True\n"
    }
  ]
}
```

## Training Samples by Category

### 1. Positive Samples (verification_passed: true)
- Worker correctly understood TaskContract
- Produced valid CandidateFileWrite
- Verification command passed
- **Count from Feiyue**: ~60–80 samples

### 2. Teacher-Retry Pairs (verification_passed: true after retry)
- Worker initially failed → Teacher provided guidance → Worker retried successfully
- Training objective: learn to incorporate teacher feedback
- **Count from Feiyue**: ~30 pairs
- **Format**: Two consecutive message pairs in the same conversation

### 3. Negative Samples (verification_passed: false)
- Used for DPO/contrastive training (future phase)
- NOT used for initial SFT

## Data Extraction

Run the extraction script to generate training data from a Feiyue checkout:

```bash
python scripts/extract_training.py /path/to/Feiyue-checkout --output data/train.jsonl
```

Options:
- `--format chatml|alpaca|sharegpt` — Output format (default: chatml)
- `--split 0.8` — Train/val split ratio
- `--include-retries` — Include teacher-retry pairs
- `--max-samples 200` — Cap on output samples

## Validation

After extraction, validate output against the schema:

```bash
python scripts/validate_training_data.py data/train.jsonl
```

Checks:
- All messages have valid role/content
- All assistant responses parse as valid CandidateFileWrite
- All paths are relative (no absolute paths)
- No API keys or secrets leaked in content
