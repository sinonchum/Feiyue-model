# Feiyue-Model

> Fine-tuning Qwen3-4B for AI agent workflows: PRD generation, code review, and structured tool-calling.

[![Hugging Face](https://img.shields.io/badge/🤗_HuggingFace-Models-yellow)](https://huggingface.co/sinonchum/Qwen3-4B-Feiyue-v1-bf16)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

## Overview

**Feiyue-v1** is a fine-tuned [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) model optimized for three core AI agent functions:

1. **PRD Generation** — Structured product requirements with task decomposition, acceptance criteria, and dependency mapping
2. **Code Review** — Bug detection, style analysis, architecture review, and security vulnerability identification
3. **Tool-calling** — Structured function calling for agent orchestration pipelines

The model was trained using LoRA on a curated dataset of 2,132 samples and evaluated on BFCL v4 and MT-Bench benchmarks.

## Model Variants

| Variant | Format | Size | Link |
|---------|--------|------|------|
| **Adapter** | LoRA (PEFT) | ~63 MB | [sinonchum/Qwen3-4B-Feiyue-v1](https://huggingface.co/sinonchum/Qwen3-4B-Feiyue-v1) |
| **Merged bf16** ⭐ | Safetensors | ~8 GB | [sinonchum/Qwen3-4B-Feiyue-v1-bf16](https://huggingface.co/sinonchum/Qwen3-4B-Feiyue-v1-bf16) |
| **Q8_0** | GGUF 8-bit | 4.0 GB | [sinonchum/Qwen3-4B-Feiyue-v1-Q8_0](https://huggingface.co/sinonchum/Qwen3-4B-Feiyue-v1-Q8_0) |
| **Q4_K_M** | GGUF 4-bit | 2.5 GB | [sinonchum/Qwen3-4B-Feiyue-v1-Q4_K_M](https://huggingface.co/sinonchum/Qwen3-4B-Feiyue-v1-Q4_K_M) |

## Benchmark Results

### BFCL v4 (Berkeley Function Calling Leaderboard)

Evaluated on BFCL v4 exec categories using NVIDIA L40S GPU with AST-based scoring against ground truth.

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| `exec_simple` | **89.0%** | 89 | 100 |
| `exec_parallel` | **88.0%** | 44 | 50 |
| `exec_multiple` | **80.0%** | 40 | 50 |
| `exec_parallel_multiple` | **82.5%** | 33 | 40 |
| **Overall** | **85.8%** | **206** | **240** |

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) |
| Method | LoRA (PEFT) → Merged |
| LoRA Rank / Alpha | 8 / 16 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Training Precision | bfloat16 |
| Dataset | Feiyue v11_8k (2,132 samples) |
| Max Sequence Length | 8,192 tokens |
| Epochs | 3 |
| Learning Rate | 2e-4 (cosine, 10% warmup) |
| Effective Batch Size | 4 |
| GPU | NVIDIA L40S (48 GB) |
| Training Time | ~44 minutes |
| Final Train Loss | 0.6486 |

## Usage

### bf16 Merged Model (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "sinonchum/Qwen3-4B-Feiyue-v1-bf16",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("sinonchum/Qwen3-4B-Feiyue-v1-bf16")

messages = [
    {"role": "system", "content": "You are Feiyue, an AI agent."},
    {"role": "user", "content": "Review this code for security issues:\n\ndef login(username, password):\n    query = f\"SELECT * FROM users WHERE name='{username}' AND pass='{password}'\"\n    return db.execute(query)"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### LoRA Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, "sinonchum/Qwen3-4B-Feiyue-v1")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
```

### GGUF (llama.cpp / Ollama)

```bash
# Using Ollama
ollama import feiyue-v1 Qwen3-4B-Feiyue-v1-Q8_0.gguf

# Using llama.cpp
./llama-cli -m Qwen3-4B-Feiyue-v1-Q8_0.gguf -p "Review this code:" -n 512
```

## Repository Structure

```
Feiyue-model/
├── README.md
├── PRD.md                      # Product requirements document
├── BENCHMARK_PLAN.md           # Benchmark evaluation plan
├── data/
│   ├── v11_8k/                 # Final training dataset (2,132 samples)
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── format.md               # Training data schema
├── scripts/
│   ├── gen_vertex_data.py      # Training data generation via Vertex AI
│   ├── build_v11_data.py       # Dataset construction pipeline
│   ├── extract_training.py     # Extract training pairs from Feiyue evidence
│   ├── modal_train.py          # LoRA training on Modal L40S
│   └── merge_lora.py           # Merge adapter into base model
├── eval/
│   ├── run_bfcl.py             # BFCL v4 evaluation (Modal GPU)
│   ├── run_mtbench.py          # MT-Bench answer generation
│   ├── judge_mtbench.py        # MT-Bench DeepSeek judge
│   ├── run_swebench.py         # SWE-bench Lite inference
│   └── eval_swebench.py        # SWE-bench Lite evaluation harness
├── configs/
│   └── unsloth_qlora.yaml      # Unsloth QLoRA training config
├── Modelfile.q8                # Ollama Modelfile (Q8_0)
└── Modelfile.q4                # Ollama Modelfile (Q4_K_M)
```

## Training Pipeline

```
1. Data Generation     scripts/gen_vertex_data.py → Gemini Flash Lite
2. Data Curation       scripts/build_v11_data.py  → v11_8k dataset
3. LoRA Training       scripts/modal_train.py     → Modal L40S GPU
4. Merge Adapter       scripts/merge_lora.py      → bf16 merged model
5. Quantize            convert_gguf.py            → Q8_0 / Q4_K_M GGUF
6. Evaluate            eval/run_bfcl.py           → BFCL v4 (85.8%)
7. Upload              upload scripts             → HuggingFace Hub
```

## License

Apache 2.0 — same as the base Qwen3-4B model.

## Citation

```bibtex
@misc{feiyue-v1-2026,
  author = {Simon Qin},
  title = {Qwen3-4B-Feiyue-v1: Fine-Tuned Model for AI Agent Workflows},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/sinonchum/Qwen3-4B-Feiyue-v1-bf16}},
}
```
