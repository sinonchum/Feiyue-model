# Feiyue-Model: Local Fine-Tuned Worker for the Feiyue AI Orchestrator

> Fine-tuning Qwen 3 8B to replace cloud API calls in the [Feiyue](https://github.com/sinonchum/Feiyue) self-evolving AI development loop. Zero API cost, sub-second latency, sovereign execution.

## Overview

Feiyue is a verification-gated AI agent orchestrator that uses **strong models** for specification and **weak models** for execution. Today, the worker layer calls DeepSeek/GPT APIs for every task. This project fine-tunes **Qwen 3 8B** to run locally on an RTX 5060, replacing those API calls entirely.

```
Before:  Human → Strong Model (API) → TaskContract → Weak Model (API $$) → Verifier
After:   Human → Strong Model (API) → TaskContract → Qwen 3 8B (local, $0) → Verifier
```

## Why This Works

Feiyue has accumulated **152 structured evidence files** across 14 development waves. Each file captures:

- **TaskContract** (what to do) → **CandidateFileWrite** (what was done) → **Verification** (did it pass?)
- **Teacher guidance** when the worker fails → **Worker retry** with corrected approach
- **Capability history** tracking 130+ runs across multiple model profiles

This is perfect training data: clean {prompt → response → verification} triples.

## Repository Structure

```
Feiyue-model/
├── README.md              ← You are here
├── PRD.md                 ← Full product requirements document
├── data/
│   ├── format.md          ← Training data schema specification
│   ├── catalog.json       ← Summary of all 152 evidence files
│   └── samples/
│       ├── worker_initial.json   ← Worker first attempt
│       ├── worker_retry.json     ← Worker after teacher guidance
│       ├── teacher_guidance.json ← Teacher correcting worker
│       └── multi_student.json    ← Multi-profile parallel execution
├── scripts/
│   └── extract_training.py ← Extract training pairs from Feiyue evidence
└── configs/
    └── unsloth_qlora.yaml  ← Unsloth QLoRA training config
```

## Quick Start

```bash
# 1. Extract training data from a Feiyue checkout
python scripts/extract_training.py /path/to/Feiyue

# 2. Fine-tune (on Serverai RTX 5060 8GB)
# See configs/unsloth_qlora.yaml

# 3. Deploy as Hermes profile
hermes config set model provider=custom:feiyue-qwen
hermes config set model model=qwen3-8b-feiyue-worker
```

## Hardware Target

| Component | Spec |
|-----------|------|
| GPU | RTX 5060 8GB (sm_120) |
| Base Model | Qwen/Qwen3-8B-Instruct |
| Method | QLoRA 4-bit (NF4), r=16 |
| Adapter Size | ~150 MB |
| Inference | < 1s per call |

## License

MIT — same as Feiyue.
