# Feiyue-Model PRD: Local Qwen 3 8B Worker Fine-Tuning

> **Version**: v0.1 · **Date**: 2026-06-22 · **Status**: Planning

---

## 1. Problem Statement

Feiyue's worker execution layer currently relies on cloud API calls:
- `feiyue-weak-deepseek-flash` (primary worker)
- `feiyue-mid-deepseek-pro` (mid worker for complex tasks)
- `feiyue-strong-gpt55` (teacher guidance)

**Cost**: ~$5–15/month at current usage (growing with scale).
**Latency**: 3–8 seconds per worker call (network + inference).
**Privacy**: All task contracts and code context sent to third-party APIs.
**Vendor lock-in**: Tied to DeepSeek/GPT availability and pricing.

## 2. Proposed Solution

Fine-tune **Qwen 3 8B Instruct** on Feiyue's own execution evidence to create a local drop-in replacement for the worker profile. The fine-tuned model:

1. Accepts a TaskContract as input
2. Produces CandidateFileWrite as structured output
3. Understands verification failure feedback for self-correction
4. Runs locally on Serverai's RTX 5060 8GB

The **teacher** (gpt-5.5) remains as cloud API — it's called rarely (< 10% of runs) and needs the strongest model for specification.

## 3. Training Data

### 3.1 Source

Feiyue's `.hermes/` directory contains **152 structured evidence files** from Wave 1–14:

| Category | Count | Content |
|----------|-------|---------|
| `workflow-smokes/` | 45 | Provider-free dry runs |
| `multi-worker-workflows/` | 22 | Multi-profile parallel runs |
| `real-multi-worker-runs/` | 19 | Live profile execution |
| `provider-runs/` | 16 | Individual profile call records |
| `capability-history/` | 1 | 130-run longitudinal history |

### 3.2 Sample Format

Each training sample is a ChatML-formatted pair:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a Feiyue worker agent. Given a TaskContract, produce CandidateFileWrite JSON. If verification fails, incorporate teacher guidance and retry."
    },
    {
      "role": "user",
      "content": "{\"task_id\":\"real-repo-3c\",\"description\":\"Update marker file with REAL_REPO_3C_RETRY_OK\",\"verification_command\":\"grep -q REAL_REPO_3C_RETRY_OK docs/file.md\",\"teacher_guidance\":\"Replace the failing marker...\"}"
    },
    {
      "role": "assistant",
      "content": "{\"writes\":[{\"path\":\"docs/file.md\",\"content\":\"# Updated\\nREAL_REPO_3C_RETRY_OK\"}]}"
    }
  ]
}
```

### 3.3 Training/Validation Split

- **Training**: ~80 positive samples (verification_passed=true) + ~30 teacher-retry pairs
- **Validation**: ~20 held-out samples
- **Test**: Feiyue's existing 725+ pytest suite (provider-free verification)

## 4. Fine-Tuning Configuration

### 4.1 Hardware

| Component | Detail |
|-----------|--------|
| Machine | Serverai (Windows 11 Pro) |
| GPU | RTX 5060 8GB VRAM |
| CUDA | sm_120 (Blackwell) |
| Python | 3.12 |
| PyTorch | 2.12+cu128 |

### 4.2 Unsloth QLoRA Config

```yaml
model: Qwen/Qwen3-8B-Instruct
load_in_4bit: true
bnb_4bit_quant_type: nf4
bnb_4bit_compute_dtype: bfloat16

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.1
num_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
max_seq_length: 4096
```

**Estimated training time**: 3–6 hours on RTX 5060 8GB.

### 4.3 Alternative: Axolotl

```yaml
base_model: Qwen/Qwen3-8B-Instruct
model_type: AutoModelForCausalLM
tokenizer_type: AutoTokenizer

adapter: qlora
lora_r: 16
lora_alpha: 32

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

bf16: auto
gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
optimizer: adamw_8bit
lr_scheduler: cosine
learning_rate: 0.0002
```

## 5. Integration with Hermes

### 5.1 Profile Setup

```bash
# On Serverai — configure custom provider pointing to local model
hermes config set model.providers.custom."feiyue-qwen".base_url "http://localhost:8080/v1"
hermes config set model.providers.custom."feiyue-qwen".api_key "not-needed"
hermes config set model default_model "qwen3-8b-feiyue-worker" --provider "custom:feiyue-qwen"
```

### 5.2 Routing Update

In Feiyue's `.hermes/model-routing.yaml`:

```yaml
routes:
  worker:
    primary: feiyue-qwen-local    # ← new local model
    fallback: feiyue-mid-deepseek-pro  # cloud fallback
```

### 5.3 Serving

Options for serving the LoRA adapter:
- **llama-cpp** with GGUF conversion (CPU-friendly, slower)
- **vLLM** with LoRA adapter (GPU, fast)
- **Ollama** with custom Modelfile

## 6. Success Metrics

| Metric | Baseline (DeepSeek API) | Target (Qwen 3 8B) |
|--------|------------------------|---------------------|
| Worker call latency | 3–8s | < 2s |
| Worker cost per call | $0.001–0.005 | $0 |
| Verification pass rate | ~75% | > 70% |
| Teacher escalation rate | ~10% | < 15% |
| Monthly cost (1000 calls) | $5 | $0 |

**Success criteria**: If Qwen verification pass rate ≥ 70% of DeepSeek baseline, the cost savings justify the swap. Teacher escalation provides a safety net for the remaining gap.

## 7. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| 8B model insufficient for complex tasks | Teacher (gpt-5.5) fallback; route complex tasks to mid-level cloud model |
| Training data too small | Feiyue's 152 evidence files provide 80+ positive samples + 30+ retry pairs — sufficient for LoRA |
| VRAM constraints | QLoRA 4-bit fits in ~5.5GB; RTX 5060 8GB is adequate |
| Model drift over time | Capability-history tracking enables continuous evaluation; re-fine-tune monthly |

## 8. Phase Plan

### Phase 1: Data Extraction (Day 1)
- Run `scripts/extract_training.py` on full Feiyue checkout
- Validate training pairs against Pydantic schemas
- Split train/val/test

### Phase 2: Fine-Tuning (Day 1–2)
- Set up Unsloth on Serverai
- Run QLoRA fine-tuning
- Save LoRA adapter (~150 MB)

### Phase 3: Integration (Day 2)
- Set up local inference server (vLLM or llama-cpp)
- Configure Hermes custom provider
- A/B test against DeepSeek on 10 representative tasks

### Phase 4: Validation (Day 2–3)
- Run `feiyue-runs capability-history` for longitudinal comparison
- Execute provider-free smoke tests
- Compare pass rate, latency, teacher escalation rate

### Phase 5: Production Rollout (Day 3+)
- Switch worker route to Qwen
- Monitor for 1 week
- Fall back to DeepSeek if pass rate drops > 30%

## 9. Open Questions

1. **Qwen 3 4B vs 8B?** 4B fits more comfortably on 8GB VRAM and may be sufficient for the worker role. Start with 8B, fall back to 4B if VRAM is tight.
2. **Multi-turn training?** Current data is single-turn (TaskContract → writes). Multi-turn teacher-retry pairs exist and could improve self-correction.
3. **Curator model?** Same approach could train a local curator for asset distillation — evaluate after worker model is stable.
