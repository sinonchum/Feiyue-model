# RTX 5060 Fine-Tuning Plan: Feiyue + Hermes Agent

> **Status:** Converged — cross-checked by Hermes Agent + GLM-5.2 (2026-06-25)
> **Hardware:** RTX 5060 8GB GDDR7
> **Goal:** Fine-tune a small model to write PRDs, break down tasks, and do multi-step tool calling as a product-aware engineering agent within the Hermes ecosystem.

---

## 0. Decision Log

| Decision | Hermes | GLM-5.2 | Converged |
|---|---|---|---|
| Model: Phi-4-mini-instruct (3.8B) | ✅ | ✅ | ✅ |
| Train 8-bit, Deploy 4-bit | ✅ | ✅ | ✅ |
| SFT → DPO (not GRPO) | ✅ | ✅ | ✅ |
| DPO LR: 4-5e-5 | 5e-6 ❌ | **4-5e-5** ✅ | **4-5e-5** |
| SFT: 2-3 epochs | 1 ❌ | 2-3 ✅ | **2-3** |
| Data: 40/40/20 split | 70/30 ❌ | Curate more ✅ | **40/40/20** |
| 5+ step tool calling | Train harder | Architecture fix | **Architecture fix** |

---

## 1. Hardware Baseline

- **GPU:** RTX 5060, 8GB GDDR7, FP16 ~25 TFLOPS
- **Constraint:** Model + LoRA + optimizer states + activations + KV cache ≤ 8GB

---

## 2. Model Selection

**Phi-4-mini-instruct (3.8B)** — chosen and locked.

| Why | Detail |
|---|---|
| MMLU | 76.2 — top-tier for 3B class |
| Function calling | Native, not bolted-on |
| VRAM fit | 8-bit training fits 8GB with safe headroom |
| License | MIT — no restrictions |

Alternative considered: Qwen2.5-3B-Instruct. Rejected — function calling less native, MMLU lower.

---

## 3. Quantization Strategy

**Split strategy: train high, deploy low.**

| Phase | Quantization | Framework | VRAM |
|---|---|---|---|
| SFT + DPO training | 8-bit | bitsandbytes | ~6.7GB |
| Inference deployment | 4-bit GPTQ/AWQ | vLLM / llama.cpp | ~2.5GB |

### Training VRAM Budget (8-bit, seq=4096)

```
8-bit model weights:         3.8 GB
LoRA r=32 params:           0.01 GB
LoRA optimizer states:      0.04 GB
Gradients (LoRA only):      0.01 GB
Activations (checkpointed): 1.5 GB
KV cache (inference pass):  0.5 GB
CUDA overhead:              0.8 GB
─────────────────────────────────
Total:                     ~6.7 GB  ✓ (8GB safe)
```

If seq=8192: activations double to ~3GB, borderline. Start at 4096, extend later.

---

## 4. Data Strategy

### 4.1 Composition

| Source | Ratio | Count | Content |
|---|---|---|---|
| Curated open-source | 40% | ~5K | OpenHermes, AgentInstruct — PRD + tool-calling subsets only |
| GPT-4 generated | 40% | ~5K | High-quality Feiyue data: PRDs, task breakdowns, tool trajectories |
| Hermes agent logs | 20% | ~2K | Real interactions, anonymized, including failure recovery cases |

**Total: ~12K examples**

### 4.2 Quality Gates

- Deduplication: semantic similarity >0.85 → keep one
- Format normalization: unified JSON tool-calling schema, PRD template
- 10-15% negative samples for DPO preference pairs
- Audit open-source data BEFORE training — quality variance is the #1 risk

### 4.3 Format (ChatML)

```json
{
  "messages": [
    {"role": "system", "content": "You are a product-aware engineering agent on Hermes Agent..."},
    {"role": "user", "content": "Write a PRD for [feature] and break it into dev tasks"},
    {"role": "assistant", "content": "## PRD\n...\n## Task Breakdown\n1. ...\n2. ..."},
    {"role": "tool", "name": "search_codebase", "content": "..."},
    {"role": "assistant", "content": "Based on codebase analysis, adjust task 2 to..."}
  ]
}
```

---

## 5. Training Pipeline

### 5.1 Phase 1 — SFT (Teach Format)

| Parameter | Value |
|---|---|
| Epochs | 2–3 |
| Learning rate | 2e-4 |
| Schedule | Cosine, warmup=10% |
| LoRA rank | r=32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (7) |
| Batch size | 1 |
| Gradient accumulation | 16 (effective batch=16) |
| Max sequence length | 4096 |
| Early stopping | Validation loss plateau 3 epochs |
| Optimizer | AdamW 8-bit |

### 5.2 Phase 2 — DPO (Teach Correctness)

| Parameter | Value |
|---|---|
| **Learning rate** | **4-5e-5** ← CRITICAL: not 5e-6 |
| Beta | 0.1 |
| Epochs | 1–2 |
| LoRA | Same as SFT, continue training (don't reset) |

**Why 4-5e-5, not 5e-6:**
- SFT_LR × 0.2–0.3 is the correct ratio for post-SFT DPO
- At 5e-6, model won't meaningfully update from preferences
- You'll mistake "no change" for successful training
- Entire data investment is wasted if LR is wrong

**Monitoring:**
- Watch DPO loss AND held-out PRD-judge scores every 500 steps
- If loss drops but judge score plateaus → LR too high, overfitting SFT behaviors

**Preference pair construction:**
- Good PRD > Bad PRD (GPT-4 judged)
- Correct tool call > Wrong tool call
- Human spot-check 10% of pairs

### 5.3 Phase 3 — GRPO (Conditional, Discouraged)

**Only consider if:**
- Tool-calling success rate <80% AFTER DPO
- You have a validated, automated tool-calling reward function
- PRD quality reward is NOT included

**Honest assessment:** If DPO didn't fix tool calling, GRPO probably won't either on 3.8B. Budget for system-prompt engineering instead. 5+ step tool flows → route to code/rules, not the LLM.

---

## 6. Evaluation Framework

**No eval pipeline → don't start training.**

### Layer 1 — Auto Metrics (every 200 steps)
- Format compliance rate (JSON schema validation)
- Tool name accuracy
- Parameter type correctness
- Perplexity on held-out set

### Layer 2 — Task Success (every 500 steps, 50 test cases)
- 1-step tool call success rate
- 2-3 step tool chain success rate
- 5+ step tool chain success rate
- PRD structural completeness (automated rubric)

### Layer 3 — LLM-Judge (every epoch, 20 PRDs)
- GPT-4 / Claude blind evaluation vs baseline (GPT-4 direct output)
- Dimensions: completeness, executability, trade-off analysis depth
- Human calibration on 10% of judgments

---

## 7. Deployment & Hermes Integration

### 7.1 Model Serving

```
LoRA adapter (~50MB)
    → Merge with base model
    → Quantize to 4-bit GPTQ/AWQ
    → Serve via vLLM (OpenAI-compatible API)
         or llama.cpp server (GGUF)
```

### 7.2 Hermes Provider Config

```yaml
# ~/.hermes/config.yaml
custom_providers:
  phi4-agent:
    base_url: http://localhost:8000/v1
    api_key: not-needed
    model: phi-4-mini-agent
```

### 7.3 System Prompt

```
You run on Hermes Agent. You are a product-aware engineering agent.

When given a feature request:
1. Write a concise PRD with scope, trade-offs, and success criteria
2. Break down into tasks ordered by dependency
3. For each task, use available tools to inspect the codebase
4. Adjust the plan based on what you find

For multi-step tool chains >4 steps, route to the orchestration layer.
```

---

## 8. Iteration Plan

| Week | Phase | Deliverables |
|---|---|---|
| 1 | Data | Curated 5K open-source, GPT-4 generation, Hermes logs, DPO pairs |
| 2 | SFT | 2-3 epochs, eval baselines established, all metrics tracked |
| 3 | DPO | 1-2 epochs, SFT vs SFT+DPO comparison, decision on GRPO |
| 4 | Deploy | 4-bit quantization, Hermes integration, end-to-end testing |

---

## 9. Risk Matrix

| Risk | Probability | Mitigation |
|---|---|---|
| 3.8B can't write good enough PRDs | 30% | System prompt engineering + template constraints; accept internal-use quality |
| DPO preference pairs insufficient quality | 25% | GPT-4 batch generation + human calibration on 10% |
| Training OOM at seq=4096 | 15% | Gradient checkpointing; reduce to seq=2048 if needed |
| 5+ step tool calling unreliable | 60% | **Architecture fix, not model fix** — route to code/rules layer |
| vLLM/GGUF serving issues | 20% | Validate deployment pipeline BEFORE starting training |

---

## 10. Success Criteria

- PRD quality: GPT-4 judge score ≥80% of GPT-4 baseline
- Tool calling: ≥85% success on 1-3 step chains, ≥60% on 4+ steps
- Format compliance: ≥95%
- Deployment: Hermes agent can complete a real feature request end-to-end

---

*Plan synthesized by Hermes Agent. Cross-checked and approved by GLM-5.2 via Tabbit CDP bridge.*
