# RTX 5060 Fine-Tuning Plan: Feiyue + Hermes Agent (Three-Capability)

> **Status:** Converged v2 — cross-checked by Hermes Agent + GLM-5.2 (2026-06-25)
> **Hardware:** RTX 5060 8GB GDDR7
> **Goal:** Fine-tune Phi-4-mini-instruct (3.8B) to write PRDs, do multi-step tool calling, AND write/review/fix code — all in one model, running locally.

---

## 0. Decision Log

| Decision | Hermes | GLM-5.2 | Converged |
|---|---|---|---|
| Model: Phi-4-mini-instruct (3.8B) | ✅ | ✅ | ✅ |
| Three-capability in one model | ✅ | ✅ | ✅ |
| Data split: 30/35/35 (PRD/tool/code) | 35/35/30 | **30/35/35** | **30/35/35** |
| Code sub-allocation | Not specified | Write 12% / Read 8% / Review 8% / Fix 7% | **Adopted** |
| Train 8-bit, Deploy 4-bit | ✅ | ✅ | ✅ |
| SFT → DPO (not GRPO) | ✅ | ✅ | ✅ |
| DPO LR: 4-5e-5 | Modified from 5e-6 | ✅ | **4-5e-5** |
| SFT: 2-3 epochs | Modified from 1 | ✅ | **2-3** |
| Multi-turn, windowed context | ✅ | ✅ | ✅ |
| Mode separation via system prompt | Added | ✅ | **Adopted** |
| Languages: max 2 | Not specified | Python + TypeScript | **Adopted** |
| 5+ step tool calling | Train harder | Architecture fix | **Architecture fix** |

---

## 1. Core Judgment

> **3.8B can do all three simultaneously.** The binding constraint is not parameter count — it's data discipline around mode separation. If the model can't distinguish "now I'm writing a PRD" from "now I'm writing code," all three capabilities degrade. With proper mode conditioning, 3.8B is adequate.

| Capability | Expected Level | Risk |
|---|---|---|
| PRD writing | Internal-use quality, structurally complete, moderate depth | Low — template-driven, saturates fast |
| Tool calling | Smooth for 2-3 steps, unreliable at 5+ | Medium — compensated by trajectory volume |
| Code (read/write/review/fix) | Reliable at function/module level, weak at multi-file | Medium — compensated by format + windowed training |

### Capability Interactions

- **Synergistic:** Code writing + tool calling share structured output and syntax adherence. PRD + task breakdown share hierarchical decomposition.
- **Competing:** Code review demands precision and narrow output; PRD demands breadth and long-form generation. Without strong task-conditioning signals, the model blends modes — verbose code reviews, terse PRDs.

---

## 2. Hardware Baseline

- **GPU:** RTX 5060, 8GB GDDR7, FP16 ~25 TFLOPS
- **Constraint:** Model + LoRA + optimizer states + activations + KV cache ≤ 8GB

---

## 3. Model Selection

**Phi-4-mini-instruct (3.8B)** — locked.

| Why | Detail |
|---|---|
| MMLU | 76.2 — top-tier for 3B class, critical for PRD reasoning |
| Function calling | Native |
| VRAM fit | 8-bit training fits 8GB |
| License | MIT |

**Language constraint:** Training data uses **Python + TypeScript only** (max 2 languages). Beyond that, per-language example count drops below 700 and the model produces syntactically mixed code.

---

## 4. Quantization Strategy

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

If code samples frequently exceed 3500 tokens: **truncate context windows, do not expand seq_len.**

---

## 5. Data Strategy

### 5.1 Composition

| Source | Ratio | Count | Content |
|---|---|---|---|
| Curated open-source | 40% | ~5K | OpenHermes, AgentInstruct — filtered subsets only |
| GPT-4 generated | 40% | ~5K | High-quality Feiyue data across all three capabilities |
| Hermes agent logs | 20% | ~2K | Real interactions, anonymized, including failure recovery |

**Total: ~12K examples**

### 5.2 Capability Split

```
PRD writing + task breakdown    30%  (~3.6K)  ← Saturates fast; 3.6K is enough
Multi-step tool calling         35%  (~4.2K)  ← Fragilest capability; needs most trajectories
Code (read/write/review/fix)    35%  (~4.2K)  ← Most sub-skills; needs balanced coverage
```

### 5.3 Code Sub-Allocation

| Sub-skill | Share | Count | Rationale |
|---|---|---|---|
| Code writing from PRD | 12% | ~1.4K | **Keystone skill** — connects PRD→code, the agent's value proposition |
| Code reading + analysis | 8% | ~1.0K | Compact models' strongest code axis |
| Code review | 8% | ~1.0K | Functional review, not style |
| Bug fix + write tests | 7% | ~0.8K | Includes self-correction examples |

### 5.4 Quality Gates

- Deduplication: semantic similarity >0.85 → keep one
- Format normalization per mode
- 10-15% negative samples for DPO preference pairs
- Audit open-source data BEFORE training

---

## 6. Data Format

### 6.1 Critical Design Rule: Multi-Turn + Windowed Context

> ⚠️ **No single-turn. No full-file context.** Single-turn is too synthetic. Full-file context teaches the model to expect complete visibility that won't exist at inference.

### 6.2 Code Trajectory Format

```
[system] You are a code agent. Read existing code → implement from PRD.

[user]
PRD excerpt: {relevant section, not full PRD}
Existing code:
```python
# auth/service.py (lines 23-58)
import ...
class AuthService:
    def __init__(self, ...): ...
    def validate_token(self, token: str) -> bool:
        {existing implementation}
```
Implement login().

[assistant]
Analysis: AuthService uses token-based validation pattern...
```python
def login(self, username: str, password: str) -> Optional[str]:
    user = self.db.find_user(username)
    if not user or not self._verify_password(password, user.hash):
        return None
    return self._generate_token(user.id)
```
Explanation: Reuses existing token generation from validate_token...
```

### 6.3 Format Requirements

- **Multi-turn** — trains the actual agent workflow (read context → reason → produce)
- **Windowed context** — function signature + 20 lines surrounding + imports. Model learns partial visibility.
- **15-20% of code samples explicitly reference a prior PRD** — trains PRD↔code linkage.
- **Include self-correction examples** — "write code → find error → fix" trajectories teach debugging-in-generation.
- **Clear EOS placement** — prevent runaway code generation.

### 6.4 Forbidden Patterns

- Single-turn "write a function that does X"
- Full repo dumps as context
- Code-only conversations with no PRD linkage
- Style-only preference pairs (naming, comments)

---

## 7. Training Pipeline

### 7.1 Phase 1 — SFT (Teach Format)

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
| Gradient checkpointing | **Required** — code samples are long |
| Early stopping | Validation loss plateau 3 epochs |
| Optimizer | AdamW 8-bit |

### 7.2 Phase 2 — DPO (Teach Correctness)

| Parameter | Value |
|---|---|
| **Learning rate** | **4-5e-5** ← CRITICAL: not 5e-6 |
| Beta | 0.1 |
| Epochs | 1–2 |
| LoRA | Same as SFT, continue training (don't reset) |

**Why 4-5e-5, not 5e-6:**
- SFT_LR × 0.2–0.3 is the correct ratio for post-SFT DPO
- At 5e-6, model won't meaningfully update from preferences
- Entire data investment is wasted if LR is wrong

**Monitoring:** Watch DPO loss AND held-out judge scores every 500 steps. If loss drops but judge score plateaus → overfitting SFT behaviors, back off LR.

### 7.3 DPO Preference Pairs — Code-Specific

| Axis | Share | Content |
|---|---|---|
| Correctness | 60% | Functionally correct vs has a bug. Both being correct teaches nothing. |
| Code quality / idiomaticity | 25% | Both correct, but one idiomatic + uses stdlib, the other "works but ugly." |
| Architecture / modularity | 15% | Both correct and clean, but one has better decomposition. |

**Forbidden preferences:**
- Style-only differences (naming, comments)
- "Longer is better" — 3.8B will learn verbose = preferred
- Syntax errors as "worse" — too easy, doesn't teach reasoning

### 7.4 Phase 3 — GRPO (Conditional, Discouraged)

Only consider if tool-calling <80% AFTER DPO, with validated reward function. Honest assessment: if DPO didn't fix it, GRPO probably won't either on 3.8B. Budget for system-prompt engineering instead.

---

## 8. Mode Separation (GLM-5.2's Strongest Recommendation)

> **3.8B lacks the capacity for implicit mode switching that larger models achieve naturally. Training data must do the work that parameters can't.**

### 8.1 System Prompt Per Mode (Every Training Example)

```
PRD mode:     "You are writing a product requirements document..."
Tool mode:    "You are calling tools to investigate and complete a task..."
Code mode:    "You are reviewing/writing code. Be concise and precise..."
```

**Cannot be vague.** The model learns mode-conditioning from the system prompt, not from inferring it from the input.

### 8.2 Format Enforcement

- PRDs → markdown headings
- Code → fenced code blocks
- Tool calls → JSON schema
- **Never let these overlap in training data.** The format itself becomes the mode signal.

### 8.3 DPO Pairs Penalizing Cross-Mode Leakage

Build pairs where the "worse" response is functionally correct but in the wrong mode (e.g., code review written as a PRD section). This directly trains against contamination.

### 8.4 Fallback

If mode contamination shows up in eval and can't be fixed with data adjustments: inference-time routing layer (lightweight classifier injecting mode-specific system prompt). But solve it in data first.

---

## 9. Evaluation Framework

### Layer 1 — Auto Metrics (every 200 steps)
- Format compliance rate (JSON schema validation per mode)
- Tool name accuracy, parameter type correctness
- Perplexity on held-out set

### Layer 2 — Task Success (every 500 steps, 50 test cases)
- 1-step, 2-3 step, 5+ step tool chain success rates
- PRD structural completeness (automated rubric)
- **Code unit-test pass rate** ← critical: measures actual correctness, not style

### Layer 3 — LLM-Judge (every epoch, 20 items)
- GPT-4 / Claude blind evaluation vs baseline
- PRD: completeness, executability, trade-off depth
- Code: correctness, idiomaticity, architecture

### Layer 4 — Mode Contamination Detection (every 500 steps) ← NEW

- Input contains code snippet, task is "write PRD" → if model generates code, contamination detected
- Input contains PRD text, task is "write code" → if model generates PRD sections, contamination detected
- Track as independent metric

---

## 10. Deployment & Hermes Integration

### 10.1 Model Serving

```
LoRA adapter (~50MB)
    → Merge with base model
    → Quantize to 4-bit GPTQ/AWQ
    → Serve via vLLM (OpenAI-compatible API)
         or llama.cpp server (GGUF)
```

### 10.2 Hermes Provider Config

```yaml
# ~/.hermes/config.yaml
custom_providers:
  phi4-agent:
    base_url: http://localhost:8000/v1
    api_key: not-needed
    model: phi-4-mini-agent
```

### 10.3 Inference-Time System Prompts

```
PRD mode:
"You are a product-aware engineering agent on Hermes. Write a concise PRD with scope, trade-offs, and success criteria."

Code mode:
"You are a code agent on Hermes. Read context, then write precise, idiomatic code. Be concise."

Tool mode:
"You are calling tools to investigate and complete a task. Use JSON for tool calls."
```

For multi-step tool chains >4 steps: route to code/rules orchestration layer, not the LLM.

---

## 11. Iteration Plan

| Week | Phase | Deliverables |
|---|---|---|
| 1 | Data | Curated open-source, GPT-4 generation (all 3 modes), Hermes logs, DPO pairs |
| 2 | SFT | 2-3 epochs, all eval baselines established, mode contamination tracked |
| 3 | DPO | 1-2 epochs, SFT vs SFT+DPO comparison, verify mode separation |
| 4 | Deploy | 4-bit quantization, Hermes integration, end-to-end testing |

---

## 12. Risk Matrix

| Risk | Probability | Mitigation |
|---|---|---|
| **Mode contamination (biggest risk)** | 40% | System prompt per mode + DPO anti-leakage pairs + dedicated eval |
| 3.8B code inadequate for multi-file tasks | 35% | Accept — scope to function/module level; complex refactors go to larger model |
| DPO preference pairs insufficient quality | 25% | GPT-4 batch generation + human calibration on 10% |
| Code token fragmentation (poor tokenizer efficiency) | 20% | Monitor token efficiency; if severe, custom tokenizer merge |
| Training OOM with long code samples | 15% | Truncate context windows, do not expand seq_len |
| 5+ step tool calling unreliable | 60% | **Architecture fix** — route to code/rules layer |

---

## 13. Success Criteria

- PRD quality: GPT-4 judge score ≥80% of GPT-4 baseline
- Tool calling: ≥85% on 1-3 steps, ≥60% on 4+
- Code: ≥70% unit-test pass rate on generated code from PRD
- Format compliance: ≥95% per mode
- Mode contamination: <5% cross-mode leakage
- Deployment: Hermes agent completes real feature request end-to-end

---

*Plan synthesized by Hermes Agent. Cross-checked and approved by GLM-5.2 via Tabbit CDP bridge.*
