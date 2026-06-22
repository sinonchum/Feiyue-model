# Feiyue-Model PRD v2.0: Text-First Agent Model for the Feiyue+Hermes Runtime

> **Version**: v2.0 · **Date**: 2026-06-22 · **Status**: Architecture Finalized
>
> **v1.0 → v2.0**: Single-turn SFT → Multi-turn SFT+GRPO pipeline. Local worker → Self-evolving agent runtime core.

---

## 0. Positioning: Why This Beats Holo3 Without Vision

Holo3 (H Company) is a **vision-language computer-use model**. It sees screens, clicks buttons, navigates GUIs. It scored 78.85% on OSWorld-Verified. It costs $0.40–$3.00 per million tokens via API.

Feiyue-Model takes the **opposite bet**: 

| Axis | Holo3 | Feiyue-Model |
|------|-------|-------------|
| **Modality** | Vision + text | Text only |
| **Core capability** | GUI perception & navigation | Tool orchestration & self-correction |
| **Training** | Synthetic env factory + curated RL | Real execution traces + verification-gated RL |
| **Self-improvement** | Static after training | **Continuous** — every Feiyue run produces new training data |
| **Cost per call** | $0.001–0.01 (API) | **$0** (local RTX 5060) |
| **Privacy** | Data sent to H Company API | **Fully local** |
| **Evolvability** | Requires H Company to retrain | **Self-evolving flywheel** — model improves with every Feiyue wave |

**The wager**: In a text-only agent runtime like Hermes, deep tool-use mastery + relentless self-improvement beats broad-but-shallow vision-based computer use. Holo3 must generalize across infinite UIs; Feiyue-Model only needs to master one runtime — Hermes — and it has 152+ executions of evidence to learn from, with more generated every day.

---

## 1. Architecture Overview

```
                   ┌──────────────────────────────────────┐
                   │        Feiyue Self-Evolution Loop     │
                   │                                      │
  Human ──► Teacher(gpt-5.5) ──► TaskContract ──┐        │
                                                  ▼        │
                   ┌──────────────────────────────────┐    │
                   │   Feiyue-Model (local, $0)        │    │
                   │   ┌──────────────────────────┐   │    │
                   │   │ SFT: Trajectory Imitation │   │    │
                   │   │  ↓                        │   │    │
                   │   │ GRPO: Verifiable RL       │   │    │
                   │   │  ↓                        │   │    │
                   │   │ Continuous Fine-tuning    │◄──┼────┤── New evidence
                   │   └──────────────────────────┘   │    │    from each run
                   └──────────────────────────────────┘    │
                              │                            │
                              ▼                            │
                   ┌──────────────────┐                    │
                   │ Verifier (pytest) │── FAIL ──► Teacher│
                   │                   │           Retry   │
                   │ PASS              │                    │
                   └──────┬───────────┘                    │
                          ▼                                │
                   CandidateFileWrite ───► Merge ───► Git  │
                                                           │
                   New Evidence ←──────────────────────────┘
```

**Three training phases, one model:**

| Phase | Method | Data Source | Purpose |
|-------|--------|-------------|---------|
| **Phase 1: SFT** | Multi-turn trajectory imitation | Hermes Agent Reasoning Traces + Feiyue evidence + synthetic augmentation | Learn Hermes tool patterns and Feiyue workflow structure |
| **Phase 2: GRPO** | Trajectory-level RL with verifiable rewards | Feiyue TaskContracts + real execution feedback | Optimize for task completion and self-correction |
| **Phase 3: Continuous** | Monthly re-fine-tuning | New evidence from Feiyue runs | Self-evolution — never stops improving |

---

## 2. Base Model Selection

### Qwen 3 — the only viable text-agent base in 2026

| Candidate | Reasoning | Tool-Use | Agent Benchmarks | Apache 2.0 | Why Not |
|-----------|-----------|----------|------------------|------------|---------|
| **Qwen3-8B** | ★★★★☆ | ★★★★☆ | SOTA at 8B | ✅ | — |
| Qwen3-14B | ★★★★★ | ★★★★★ | Stronger | ✅ | Needs >8GB VRAM for training |
| Qwen3-32B | ★★★★★ | ★★★★★ | Best open 32B | ✅ | Needs A100/H100 |
| DeepSeek-V3 | ★★★★★ | ★★★☆☆ | Strong reasoning, weak tool-use | ❌ (custom license) | License, no fine-tune support |
| Llama-4 | ★★★☆☆ | ★★☆☆☆ | Agent-lagging | ❌ (Llama license) | Weak tool-calling natively |
| Mistral-Large | ★★★★☆ | ★★★☆☆ | Good generalist | ❌ (research only) | Not open for commercial use |

**Decision: Qwen3-8B-Instruct as primary target, Qwen3-14B as stretch.**

Qwen3 was explicitly trained on agent tasks (Qwen3 technical report, May 2025). It has thinking/non-thinking mode switching built in. Its tool-calling capability is native, not bolted on. And it's Apache 2.0.

For RTX 5060 8GB: Qwen3-8B with QLoRA fits comfortably (~5.5GB VRAM training, ~5GB inference).
For better hardware: Qwen3-14B-QLoRA fits in 12GB, offers meaningfully stronger reasoning.

---

## 3. Training Data Strategy

### 3.1 Real Data: Feiyue Execution Evidence (152+ files, growing)

**Categories and their training value:**

| Source | Count | Training Signal |
|--------|-------|-----------------|
| `workflow-smokes/` | 45 | Basic task→write→verify patterns |
| `multi-worker-workflows/` | 22 | Parallel execution, role assignment |
| `real-multi-worker-runs/` | 19 | Live profile execution with real models |
| `provider-runs/` | 16 | Individual model call logs — raw LLM output |
| `capability-history/` | 1 | Longitudinal 130+ run performance data |
| Teacher-retry pairs | ~30 | Failure→guidance→correction loops |

**What v1.0 got right**: These are gold. Real execution traces with verifiable outcomes.

**What v1.0 missed**: These need to be reformatted as **multi-turn trajectories**, not single-turn pairs. A Feiyue worker session isn't one message — it's a sequence:
1. System prompt → 2. TaskContract → 3. Initial attempt → 4. Verification result → 5. Teacher guidance → 6. Retry → 7. Final verification

Each of these is a **trajectory** with reward signals at each verification point.

### 3.2 External Data: Hermes Agent Reasoning Traces

[lambda/hermes-agent-reasoning-traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) — **14,701 real multi-turn tool-calling trajectories** captured from Kimi-K2.5 and GLM-5.1-FP8 running inside Hermes Agent.

Key stats:
- Average **24.3 turns** per sample
- Average **13.9 tool calls** per sample
- Nine categories: terminal, coding, browser, repo work, file ops, planning, etc.
- All tool executions are **real** (not synthetic) — Playwright browsers, file operations, code compilation
- Includes `<think>` reasoning blocks, `<tool_call>` invocations, `<tool_response>` results

**Why this matters**: This dataset teaches the model what Hermes Agent conversations look like — the rhythm of think→act→observe→think→act. Combined with Feiyue's verification-gated structure, it provides the foundation for multi-turn agentic behavior.

### 3.3 Synthetic Augmentation: Teacher-Generated Trajectories

Feiyue's teacher model (gpt-5.5) generates additional training samples:

1. **Difficulty curriculum**: Takes existing TaskContracts and modifies them to be harder (more files, more complex verification, multi-step dependencies)
2. **Tool diversity**: Generates tasks requiring terminal, git, search, and web tools (not just file writes)
3. **Error injection**: Creates TaskContracts with deliberate ambiguities to train self-correction

**Synthetic data quality gate**: All teacher-generated samples must pass through Feiyue's verification pipeline (dry-run with a real worker model) before entering the training set. Only samples where the verification command produces a clear pass/fail signal are included.

### 3.4 Data Format: Multi-Turn ChatML with Tool Calls

```json
{
  "messages": [
    {"role": "system", "content": "<Hermes Agent persona + Feiyue worker rules>"},
    {"role": "user", "content": "<TaskContract JSON>"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {...}}\n</tool_call>"},
    {"role": "tool", "content": "{\"success\": true, \"path\": \"...\"}"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"pytest -q\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"exit_code\": 1, \"output\": \"FAILED test_x\"}"},
    {"role": "assistant", "content": "Verification failed. The test expects X but got Y. Fixing..."},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {...}}\n</tool_call>"},
    {"role": "tool", "content": "{\"success\": true}"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"pytest -q\"}}\n</tool_call>"},
    {"role": "tool", "content": "{\"exit_code\": 0, \"output\": \"PASSED\"}"}
  ],
  "metadata": {
    "task_id": "real-repo-3c",
    "verification_passed": true,
    "teacher_used": true,
    "attempts": 2,
    "tools_used": ["write_file", "terminal"],
    "difficulty": "medium"
  }
}
```

---

## 4. Training Pipeline

### Phase 1: Supervised Fine-Tuning (Multi-Turn Trajectory Imitation)

**Goal**: Teach the model the rhythm of Hermes Agent tool use — think, act, observe, adapt.

**Data**:
- Hermes Agent Reasoning Traces: 10,000 samples (filter for code+terminal categories)
- Feiyue multi-turn trajectories: ~110 samples (reformatted from 152 evidence files)
- Teacher-generated synthetic trajectories: ~500 samples

**Config** (Unsloth QLoRA):

```yaml
model: unsloth/Qwen3-8B-Instruct-bnb-4bit
max_seq_length: 8192            # ↑ from 4096 — multi-turn needs longer context
load_in_4bit: true

lora_r: 32                       # ↑ from 16 — more capacity for tool patterns
lora_alpha: 64                   # ↑ from 32
lora_dropout: 0.05
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

learning_rate: 1e-4              # ↓ from 2e-4 — more stable for multi-turn
lr_scheduler: cosine
warmup_ratio: 0.1
num_epochs: 3
per_device_train_batch_size: 1   # ↓ from 2 — multi-turn samples are larger
gradient_accumulation_steps: 8   # ↑ from 4
packing: false                   # multi-turn + tool calls don't pack well
```

**Estimated time**: 4–6 hours on RTX 5060 8GB.

**Success criteria**: Model produces valid tool calls in correct Hermes format, follows verification feedback, makes at least one self-correction attempt on failure.

### Phase 2: GRPO — Reinforcement Learning with Verifiable Rewards

**Goal**: Optimize the model to maximize task completion rate, minimize teacher escalations.

**Why GRPO (not PPO/DPO)**:
- No critic model needed → fits in 8GB VRAM
- Group-relative scoring → stable on small batch sizes
- Proven for reasoning (DeepSeek-R1) and agent training (Fireworks, Bespoke Labs)
- Unsloth's GRPO implementation supports 7x longer context (up to 110K for Qwen3-8B on H100, ~16K on RTX 5060)

**Reward function** — trajectory-level, multi-component:

```python
def compute_reward(trajectory: list[Message]) -> float:
    """
    Components:
    1. Verification pass (0 or 1): Did the final verification command succeed?
    2. Efficiency bonus (-0.1 per extra attempt): Penalty for excessive retries
    3. Self-correction bonus (+0.2): Reward for successfully fixing own error
    4. Teacher avoidance bonus (+0.3): Reward for passing without teacher help
    5. Tool usage penalty (-0.05 per unnecessary tool call): Avoid tool spam
    """
    passed = trajectory_final_verification_passed(trajectory)
    attempts = count_attempts(trajectory)
    self_corrected = had_successful_self_correction(trajectory)
    used_teacher = had_teacher_guidance(trajectory)
    tool_calls = count_tool_calls(trajectory)
    min_expected_tools = estimate_expected_tools(trajectory)

    reward = 0.0
    if passed:
        reward += 1.0
    reward -= 0.1 * (attempts - 1)          # Efficiency
    if self_corrected:
        reward += 0.2                         # Self-correction
    if passed and not used_teacher:
        reward += 0.3                         # Teacher-free success
    if tool_calls > min_expected_tools * 1.5:
        reward -= 0.05 * (tool_calls - int(min_expected_tools * 1.5))

    return reward
```

**Training environment**: Feiyue's provider-free dry-run mode — tasks execute in isolated temp directories, no API calls, fully deterministic.

```yaml
# GRPO config (Unsloth GRPOTrainer)
model: ./feiyue-qwen-8b-sft     # From Phase 1
max_prompt_length: 4096
max_completion_length: 4096
num_generations: 4               # 4 completions per prompt for group scoring
temperature: 0.9
learning_rate: 5e-6
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
beta: 0.001                      # KL penalty coefficient
```

**Training data for GRPO**: 50–80 Feiyue TaskContracts with verifiable reward signals. Each contract is run in the dry-run environment, the model's trajectory is scored, and GRPO updates the policy.

**Estimated time**: 6–10 hours on RTX 5060 8GB.

### Phase 3: Continuous Self-Evolution

After initial deployment, every Feiyue run produces new evidence. Monthly re-fine-tuning:

```bash
# 1. Collect new evidence from the past month
python scripts/extract_training.py /path/to/Feiyue --since 2026-07-01

# 2. Filter: only include samples where the local model FAILED but teacher succeeded
#    (these are the highest-signal training examples)

# 3. Merge with existing SFT data (80% old, 20% new)
# 4. Re-run Phase 1 SFT for 1 epoch (not full retraining)
# 5. Re-run Phase 2 GRPO on challenging new contracts

# 6. A/B test: old model vs new model on 20 held-out tasks
# 7. Deploy if pass rate improves by >2 percentage points
```

This is the **self-evolution flywheel** — the model gets better every month, autonomously, from its own failures.

---

## 5. Inference Deployment

### 5.1 Serving Stack

| Option | Latency | VRAM | Throughput | Recommendation |
|--------|---------|------|------------|----------------|
| **vLLM + LoRA** | ~0.5s/token | ~6GB | High | **Primary** — production-grade |
| llama.cpp (GGUF) | ~1.2s/token | ~5GB | Low | Fallback if vLLM doesn't build on Windows |
| Ollama | ~0.8s/token | ~5.5GB | Medium | Easiest setup, less configurable |

### 5.2 Hermes Integration

```bash
# Serverai — custom provider pointing to local vLLM
hermes config set model.provider "custom:feiyue-qwen"
hermes config set model.base_url "http://localhost:8000/v1"
hermes config set model.api_key "not-needed"
hermes config set model.default_model "feiyue-qwen-8b-worker"
```

### 5.3 Routing

Feiyue's routing auto-escalates when the local model's verification fails:

```yaml
routes:
  worker:
    primary: feiyue-qwen-local       # $0, <2s
    fallback: feiyue-mid-deepseek-pro # API, used only on failure
```

Teacher (specification) always stays on gpt-5.5 — it's called <10% of runs and needs the strongest model.

---

## 6. Success Metrics

### 6.1 vs DeepSeek API (current baseline)

| Metric | DeepSeek API | Feiyue-Model v2 Target | Stretch |
|--------|-------------|----------------------|---------|
| Verification pass rate | ~75% | > 80% | > 85% |
| Self-correction rate | ~30% of failures | > 50% | > 65% |
| Teacher escalation rate | ~10% | < 8% | < 5% |
| Latency per call | 3–8s | < 2s | < 1s |
| Cost per 1000 calls | $3–5 | **$0** | **$0** |
| Monthly improvement | 0 (static) | +2pp pass rate | +5pp pass rate |

### 6.2 vs Holo3 (competitive benchmark)

Since Holo3 is vision-based and Feiyue-Model is text-only, direct benchmark comparison requires a text-only agent benchmark. Proposed:

| Benchmark | What It Measures | Holo3 (estimated) | Feiyue-Model Target |
|-----------|-----------------|-------------------|---------------------|
| **BFCL v3 Multi-Turn** | Multi-turn tool orchestration | ~45% (zero-shot, no vision tasks) | > 55% |
| **SWE-bench Lite** | Real-world bug fixes | N/A (vision model, not code) | > 20% (strong for 8B) |
| **Feiyue-Internal-50** | 50 held-out Feiyue TaskContracts | — | > 80% pass rate |
| **Hermes Tool Suite** | Hermes-native tool use across 9 categories | — | > 70% pass rate |

### 6.3 Self-Evolution Metrics

| Metric | Month 1 | Month 3 | Month 6 |
|--------|---------|---------|---------|
| Training samples | 600+ | 800+ | 1200+ |
| Pass rate (internal) | 80% | 84% | 88% |
| Novel task pass rate | 65% | 72% | 78% |
| Teacher escalations | 8% | 5% | 3% |

---

## 7. Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| **8B model insufficient capacity** | Medium | Qwen3-14B stretch target; route complex tasks to cloud fallback |
| **GRPO unstable on small batch** | Medium | Use Unsloth's memory-efficient GRPO with group size 4; fall back to DPO if unstable |
| **Multi-turn training OOM** | Medium | Gradient checkpointing + sequence chunking; Unsloth supports 7x longer context |
| **Synthetic data quality collapse** | High | **Verification gate**: every synthetic sample must pass through Feiyue dry-run before inclusion |
| **Model overfits to Feiyue patterns** | Medium | 80% Hermes traces + 20% Feiyue data in SFT mix; hold-out task evaluation |
| **vLLM doesn't build on Windows** | Low | llama.cpp GGUF fallback; Ollama as simplest path |
| **Continuous training causes regressions** | Medium | A/B test before deploy; keep previous checkpoint as rollback |

---

## 8. Phase Plan

### Phase 0: Data Preparation (Day 1)

- [ ] Download Hermes Agent Reasoning Traces from Hugging Face
- [ ] Reformat Feiyue evidence files as multi-turn trajectories with tool calls
- [ ] Generate 500 synthetic trajectories via teacher model (gpt-5.5)
- [ ] Verification-gate all synthetic data through Feiyue dry-run
- [ ] Create train/val/test splits (80/10/10)
- [ ] Build reward scoring harness for GRPO

### Phase 1: SFT Training (Day 2)

- [ ] Set up Unsloth on Serverai
- [ ] Run multi-turn SFT with config from §4-Phase1
- [ ] Evaluate: does model produce valid tool calls? does it self-correct on failure?
- [ ] Save LoRA adapter (~300MB with r=32)

### Phase 2: GRPO Training (Day 3–4)

- [ ] Set up Feiyue dry-run environment on Serverai
- [ ] Implement trajectory-level reward function
- [ ] Run GRPO with 50–80 contracts
- [ ] Monitor reward curve for convergence
- [ ] Evaluate against SFT-only baseline

### Phase 3: Integration & A/B Test (Day 4–5)

- [ ] Deploy vLLM + LoRA on Serverai
- [ ] Configure Hermes custom provider
- [ ] Run 50-task A/B test: Feiyue-Model vs DeepSeek API
- [ ] Measure pass rate, latency, self-correction rate

### Phase 4: Production Rollout (Day 5–7)

- [ ] Switch worker route to Feiyue-Model (primary) + DeepSeek (fallback)
- [ ] Monitor for 1 week
- [ ] Collect new training data from production runs
- [ ] Roll back to DeepSeek if pass rate drops >5pp

### Phase 5: Continuous Evolution (Monthly, ongoing)

- [ ] Extract new evidence from past month
- [ ] Filter for high-signal samples (model failures, teacher successes)
- [ ] Re-run SFT + GRPO with expanded dataset
- [ ] A/B test and deploy if improved
- [ ] Track self-evolution metrics

---

## 9. Competitive Analysis: Why This Architecture Beats Holo3 at Text Agency

### Holo3's Strengths (we don't compete on these)
- Sees and understands any GUI → we don't do vision
- Navigates web pages, fills forms, clicks buttons → not our lane
- 78.85% OSWorld → irrelevant benchmark for text agents

### Holo3's Weaknesses (we attack these)
- **Static**: Once trained, Holo3 never improves until H Company retrains it
- **Expensive**: $0.40–$3.00/M tokens. A 1000-call day costs $4–30
- **Generic**: Trained for "any enterprise UI" — no deep understanding of any specific runtime
- **No self-correction loop**: Holo3 doesn't learn from its own failures in production
- **Cloud-dependent**: All inference goes through H Company API

### Feiyue-Model's Moat
1. **Self-evolution**: Every Feiyue run produces training data. The model improves monthly without human intervention.
2. **Zero marginal cost**: Once trained, inference is free forever on existing hardware.
3. **Runtime mastery**: Instead of shallow coverage of infinite UIs, deep mastery of one runtime (Hermes) with its full tool surface.
4. **Verification-gated training**: Every training sample has a ground-truth pass/fail signal — no reward model hallucination, no RLHF ambiguity.
5. **Privacy**: All data stays local. No third party sees task contracts, code, or verification results.

---

## 10. Open Questions & Decisions

| Question | Current Decision | Rationale |
|----------|-----------------|-----------|
| Qwen3-8B vs 14B? | 8B first, 14B stretch | 8B fits RTX 5060 for both training and inference |
| SFT+GRPO vs SFT+DPO? | GRPO | No critic model needed; proven for agent training (Bespoke Labs +23%) |
| Fine-tune thinking mode too? | Yes | Feiyue tasks benefit from CoT before tool calls |
| Multi-turn context length? | 8192 | 7 tool calls × ~1K tokens each = 7K + prompt. 8K is safe ceiling on 8GB |
| Continuous training frequency? | Monthly | Balances improvement rate vs training cost/risk |
| Release model publicly? | TBD | Feiyue is public (MIT); model weights could be too. Decision after v1 stable. |

---

## Appendix A: Key References

| Source | Relevance |
|--------|-----------|
| [ASTRA (arXiv 2601.21558)](https://arxiv.org/abs/2601.21558) | SOTA tool-agent training: SFT + verifiable RL pipeline |
| [Hermes Agent Reasoning Traces](https://huggingface.co/datasets/lambda/hermes-agent-reasoning-traces) | 14,701 real multi-turn Hermes tool trajectories |
| [Unsloth GRPO for Agents](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/training-ai-agents-with-rl) | Memory-efficient multi-turn agent RL |
| [ART (Agent Reinforcement Trainer)](https://github.com/openpipe/art) | Multi-turn agent RL on top of Unsloth GRPO |
| [Fireworks: Best Practices for Multi-Turn RL](https://fireworks.ai/blog/best-practices-for-multi-turn-RL) | RL > SFT alone; trajectory-level rewards; strong base model |
| [Bespoke Labs: RL for Tool Use](https://www.bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning) | GRPO improved Qwen2.5-7B tool use by 23% with 100 samples |
| [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) | Qwen3 trained on agent tasks; thinking mode; Apache 2.0 |
| [Holo3 (H Company)](https://hcompany.ai/holo3) | Competitive baseline: 78.85% OSWorld, MoE, vision-only |

---

> **Bottom line**: Feiyue-Model v2.0 doesn't try to be a cheaper Holo3. It's a fundamentally different bet — a model that masters one runtime deeply, improves itself continuously from real execution evidence, and costs nothing to run. In the text-agent space where Hermes operates, that's a winning hand.
