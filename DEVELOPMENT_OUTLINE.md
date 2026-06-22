# Feiyue-Model Development Outline

> **Canonical development outline for Feiyue-Model v1.0.**
> Derived from [PRD.md](./PRD.md). All future development follows this outline unless overridden by the user.
>
> **Last updated**: 2026-06-22 · **PRD version**: v1.0

---

## Current Asset Inventory

| Asset | Path | Status | Purpose |
|-------|------|--------|---------|
| Data catalog | `data/catalog.json` | ✅ **Exists** — 152 evidence files indexed | Maps evidence to training categories |
| Data format spec | `data/format.md` | ⚠️ **Needs update** — v1.0 single-turn schema | Must be rewritten for multi-turn ChatML with tool calls |
| Training extraction | `scripts/extract_training.py` | ⚠️ **Needs rewrite** — v1.0 single-turn extraction | Must produce multi-turn trajectories with tool-call blocks |
| SFT config | `configs/sft_config.yaml` | ⚠️ **Needs update** — v1.0 params | r=16, seq_len=4096, lr=1e-4 (PatentFlow stack) |
| Training sample | `data/samples/train_sample.jsonl` | ❌ **Obsolete** — single-turn format | Replace with multi-turn trajectory examples |
| GRPO config | — | ❌ **Missing** | New file: `configs/grpo.yaml` |
| Reward scorer | — | ❌ **Missing** | New file: `scripts/reward_scorer.py` |
| Synthetic data gen | — | ❌ **Missing** | New file: `scripts/synth_trajectories.py` |
| A/B evaluation harness | — | ❌ **Missing** | New file: `scripts/ab_eval.py` |
| Continuous training cron | — | ❌ **Missing** | New file: `scripts/monthly_retrain.py` |

---

## Milestone Map

```
M0: Foundation (prerequisite for everything)
 │
 ├──► M1a: Data Pipeline (parallel)     ├──► M1b: Synthetic Data Gen (parallel)
 │         │                                      │
 │         └──────────┬───────────────────────────┘
 │                    ▼
 ├──► M2: SFT Training
 │         │
 │         ▼
 ├──► M3: GRPO Training
 │         │
 │         ▼
 ├──► M4: Inference Deployment
 │         │
 │         ▼
 ├──► M5: Integration & A/B Test
 │         │
 │         ▼
 └──► M6: Continuous Evolution
```

### Dependency Graph

```
                    M0
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
         M1a       M1b    (configs)
          │         │
          └────┬────┘
               ▼
              M2 ◄──── Configs updated
               │
               ▼
              M3 ◄──── Reward scorer built
               │
               ▼
              M4
               │
               ▼
              M5
               │
               ▼
              M6 ◄──── Cron job deployed
```

| Edge | Type | Reason |
|------|------|--------|
| M0 → M1a, M1b | Serial | Foundation tools/configs must exist before data work |
| M1a ∥ M1b | **Parallel** | Independent data sources, no shared state |
| M1a + M1b → M2 | Serial | Training needs the merged dataset |
| M2 → M3 | Serial | GRPO starts from SFT checkpoint |
| M3 → M4 | Serial | Need trained model to deploy |
| M4 → M5 | Serial | Need running inference server to integrate |
| M5 → M6 | Serial | Need production baseline to measure evolution |

---

## M0: Foundation — Repo & Toolchain Setup

**Goal**: All tools, configs, and environments ready. No training data work starts before this is verified.

### M0.1: Serverai Environment

| Task | Technology | Detail |
|------|-----------|--------|
| CUDA verification | `nvidia-smi`, `python -c "import torch; print(torch.cuda.is_available())"` | RTX 5060 must report sm_120, 8GB VRAM |
| Python env | `uv` or `conda` | Python 3.12, PyTorch 2.12+cu128 |
|| BitsAndBytes | `pip install bitsandbytes>=0.45` | 4-bit NF4 quantization |
|| TRL install | `pip install trl>=0.14` | For SFTTrainer + GRPOTrainer |
|| PEFT install | `pip install peft>=0.14` | LoRA adapters |
| vLLM install | `pip install vllm` | For inference serving; Windows may need WSL |
| Hugging Face login | `huggingface-cli login` | For model download and push |

**Files created**: None (environment only)

**Dependencies**: Serverai machine accessible, RTX 5060 driver installed

**Parallel with**: Nothing — everything depends on this

### M0.2: Repo Structure

```bash
Feiyue-model/
├── configs/
│   ├── sft_config.yaml          # ← Update from v1.0
│   └── grpo.yaml                 # ← NEW
├── scripts/
│   ├── extract_training.py       # ← Rewrite for multi-turn
│   ├── synth_trajectories.py     # ← NEW
│   ├── reward_scorer.py          # ← NEW
│   ├── ab_eval.py                # ← NEW
│   └── monthly_retrain.py        # ← NEW
├── data/
│   ├── catalog.json              # ← Existing (OK)
│   ├── format.md                 # ← Rewrite for multi-turn
│   ├── train.jsonl               # ← Will be generated
│   ├── val.jsonl                 # ← Will be generated
│   ├── test.jsonl                # ← Will be generated
│   └── samples/
│       └── multi_turn_sample.jsonl  # ← NEW
├── reward_envs/                  # ← NEW directory
│   └── (GRPO task environments)
├── PRD.md                        # ← Existing (v2.0)
├── DEVELOPMENT_OUTLINE.md        # ← This file
└── README.md                     # ← Update
```

### M0.3: Config Files

**File**: `configs/sft_config.yaml` (update from v1.0)

```yaml
# QLoRA SFT — multi-turn trajectory imitation
# Stack: trl.SFTTrainer + peft + bitsandbytes (PatentFlow proven)
model_name: "Qwen/Qwen3-4B-Instruct-2507"
max_seq_length: 4096
load_in_4bit: true
bnb_4bit_quant_type: "nf4"
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_use_double_quant: true

r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"

learning_rate: 1.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.1
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false

optim: "adamw_8bit"
weight_decay: 0.01
max_grad_norm: 1.0

dataset_kwargs:
  data_files:
    train: "data/train.jsonl"
    validation: "data/val.jsonl"
packing: false

output_dir: "./feiyue-qwen-4b-sft"
save_strategy: "epoch"
save_total_limit: 2

logging_steps: 5
report_to: "none"
```

**File**: `configs/grpo.yaml` (NEW)

```yaml
# GRPO — trajectory-level RL with verifiable rewards
model_path: "./feiyue-qwen-4b-sft"
max_prompt_length: 2048
max_completion_length: 2048
num_generations: 4
temperature: 0.9
learning_rate: 5.0e-6
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
beta: 0.001

reward_function: "scripts/reward_scorer.py"
environment: "feiyue_dry_run"  # provider-free, deterministic

output_dir: "./feiyue-qwen-4b-grpo"
save_strategy: "steps"
save_steps: 50
logging_steps: 5
report_to: "none"
```

### M0 Acceptance

- [ ] `nvidia-smi` shows RTX 5060 with ≥4GB free VRAM
- [ ] `python -c "import torch, trl, peft, bitsandbytes; print('OK')"` exits 0
- [ ] `huggingface-cli whoami` returns authenticated user
- [ ] All config files pass YAML syntax check
- [ ] All `scripts/*.py` files exist as stubs (can be empty)
- [ ] `data/format.md` rewritten for multi-turn ChatML

---

## M1a: Data Pipeline — Multi-Turn Extraction

**Goal**: Convert 152 Feiyue evidence files into multi-turn ChatML trajectories with tool-call blocks. Produce train/val/test splits.

### M1a.1: Update Data Format Spec

**File**: `data/format.md`

Rewrite from single-turn to multi-turn tool-calling ChatML. The key changes:

```
OLD (v1.0):  system → user(TaskContract) → assistant(CandidateFileWrite)
NEW (v2.0):  system → user(TaskContract) → assistant(<tool_call>) → tool(response)
               → assistant(<tool_call>) → tool(response)
               → ... (multi-turn) ...
               → assistant(final result)
```

Include the [Hermes Agent persona + Feiyue worker rules] system prompt, `<tool_call>`/`<tool_response>` format, and the multi-component metadata schema.

### M1a.2: Rewrite Extraction Script

**File**: `scripts/extract_training.py`

Rewrite from single-turn to multi-turn trajectory extraction. Key modules:

```python
# Module structure:
def extract_provider_runs_multi_turn(hermes_dir: Path) -> list[dict]:
    """Extract multi-turn trajectories from provider-runs/.
    Each trajectory includes: initial attempt → verification → retry → re-verification.
    Tool calls (write_file, terminal) are extracted from command history."""

def extract_workflow_smokes_multi_turn(hermes_dir: Path) -> list[dict]:
    """Extract from workflow-smokes/ with execution stages as turns."""

def extract_multi_worker_multi_turn(hermes_dir: Path) -> list[dict]:
    """Extract parallel worker trajectories with role-specific tool calls."""

def format_as_chatml(trajectory_events: list[dict]) -> dict:
    """Convert raw evidence events into ChatML messages with tool-call blocks."""

def reconstruct_tool_calls(evidence: dict) -> list[dict]:
    """Reconstruct the sequence of tool calls from evidence artifacts.
    Each evidence file has artifacts showing which files were written
    and which verification commands were run."""
```

**Technology**: Python stdlib `json`, `pathlib`, `argparse`. No heavy deps.

**Output format**:

```json
{
  "messages": [
    {"role": "system", "content": "<system prompt>"},
    {"role": "user", "content": "{\"task_id\": \"...\", ...}"},
    {"role": "assistant", "content": "<tool_call>...</tool_call>"},
    {"role": "tool", "content": "{\"success\": true, ...}"},
    ...
  ],
  "metadata": {
    "task_id": "...",
    "verification_passed": true,
    "teacher_used": false,
    "attempts": 1,
    "tools_used": ["write_file"],
    "difficulty": "easy",
    "source": "workflow-smokes",
    "source_run_id": "..."
  }
}
```

### M1a.3: Generate Dataset Splits

**Script**: Same extraction script, with `--split 0.8 --val-split 0.1` flags.

| Split | Count (est.) | Purpose |
|-------|-------------|---------|
| `train.jsonl` | ~90 samples | SFT training |
| `val.jsonl` | ~12 samples | Hyperparameter tuning |
| `test.jsonl` | ~12 samples | Final evaluation (held out) |

Stratified by difficulty and source category to ensure balanced splits.

### M1a.4: Download Hermes Agent Reasoning Traces

**Script**: New script section or standalone.

```bash
# From Hugging Face
python -c "
from datasets import load_dataset
ds = load_dataset('lambda/hermes-agent-reasoning-traces', 'kimi', split='train')
# Filter for code + terminal categories
filtered = ds.filter(lambda x: x['category'] in ('Terminal & Coding', 'Repository Work'))
# Down-sample to ~10K
filtered = filtered.select(range(10000))
filtered.to_json('data/hermes_traces.jsonl')
"
```

### M1a.5: Merge Feiyue + Hermes Datasets

**Script**: In `extract_training.py`, add `--merge-hermes` flag.

Merge strategy: 80% Hermes traces + 20% Feiyue trajectories.
This ratio prevents overfitting to Feiyue's specific patterns while ensuring the model learns Feiyue's verification-gated workflow.

### M1a Dependencies

| Depends on | Status |
|-----------|--------|
| M0 complete | Required |
| M1b | None — runs in parallel |

### M1a Acceptance

- [ ] `python scripts/extract_training.py /path/to/Feiyue --merge-hermes` exits 0
- [ ] `data/train.jsonl` contains ≥10,000 valid ChatML samples
- [ ] Every sample has `messages` list where all `role` values are in `{system, user, assistant, tool}`
- [ ] All tool-call messages parse as valid JSON with `name` and `arguments` keys
- [ ] All tool-response messages parse as valid JSON
- [ ] Validation: 100% of `assistant` content containing `"writes"` is valid CandidateFileWrite schema
- [ ] No API keys, absolute paths, or secrets in any sample
- [ ] `data/val.jsonl` and `data/test.jsonl` contain held-out samples not in `train.jsonl`
- [ ] Difficulty distribution: easy 40%, medium 40%, hard 20% (±10pp each)

---

## M1b: Synthetic Data Generation

**Goal**: Generate 500+ multi-turn trajectories from teacher model (gpt-5.5). All must pass the verification gate.

**Runs in parallel with M1a**.

### M1b.1: Build Synthetic Trajectory Generator

**File**: `scripts/synth_trajectories.py`

```python
"""
Generate synthetic multi-turn trajectories using the Feiyue teacher model.

Pipeline:
1. Template selection: pick a TaskContract template
2. Difficulty scaling: modify parameters to increase complexity
3. Teacher generation: ask gpt-5.5 to produce a full trajectory
4. Verification gate: run the trajectory through Feiyue dry-run
5. Output: valid ChatML trajectory or discard
"""

def generate_difficulty_curriculum(base_contracts: list[dict]) -> list[dict]:
    """Take existing contracts and create harder variants:
    - More files to modify (1 → 3 → 5+)
    - More complex verification (single grep → multi-file pytest)
    - Multi-step dependencies (file A depends on file B)"""

def generate_tool_diversity(hermes_tools: list[str]) -> list[dict]:
    """Generate contracts requiring under-represented tools:
    - terminal: shell commands, git operations
    - web_search: research tasks
    - read_file: code understanding
    - patch: targeted edits"""

def generate_error_injection(contracts: list[dict]) -> list[dict]:
    """Deliberately ambiguous contracts to train self-correction:
    - Missing context that the worker must request
    - Ambiguous verification criteria requiring interpretation
    - Contracts where initial approach must fail"""

def verification_gate(trajectory: dict, feiyue_root: Path) -> bool:
    """Run the trajectory through Feiyue's provider-free dry-run.
    Returns True only if verification command produces clear pass/fail."""
```

### M1b.2: Integration with Feiyue Teacher

The teacher model is gpt-5.5 (via OpenAI Codex provider). The script calls Hermes in one-shot mode:

```bash
hermes chat -q "
Generate a multi-turn Feiyue worker trajectory for this TaskContract:
{contract_json}

The trajectory must include:
1. Worker receiving the contract
2. Worker making an initial attempt (write_file tool call)
3. Verification running (terminal tool call)
4. If verification fails: worker analyzing the error and retrying
5. Final verification passing

Output format: ChatML JSONL with tool_call and tool_response blocks.
" --model gpt-5.5 --provider openai-codex
```

### M1b.3: Quality Gate Implementation

Every synthetic trajectory must pass through Feiyue's provider-free dry-run mode BEFORE entering the training set.

Implementation approach:
1. Extract the TaskContract from the trajectory
2. Execute the contract in Feiyue's dry-run env (isolated temp dir)
3. Run the verification command
4. If exit code == 0 → accept trajectory into training set
5. If exit code != 0 → discard (synthetic data hallucinated a bad solution)

This is the **only** quality mechanism that can prevent synthetic data poisoning. No heuristic filtering, no embedding similarity — only real verification.

### M1b Dependencies

| Depends on | Status |
|-----------|--------|
| M0 complete | Required |
| M1a | None — runs in parallel |
| Feiyue checkout accessible | Required for verification gate |
| Teacher model (gpt-5.5) accessible | Required for generation |

### M1b Acceptance

- [ ] `scripts/synth_trajectories.py` runs without errors
- [ ] Generates ≥500 trajectories across all three strategies (curriculum, diversity, error-injection)
- [ ] ≥80% of generated trajectories pass the verification gate
- [ ] No trajectory in output has `verification_passed: false` as final state
- [ ] Tool diversity: all of {write_file, terminal, read_file, patch} appear in ≥10% of trajectories each
- [ ] Merged with M1a output: `data/train.jsonl` now contains ≥10,500 samples

---

## M2: SFT Training — Multi-Turn Trajectory Imitation

**Goal**: Produce a Qwen3-4B LoRA adapter that can execute multi-turn Hermes tool-calling trajectories. Model must produce valid tool calls, follow verification feedback, and attempt self-correction.

**Depends on**: M1a + M1b complete (merged dataset ready)

### M2.1: Run SFT Training

**Environment**: Serverai RTX 5060 8GB, Windows 11 Pro

```bash
# On Serverai
cd C:\Users\Simon\feiyue-model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
# ... training script using configs/sft_config.yaml
"
```

Estimated: 2–3 hours, ~3.5GB VRAM.

### M2.2: Training Monitoring

Track during training:
- Loss curve (should decrease monotonically)
- GPU memory (must stay <7.5GB)
- Perplexity on val set (every 50 steps)

Save checkpoints at each epoch.

### M2.3: Basic Behavioral Eval (Pre-GRPO)

Before proceeding to GRPO, verify the model learned the basics. Run 20 hand-crafted evaluation prompts:

| Test | Expected Behavior |
|------|-------------------|
| Simple write_file | Produces valid `<tool_call>` with correct JSON |
| Write + verify | Writes file, calls terminal for verification |
| Verification fail → retry | Sees FAIL, modifies file, re-verifies |
| Multi-file task | Writes 2+ files in sequence |
| No tool spam | Doesn't repeat the same tool call without new info |
| Follows system prompt | Doesn't output markdown fences around JSON |

### M2 Acceptance

- [ ] Loss converged (final loss < 0.5 × initial loss)
- [ ] No OOM during training
- [ ] Basic eval: ≥15/20 prompts produce valid tool calls
- [ ] ≥10/20 prompts show correct multi-turn behavior (write → verify → retry)
- [ ] Self-correction: on verification-fail prompts, model makes ≥1 retry attempt in ≥50% of cases
- [ ] LoRA adapter saved: `./feiyue-qwen-4b-sft/adapter_model.safetensors` exists (~150MB)
- [ ] Adapter loads and infers without error

**Phase promotion gate**: Basic eval ≥15/20. If not met, debug training data or config before proceeding to M3.

---

## M3: GRPO Training — Verifiable Reinforcement Learning

**Goal**: Optimize the SFT model to maximize task completion rate and minimize teacher escalations. The model learns through trial-and-error in a deterministic, verifiable environment.

**Depends on**: M2 complete (SFT checkpoint ready)

### M3.1: Build Reward Scoring Harness

**File**: `scripts/reward_scorer.py`

```python
"""
Trajectory-level reward scorer for GRPO training.

Operates inside Feiyue's provider-free dry-run environment.
Each trajectory is executed against a sandbox temp directory,
and the reward is computed from the execution outcome.
"""

def compute_reward(trajectory: list[dict], task_contract: dict, sandbox_dir: Path) -> float:
    """
    1. Replay the trajectory in sandbox_dir (execute each tool call)
    2. Run the verification command from task_contract
    3. Score based on:
       - Verification pass: +1.0
       - Efficiency penalty: -0.1 per extra attempt
       - Self-correction bonus: +0.2 if fixed own error
       - Teacher-free bonus: +0.3 if passed without teacher help
       - Tool spam penalty: -0.05 per unnecessary tool call
    Returns float in approximately [-0.5, 1.5] range.
    """

def replay_tool_call(tool_call: dict, sandbox_dir: Path) -> dict:
    """Execute a tool call in the sandbox. Supports:
    - write_file: write content to sandbox_dir/path
    - terminal: execute command in sandbox_dir (whitelist: grep, pytest, python, cat)
    - read_file: read from sandbox_dir/path
    Returns tool response dict."""

def verify(contract: dict, sandbox_dir: Path) -> bool:
    """Run the verification command. Returns True if exit_code == 0."""

def get_teacher_guidance(contract: dict, sandbox_dir: Path) -> str:
    """Call teacher model to get guidance on a failed task.
    Only called when verification fails and trajectory indicates retry."""
```

### M3.2: Build GRPO Task Environments

**Directory**: `reward_envs/`

Each task environment is a self-contained Python module:

```python
# reward_envs/task_01_single_file.py
TASK_CONTRACT = {
    "task_id": "grpo-01-single-file",
    "description": "Fix calc.py: make add(a,b) return a+b",
    "verification_command": "python -c 'from calc import add; assert add(2,3)==5'",
    "allowed_files": ["calc.py"],
    "initial_state": {
        "calc.py": "def add(a, b):\n    return a - b\n"  # Bug: subtracts instead of adds
    }
}
```

Create 50–80 such environments covering:
- Single-file fixes (20)
- Multi-file features (20)
- Test-driven tasks (15)
- Config/documentation updates (5)
- Error recovery from ambiguous contracts (10)

### M3.3: Run GRPO Training

```bash
# On Serverai
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import yaml

# Load SFT adapter
model, tokenizer = load_model(
    model_name='Qwen/Qwen3-4B-Instruct-2507',
    adapter_path='./feiyue-qwen-4b-sft',
)

# Load GRPO config
with open('configs/grpo.yaml') as f:
    config = yaml.safe_load(f)

# Train
trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=GRPOConfig(**config),
    train_dataset=grpo_tasks_dataset,
    reward_funcs=[compute_reward],
)
trainer.train()
trainer.save_model('./feiyue-qwen-4b-grpo')
"
```

Estimated: 3–5 hours, ~4.5GB VRAM.

### M3.4: Monitor GRPO Metrics

Track during training:
- Mean reward per batch (should trend upward)
- Reward variance (should decrease as policy converges)
- KL divergence from SFT model (should stay <0.01)
- Pass rate on held-out GRPO tasks

### M3 Acceptance

- [ ] Mean reward increased by ≥0.2 from start to end of training
- [ ] Final pass rate on GRPO training tasks ≥75%
- [ ] KL divergence from SFT stayed <0.01 (model didn't collapse)
- [ ] No OOM during training
- [ ] GRPO adapter saved: `./feiyue-qwen-4b-grpo/adapter_model.safetensors` exists
- [ ] On 10 held-out tasks NOT used in GRPO training: pass rate ≥60%
- [ ] Compared to SFT-only baseline: GRPO model pass rate ≥10pp higher

**Phase promotion gate**: Held-out pass rate ≥60%. If not met, increase GRPO training data or adjust reward function before M4.

---

## M4: Inference Deployment

**Goal**: Feiyue-Model running as a local OpenAI-compatible API endpoint on Serverai, ready for Hermes to connect.

**Depends on**: M3 complete (GRPO checkpoint ready)

### M4.1: Merge LoRA and Export

```bash
# Merge LoRA adapter into base model weights
python scripts/merge_lora.py \
    --base Qwen/Qwen3-4B-Instruct-2507 \
    --adapter ./feiyue-qwen-4b-grpo \
    --output ./feiyue-qwen-4b-merged

# Convert to GGUF (for llama.cpp fallback)
python scripts/convert_to_gguf.py \
    --model ./feiyue-qwen-4b-merged \
    --output ./feiyue-qwen-4b-grpo.Q4_K_M.gguf \
    --quantization Q4_K_M
```

### M4.2: Deploy vLLM Server

```bash
# Primary: vLLM with LoRA adapter
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --enable-lora \
    --lora-modules feiyue-worker=./feiyue-qwen-4b-grpo \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

### M4.3: Fallback: llama.cpp Server

If vLLM fails to build on Windows:

```bash
# llama.cpp with GGUF
./llama-server \
    -m ./feiyue-qwen-4b-grpo.Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 \
    -c 8192 \
    -ngl 99  # Offload all layers to GPU
```

### M4.4: Smoke Test

```bash
# Test the endpoint
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "feiyue-worker",
    "messages": [
      {"role": "system", "content": "You are a Feiyue worker agent."},
      {"role": "user", "content": "{\"task_id\":\"smoke-1\",\"description\":\"Write a Python function add(a,b) that returns sum\",\"verification_command\":\"python -c \\\"from calc import add; assert add(2,3)==5\\\"\"}"}
    ],
    "temperature": 0.0,
    "max_tokens": 2048
  }'
```

### M4 Acceptance

- [ ] Server starts without errors, `curl /v1/models` returns 200
- [ ] Single-turn inference latency <2s per call
- [ ] Smoke test: model produces valid tool-call JSON
- [ ] Server stable for 1 hour under load (10 requests/minute)
- [ ] VRAM usage <5GB during inference
- [ ] Both vLLM and llama.cpp fallback tested (at least one working)

**Phase promotion gate**: Smoke test passes with valid tool-call output. Server stable.

---

## M5: Integration & A/B Test

**Goal**: Connect Hermes to the local model, run 50-task A/B test against DeepSeek API baseline.

**Depends on**: M4 complete (inference server running)

### M5.1: Hermes Configuration

```bash
# On Serverai
hermes config set model.provider "custom:feiyue-qwen"
hermes config set model.base_url "http://localhost:8000/v1"
hermes config set model.api_key "not-needed"
hermes config set model.default_model "feiyue-worker"
```

### M5.2: Feiyue Routing Update

In `~/.hermes/model-routing.yaml` (or Feiyue's routing config):

```yaml
routes:
  worker:
    primary:
      provider: custom:feiyue-qwen
      model: feiyue-worker
    fallback:
      provider: deepseek
      model: deepseek-chat
```

### M5.3: A/B Test Harness

**File**: `scripts/ab_eval.py`

```python
"""
Run 50 held-out Feiyue TaskContracts against both:
A: Feiyue-Model (local vLLM)
B: DeepSeek API (current baseline)

Compare: pass rate, latency, self-correction rate, teacher escalations.
"""

def run_ab_test(contracts: list[dict], model_a: str, model_b: str) -> dict:
    """For each contract:
    1. Run with model_a, record trajectory + pass/fail + latency
    2. Run with model_b, record trajectory + pass/fail + latency
    3. Compare results
    Returns summary statistics.
    """

def compare_latency(results_a: list, results_b: list) -> dict:
    """p50, p95, p99 latency comparison."""

def compare_pass_rate(results_a: list, results_b: list) -> dict:
    """Overall + per-difficulty pass rate."""

def compare_self_correction(results_a: list, results_b: list) -> dict:
    """Rate of successful retry after initial failure."""
```

### M5.4: A/B Test Execution

Run on 50 contracts from `data/test.jsonl` (held out, never seen in training):

```bash
python scripts/ab_eval.py \
    --contracts data/test.jsonl \
    --model-a "custom:feiyue-qwen/feiyue-worker" \
    --model-b "deepseek/deepseek-chat" \
    --output results/ab_test_$(date +%Y%m%d).json
```

### M5 Acceptance

- [ ] A/B test completed on all 50 contracts
- [ ] Feiyue-Model pass rate ≥80% or within 5pp of DeepSeek baseline
- [ ] Feiyue-Model latency <2s (p95), <3s (p99)
- [ ] Self-correction rate ≥50% of initial failures
- [ ] Teacher escalation rate <8%
- [ ] Results saved to `results/` directory

**Phase promotion gate**: Pass rate ≥80% OR within 5pp of DeepSeek. If below, return to M3 for more GRPO training.

---

## M6: Continuous Self-Evolution

**Goal**: The model improves every month from its own production failures. Fully automated.

**Depends on**: M5 complete (model in production)

### M6.1: Evidence Collector

**File**: `scripts/monthly_retrain.py` — section 1

```python
def collect_new_evidence(feiyue_root: Path, since_date: str) -> list[dict]:
    """Scan Feiyue's .hermes/ directory for evidence files modified since since_date.
    Only collect samples where:
    - The local model (feiyue-qwen) was the primary worker
    - The task FAILED verification initially
    - Teacher guidance was provided
    - The retry SUCCEEDED
    These are the highest-signal training examples."""

def filter_high_signal(samples: list[dict]) -> list[dict]:
    """Keep only: model_failed AND teacher_succeeded.
    These teach the model what it currently gets wrong."""
```

### M6.2: Merge + Retrain

**File**: `scripts/monthly_retrain.py` — section 2

```python
def merge_datasets(old_train: Path, new_samples: list[dict], ratio: float = 0.8):
    """Merge 80% old + 20% new. Prevents catastrophic forgetting."""

def retrain_sft(base_model: str, merged_data: Path, output: Path):
    """Run SFT for 1 epoch (not full retraining) on merged dataset."""

def retrain_grpo(sft_checkpoint: Path, grpo_tasks: Path, output: Path):
    """Run GRPO on challenging new contracts."""
```

### M6.3: A/B Deploy Gate

**File**: `scripts/monthly_retrain.py` — section 3

```python
def ab_test_and_deploy(old_model: Path, new_model: Path, test_contracts: Path):
    """Run A/B test on 20 held-out contracts.
    If new_model pass rate > old_model pass rate + 2pp:
        Deploy new_model (swap LoRA adapter in vLLM)
    Else:
        Keep old_model, log reason for rejection."""
```

### M6.4: Cron Job Setup

Deploy as a Hermes cron job or Windows Scheduled Task:

```bash
# As Hermes cron (preferred — handles delivery and logging)
hermes cron create "0 3 1 * *" \
  --name "feiyue-monthly-retrain" \
  --prompt "Run scripts/monthly_retrain.py to collect new evidence, retrain if enough data, and A/B test. Report results." \
  --workdir /path/to/Feiyue-model
```

### M6.5: Self-Evolution Metrics Dashboard

**File**: `results/evolution_log.jsonl`

Each monthly run appends a line:

```json
{
  "date": "2026-07-01",
  "new_samples": 23,
  "total_samples": 623,
  "old_pass_rate": 0.81,
  "new_pass_rate": 0.83,
  "deployed": true,
  "improvement_pp": 2.0
}
```

### M6 Acceptance

- [ ] `scripts/monthly_retrain.py` runs end-to-end without errors
- [ ] Evidence collector correctly identifies model-failed/teacher-succeeded pairs
- [ ] Merge ratio verified: 80% old + 20% new = 100% total samples
- [ ] Retraining completes within 6 hours
- [ ] A/B test gate functions: deploys only if improvement ≥2pp
- [ ] Rollback mechanism: previous checkpoint preserved before deploy
- [ ] Metrics log appended correctly after each run
- [ ] Cron job scheduled and verified (test run with small data)

---

## Parallelizable Workstreams

```
Week 1:
  Day 1 AM:  M0 (Foundation)                           [SERIAL — everything depends on it]
  Day 1 PM:  M1a (Data Pipeline)  ∥  M1b (Synthetic Data Gen)  [PARALLEL]
  Day 2:     M1a + M1b complete → M2 (SFT Training)    [SERIAL]
  Day 2-3:   M2 running (6h, can be overnight)

Week 2:
  Day 3:     M3 prep (reward scorer + envs)             [SERIAL — depends on M2]
  Day 3-4:   M3 (GRPO Training, 10h, overnight)

Week 3:
  Day 4-5:   M4 (Inference Deployment)                  [SERIAL — depends on M3]
  Day 5-6:   M5 (Integration + A/B Test)               [SERIAL — depends on M4]
  Day 7:     M5 results analysis, go/no-go decision

Month 2+:
  Monthly:   M6 (Continuous Evolution)                  [ONGOING]
```

### What CAN Run in Parallel

| Pair | Why Safe |
|------|----------|
| M1a ∥ M1b | Different data sources, no shared state. Both read from M0 configs, write to `data/`. |
| M3 reward scorer ∥ M3 task environments | Scorer is framework code; environments are task data. Different files. |
| M4 vLLM ∥ M4 llama.cpp fallback | Different binaries. Can test both simultaneously. |
| M6 evidence collection ∥ M6 dashboard | Collector writes data; dashboard reads from log file. No conflict. |

### What MUST Run Serially

| Edge | Why |
|------|-----|
| M0 → Everything | Tools, configs, env must exist first |
| M1 → M2 | Training needs dataset |
| M2 → M3 | GRPO needs SFT checkpoint as starting policy |
| M3 → M4 | Need trained model to deploy |
| M4 → M5 | Need running server to integrate |
| M5 → M6 | Need production baseline to measure evolution |

---

## Testing & Acceptance System

### Phase Gates

Each milestone has a **phase promotion gate** — a hard requirement that must be met before the next milestone can begin.

| Gate | Milestone | Criterion | Fail Action |
|------|-----------|-----------|-------------|
| G0 | M0 | `nvidia-smi` + `import torch, trl, peft, bitsandbytes` pass | Fix environment |
| G1 | M1a + M1b | ≥10,500 valid ChatML samples in train.jsonl | Debug extraction, generate more synthetic data |
| G2 | M2 | Basic eval ≥15/20 valid tool calls | Debug training data format, adjust SFT config |
| G3 | M3 | Held-out pass rate ≥60%, ≥10pp over SFT baseline | Increase GRPO data, adjust reward weights |
| G4 | M4 | Server stable, smoke test produces valid tool call | Debug serving config |
| G5 | M5 | Pass rate ≥80% OR within 5pp of DeepSeek | Return to M3 for more GRPO |
| G6 | M6 | Monthly pipeline runs end-to-end, A/B gate functions | Debug automation scripts |

### Per-Milestone Testing

| Milestone | Test Type | Command | Expected |
|-----------|-----------|---------|----------|
| M0 | Env check | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| M0 | VRAM check | `nvidia-smi --query-gpu=memory.free --format=csv,noheader` | ≥7500 MB |
| M1a | Schema validation | `python scripts/validate_data.py data/train.jsonl` | 0 errors |
| M1a | No data leak | `python scripts/check_leak.py data/train.jsonl data/test.jsonl` | 0 overlapping task_ids |
| M1b | Verification gate rate | `python scripts/synth_trajectories.py --dry-run | grep "gate_pass_rate"` | ≥0.80 |
| M2 | SFT basic eval | `python scripts/eval_sft_basic.py --checkpoint ./feiyue-qwen-4b-sft` | ≥15/20 |
| M3 | GRPO reward curve | Plot reward over steps | Monotonic increase, final > initial + 0.2 |
| M3 | GRPO hold-out eval | `python scripts/eval_grpo_holdout.py --checkpoint ./feiyue-qwen-4b-grpo` | ≥60% pass rate |
| M4 | Server health | `curl -s http://localhost:8000/health` | HTTP 200 |
| M4 | Latency p95 | `python scripts/bench_latency.py --requests 100` | <2s |
| M5 | A/B pass rate | `python scripts/ab_eval.py` results | ≥80% OR ≤5pp gap |
| M5 | Self-correction rate | Parse A/B results | ≥50% |
| M6 | Pipeline dry-run | `python scripts/monthly_retrain.py --dry-run --since 2026-06-01` | Exits 0, reports sample counts |
| M6 | A/B gate logic | Unit test with mock pass rates | Deploys only when +2pp |

### Code Quality & Cleanliness Gates

Applied at every commit. Enforced via pre-commit or CI.

| Check | Tool | Threshold |
|-------|------|-----------|
| No secrets | `grep -rE "sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{36}" scripts/ data/ configs/` | 0 matches |
| Python lint | `ruff check scripts/` | 0 errors |
| Python format | `ruff format --check scripts/` | Would leave files unchanged |
| YAML syntax | `python -c "import yaml; yaml.safe_load(open(f))"` for each .yaml | No exceptions |
| JSONL validity | `python -c "import json; [json.loads(l) for l in open(f)]"` for each .jsonl | No exceptions |
| File size | `find . -name "*.safetensors" -size +500M` | No adapter >500MB |

---

## Immediate Next Slice: M0

**The next action is M0 — Foundation setup.** This is the only prerequisite for all other work.

Specific tasks for M0:
1. SSH into Serverai, verify CUDA/PyTorch/VRAM
2. Install bitsandbytes, TRL, PEFT, vLLM, Hugging Face CLI
3. Create repo directory structure on Serverai
4. Write updated `configs/sft_config.yaml`
5. Write `configs/grpo.yaml`
6. Rewrite `data/format.md` for multi-turn ChatML
7. Create stub files for all new scripts
8. Push to GitHub

After M0 is verified, M1a and M1b can start **in parallel**.

---

> **This outline is canonical for Feiyue-Model v1.0 development. All implementation work follows the milestones, dependencies, acceptance gates, and quality standards defined here unless explicitly overridden by the user.**
