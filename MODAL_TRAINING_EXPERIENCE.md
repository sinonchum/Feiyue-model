# Feiyue-v1: Modal Training Experience Log

> A detailed record of fine-tuning Qwen3-4B on Modal cloud GPU, including problems encountered, solutions, and cost breakdown.

## Overview

| Item | Value |
|------|-------|
| Model | Qwen3-4B-Instruct-2507 → Feiyue-v1 |
| Method | LoRA SFT (r=8, α=16, bf16) |
| GPU | Modal L40S (48 GB VRAM) |
| Dataset | v11_8k (2,132 train / 237 val) |
| Training Time | ~44 minutes |
| Training Cost | ~$1.45 |
| Final Train Loss | 0.6486 |
| Steps | 1,599 (3 epochs) |

## Pipeline

```
1. Data Generation     scripts/gen_vertex_data.py   → Gemini Flash Lite (Vertex AI)
2. Data Curation       scripts/build_v11_data.py    → v11_8k dataset (ChatML JSONL)
3. LoRA Training       modal run train script       → Modal L40S GPU
4. Merge Adapter       merge_lora.py                → Modal L4 (bf16 merged)
5. GGUF Quantize       convert_gguf.py              → Modal L4 (Q8_0 + Q4_K_M)
6. Upload to HF        upload scripts               → HuggingFace Hub
```

## Step-by-Step Details

### 1. Data Generation

Training data was generated using Gemini Flash Lite via Vertex AI (`scripts/gen_vertex_data.py`). The script generated:

- **988** instruction/input/output samples for PRD writing, code review, and tool-calling
- **1,000** code review samples from GitHub-style diffs

Total: **1,988 → curated to 2,132** after quality filtering.

Data format: ChatML JSONL with `messages` field containing `system`, `user`, `assistant` roles.

### 2. Data Curation

`scripts/build_v11_data.py` combined multiple data sources:

- Synthetic PRD data (Vertex AI generated)
- GitHub issues and code review examples
- Magicoder OSS-Instruct code generation data
- Tool-calling examples with diverse JSON schemas

All examples were formatted to max 8,192 tokens using the Qwen3 chat template. Final split: 2,132 train / 237 validation.

### 3. LoRA Training (Modal L40S)

**First attempt — modal_train.py (initial plan):**

The initial script used:
- A10G GPU ($1.10/hr)
- 8-bit quantization (BitsAndBytes)
- r=16, α=32, seq_len=4096
- transformers==4.48.0, peft==0.14.0

This script was a draft and was **not used for the final training**. It had several issues:
- Dependency version conflicts (transformers 4.48 vs Qwen3-4B-Instruct-2507 requiring newer versions)
- 8-bit quantization unnecessary on L40S with 48 GB VRAM

**Final training — bf16 on L40S:**

The actual training used a different configuration:

| Parameter | Initial Plan | Final |
|-----------|-------------|-------|
| GPU | A10G ($1.10/hr) | L40S ($1.78/hr) |
| Precision | 8-bit quantized | bf16 full precision |
| LoRA r | 16 | 8 |
| LoRA α | 32 | 16 |
| Seq Length | 4,096 | 8,192 |
| Grad Accum | 16 | 4 |
| transformers | 4.48.0 | >=4.55 |

Why the change: L40S has 48 GB VRAM — no need for quantization. bf16 gives better quality. Smaller r=8 is sufficient for a 4B model with 2K training samples.

### 4. Merge LoRA Adapter

`merge_lora.py` runs on Modal L4 (cheaper than L40S for non-training work):

```python
# Load base model
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16)
# Load adapter
model = PeftModel.from_pretrained(model, adapter_dir)
# Merge
model = model.merge_and_unload()
model.save_pretrained(output_dir, safe_serialization=True)
```

Output: bf16 merged model (~8 GB) saved to Modal volume.

### 5. GGUF Quantization

`convert_gguf.py` runs on Modal L4:

1. Clones llama.cpp
2. Builds with cmake
3. Converts HF model to GGUF (f16)
4. Quantizes to Q8_0 (4.0 GB) and Q4_K_M (2.5 GB)

Both quantized files saved to Modal volume, then uploaded to HuggingFace.

### 6. Upload to HuggingFace

Four repos created:
- `sinonchum/Qwen3-4B-Feiyue-v1` — LoRA adapter (~63 MB)
- `sinonchum/Qwen3-4B-Feiyue-v1-bf16` — Merged bf16 (~8 GB)
- `sinonchum/Qwen3-4B-Feiyue-v1-Q8_0` — GGUF 8-bit (4.0 GB)
- `sinonchum/Qwen3-4B-Feiyue-v1-Q4_K_M` — GGUF 4-bit (2.5 GB)

## Problems Encountered & Solutions

### Problem 1: HFValidationError on Modal Volume

**Error:**
```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
'/mnt/volume/output/sft/merged'. Use `repo_type` argument if needed.
```

**Cause:** `transformers` tried to interpret the local path as a HuggingFace repo ID and attempted to check the HF Hub cache.

**Fix:** Add `local_files_only=True` to both `from_pretrained` calls:
```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, ..., local_files_only=True)
```

**Lesson:** When loading from Modal volumes (or any non-standard path), always set `local_files_only=True` to prevent transformers from trying to contact HF Hub.

### Problem 2: Modal Credential Check in SWE-bench

**Error:**
```
RuntimeError: ~/.modal.toml not found - it looks like you haven't configured
credentials for Modal.
```

**Cause:** SWE-bench's `--modal true` mode checks for `~/.modal.toml` before spawning containers. Inside a Modal container, credentials are available via environment variables but not via the toml file.

**Fix attempt:** Write a dummy `~/.modal.toml` + monkey-patch `validate_modal_credentials()`. This bypassed the check but the evaluation itself failed due to Docker-in-Docker limitations.

**Lesson:** SWE-bench's Modal mode is designed to be called from a local machine (which has `~/.modal.toml`), not from within a Modal container. Running the full evaluation harness on Modal is not straightforward.

### Problem 3: BFCL Data Path in Gorilla Repo

**Error:** BFCL data files not found at expected paths.

**Cause:** The gorilla repo stores eval data in `unused_datasets/question/` and `unused_datasets/possible_answer/`, not in the main `data/question/` path.

**Fix:** Update data paths to use `unused_datasets/` subdirectory.

### Problem 4: BFCL Ground Truth Format Mismatch

**Error:** 0% accuracy on first BFCL run despite correct model outputs.

**Cause:** Ground truth is in Python function call format (`func_name(k=v, ...)`), but the parser expected JSON. The model outputs JSON `{"name": ..., "arguments": {...}}`, which never matched the string-based ground truth.

**Fix:** Implemented structured comparison:
- `model_to_struct()` — Parse model JSON output into `{name, arguments}` dict
- `gt_to_struct()` — Parse ground truth `func_name(a=1, b=[0.5])` into same dict format
- `struct_equal()` — Compare structs with tolerance for extra parameters

### Problem 5: `torch_dtype` Deprecation Warning

**Warning:**
```
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
```

**Cause:** Newer versions of transformers renamed the parameter.

**Fix:** Use `dtype=torch.bfloat16` instead of `torch_dtype=torch.bfloat16`. Non-critical — the warning doesn't affect functionality.

### Problem 6: Modal Billing Cycle Limit

**Error:**
```
App creation failed: workspace billing cycle spend limit reached
```

**Cause:** The Modal account hit its spending cap. This happened primarily due to Docker-based SWE-bench evaluation attempts that spawned hundreds of containers.

**Fix:** Switched to a new Modal account for subsequent work.

**Lesson:** Set hard budget caps on Modal. Docker-based benchmarks (SWE-bench) are extremely expensive on Modal because each instance needs its own container with a full repo environment.

## Cost Breakdown

| Operation | GPU | Time | Cost |
|-----------|-----|------|------|
| LoRA Training | L40S | ~44 min | ~$1.45 |
| Merge Adapter | L4 | ~5 min | ~$0.05 |
| GGUF Conversion | L4 | ~10 min | ~$0.10 |
| BFCL Evaluation | L40S | ~3 min | ~$0.10 |
| **Training Total** | | | **~$1.70** |

### Wasted Costs (Lessons Learned)

| Operation | Issue | Cost |
|-----------|-------|------|
| SWE-bench Docker eval | 300 containers × repo builds | ~$60+ |
| Multiple BFCL retries | Script bugs, dependency issues | ~$5 |
| **Total Wasted** | | **~$70** |

**Key takeaway:** Modal is excellent for training and inference. Do NOT use Modal for Docker-heavy evaluation benchmarks (SWE-bench). The cost of spinning up hundreds of containers with full repo environments far exceeds the training cost itself.

## Hyperparameters That Worked

```yaml
# Final configuration
base_model: Qwen/Qwen3-4B-Instruct-2507
method: LoRA (PEFT)
r: 8
alpha: 16
dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
precision: bf16
attention: sdpa
gradient_checkpointing: true
epochs: 3
learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.1
batch_size: 1
gradient_accumulation: 4
effective_batch: 4
max_seq_length: 8192
optimizer: AdamW (default)
seed: 42
```

## Recommendations for Future Training

1. **Budget caps** — Always set `MAX_SPEND` in Modal scripts. Use `modal app list` to monitor active apps.
2. **Use L40S for bf16** — Don't use 8-bit quantization on L40S. The 48 GB VRAM is more than enough for a 4B model in bf16.
3. **L4 for non-training** — Use L4 (cheaper) for merge, GGUF conversion, and inference-only tasks.
4. **local_files_only** — Always set this when loading from Modal volumes.
5. **Avoid Docker benchmarks on Modal** — SWE-bench and similar Docker-heavy evaluations are prohibitively expensive on Modal. Run them locally or on a fixed-cost VPS instead.
6. **Data generation is cheap** — Vertex AI / Gemini for synthetic data generation is negligible cost compared to training.
7. **Volume commit** — Always call `volume.commit()` after writing to Modal volumes, otherwise changes are lost.

## Benchmark Results

### BFCL v4 (Berkeley Function Calling Leaderboard)

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
| exec_simple | 89.0% | 89 | 100 |
| exec_parallel | 88.0% | 44 | 50 |
| exec_multiple | 80.0% | 40 | 50 |
| exec_parallel_multiple | 82.5% | 33 | 40 |
| **Overall** | **85.8%** | **206** | **240** |

### SWE-bench Lite

Result: 0/280 resolved (0%). Expected for a 4B model — SWE-bench Lite requires multi-file understanding, repository-level context, and iterative debugging that exceeds the capacity of small models.

## Files Reference

| File | Purpose |
|------|---------|
| `scripts/modal_train.py` | Initial training plan (A10G, 8-bit) — draft only |
| `scripts/gen_vertex_data.py` | Training data generation via Vertex AI |
| `scripts/build_v11_data.py` | Dataset construction pipeline |
| `merge_lora.py` | Merge LoRA adapter into base model (Modal L4) |
| `convert_gguf.py` | HF → GGUF conversion + quantization (Modal L4) |
| `run_bfcl.py` | BFCL v4 evaluation (standalone, no framework deps) |
| `run_swebench.py` | SWE-bench Lite inference (300 patches) |
| `eval_swebench.py` | SWE-bench Lite evaluation harness |
| `upload_gguf.py` | Upload GGUF files to HuggingFace |
| `set_gated.py` | Set HuggingFace repo to gated access |
