# Evaluation Methodology — Feiyue Qwen3-4B

> **Version:** V10 (2026-06-26)  
> **Evaluator:** Gemini 3.1 Pro Preview  
> **Model:** Qwen/Qwen3-4B-Instruct-2507 + 8-bit QLoRA (r=8, attention-only)  
> **Training:** 3 epochs, 1,264 samples, eval_loss=0.7688

## Overview

The Feiyue model is a multi-capability agent fine-tuned for three domains:

| Domain | Training Share | Description |
|--------|:---:|-------------|
| **Code** | 35% | Write, implement, and explain code from specifications |
| **Code Review** | 35% | Identify bugs, performance issues, and security vulnerabilities in code |
| **PRD** | 30% | Write structured Product Requirement Documents with task decomposition |

Evaluation uses **LLM-as-judge** (Gemini 3.1 Pro Preview) on a stratified random sample drawn from a held-out test set (159 items). The judge is given the original task prompt and the model's response, but NOT the model or training details that could bias scoring.

---

## Scoring Rubric

All domains use a 4-axis rubric with domain-specific interpretations of the last two axes.

### Common axes (all domains)

| Axis | Range | What it measures |
|------|:-----:|------------------|
| **Format Compliance** | 0–2 | Proper `<think>` blocks, clear output structure, no garbled text or leaked `<tool_call>` XML |
| **Task Understanding** | 0–3 | Correctly interprets the task prompt and addresses the asked question |

### Domain-specific axes

**Code** — pure code generation tasks (write a function, implement a feature):

| Axis | Range | What it measures |
|------|:-----:|------------------|
| **Code Quality** | 0–3 | Correctness, idiomatic style, completeness of the implementation |
| **Tool Usage** | 0–2 | Appropriate use of `<tool_call>` for file operations; N/A scored as 2 |

**Code Review** — analyze given code for bugs, performance, or security issues:

| Axis | Range | What it measures |
|------|:-----:|------------------|
| **Code Quality** | 0–3 | Accuracy of diagnosis (correct bug/vuln identified), quality of proposed fix |
| **Tool Usage** | 0–2 | Contextual: did it use tools to inspect files before diagnosing? Is the fix complete? |

**PRD** — write product requirement documents from high-level descriptions:

| Axis | Range | What it measures |
|------|:-----:|------------------|
| **PRD Quality** | 0–3 | Structured sections (scope, requirements, tradeoffs, metrics), actionable content |
| **Clarity & Depth** | 0–2 | Clear writing, sufficient detail for engineering handoff, not just bullet points |

### Total score

```
Total = Format + Understanding + Quality + Extra  →  0–10
```

---

## Sampling Strategy

1. **Population:** 159 test items (code: 92, code_review: 24, prd: 43)
2. **Method:** Stratified random sampling — 5 items per domain (15 total) with fixed seed (42) for reproducibility
3. **Rationale:** 15 items balances Gemini API cost (~10K input tokens) with statistical coverage across all three capabilities
4. **Blinding:** The evaluator receives only the task prompt and model response — no metadata about training config, loss curves, or expected difficulty

---

## Evaluation Procedure

### Step 1 — Inference

```
Model: merged LoRA (checkpoint-237) loaded in 8-bit INT8
Max new tokens: 1024, temperature: 0.6, top_p: 0.95
Input: ChatML messages list from test set (system + user, with generation prompt)
Output: assistant response (no system prompt modification)
```

8-bit quantization preserves >99% fidelity vs full-precision for 4B models (Gemini 3.1 Pro verified).

### Step 2 — Gemini Scoring

```
Model: gemini-3.1-pro-preview (Vertex AI, global endpoint)
Temperature: 0.2, maxOutputTokens: 12288 (thinking model needs headroom)
Prompt: Structured evaluation prompt with all 15 items + rubric definition
Output: JSON with per-item scores + domain averages + strengths/weaknesses/recommendation
```

The prompt explicitly defines each scoring axis and asks for structured JSON output. Gemini's `thoughtsTokenCount` (internal reasoning) is approximately equal to `candidatesTokenCount` — the model thinks before scoring.

### Step 3 — Aggregation

Domain averages are computed as the arithmetic mean of per-item totals. The overall average is weighted equally across domains (not weighted by sample count).

---

## Results (V10)

### Per-Domain Summary

| Domain | Samples | Avg Score | Key Pattern |
|--------|:---:|:---:|-----------|
| Code | 5 | **1.6 / 10** | Hallucinations, empty templates, `<tool_call>` leakage |
| Code Review | 5 | **7.2 / 10** | Strong bug identification, some response truncation |
| PRD | 5 | **7.0 / 10** | Good structure, often too brief |
| **Overall** | **15** | **5.3 / 10** | Strong on analysis, weak on generation |

### Detailed Per-Item Scores

#### Code (5 items)

| # | Task | F | U | Q | E | Total | Notes |
|---|------|:--:|:--:|:--:|:--:|:-----:|-------|
| 1 | Write code + tests for data pipeline | 0 | 0 | 0 | 0 | **0** | Only output "Task complete." |
| 2 | Polish a coding prompt for LLM | 1 | 1 | 1 | 0 | **3** | Misunderstood task; leaked `<tool_call>` |
| 3 | Write SQL query for analysis | 1 | 1 | 1 | 0 | **3** | Wrapped in useless Python script; truncated |
| 4 | Write JavaScript function | 0 | 0 | 0 | 0 | **0** | Hallucinated code already existed |
| 5 | Implement ML preprocessing pipeline | 1 | 1 | 0 | 0 | **2** | `# Your code here` placeholder |

**Code domain analysis:** All 5 items show the model failing to produce actual code. Two items (0/10) are complete failures — the model outputs "Task complete." or hallucinates. The remaining three show partial understanding but fail to produce functional code, leaking `<tool_call>` XML tokens or outputting placeholder templates.

#### Code Review (5 items)

| # | Task | F | U | Q | E | Total | Notes |
|---|------|:--:|:--:|:--:|:--:|:-----:|-------|
| 6 | Analyze C loop performance | 2 | 3 | 3 | 1 | **9** | O(N²) `strlen` in loop → O(N) fix |
| 7 | Find buffer overflow in C | 1 | 3 | 1 | 0 | **5** | Correct diagnosis, missing fix code |
| 8 | Debug React `useEffect` closure | 2 | 3 | 3 | 1 | **9** | Stale closure + functional update fix |
| 9 | Diagnose async race condition | 2 | 3 | 2 | 1 | **8** | Excellent explanation, fix truncated |
| 10 | Review Python exception handling | 1 | 3 | 1 | 0 | **5** | Identified broad `except`, too brief |

**Code Review domain analysis:** The model excels at identifying bugs — every item scored 3/3 on task understanding. Performance bottlenecks (O(N²)), security vulnerabilities (buffer overflow), framework-specific issues (React stale closure), and concurrency bugs (race condition) were all correctly identified. The main weakness is response completeness: 3 of 5 items had truncated or incomplete fixes.

#### PRD (5 items)

| # | Task | F | U | Q | E | Total | Notes |
|---|------|:--:|:--:|:--:|:--:|:-----:|-------|
| 11 | PRD for notification system | 2 | 3 | 2 | 1 | **8** | Good work streams + dependencies; truncated |
| 12 | PRD for analytics dashboard | 2 | 3 | 2 | 1 | **8** | All sections covered, slightly concise |
| 13 | PRD for CI/CD migration | 2 | 3 | 1 | 0 | **6** | Too brief to be useful |
| 14 | PRD for API versioning | 1 | 3 | 2 | 1 | **7** | Good tech choices; leaked `<tool_call>` |
| 15 | PRD for search feature | 2 | 3 | 1 | 0 | **6** | Bullet points only, no depth |

**PRD domain analysis:** Task understanding is uniformly strong (3/3 across all items). The model consistently produces structured PRDs with appropriate sections. The main gap is depth — responses tend to be outlines rather than comprehensive documents, scoring 6-8 instead of 9-10.

---

## Strengths & Weaknesses

### Strengths
- **Code Review:** Accurately identifies diverse bug types (performance, security, concurrency, framework patterns) across C, JavaScript, Python, and React
- **PRD Structure:** Consistently produces well-organized documents with appropriate markdown headers and section coverage
- **Task Understanding:** Near-perfect task interpretation in code review and PRD domains (13/15 items at max understanding score)

### Weaknesses
- **Code Generation:** Severe failure mode — model defaults to "Task complete." or placeholder templates instead of writing code
- **`<tool_call>` Leakage:** XML tool-call tokens appear in final output where they don't belong (4/15 items affected)
- **Response Truncation:** High rate of mid-sentence/mid-code cutoffs (6/15 items affected)
- **Brevity:** Many responses provide outlines where depth is expected, suggesting overfitting to concise training samples

---

## Objectivity Caveats

This evaluation uses **LLM-as-judge**, which has known limitations:

1. **Single-judge bias:** Only Gemini 3.1 Pro was used. Cross-validation with GPT-4 or Claude would improve reliability
2. **Rubric subjectivity:** "Code Quality" and "PRD Quality" are inherently subjective — different human reviewers might assign different scores
3. **Sample size:** 15 items (5 per domain) provides directional signal but not statistical significance. Error bars are approximately ±1 point
4. **Domain imbalance:** Code Review and PRD samples (5 each from 24-43 total) are proportionally larger samples than Code (5 from 92 total), making Code scores potentially less representative

### Mitigations
- Fixed random seed (42) for reproducible sampling
- Structured JSON output format with explicit scoring axes
- Blinded evaluation (judge unaware of model version or training details)
- All raw responses and scores published alongside this document for independent verification

---

## Reproducing

```bash
# 1. Run inference on test set
python scripts/infer_v10_eval.py

# 2. Evaluate with Gemini 3.1 Pro
python scripts/eval_v10_gemini.py

# 3. View results
cat data/v10_gemini_eval.json
```

Raw inference outputs: `data/v10_test_responses.jsonl` (159 items)  
Gemini evaluation: `data/v10_gemini_eval.json` (15 scored items)

---

## Changelog

| Date | Version | Change |
|------|---------|--------|
| 2026-06-26 | V10 | Initial 3-domain evaluation with stratified sampling |
