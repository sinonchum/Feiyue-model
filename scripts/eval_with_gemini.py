#!/usr/bin/env python3
"""Gemini 2.5 Pro evaluation of SFT training run."""
import json, urllib.request, os

def call_gemini(prompt, max_tokens=4096):
    with open(os.path.expanduser('~/gcloud_token.txt')) as f:
        token = f.read().strip()
    project = 'gen-lang-client-0869773739'
    region = 'us-central1'
    url = f'https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/google/models/gemini-2.5-pro:generateContent'
    body = {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {'temperature': 0.3, 'maxOutputTokens': max_tokens}
    }
    req = urllib.request.Request(url,
        data=json.dumps(body).encode(),
        headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data['candidates'][0]['content']['parts'][0]['text'], data.get('usageMetadata', {})


with open('data/prepare_report.json') as f:
    report = json.load(f)

train_stats = report['train']
val_stats = report['val']

prompt = f"""Evaluate this Feiyue-Model SFT training run vs the smoke test baseline.

=== SMOKE TEST (1st run) ===
Data: 36 samples, all easy
Training: 1 epoch, 35.44s, VRAM 4.08GB/8GB
Loss: 0.67, Token accuracy: 95.99% (suspected overfitting)
Inference: tested on irrelevant "robot story" prompt

=== FULL TRAINING (2nd run) ===
Data: {train_stats['count']} train, {val_stats['count']} val
Difficulty: {train_stats['difficulty_distribution']}
Training: 3 epochs, batch=4, grad_accum=2, QLoRA r=16 alpha=32
eval_loss (epoch 2): 1.054
eval_token_accuracy (epoch 2): 78.76%
Inference: still using "robot story" test (coding eval set created but not yet run)

=== NEW EVAL SET ===
Created 8 coding problems: 3 easy + 3 medium + 2 hard

=== EVALUATE ===
Score (1-10):
1. Data improvement vs smoke test
2. Training quality (interpret eval_loss 1.054 + accuracy drop from 95%→79%)
3. Readiness for GRPO RL phase
4. Top 2 actions needed NOW

Be brutally honest. If accuracy dropped from 95% to 79%, explain what this means about overfitting. The new data is mostly easy (43/54) — is this a problem?"""

text, usage = call_gemini(prompt)
print(text)
print()
print(f"Tokens: prompt={usage.get('promptTokenCount')}, output={usage.get('candidatesTokenCount')}, thoughts={usage.get('thoughtsTokenCount')}")
