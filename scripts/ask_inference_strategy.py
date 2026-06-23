"""
Ask Gemini 3.1 Pro to decide the best inference strategy for 8GB VRAM.
"""
import json
import subprocess
import urllib.request
import urllib.error

# Get access token
result = subprocess.run(
    ["C:/Users/simon/google-cloud-sdk/bin/gcloud.cmd", "auth", "application-default", "print-access-token"],
    capture_output=True, text=True
)
token = result.stdout.strip()

prompt = """I have a fine-tuned Qwen3-4B-Instruct model (LoRA adapter merged into base model, saved as fp16).
I need to run inference on 8 eval items for evaluation.

Hardware constraint: RTX 5060, 8GB VRAM total.

The merged model in fp16 is ~8GB which doesn't fit. I need to load it for inference.

Options I'm considering:
1. 4-bit quantization (bitsandbytes NF4) - loads ~2.5GB, leaves room for inference
2. 8-bit quantization (bitsandbytes INT8) - loads ~4GB, leaves some room
3. CPU offloading with accelerate - slow but full precision
4. Other approach?

Key concern: This is an EVALUATION run. Quantization during inference may affect the model's output quality and unfairly penalize the model. But I also need it to actually run.

Questions:
1. For a fair evaluation of a 4B model's coding ability, which quantization level should I use?
2. Is 8-bit significantly better than 4-bit for preserving coding quality?
3. Should I use CPU offloading instead to preserve full precision?
4. What's the recommended approach given 8GB VRAM?

Please provide a specific recommendation with the exact code/configuration to use."""

url = "https://aiplatform.googleapis.com/v1/projects/project-7c8d85fa-4f69-41a0-abb/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"

payload = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("Asking Gemini 3.1 Pro about inference strategy...")
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode())
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    print("\n--- Gemini 3.1 Pro Recommendation ---\n")
    print(text)
    with open("data/inference_strategy_gemini.txt", "w") as f:
        f.write(text)
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP Error {e.code}: {body[:500]}")
