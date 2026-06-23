"""
Call Gemini 3.1 Pro Preview to evaluate coding eval responses.
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
print(f"Token: {token[:15]}...")

# Read prompt
with open("data/gemini_eval_prompt.txt") as f:
    prompt = f.read()

# Call Gemini 3.1 Pro via Vertex AI
project_id = "project-7c8d85fa-4f69-41a0-abb"
url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent"

payload = {
    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096}
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print(f"Calling Gemini 3.1 Pro Preview...")
req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode())
    
    text = result["candidates"][0]["content"]["parts"][0]["text"]
    
    with open("data/gemini_eval_result.txt", "w") as f:
        f.write(text)
    
    print("\n--- RESULT ---\n")
    print(text)
    
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP Error {e.code}: {body[:500]}")
