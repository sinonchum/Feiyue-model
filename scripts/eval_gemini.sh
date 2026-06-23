#!/bin/bash
TOKEN=$(gcloud auth application-default print-access-token 2>/dev/null)
echo "Token: ${TOKEN:0:10}..."

PROMPT=$(cat ~/Feiyue-model/data/gemini_eval_prompt.txt)

# Build JSON payload
PAYLOAD=$(cat <<'HEREDOC'
{"contents":[{"role":"user","parts":[{"text":"PLACEHOLDER"}]}],"generationConfig":{"temperature":0.2,"maxOutputTokens":4096}}
HEREDOC
)

# Try Vertex AI with the correct endpoint
curl -s -X POST \
  "https://aiplatform.googleapis.com/v1/projects/project-7c8d85fa-4f69-41a0-abb/locations/global/publishers/google/models/gemini-3.1-pro-preview:generateContent" \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" 2>&1 | head -30
