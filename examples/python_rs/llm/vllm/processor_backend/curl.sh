#!/bin/bash
PORT=8080

# list models
echo "\n\n### Listing models"
curl http://localhost:$PORT/v1/models

# create completion
echo "\n\n### Creating completions"
curl -X 'POST' \
  "http://localhost:$PORT/v1/chat/completions" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "messages": [
      {
        "role":"user",
        "content":"what is deep learning?"
      }
    ],
    "max_tokens": 64,
    "stream": true,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
    "top_k": 5
  }'
