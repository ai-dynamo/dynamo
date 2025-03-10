#!/bin/bash

# list models
echo "\n\n### Listing models"
curl http://localhost:8000/v1/models

# create completion
echo "\n\n### Creating completions"
curl -X POST http://localhost:8000/v1/chat/completions \
-H 'accept: application/json' \
-H 'Content-Type: application/json' \
-d '{
    "model": "mock_model",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
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