#!/bin/bash

# Script to send a completion request to vLLM
# Endpoint: http://127.0.0.1:8000/v1/completions

curl -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "prompt": "what is a dynamo? and how did it signify the start of the industrial revolution?",
    "max_tokens": 32,
    "temperature": 0
  }'
