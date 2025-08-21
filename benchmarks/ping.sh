#!/bin/bash

# Get port from first argument, default to 8080 if not provided
PORT=${1:-8080}

curl -X POST http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d '{
    "model": "6a6f4aa4197940add57724a7707d069478df56b1",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Answer in 5 words."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "stream": true,
    "max_tokens": 10,
    "ignore_eos": true,
    "nvext": {"ignore_eos": true}
    }'