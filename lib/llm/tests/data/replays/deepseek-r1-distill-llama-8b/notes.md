captured from 0.9.0.2.dev22+gbc825748a.d20250715.precompiled

script to generate deepseek-r1-distill-llama-8b/chat-completions.stream.logprobs.1

```
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [{"role": "user", "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."}],
    "max_tokens": 32,
    "temperature": 0.0,
    "top_p": 0.001,"stream":true,"logprobs":1,"top_logprobs":2
}'
```
