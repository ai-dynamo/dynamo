Run commands below in 4 terminals -

- Terminal.1 [http service]
```
TRD_LOG=DEBUG http
```

- Terminal.2 [vLLM Backend]
```
python -m processor_backend.vllm_backend --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

- Terminal.3 [Processor]
```
python -m preprocessor_backend.preprocessor --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

- Terminal.4
```
llmctl http add chat-models deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B triton-init.preprocessor.generate

curl localhost:8080/v1/models | jq .

curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```