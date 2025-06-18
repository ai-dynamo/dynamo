genai-perf profile \
  --input-file new.jsonl \
  --extra-inputs payload.j2 \
  --endpoint /v1/experimental/dynamo/completions \
  --endpoint-type completions \
  --url localhost:8000 \
  --model deepseek-ai/DeepSeek-R1 \
  --tokenizer deepseek-ai/DeepSeek-R1 \
  --streaming \
  --concurrency 16 \
  --request-count 160 \
  --warmup-request-count 16 \
  -H 'Accept: text/event-stream'

