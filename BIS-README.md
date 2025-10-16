DYN_REQUEST_PLANE=http DYN_HTTP_RPC_PORT=8084 DYN_HTTP_RPC_HOST=10.0.8.158  DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8085    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --connector none

DYN_REQUEST_PLANE=http python -m dynamo.frontend --http-port=8000

# -------------

DYN_REQUEST_PLANE=http python -m dynamo.frontend --http-port=8000

DYN_REQUEST_PLANE=http DYN_HTTP_RPC_PORT=8084  DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8085    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --connector none

## curl
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream": false,
    "max_tokens": 30
  }'


## etcdctl get --prefix ""
v1/instances/dynamo/backend/generate/694d99eabb918920
{
  "component": "backend",
  "endpoint": "generate",
  "namespace": "dynamo",
  "instance_id": 7587890180637428000,
  "transport": {
    "http_tcp": {
      "http_endpoint": "http://0.0.0.0:8084/v1/rpc/dynamo_backend.generate-694d99eabb918920",
      "tcp_endpoint": "0.0.0.0:8084"
    }
  }
}


export ARTIFACT_DIR=~/tmp/aiperf/a1
export ENDPOINT=http://localhost:8000

isl=100
osl=100
concurrency=10

aiperf profile --artifact-dir $ARTIFACT_DIR \
    --model $TARGET_MODEL \
    --tokenizer ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/  \
    --endpoint-type chat  --endpoint /v1/chat/completions \
    --streaming \
    --url http://$ENDPOINT \
    --synthetic-input-tokens-mean $isl \
    --synthetic-input-tokens-stddev 0 \
    --output-tokens-mean $osl \
    --output-tokens-stddev 0 \
    --extra-inputs "{\"max_tokens\":$osl}" \
    --extra-inputs "{\"min_tokens\":$osl}" \
    --extra-inputs "{\"ignore_eos\":true}" \
    --extra-inputs "{\"nvext\":{\"ignore_eos\":true}}" \
    --extra-inputs "{\"repetition_penalty\":1.0}" \
    --extra-inputs "{\"temperature\": 0.0}" \
    --concurrency $concurrency \
    --request-count $((10*concurrency)) \
    --warmup-request-count $concurrency \
    --conversation-num 12800 \
    --random-seed 100 \
    --workers-max 252 \
    -H 'Authorization: Bearer NOT USED' \
    -H 'Accept: text/event-stream'\
    --record-processors 32 \
    --ui simple
