## Setup


```
uv pip uninstall ai-dynamo-vllm
git clone --branch upstream-nixl-clean https://github.com/robertgshaw2-redhat/vllm.git
VLLM_USE_PRECOMPILED=1 uv pip install --editable ./vllm/

Start required services (etcd and NATS) using [Docker Compose](../../deploy/docker-compose.yml)
```bash
docker compose -f deploy/docker-compose.yml up -d
```


## Run server

```
cd examples/vllm_v1
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
```

## Run client

```
curl eos0340:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "prompt": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
    "stream":false,
    "max_tokens": 30
  }'
```