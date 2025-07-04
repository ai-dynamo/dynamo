
## Setup checklist:
Using base vLLM container
```
uv pip uninstall ai-dynamo-vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout 657f2f3
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```
This has this specific commit - https://github.com/vllm-project/vllm/pull/19790
which will be useful for supporting DEP, as we can have each DP Rank be its own Dynamo component.

```
docker compose -f deploy/metrics/docker-compose.yml up -d
```

## Go

```bash
cd examples/vllm_v1
bash launch/agg.sh
```

```bash
cd examples/vllm_v1
bash launch/agg_router.sh
```

```
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
    {
        "role": "user",
        "content": "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden."
    }
    ],
    "stream":false,
    "max_tokens": 30
  }'
```
