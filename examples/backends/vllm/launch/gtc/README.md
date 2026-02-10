# Spec

Qwen3 VL
https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor
Default patch size: 16

# Scenario P

## Input

System prompts
8,600 = 5,500 + 2,600 + 500

User input
4,300 = 3,000 + 300 + 1,000

Output token
500

3 512*512 images per request, drawn from a pool of unique images.
A smaller pool relative to total image slots produces more cross-request reuse.

## Generate

```
python examples/backends/vllm/launch/gtc/main.py \
  -n 100 --images-pool 240
```

## Run

```
aiperf profile \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-file examples/backends/vllm/launch/gtc/requests.jsonl \
  --custom-dataset-type single_turn \
  --shared-system-prompt-length 8600 \
  --extra-inputs "max_tokens:500" \
  --extra-inputs "min_tokens:500" \
  --extra-inputs "ignore_eos:true"
```

# Scenario R

## Input

9K Text input tokens
- 6K fixed System prompt
- 3K User input

1 Output tokens

Images
- 20 images per request
  - "The number of images is between 10 and 50 per request"
- choose 256 tokens per image
  - "There are two image token configurations (128 and 256)"
- Images drawn from a pool; pool size controls cross-request reuse.
  - "within a 24-hour period, the duplication rate of images was 27%."


## Generate

```
python main.py \
  --image-mode http \
  -n 100 \
  --images-per-request 10 \
  --images-pool 500 \
  --user-text-tokens 3000
```

## Run

```
aiperf profile \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-file examples/backends/vllm/launch/gtc/100req_10img_500pool_3000word_http.jsonl \
  --custom-dataset-type single_turn \
  --shared-system-prompt-length 6000 \
  --osl 1 \
  --request-count 100 \
  --concurrency 1 \
  --warmup-request-count 3 \
  --artifact-dir /workspace/logs/aiperf
```

# Temp

```
python main.py \
  --image-mode http \
  -n 100 \
  --images-per-request 10 \
  --images-pool 500 \
  --user-text-tokens 300
```

```
aiperf profile \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-file examples/backends/vllm/launch/gtc/1000req_1img_200pool_300word_base64.jsonl.jsonl \
  --endpoint-type chat \
  --osl 1 \
  --request-count 1000 \
  --concurrency 1 \
  --warmup-request-count 3 \
  --artifact-dir /workspace/logs/aiperf
```