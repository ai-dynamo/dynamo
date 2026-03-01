# Summary

We would like to test 2 workflow to confirm embedding cache is working as expected

- vLLM E/PD, and
- TRT-LLM E/PD

We can use the below instructions to start workflow and send requests.
We don't need to wait for the aiperf to finish, the idea is to verify two things
- [functional] when multimodal embedding cache is enabled, workflow should continue function
- [non-functional] performance gain on certain hardware (e.g. GB200)

# Start Workflow

## vLLM

```
bash examples/backends/vllm/launch/disagg_multimodal_e_pd.sh \
  --multimodal-embedding-cache-capacity-gb 10 \
  &> logs/dynamo_vllm_e_pd_mm_embedding_cache_enabled.txt &
```

## TRT-LLM

```
bash examples/backends/trtllm/launch/disagg_e_pd.sh \
  --multimodal-embedding-cache-capacity-gb 10 \
  &> logs/dynamo_trtllm_e_pd_mm_embedding_cache_enabled.txt &
```

# Send Requests

generate files with image URLs

```
python benchmarks/multimodal/jsonl/main.py \
  --image-mode http \
  -n 700 \
  --images-per-request 30 \
  --images-pool 1000 \
  --user-text-tokens 300
```

```
aiperf profile \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-file benchmarks/multimodal/jsonl/700req_30img_1000pool_300word_http.jsonl \
  --custom-dataset-type single_turn \
  --osl 1 \
  --request-count 700 \
  --concurrency 1 \
  --warmup-request-count 3 \
  --artifact-dir /workspace/logs/aiperf
```
