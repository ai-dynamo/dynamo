# LMCache Dynamo MMLU Testing Suite

## Overview
Test the correctness of Dynamo integration with LMCache by comparing MMLU benchmark results with and without LMCache enabled.

## Testing Principle
Compare MMLU test results under two configurations:
- **Baseline Test**: Dynamo without LMCache
- **LMCache Test**: Dynamo with LMCache enabled

If both configurations produce the same inference results, it verifies that LMCache functionality is correct.

## Quick Start

### Prerequisites
1. Ensure dynamo and its dependencies are properly installed (i.e. nats and etcd are running)
2. Download MMLU dataset to `data` directory
3. Ensure HuggingFace models are accessible

### Download MMLU Dataset

```bash
cd ./tests/lmcache

# Auto-download and organize data
python3 download_mmlu.py
```

### Run Single Model Test
Change model name in the script to test other models.
```bash
cd ./tests/lmcache

# 1. Baseline test (without LMCache)
./deploy-baseline-dynamo.sh Qwen/Qwen3-0.6B
# Wait for model to load, then run test in another terminal:
python3 mmlu-baseline-dynamo.py --model Qwen/Qwen3-0.6B --number-of-subjects 15
# Stop services with Ctrl+C in the deploy script terminal

# 2. LMCache test (with LMCache enabled)
./deploy-lmcache_enabled-dynamo.sh Qwen/Qwen3-0.6B
# Wait for model to load, then run test in another terminal:
python3 mmlu-lmcache_enabled-dynamo.py --model Qwen/Qwen3-0.6B --number-of-subjects 15
# Stop services with Ctrl+C in the deploy script terminal

# 3. Compare results
python3 summarize_scores_dynamo.py
```

## File Description

### Deployment Scripts
- **`deploy-baseline-dynamo.sh`**: Deploy Dynamo without LMCache (baseline)
- **`deploy-lmcache_enabled-dynamo.sh`**: Deploy Dynamo with LMCache enabled (test)

### Test Scripts
- **`mmlu-baseline-dynamo.py`**: Run MMLU test on baseline Dynamo
- **`mmlu-lmcache_enabled-dynamo.py`**: Run MMLU test on Dynamo with LMCache
- **`summarize_scores_dynamo.py`**: Compare and analyze test results

## Architecture Differences

### Baseline Architecture (deploy-baseline-dynamo.sh)
```
HTTP Request → Dynamo Ingress(8000) → Dynamo Worker → Direct Inference
```

### LMCache Architecture (deploy-lmcache_enabled-dynamo.sh)
```
HTTP Request → Dynamo Ingress(8000) → Dynamo Worker → LMCache-enabled Inference
Environment:LMCACHE_CHUNK_SIZE=256
            LMCACHE_LOCAL_CPU=True
            LMCACHE_MAX_LOCAL_CPU_SIZE=1.0
```

## API Format

Test scripts use Dynamo's Chat Completions API:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": Qwen/Qwen3-0.6B,
    "messages": [{"role": "user", "content": "question content"}],
    "temperature": 0,
    "max_tokens": 3,
    "stream": false,
    "seed": 42
  }'
```


## Result Interpretation

After testing completes, the following files will be generated:
- `dynamo-baseline-{model_name}.jsonl`: Baseline test results
- `dynamo-lmcache-{model_name}.jsonl`: LMCache test results

If the accuracy in both result files is very close (difference < 1%), it indicates LMCache functionality is correct.

## Notes

1. **Determinism guarantee**: All tests use the same seed (42) and zero temperature to ensure reproducible results
2. **Pre-requisites**: Ensure nats and etcd are running.
3. **Sequential execution**: Must stop the first test before starting the second to avoid port conflicts