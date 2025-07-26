# Dynamo LMCache tests Quick Start Guide

```bash
# 1. Enter test directory
cd ./tests/lmcache

# 2. Download MMLU dataset (run once only)
python3 download_mmlu.py

# 3. Run test
./run_test.sh "Qwen/Qwen3-0.6B"
```

## Detailed Test Steps

### Step 1: Download Data
```bash
# Python method (recommended)
python3 download_mmlu.py
```

### Step 2: Baseline Test
```bash
# Start dynamo without LMCache
./deploy-1-dynamo.sh "Qwen/Qwen3-0.6B"
# Wait for model to load, then in another terminal run:
python3 1-mmlu-dynamo.py --model "Qwen/Qwen3-0.6B" --number-of-subjects 15
# Stop services with Ctrl+C in deploy script terminal
```

### Step 3: LMCache Test
```bash
# Start dynamo with LMCache
./deploy-2-dynamo.sh "Qwen/Qwen3-0.6B"
# Wait for model to load, then in another terminal run:
python3 2-mmlu-dynamo.py --model "Qwen/Qwen3-0.6B" --number-of-subjects 15
# Stop services with Ctrl+C in deploy script terminal
```

### Step 4: Result Comparison
```bash
# Analyze results
python3 summarize_scores_dynamo.py
```

## Success Criteria
- MMLU accuracy difference between two configurations < 1%
- If difference is small, LMCache functionality is correct

## Test Models
- `Qwen/Qwen3-0.6B` - Lightweight, recommended for quick testing
- Other HuggingFace models - Modify model name as needed
