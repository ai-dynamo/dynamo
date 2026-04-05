# Qwen3-30B-A3B (MoE) Results

## Model info
- **Model**: Qwen/Qwen3-30B-A3B
- **Type**: Mixture of Experts (128 experts, top-2 routing)
- **Total parameters**: 30.5B (BF16)
- **Active parameters per token**: 3.3B
- **Architecture**: Qwen3MoeForCausalLM
- **Engine image**: `multinode-failover-650234f660-vllm-runtime`

## Test 1: Baseline eager (TP=2, 0.85 util)

| Metric | Value |
|--------|-------|
| nvidia-smi per GPU | 70,871 MiB used / 10,282 MiB free |
| KV cache | 37.77 GiB, 824,976 tokens, 51,561 blocks |
| Weights load time | 14.68s |
| Init time | 6.94s |
| Inference | 200, 2022ms |
| MoE backend | TRITON (unquantized) |

Note: Missing MoE config warning — no tuned config for E=128,N=384 on A100.

## Test 2: Baseline CUDA graphs (TP=2, 0.85 util)

| Metric | Value |
|--------|-------|
| nvidia-smi per GPU | 71,723 MiB used / 9,430 MiB free |
| KV cache | 37.70 GiB, 823,616 tokens |
| torch.compile | 28.57s |
| Init time | 47.87s |
| Inference | 200, 213ms |
| CUDA graphs overhead | 71,723 - 70,871 = 852 MiB/GPU |

## Test 3: Failover eager (TP=2, GMS, 0.85 util)

### Bug fix verification

| | Engine-0 (RW) | Engine-1 (RO) |
|--|---|---|
| non_kv | 29.13 GiB | 29.24 GiB |
| projected KV | 38.24 GiB | 38.12 GiB |
| torch_peak | 29.13 GiB | 0.68 GiB |
| weights | 28.56 GiB | 28.56 GiB |

Both engines compute matching KV cache sizes. Bug fix confirmed for MoE.

### MoE-specific metrics
- GMS committed: 28.56 GiB/GPU, 162 mappings
- Compared to Qwen3-32B dense: 15.51 GiB/GPU, 262 mappings
- MoE has ~2x weight per GPU (TP=2 vs TP=4) but fewer GMS mappings

### Results

| Step | Result |
|------|--------|
| Deploy | 0 restarts, both engines ready |
| nvidia-smi | 74,385 MiB/GPU |
| Inference | 200, 2020ms |
| Failover (kill e0) | ~3s |
| KV cache on wake | 38.12 GiB (48 tensors) |
| Inference post-failover | 200, 2409ms |

## Test 4: Failover CUDA graphs (TP=2, GMS, 0.85 util)

### Bug fix verification

| | Engine-0 (RW) | Engine-1 (RO) |
|--|---|---|
| non_kv | 29.13 GiB | 29.24 GiB |
| projected KV | 38.24 GiB | 38.12 GiB |

Matching. Confirmed.

### Results

| Step | Result |
|------|--------|
| Deploy | 0 restarts |
| nvidia-smi | 75,383 MiB/GPU |
| Inference | 200, 264ms |
| Failover | ~2s |
| KV cache on wake | 38.12 GiB (48 tensors) |
| Inference post-failover | 200, 259ms |

## MoE vs Dense comparison

| Metric | Qwen3-32B (dense, TP=4) | Qwen3-30B-A3B (MoE, TP=2) |
|--------|------------------------|---------------------------|
| Weights per GPU | 15.51 GiB | 28.56 GiB |
| GMS mappings | 262 | 162 |
| KV cache (eager, failover) | 51.40 GiB | 38.24 GiB |
| Failover time | ~3s | ~2-3s |
| Inference (graphs) | 358ms | 264ms |
| CUDA graph overhead | 904 MiB | 852 MiB |

MoE has higher per-GPU weight footprint but faster inference due to lower
active parameter count per token (3.3B vs 32.8B). KV cache is smaller because
weights consume more of the 0.85 utilization budget. Failover timing is
comparable — GMS remap + KV allocation is the bottleneck regardless of model
architecture.
