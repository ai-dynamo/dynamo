# Qwen3-8B Results

## Model info
- **Model**: Qwen/Qwen3-8B
- **Type**: Dense
- **Parameters**: 8.2B (BF16 auto-detected)
- **TP**: 2, **GPUs**: 2 (A100-SXM4-80GB)

## Baseline (no failover)

### Memory accounting (per GPU)

| Category | Per-GPU | Source |
|----------|---------|--------|
| CUDA context + PyTorch + NCCL | ~1.5 GB | Residual (can't isolate) |
| Model weights | ~8.2 GB | Calculated: 8.2B x 2 bytes / 2 GPUs |
| Peak activations headroom | ~1.1 GB | Residual (can't isolate from context) |
| KV cache | 62.56 GB | Logged: `Available KV cache memory: 62.56 GiB` |
| CUDA graphs + torch.compile | ~0.4 GB | Measured: nvidia-smi delta vs enforce-eager |
| **Total used** | **73.1 GB** (74,819 MiB) | nvidia-smi |
| **Free (safety buffer)** | **6.2 GB** (6,334 MiB) | nvidia-smi |

### Enforce-eager variant

| Metric | With CUDA Graphs | Enforce Eager | Delta |
|--------|-----------------|---------------|-------|
| nvidia-smi used (per GPU) | 74,819 MiB | 74,385 MiB | -434 MiB |
| KV cache available | 62.56 GiB | 62.66 GiB | +0.10 GiB |
| Init time | 34.4s | 4.1s | -30.3s |

### Key observations
- Memory is fully pre-allocated at init (nvidia-smi identical idle vs 50 concurrent requests)
- Weights loaded in 3.4s, torch.compile 25.5s, CUDA graphs ~7s

### Inference
- [x] 200 response: 208ms (graphs), 275ms (eager)

## Failover (GMS shadow mode, gpu-memory-utilization=0.80)

### Bug: KV cache oversizing on RO engine (fixed in image 650234f660)
With gpu_memory_utilization=0.9, failover failed because the RO engine
calculated 66.78 GiB KV cache (vs 51.40 GiB on RW engine). The RO engine's
`torch.cuda.max_memory_allocated()` doesn't see GMS-mapped weights. On wake,
the restarting peer's CUDA context consumed ~2 GiB, leaving insufficient room.

Lowering to 0.80 worked around this by reducing the KV cache request. The
proper fix (in image 650234f660) adds `model_memory_usage` to `non_kv` in
`GMSWorker.determine_available_memory()`.

### Failover with 0.80 utilization — enforce-eager

| Step | Result |
|------|--------|
| Deploy | 0 restarts |
| Engine-0 projected KV | 55.20 GiB |
| Engine-1 projected KV | 62.83 GiB |
| nvidia-smi | 69,311 MiB/GPU |
| Inference | 200, 846ms |
| Failover (kill e0 -> e1 takes over) | ~1s |
| Inference post-failover | 200, 625ms |

### Failover with 0.80 utilization — CUDA graphs

| Step | Result |
|------|--------|
| Deploy | 0 restarts |
| Engine-0 projected KV | 55.10 GiB |
| nvidia-smi | 70,021 MiB/GPU |
| Inference | 200, 283ms |
| Failover | ~1s |
| Inference post-failover | 200, 279ms |

### Standby wake observation
The standby engine gets more KV cache on wake (62.83 GiB) than the active
had (55.20 GiB) because the active engine's KV cache is freed on death and
the standby re-profiles with more available memory. The 0.80 utilization
only constrains the initial active engine's allocation.
