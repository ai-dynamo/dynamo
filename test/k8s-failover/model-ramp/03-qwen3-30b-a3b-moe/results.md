# Qwen3-30B-A3B (MoE) Results

## Model info
- **Model**: Qwen/Qwen3-30B-A3B
- **Type**: Mixture of Experts (128 experts, top-2 routing)
- **Total parameters**: 30.5B
- **Active parameters per token**: 3.3B
- **TP**: 2, **GPUs**: 2

## Baseline (no failover)

### Memory accounting

| Category | Per-GPU | Total |
|----------|---------|-------|
| Model weights (all experts) | | |
| Activations + overhead | | |
| CUDA graphs | | |
| KV cache | | |
| **Total used** | | |
| GPU capacity | 80 GB | 160 GB |
| Free | | |

### Inference test
- [ ] Model loads successfully
- [ ] Inference returns valid response
- [ ] Response time: ___ms

## Failover (GMS shadow mode)

### Memory accounting (with GMS)

| Category | Per-GPU | Total |
|----------|---------|-------|
| GMS weights (shared, all experts) | | |
| Engine-0 non-KV | | |
| Engine-1 non-KV | | |
| Active engine KV cache | | |
| **Total used** | | |
| GPU capacity | 80 GB | 160 GB |
| Free | | |

### Failover test
- [ ] Both engines init in shadow mode
- [ ] Active engine acquires lock and serves
- [ ] Standby engine sleeps
- [ ] Inference works
- [ ] Kill active -> standby takes over
- [ ] Inference works after failover
- [ ] Killed engine recovers as standby

### MoE-specific observations
- Expert routing during failover: ___
- Weight sharing of expert blocks via GMS: ___
