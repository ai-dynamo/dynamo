# AFD (Attention-FFN Disaggregation) Implementation

## Overview

AFD is a disaggregation architecture for LLM decode phase that separates stateful Attention layers from stateless FFN layers, enabling independent scaling of memory and compute resources.

**Reference:** [Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](https://arxiv.org/abs/2601.21351)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    rA : 1F Topology                         │
│                                                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │  Attention  │  │  Attention  │  │  Attention  │  ...    │
│   │  Worker 1   │  │  Worker 2   │  │  Worker r   │        │
│   │  (KV Cache) │  │  (KV Cache) │  │  (KV Cache) │        │
│   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│          │                │                │                │
│          └────────────────┼────────────────┘                │
│                           │                                 │
│                           ▼                                 │
│                    ┌─────────────┐                          │
│                    │  FFN Worker │                          │
│                    │  (Shared)   │                          │
│                    │  Compute    │                          │
│                    └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Key Characteristics

| Component | Attention Worker | FFN Worker |
|-----------|------------------|------------|
| **State** | Stateful (KV cache) | Stateless |
| **Bottleneck** | Memory bandwidth | Compute |
| **Scaling** | Memory resources | Compute resources |
| **Work per step** | Grows with context length | Constant per batch |

## Implementation Status

### Phase 1: Core Infrastructure (Current)

- [x] Add `ATTENTION` and `FFN` to `DisaggregationMode` enum
- [x] Create `AFDAttentionHandler` placeholder
- [x] Create `AFDFFNHandler` placeholder
- [x] Update `Config._set_serving_strategy()` to recognize AFD modes

### Phase 2: Communication Protocol

- [ ] Implement NIXL-based activation transfer
- [ ] Add microbatch pipelining support
- [ ] Implement synchronization barriers
- [ ] Add A/F ratio configuration parameters

### Phase 3: Integration

- [ ] Integrate with SGLang backend
- [ ] Add AFD-specific metrics and monitoring
- [ ] Update documentation

### Phase 4: Optimization

- [ ] Implement optimal A/F ratio calculation
- [ ] Add load balancing across Attention workers
- [ ] Performance tuning and benchmarking

## Usage

```bash
# Attention worker (r instances)
python -m dynamo.sglang \
    --model-path <model> \
    --disaggregation-mode attention \
    --afd-attention-ratio 8 \
    --afd-ffn-endpoint dyn://dynamo.ffn.generate

# FFN worker (1 instance, shared by r Attention workers)
python -m dynamo.sglang \
    --model-path <model> \
    --disaggregation-mode ffn \
    --afd-attention-ratio 8
```

## Configuration Parameters (Planned)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--afd-attention-ratio` | Number of Attention workers per FFN (r in r:1) | 1 |
| `--afd-ffn-endpoint` | Endpoint for FFN worker communication | None |
| `--afd-microbatch-size` | Microbatch size for pipelining | 256 |
| `--afd-sync-timeout-ms` | Synchronization timeout in milliseconds | 1000 |

## Related Files

- `components/src/dynamo/common/constants.py` - DisaggregationMode enum
- `components/src/dynamo/sglang/args.py` - Configuration parsing
- `components/src/dynamo/sglang/request_handlers/llm/afd_attention_handler.py`
- `components/src/dynamo/sglang/request_handlers/llm/afd_ffn_handler.py`

## References

1. [Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](https://arxiv.org/abs/2601.21351)
2. [Step-3 is Large yet Affordable](https://arxiv.org/abs/2507.19427) - First production AFD implementation
3. [SGLang Disaggregated Serving](../../docs/pages/backends/sglang/README.md)
