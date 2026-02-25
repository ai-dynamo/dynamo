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

### Phase 1: Core Infrastructure ✅

- [x] Add `ATTENTION` and `FFN` to `DisaggregationMode` enum
- [x] Create `AFDAttentionHandler` placeholder for attention workers
- [x] Create `AFDFFNHandler` placeholder for FFN workers
- [x] Update `Config._set_serving_strategy()` to recognize AFD modes
- [x] Add AFD_DESIGN.md with architecture documentation

### Phase 2: Communication Protocol ✅

- [x] Create `AFDCommunicationManager` for Attention-FFN communication
- [x] Implement `AFDActivationBatch` serialization/deserialization
- [x] Implement `AFDFFNResult` serialization/deserialization
- [x] Create `AFDMicrobatchPipeline` for pipelined execution
- [x] Add AFD configuration parameters to `DynamoSGLangConfig`
- [x] Create `init_afd.py` with initialization functions
- [x] Update `main.py` to support AFD modes

### Phase 3: Integration 🚧

- [ ] Implement NIXL-based zero-copy activation transfer
- [ ] Add model partitioning support (split attention/FFN layers)
- [ ] Integrate with SGLang scheduler
- [ ] Add AFD-specific metrics and monitoring
- [ ] End-to-end testing with sample models

### Phase 4: Optimization 🔜

- [ ] Implement optimal A/F ratio calculation
- [ ] Add load balancing across Attention workers
- [ ] Performance tuning and benchmarking
- [ ] Compare with baseline (aggregated) performance

## Usage

```bash
# Attention worker (r instances)
python -m dynamo.sglang \
    --model-path <model> \
    --disaggregation-mode attention \
    --afd-attention-ratio 8 \
    --afd-ffn-endpoint dynamo.ffn.generate

# FFN worker (1 instance, shared by r Attention workers)
python -m dynamo.sglang \
    --model-path <model> \
    --disaggregation-mode ffn \
    --afd-attention-ratio 8
```

## Configuration Parameters

| Parameter | CLI Flag | Env Var | Default | Description |
|-----------|----------|---------|---------|-------------|
| `afd_attention_ratio` | `--afd-attention-ratio` | `DYN_SGL_AFD_ATTENTION_RATIO` | None | Number of Attention workers per FFN (r in r:1) |
| `afd_ffn_endpoint` | `--afd-ffn-endpoint` | `DYN_SGL_AFD_FFN_ENDPOINT` | None | FFN worker endpoint for Attention workers |
| `afd_microbatch_size` | `--afd-microbatch-size` | `DYN_SGL_AFD_MICROBATCH_SIZE` | 256 | Microbatch size for pipelining |
| `afd_sync_timeout_ms` | `--afd-sync-timeout-ms` | `DYN_SGL_AFD_SYNC_TIMEOUT_MS` | 1000 | Synchronization timeout (ms) |

## Communication Protocol

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `ACTIVATION_TRANSFER` | Attention → FFN | Send activation batch |
| `ACTIVATION_RESULT` | FFN → Attention | Return FFN output |
| `SYNC_REQUEST/ACK` | Bidirectional | Synchronization |
| `HEARTBEAT` | Bidirectional | Health check |

### Activation Batch Format

```
┌─────────────────────────────────────────────────────┐
│ Request ID (variable length)                        │
│ Layer Index (4 bytes)                               │
│ Activations Shape (12 bytes: batch x seq x hidden)  │
│ Activations Data (variable: float32)                │
│ Attention Mask Shape + Data (optional)              │
│ Position IDs Shape + Data (optional)                │
│ Metadata JSON (variable length)                     │
└─────────────────────────────────────────────────────┘
```

## Related Files

| File | Purpose |
|------|---------|
| `common/constants.py` | `DisaggregationMode` enum with ATTENTION/FFN |
| `sglang/args.py` | Configuration parsing |
| `sglang/backend_args.py` | AFD CLI arguments |
| `sglang/init_afd.py` | AFD worker initialization |
| `sglang/main.py` | Entry point with AFD mode support |
| `sglang/afd_communication.py` | Communication protocol implementation |
| `sglang/request_handlers/llm/afd_attention_handler.py` | Attention worker handler |
| `sglang/request_handlers/llm/afd_ffn_handler.py` | FFN worker handler |

## References

1. [Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](https://arxiv.org/abs/2601.21351)
2. [Step-3 is Large yet Affordable](https://arxiv.org/abs/2507.19427) - First production AFD implementation
3. [SGLang Disaggregated Serving](../../docs/pages/backends/sglang/README.md)

## Performance Considerations

### Optimal A/F Ratio

The paper recommends calculating the optimal ratio based on:
- Model architecture (hidden dim, FFN expansion factor)
- Hardware specs (memory bandwidth, compute throughput)
- Workload characteristics (batch size, sequence length)

Formula (simplified):
```
r* = ceil(T_attn / T_ffn)
```
Where:
- `T_attn` = time for attention computation
- `T_ffn` = time for FFN computation

### Memory vs Compute Trade-off

| Ratio | Memory Utilization | Compute Utilization | Best For |
|-------|-------------------|--------------------|----------|
| 1:1 | Balanced | Balanced | Small models, short sequences |
| 4:1 | High attention memory | High FFN compute | Large batch, short sequences |
| 8:1 | Very high attention | Max FFN throughput | Long sequences, large batch |
| 16:1 | Extreme memory | Extreme compute | Very long contexts |
