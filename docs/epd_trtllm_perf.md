# EPD vs PD vs Aggregated: TRT-LLM Multimodal Performance Analysis

This document presents benchmark results comparing three inference architectures for multimodal (vision-language) models using TRT-LLM backend.

## Test Configuration

| Parameter | Value |
|-----------|-------|
| **Model** | `llava-hf/llava-v1.6-mistral-7b-hf` |
| **Requests** | 200 per test |
| **Max Tokens** | 500 |
| **Streaming** | Enabled |
| **Images** | 3 URLs (rotating) |
| **Hardware** | DGX A100 |

## Architecture Overview

| Architecture | Workers | Description |
|--------------|---------|-------------|
| **Aggregated** | 1 | Single worker handles encode + prefill + decode |
| **PD** (Prefill-Decode) | 2 | Prefill worker handles encode + prefill, separate decode worker |
| **EPD** (Encode-Prefill-Decode) | 3 | Dedicated encode, prefill, and decode workers |

## Benchmark Results

### Median TTFT (p50) in ms - Lower is Better

```
Concurrency │ Aggregated │     PD     │    EPD     │   Winner
────────────┼────────────┼────────────┼────────────┼──────────────────
     1      │   769 ms   │   871 ms   │   915 ms   │ Aggregated ✓ (12% faster than PD)
     8      │ 1,220 ms   │ 1,152 ms   │ 1,257 ms   │ PD ✓ (6% faster than Agg)
    16      │ 2,067 ms   │ 1,812 ms   │ 2,145 ms   │ PD ✓ (12% faster than Agg)
    32      │ 6,333 ms   │ 5,160 ms   │ 3,956 ms   │ EPD ✓ (23% faster than PD)
    64      │18,169 ms   │11,772 ms   │ 9,905 ms   │ EPD ✓ (16% faster than PD)
```

### Throughput (req/s) - Higher is Better

```
Concurrency │ Aggregated │     PD     │    EPD     │   Winner
────────────┼────────────┼────────────┼────────────┼──────────────────
     1      │    0.35    │    0.35    │    0.32    │ Agg/PD ✓ (9% higher than EPD)
     8      │    1.64    │    2.14    │    2.10    │ PD ✓ (2% higher than EPD)
    16      │    2.23    │    3.31    │    3.11    │ PD ✓ (6% higher than EPD)
    32      │    2.44    │    3.56    │    3.76    │ EPD ✓ (6% higher than PD)
    64      │    2.30    │    3.47    │    3.29    │ PD ✓ (5% higher than EPD)
```

## Key Findings

### 1. EPD Wins at High Concurrency (32+)

At concurrency 32, EPD delivers:
- **23% lower TTFT** than PD (3,956ms vs 5,160ms)
- **37% lower TTFT** than Aggregated (3,956ms vs 6,333ms)
- **6% higher throughput** than PD (3.76 vs 3.56 req/s)

### 2. PD is Best for Low-Medium Concurrency (1-16)

- Lowest TTFT at concurrency 8 and 16
- Highest throughput at concurrency 8 and 16
- Simpler architecture than EPD (2 workers vs 3)

### 3. Aggregated Has Lowest Single-Request Latency

At concurrency 1:
- **769ms median TTFT** (12% faster than PD, 16% faster than EPD)
- Zero network hop overhead
- Simplest deployment

### 4. EPD Has Overhead at Low Concurrency

At concurrency 1, EPD shows:
- 19% higher latency than Aggregated (915ms vs 769ms)
- 9% lower throughput than PD/Aggregated (0.32 vs 0.35 req/s)

This is due to the 3-hop architecture (Frontend → Prefill → Encode → Prefill → Decode).

## The Crossover Point

```
           Low Concurrency          │           High Concurrency
           (1-16)                   │           (32+)
                                    │
    Aggregated: Best @1             │
                                    │
         PD: Best @8-16             │          EPD: Best @32+
                                    │
                                    │
    EPD has overhead (3 hops)       │    EPD parallelism pays off
```

The crossover happens around **concurrency 16-32**, where EPD's parallel processing outweighs its architectural overhead.

## Recommendations

| Traffic Level | Best Architecture | Median TTFT | Throughput |
|---------------|-------------------|-------------|------------|
| **Single request / Interactive** | Aggregated | 769ms | 0.35 req/s |
| **Light traffic (1-8)** | PD | 1,152ms | 2.14 req/s |
| **Medium traffic (16)** | PD | 1,812ms | 3.31 req/s |
| **Heavy traffic (32)** | EPD | 3,956ms | 3.76 req/s |
| **Very heavy traffic (64)** | EPD | 9,905ms | 3.29 req/s |

## Resource Requirements

| Architecture | GPUs Required | Complexity |
|--------------|---------------|------------|
| Aggregated | 1 | Low |
| PD | 2 | Medium |
| EPD | 3 | High |

## Performance Optimization Applied

A key optimization that enabled EPD's performance at high concurrency was wrapping `default_multimodal_input_loader` in `asyncio.to_thread()`:

```python
# Before (blocking):
inputs = default_multimodal_input_loader(...)

# After (non-blocking):
inputs = await asyncio.to_thread(
    lambda: default_multimodal_input_loader(...)
)
```

This change was applied to:
- `components/src/dynamo/trtllm/encode_helper.py` (EPD flow)
- `components/src/dynamo/trtllm/multimodal_processor.py` (PD flow)

### Impact of Optimization

| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| EPD TTFT @32 | 17,483ms | 3,956ms | **77% reduction** |
| PD TTFT @32 | 17,425ms | 5,160ms | **70% reduction** |

## Running the Benchmarks

### Launch Scripts

```bash
# Aggregated (1 worker)
./examples/backends/trtllm/launch/agg_multimodal.sh

# PD - Prefill/Decode (2 workers)
./examples/backends/trtllm/launch/disagg_multimodal.sh

# EPD - Encode/Prefill/Decode (3 workers)
./examples/backends/trtllm/launch/epd_multimodal_image.sh
```

### Benchmark Script

```bash
# Run benchmark
python3 epd_bench.py --concurrency 32 --num-requests 200

# With sample responses for quality verification
python3 epd_bench.py --concurrency 32 --num-requests 200 --show-responses 5
```

## Conclusion

- **Use Aggregated** for development and single-user interactive use cases
- **Use PD** for production with low-medium traffic (best balance of simplicity and performance)
- **Use EPD** for production with high traffic where latency at scale matters most

EPD's dedicated encode worker architecture provides significant benefits at high concurrency, but the additional complexity and resource requirements make PD the better choice for most workloads below 32 concurrent requests.
