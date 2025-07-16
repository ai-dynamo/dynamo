# Rate Limiting Guide

## Overview

The Dynamo LLM service includes an intelligent rate limiter that monitors service performance metrics and automatically throttles requests when quality degrades. Unlike traditional rate limiters that count requests, this system focuses on maintaining good user experience by monitoring:

- **Time to First Token (TTFT)** - How long users wait for the first response
- **Inter-Token Latency (ITL)** - How long between subsequent tokens

## How It Works

### Time-Weighted Exponential Moving Average

The rate limiter uses a sophisticated time-weighted exponential moving average (EMA) algorithm:

```text
average = sum(value * weight) / sum(weight)
weight = exp(-age / time_constant_secs)
```


This means:
- Recent samples have higher influence on the average
- Old samples decay exponentially over time
- System "recovers" during idle periods

### Decision Logic

For each incoming request, the system:
1. Computes current decayed EMA for TTFT and ITL
2. Compares against configured thresholds
3. Rejects request if either threshold is exceeded
4. Logs detailed metrics for observability

## Configuration

### Environment Variables

```bash
# Enable rate limiting
export DYN_RATE_LIMITER_ENABLED=true

# TTFT threshold in milliseconds (default: 1000ms = 1s)
export DYN_RATE_LIMITER_TTFT_THRESHOLD_MS=1500

# ITL threshold in milliseconds (default: 10ms)
export DYN_RATE_LIMITER_ITL_THRESHOLD_MS=15

# Time constant for EMA decay (default: 30s)
export DYN_RATE_LIMITER_TIME_CONSTANT_SECS=60

# Enable per-model vs global limits (default: false)
export DYN_RATE_LIMITER_PER_MODEL_LIMITS=true
```

### Command Line Arguments

```bash
dynamo-http \
  --enable-rate-limiting \
  --ttft-threshold-ms 1500 \
  --itl-threshold-ms 15 \
  --time-constant-secs 60 \
  --per-model-limits
```

### Programmatic Configuration

```rust
use dynamo_llm::http::service::rate_limiter::RateLimiterConfig;

let config = RateLimiterConfig::new(
    1500.0,  // TTFT threshold (ms)
    15.0,    // ITL threshold (ms)
    60.0,    // Time constant (s)
    true,    // Per-model limits
);

let http_service = HttpService::builder()
    .with_rate_limiter_config(config)
    .build()?;
```

## Monitoring

### Prometheus Metrics

The rate limiter exposes several Prometheus metrics:

**Requests rejected by rate limiter:**

```text
nv_llm_http_service_rate_limit_requests_total{model, endpoint, request_type, status}
```

**Current TTFT metrics:**

```text
nv_llm_http_service_time_to_first_token_seconds{model}
```

**Current ITL metrics:**

```text
nv_llm_http_service_inter_token_latency_seconds{model}
```

### Log Messages

When requests are rejected, detailed log messages are emitted:

```text
WARN Rate limit exceeded for model deepseek-ai/DeepSeek-R1: RateLimiterMetrics {
TTFT: TimeWeightedDiagnostics { decayed_time_weighted_average: 2.450, time_constant_secs: 30.0, last_weighted_sum: 1.245, duration_since_last_update: 0.125 },
ITL: TimeWeightedDiagnostics { decayed_time_weighted_average: 0.025, time_constant_secs: 30.0, last_weighted_sum: 1.245, duration_since_last_update: 0.125 }
}
```


## Tuning Guidelines

### Time Constant
- **Shorter (10-30s)**: Faster reaction to load changes, more sensitive
- **Longer (60-120s)**: Smoother operation, less reactive to spikes

### TTFT Threshold
- **Conservative (500-1000ms)**: Maintains very responsive feel
- **Moderate (1000-2000ms)**: Balances throughput with responsiveness
- **Aggressive (2000ms+)**: Prioritizes throughput over latency

### ITL Threshold
- **Conservative (5-10ms)**: Ensures smooth streaming experience
- **Moderate (10-20ms)**: Allows some latency for higher throughput
- **Aggressive (20ms+)**: Accepts choppier streaming for max throughput

### Per-Model vs Global
- **Per-Model**: Better for multi-tenant scenarios with different SLAs
- **Global**: Simpler for single-tenant or uniform SLA scenarios

## Best Practices

1. **Start Conservative**: Begin with lower thresholds and increase based on user feedback
2. **Monitor Closely**: Watch both rate limit counters and user-facing metrics
3. **Load Test**: Validate behavior under realistic load patterns
4. **Document SLAs**: Clearly communicate expected performance to users
5. **Alert on Rejections**: Set up alerts when rejection rates exceed acceptable levels

## Troubleshooting

### High Rejection Rates
- Check if system is genuinely overloaded
- Consider increasing thresholds temporarily
- Scale backend resources
- Investigate specific models causing issues

### No Rejections During Overload
- Verify rate limiter is enabled
- Check threshold configuration
- Ensure metrics are being recorded properly
- Review time constant settings

### Inconsistent Behavior
- Check if per-model limits are configured correctly
- Review metric collection for gaps
- Validate system clock stability