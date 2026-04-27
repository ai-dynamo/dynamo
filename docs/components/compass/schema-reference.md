# Compass Attribution Report Schema

## AttributionReport

```json
{
  "compass_version": "1.0.0",
  "window": {
    "start": "2025-01-15T10:00:00Z",
    "end": "2025-01-15T10:15:00Z"
  },
  "deployment": {
    "name": "my-vllm-disagg",
    "engine": "vllm",
    "topology": "disaggregated"
  },
  "workload": {
    "qps": 12.3,
    "isl_p50": 2048,
    "osl_p50": 256
  },
  "end_to_end": {
    "ttft_ms": { "p50": 142.0, "p95": 380.0, "p99": 620.0 },
    "itl_ms": { "p50": 18.0, "p95": 31.0, "p99": 47.0 }
  },
  "verdict": {
    "primary_bottleneck": "kvbm.allocate",
    "attribution_pct": 47.2,
    "confidence": "high",
    "evidence": ["trace://span-id-1", "saturation://kvbm.radix_tree_lock"],
    "recommended_action": "Reduce lock contention in KVBM radix tree..."
  },
  "per_component": [
    {
      "component": "kvbm",
      "sub_component": "allocate",
      "contribution_pct": 47.2,
      "latency_ms": { "p50": 2.0, "p95": 3.0, "p99": 3.51 },
      "score": 0.62
    }
  ],
  "saturation": [
    {
      "component": "kvbm.radix_tree_lock",
      "utilization": 0.91,
      "queue_trend": "growing",
      "warning": true
    }
  ],
  "sub_component_breakdown": [
    {
      "component": "kvbm",
      "phases": [
        { "name": "hash", "p99_ms": 0.12 },
        { "name": "allocate", "p99_ms": 3.51, "on_cpu_ms": 1.20, "lock_wait_ms": 2.31 }
      ],
      "theoretical_floor_ms": 0.80
    }
  ],
  "floor_checks": [
    {
      "component": "kvbm.allocate",
      "observed_ms": 3.51,
      "floor_ms": 0.80,
      "ratio": 4.39,
      "is_optimization_candidate": true
    }
  ]
}
```

## Field Reference

### Verdict
| Field | Type | Description |
|-------|------|-------------|
| `primary_bottleneck` | string | Component identified as the main bottleneck |
| `attribution_pct` | float | Percentage of p99 TTFT excess attributed to this component |
| `confidence` | enum | `high`, `medium`, or `low` based on score gap between top-2 |
| `evidence` | string[] | Links to supporting traces/metrics |
| `recommended_action` | string | Actionable recommendation |

### Confidence Levels
| Level | Condition | Meaning |
|-------|-----------|---------|
| `high` | Score gap > 0.20 | Clear single bottleneck |
| `medium` | Score gap 0.05-0.20 | Likely bottleneck but other components contribute |
| `low` | Score gap < 0.05 | Multiple components contribute roughly equally |

### QueueTrend
| Value | Meaning |
|-------|---------|
| `growing` | Queue length increasing at steady arrival rate (saturated) |
| `stable` | Queue length steady |
| `draining` | Queue length decreasing |

### SweepResult (Sensitivity Matrix)
```json
{
  "perturbation": "kvbm-allocate-ms",
  "multiplier": 0.5,
  "concurrency": 32,
  "predicted_ttft_p99_ms": 474.3,
  "predicted_throughput_rps": 65.2
}
```

### CalibrationResult
```json
{
  "trace_source": "recorded.jsonl",
  "mocker_ttft_p99_ms": 635.0,
  "real_ttft_p99_ms": 620.0,
  "residual_pct": 2.4,
  "is_calibrated": true,
  "threshold_pct": 15.0
}
```
