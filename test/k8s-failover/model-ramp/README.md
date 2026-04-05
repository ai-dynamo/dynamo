# Model Ramp Testing

Systematic validation of failover across progressively larger/different Qwen3 models.

## Test matrix

For each model: deploy baseline (no failover) to validate the model works and
capture memory accounting, then deploy with failover enabled to check for regressions.

| Test | Model | Type | TP | GPUs | Est. weights (BF16) | Status |
|------|-------|------|----|------|---------------------|--------|
| 1 | Qwen/Qwen3-8B | Dense | 2 | 2 | ~16 GB | PASS (baseline + failover) |
| 2 | Qwen/Qwen3-32B | Dense | 4 | 4 | ~64 GB | PASS (baseline + failover + multinode) |
| 3 | Qwen/Qwen3-30B-A3B | MoE | 2 | 2 | ~60 GB | PASS (baseline + failover) |

## Directory structure

```
model-ramp/
  README.md
  01-qwen3-8b/
    baseline.yaml
    failover.yaml
    results.md
  02-qwen3-32b/
    baseline.yaml
    failover.yaml
    multinode-baseline.yaml
    multinode-failover.yaml
    results.md
  03-qwen3-30b-a3b-moe/
    baseline.yaml
    failover.yaml
    results.md
```

## Key findings

### gpu_memory_utilization=0.85
All failover tests use `--gpu-memory-utilization 0.85` (default is 0.9). This
provides headroom for the restarting engine's CUDA context on the shared GPUs.

### Engine bug: GMS RO engine KV cache oversizing (fixed)
The RO engine's `torch.cuda.max_memory_allocated()` doesn't see GMS-mapped
weights, causing it to compute a much larger KV cache than the RW engine.
Fixed in engine image `650234f660` by adding `model_memory_usage` to `non_kv`
in `GMSWorker.determine_available_memory()`.

### Harness bug: etcdctl get polling timeout (fixed)
The multinode harness monitoring loop used `etcdctl get` polling (new gRPC
connection per call). During model download, these connections timed out
(`context deadline exceeded`), causing false failure detections. Fixed by
switching to persistent `etcdctl watch` streams.

### Engine bug: wake failure doesn't crash engine (open)
If `allocate_kv_cache_on_wake()` times out, the engine still registers the
`generate` endpoint. Requests route to an engine with no KV cache. Wake
failure should be fatal (`sys.exit(1)`).

## Cluster

- 2x A100-SXM4-80GB nodes, 8 GPUs each
- Tests target vmss000001
- Capacity freed as needed from resident workloads

## Images

- Engine: `dynamoci.azurecr.io/ai-dynamo/dynamo:multinode-failover-650234f660-vllm-runtime`
- Operator: to be rebuilt with watch-based harness from this branch
