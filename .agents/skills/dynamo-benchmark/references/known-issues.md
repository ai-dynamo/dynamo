# Benchmarking Known Issues

Stable issue patterns relevant to performance measurement. Strict
6-element shape per.

---

### TTFT measurement spikes during the first ~30 s

**Symptom:** AIPerf reports TTFT p50 100-200% above the recipe baseline for the first 30-60 seconds of the run, then settles.

**Root cause:** Cold KV cache, JIT-compiled CUDA kernels not yet primed, and the worker's connection-pool warm-up. The first burst of requests pays a one-time cost.

**Affected:** All Dynamo releases, all backends. More pronounced with TensorRT-LLM (kernel JIT) and disaggregated mode.

**Fix:** Add a warm-up period before the measurement window. AIPerf has `--warmup-request-count`:

```bash
aiperf benchmark ... --warmup-request-count 100 --measurement-interval 300
```

Alternatively, run the benchmark twice and discard the first run.

**Verify:** Compare TTFT distributions of the warm-up run vs the measurement run; the latter should be stable.

Source: NVIDIA AIPerf documentation.

---

### `cache_hit_rate` is null on a KV-aware deployment

**Symptom:** AIPerf output JSON has `"cache_hit_rate": null` even though the deployment has a KV router and the Frontend has `--router kv-aware`.

**Root cause:** Either (a) AIPerf is not configured to drive multi-turn conversations (`--multi-turn-mean` unset), (b) the Frontend does not yet expose the cache-hit metric for the AIPerf to read, or (c) the Router is not actually in the request path (check the deployment topology).

**Affected:** Dynamo 1.2.x with KV-aware routing.

**Fix:**

1. Set `--multi-turn-mean 4` (or higher) to give the router something to cache.
2. Verify the router is in the path: `kubectl get pods -l app.kubernetes.io/component=router` should return the router pod; check Frontend logs for "delegating to router".
3. If the metric is still null, scrape Prometheus directly: `curl http://<router-svc>:8002/metrics | grep cache_hit`.

**Verify:** Subsequent runs report a non-null `cache_hit_rate`.

Source: (container image conventions, KV router).

---

### Benchmark saturates worker but `kubectl top` shows GPU idle

**Symptom:** TTFT degrades dramatically under load, throughput plateaus, but `kubectl top pods` shows the worker pod using <50% GPU.

**Root cause:** Bottleneck is somewhere other than the GPU compute. Likely candidates: (a) tokenizer is single-threaded and saturates CPU, (b) Frontend → worker NATS hop is the bottleneck, (c) memory bandwidth (not flops) is saturated, (d) request queueing in the engine.

**Affected:** All releases. More common with very large prompts (ISL > 8000) where the tokenizer cost dominates.

**Fix:**

1. `kubectl exec <worker> -- top` — CPU usage on the tokenizer thread?
2. Check the Frontend's request-queue depth metric (`/metrics`, look for `request_queue_depth`).
3. Reduce ISL or increase tokenizer parallelism (varies by backend).

**Verify:** GPU utilization rises after the bottleneck is removed.

Source: NVIDIA AIPerf documentation;.

---

### Recipe baseline mismatch by >10% on identical hardware

**Symptom:** Reproducing a recipe's `benchmark/run.sh` produces numbers 10-30% worse than the recipe's published `expected.json`.

**Root cause:** Likely candidates (in order): (a) different release line of Dynamo (recipes are pinned per release), (b) different backend container tag, (c) NIXL version mismatch on a disagg deploy (per, `nixl_ref` in `container/context.yaml` drives the actual NIXL in the runtime container), (d) noisy-neighbor on the cluster.

**Affected:** Any recipe reproduction.

**Fix:**

1. `git log -1 --format='%h' /Users/dagil/dynamo` — confirm the source matches the recipe's recorded commit.
2. `kubectl describe pod <worker> | grep Image:` — image tag matches the recipe's pin?
3. `kubectl get dgd <name> -o yaml | grep -i nixl` — NIXL version per the pod?
4. `kubectl get nodes -L nvidia.com/gpu.product` — identical SKU?

If all match, the deployment is on a different release line and the
recipe's `expected.json` no longer applies. Re-publish the recipe
benchmark after a re-validation run.

**Verify:** After alignment, reproduction lands within 5%.

Source:,.
