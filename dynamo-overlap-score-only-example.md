# Dynamo Overlap-Score-Only Router Example

This branch adds a load-weight control to the Rust KV router used by both the
embedded frontend and the standalone router.

Set the load weight to zero to ignore active prefill and decode load during
worker selection:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-load-weight 0 \
    --router-temperature 0 \
    --router-queue-threshold None \
    --admission-control none
```

The equivalent Kubernetes environment variables are:

```yaml
envs:
  - name: DYN_ROUTER_MODE
    value: kv
  - name: DYN_ROUTER_LOAD_WEIGHT
    value: "0"
  - name: DYN_ROUTER_TEMPERATURE
    value: "0"
  - name: DYN_ROUTER_QUEUE_THRESHOLD
    value: "None"
  - name: DYN_ADMISSION_CONTROL
    value: none
```

## Scoring Behavior

The default value is:

```text
router_load_weight = 1
```

This preserves the existing KV router cost calculation.

With:

```text
router_load_weight = 0
```

active prefill and decode load contribute nothing to the score. Eligible
workers are ranked by weighted KV overlap:

- device-local overlap uses `router_kv_overlap_score_credit`
- host-pinned overlap uses `router_host_cache_hit_weight`
- disk overlap uses `router_disk_cache_hit_weight`
- configured shared-cache overlap uses `shared_cache_multiplier`

The worker with the greatest weighted overlap wins.

## Cold Ties and Approximate Routing

No agent or trajectory ID is required.

With `router_temperature = 0`, workers with equal overlap, including a fully
cold request, use a round-robin tie-break. In approximate mode, the embedded
push router records that selected worker in the approximate index:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-load-weight 0 \
    --no-router-kv-events \
    --router-queue-threshold None \
    --admission-control none
```

Later requests with the same cached prefix see predicted overlap on that worker
and naturally route back to it. This provides cache affinity, but it is not a
hard sticky-session guarantee: TTL expiration, cache eviction, routing
constraints, or better overlap on another worker can change the destination.

## Load Gates

`--router-load-weight 0` changes worker scoring only.

Use both of these when load must not affect candidate eligibility or delay
routing:

```text
--admission-control none
--router-queue-threshold None
```

Explicit worker pins, required routing constraints, and preferred-taint score
multipliers still apply.
