# Live Snapshot — Working Multinode TP Failover (2026-03-11)

Captured from the `dynamo-exp` cluster, namespace `multinode-failover`.
This represents a fully working bidirectional failover setup with TP=2 across 2 pods.

## What's here

### Harness (etcd-based coordination)
- `harness_leader.sh` — Leader wrapper: etcd lease, formation barrier, go signal, worker monitoring, flock delay
- `harness_worker.sh` — Worker wrapper: leader discovery, registration, go signal wait, leader monitoring
- `barrier_patch.py` — Runtime patch applied at container startup; overrides `needed_bytes` to 70% of total GPU memory, adds verbose barrier logging with nvidia-smi snapshots and timing

### Live patched source
- `patches.py` — Full `gpu_memory_service/integrations/vllm/patches.py` as it runs on the container (after `barrier_patch.py` is applied). Contains all shadow mode patches: skip KV cache init, allocate on wake with memory barrier, cudagraph mode clamping, GMS memory adjustments.

### K8s manifests
- `pod-debug-ldr.yaml` — Leader pod (engine-0 + engine-1 containers + GMS sidecar)
- `pod-debug-wkr.yaml` — Worker pod (same but `--headless --node-rank 1`)
- `pod-etcd.yaml` — etcd v3.5.17 for harness coordination
- `svc-debug-headless.yaml` — Headless service for torch.distributed (ports 29500, 29600)
- `sa-debug-k8s-discovery.yaml` — ServiceAccount for K8s discovery backend
- `rbac.yaml` — Role + RoleBinding for K8s discovery

## Image
```
dynamoci.azurecr.io/ai-dynamo/dynamo:multinode-failover-ae51ca3f1-vllm-runtime
```

## Key parameters
- Model: Qwen/Qwen3-0.6B (TP=2, 2 nodes)
- GPU: A100-80GB (shared via DRA between engine-0 and engine-1)
- etcd lease TTL: 5s
- Formation timeout: 120s
- Barrier threshold: 70% of total GPU memory
- Flock delay (leader): 60s when lock already held (avoids concurrent startup OOM)

## Validated failover metrics
| Metric | Value |
|--------|-------|
| Leader pod GPU free | ~0.5s |
| Worker pod GPU free (harness cascade) | ~2.3-2.7s |
| KV cache allocation | ~30ms (70.19 GiB, 28 tensors) |
| Total wake (collective_rpc) | ~4.5s |
| Bidirectional | engine-0→engine-1 and engine-1→engine-0 confirmed |
