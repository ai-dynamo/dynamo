# K8s Discovery Validation Progress

## Status: E2E VALIDATED ✓

### All Steps Complete
- [x] Per-container K8s discovery implementation (Rust + Go)
- [x] Remove etcd requirement from failover.go
- [x] All Go tests pass locally
- [x] All Rust tests pass locally (8/8)
- [x] Build and push vLLM runtime image
- [x] Build and push operator image
- [x] Deploy platform (K8s discovery, no etcd) to `failover-k8s-disc`
- [x] Deploy failover DGD
- [x] Validate pods come up (3/3 containers: gms-weights + engine-0 + engine-1)
- [x] Check DynamoWorkerMetadata CRs — per-container CRs working correctly
- [x] Test inference on primary engine — PASS
- [x] Simulate failover (kill engine-0) — engine-1 woke and registered
- [x] Stale CR cleanup — engine-0's CR deleted on restart
- [x] Test inference after failover — PASS
- [ ] Encode as reusable test script

## E2E Test Results

### Pre-failover
- **Pod**: 3/3 containers running (gms-weights sidecar + engine-0 + engine-1)
- **CRs**: engine-0 CR has `generate` + `clear_kv_blocks` endpoints; engine-1 CR has only `worker_kv_indexer_query_dp0` (no `generate` — correctly invisible to frontend)
- **Inference**: ✓ Response in ~251ms

### Failover
- Killed engine-0 (`kubectl exec ... -c engine-0 -- kill 1`)
- Engine-1 acquired lock, allocated KV cache, registered `generate` endpoint with K8s discovery
- Engine-0 CR deleted by constructor on restart (stale CR cleanup working)

### Post-failover
- **CRs**: Only engine-1 CR remains with `generate` endpoint
- **Inference**: ✓ Response in ~234ms (routed to engine-1 via K8s discovery)
- Engine-0 restarting as new standby (loading weights, will sleep on lock)

## Key Observation

Engine-1's CR existed before failover (due to KV event/indexer registrations), but it did NOT advertise the `generate` endpoint — so the frontend correctly didn't route to it. The per-container discovery scheme works: **only engines that call `register_vllm_model()` become visible for inference**.

## Image Tags
| Image | Tag |
|-------|-----|
| vLLM Runtime | `dynamoci.azurecr.io/ai-dynamo/dynamo:failover-m6-discovery-0a5eace8c-vllm-runtime` |
| Operator | `dynamoci.azurecr.io/ai-dynamo/dynamo:failover-m6-discovery-0a5eace8c-operator` |

## Deployment Config
- **Namespace**: `failover-k8s-disc`
- **Discovery**: kubernetes (no etcd)
- **Cluster**: dynamo-exp (AKS, 2x A100 nodes)
- **Model**: Qwen/Qwen3-0.6B, TP=2
