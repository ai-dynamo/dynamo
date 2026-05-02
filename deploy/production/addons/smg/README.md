# SMG (Shepherd Model Gateway) — Production Addon

SMG sits in front of the Dynamo Frontend Service as the OpenAI-compatible client entry point. It owns gateway concerns (model selection, retry, circuit-breaker, observability, optional history); Dynamo owns KV-aware routing across SGLang prefill/decode workers; SGLang owns the kernel-level HiSparse / IndexCache / TurboQuant work. The PyTorch blog from the SMG team makes this composability the whole point — these layers exist precisely because the responsibilities are different. See [docs/smg-integration.md](../../docs/smg-integration.md) for the full boundary diagram and rationale.

## What this addon adds

- An ArgoCD `Application` (`gitops/apps/70-smg.yaml`) syncing the upstream `lightseekorg/smg` Helm chart at version `1.4.1` into the `smg` namespace, with the values in this directory.
- A single SMG router replica configured with `policy: round_robin` and a single backend (`deepseek-v32-reap-sglang-frontend.dynamo-system.svc.cluster.local:8000`).
- A ServiceMonitor wired to `kube-prometheus-stack` so SMG metrics flow into the existing Prometheus.

## What this addon does NOT change

- The 4-GPU prefill + 4-GPU decode disaggregation in [`examples/deepseek-v32-reap-sglang.yaml`](../../examples/deepseek-v32-reap-sglang.yaml) is untouched. SMG runs CPU-only (no `nvidia.com/gpu` requests).
- Dynamo Frontend's `--router-mode kv --router-kv-events` configuration is unchanged. SMG does NOT do prefix-aware routing here.
- HiSparse, IndexCache, TurboQuant, and SMC-SD remain configured at the SGLang worker level.

## Why `policy: round_robin` and not `cache_aware`

SMG's `cache_aware` policy and Dynamo Frontend's KV router both maintain prefix trees. Putting both in the path would make two trees disagree on hot prefixes — Dynamo sees the *real* SGLang KV cache state via ZMQ kv-events, SMG would see only its routing-decision history. Result: more cross-worker shuffles than either alone. With `round_robin` and a single Dynamo Frontend backend, SMG is a pure pass-through for routing decisions; Dynamo is the single owner of "which worker gets this prefix".

If/when this cluster grows to multiple Dynamo Frontends in front of multiple SGLang fleets (e.g., per-region or per-model), `policy: power_of_two` across Dynamo Frontends becomes appropriate — but each Dynamo Frontend still owns prefix routing within its own fleet.

## Verifying after deploy

```bash
# Port-forward the SMG router service from the cluster
kubectl -n smg port-forward svc/smg-router 8080:80

# Hit it with an OpenAI-compatible request
curl -sS http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-GatedNorm-G1",
    "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
    "max_tokens": 8
  }'
```

A 200 with a `chat.completion` response means the full chain (SMG → Dynamo Frontend → SGLang decode worker via Dynamo's KV router) is live. The end-to-end script lives at [`tests/smg-roundtrip.sh`](../../../../tests/smg-roundtrip.sh).
