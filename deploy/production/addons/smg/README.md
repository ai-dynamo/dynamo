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

## What this addon deliberately does NOT enable

These SMG capabilities are off by default in this profile. Each item lists why it's off and how to turn it on.

### Tokenization-based routing
**Off.** Round-robin over a single Dynamo Frontend doesn't tokenize the request — there's nothing to hash a prefix against when there's only one backend, and Dynamo's own KV router does prefix matching downstream against the live SGLang KV state. Turn on only if you scale to >1 Dynamo Frontend in front of separate SGLang fleets and switch `policy:` to `prefix_hash` or `power_of_two`. To enable, set `router.tokenizerPath` to the model path or HF id (must match the SGLang `--served-model-name`).

### Tool-call and reasoning parsing
**At the SGLang layer, not at SMG.** The decode worker already runs `--tool-call-parser deepseekv32 --reasoning-parser deepseek-v3` (see `examples/deepseek-v32-reap-sglang.yaml`). The parsed `tool_calls` array and reasoning fields appear in the engine's response; SMG passes them through unchanged. Re-parsing at SMG would duplicate work and risk drift from the engine's view of the model's chat template.

### Chat history / audit log
**Off (`postgresql.enabled: false`).** SMG can persist requests + responses to Postgres for audit, replay, and per-user usage rollups. Pulling in Postgres is a real ops cost (backups, schema migrations, secrets). Turn on by either:

- `postgresql.enabled: true` — spins up a co-located Bitnami Postgres (fastest path; OK for dev).
- Pointing `history.postgres.dsn` at an existing managed Postgres (recommended for production; requires an `external-secrets` `ExternalSecret` for the DSN).

### MCP tool registry
**Off.** SMG can host MCP tool definitions and proxy tool invocations from the model; not configured here. Enable via `mcp.tools` in values.yaml when the application layer wants centralized tool management.

## Pulling in SMG upstream updates

The chart pin lives in two places:

- `gitops/apps/70-smg.yaml`: `targetRevision: vX.Y.Z` (chart version)
- `addons/smg/values.yaml`: `global.image.tag: "X.Y.Z"` (image tag)

Both are watched by the Renovate config at [`.github/renovate.json5`](../../../../.github/renovate.json5):

- **Patch and minor releases**: Renovate opens an auto-merge PR (you still see it; CI must pass).
- **Major releases**: Renovate opens a `needs-human-review` PR. SMG's auth, retry, and circuit-breaker semantics can shift across majors; manual review preserves the boundary contract above.

To bump manually outside of Renovate, edit both files together so the chart and image stay in sync, then re-deploy via the GitOps sync.
