# SMG (Shepherd Model Gateway) — Production Addon

SMG is the client-facing OpenAI-compatible entry point in front of the Dynamo Frontend Service. It owns *gateway concerns* — chat-template structured output (tool-calls + reasoning), retry, circuit-breaker, observability, history/audit, MCP tool registry, optional auth — and explicitly does **not** own KV-aware routing. Dynamo's KV router (consuming the SGLang ZMQ kv-events stream) remains the single source of truth for prefix placement; SGLang owns the kernel-level HiSparse / IndexCache / TurboQuant work below it. The full responsibility split with rationale is in [`docs/smg-integration.md`](../../docs/smg-integration.md).

## What's wired on by default

| Capability | Where | Knob |
|---|---|---|
| **OpenAI-compatible HTTP surface** | SMG router :30000 (Service :80) | `router.port`, `router.service` |
| **Round-robin to Dynamo Frontend** | single backend, no SMG prefix tree | `router.policy: round_robin`, `router.workerUrls` |
| **Tokenization** for token-counting + parser invocation | SMG router | `router.tokenizerPath` (HF id, downloaded at startup with `HF_TOKEN` from `smg-hf-token` secret) |
| **Reasoning parsing** (DeepSeek V3.x reasoning syntax) | SMG router | `router.reasoningParser: deepseek_v31` |
| **Tool-call parsing** (DeepSeek V3.2 tool-call format) | SMG router | `router.toolCallParser: deepseek32` |
| **MCP tool registry** | SMG router | `router.mcp.enabled: true`, `configPath: /etc/smg/mcp/tools.json` (see caveat below) |
| **Chat history / audit log** | sibling Postgres in `smg` namespace | `history.backend: postgres`, `history.postgres.url` → `smg-postgres.smg.svc.cluster.local:5432` |
| **OpenTelemetry tracing** | Otel collector in `observability` namespace | `router.tracing.enabled: true`, `otlpEndpoint: http://dynamo-collector.observability.svc:4317` |
| **Prometheus metrics** | ServiceMonitor for kube-prometheus-stack | `router.metrics.serviceMonitor.enabled: true` |
| **Grafana dashboard** | ConfigMap with sidecar discovery label | `grafana.dashboard.enabled: true` |
| **Retry + circuit breaker + health check** | per-request, against Dynamo Frontend | `router.retry`, `router.circuitBreaker`, `router.healthCheck` |
| **Structured (JSON) logging** | router stdout for fluentd/loki | `router.logging.json: true` |

### Tool-call + reasoning parsing — moved from SGLang to SMG

The DGD example previously ran `--tool-call-parser deepseekv32 --reasoning-parser deepseek-v3` on both prefill and decode workers. With SMG enabled, those flags are removed (see the comment in [`examples/deepseek-v32-reap-sglang.yaml`](../../examples/deepseek-v32-reap-sglang.yaml)) and SMG owns parsing. Two parsers attempting the same job would either double-parse or disagree — so the responsibility moves up the stack to the gateway.

Mapping between the SGLang and SMG parser names:

| SGLang flag | SMG values.yaml field |
|---|---|
| `--tool-call-parser deepseekv32` | `router.toolCallParser: deepseek32` (registered in `smg/crates/tool_parser/src/factory.rs:319` as `DeepSeekDsmlParser::v32()`) |
| `--reasoning-parser deepseek-v3` | `router.reasoningParser: deepseek_v31` (V3.2 inherits V3.1's reasoning template; SMG's V3.1 parser matches) |

### Cross-namespace secrets

The router pod consumes two cluster secrets:

- `smg-hf-token` — the HuggingFace token, mirrored from the existing `dynamo-system/hf-token-secret` by an `ExternalSecret` in [`addons/smg-secrets/`](../smg-secrets/). Required because the target model is gated and SMG downloads the tokenizer at startup.
- `smg-history-postgres` — the password for the sibling Postgres, created by [`addons/smg-postgres/postgres.yaml`](../smg-postgres/postgres.yaml). Production deployments should overwrite this Secret via an `ExternalSecret` rather than relying on the default.

## What this addon does NOT change

- The 4-GPU prefill + 4-GPU decode disaggregation in [`examples/deepseek-v32-reap-sglang.yaml`](../../examples/deepseek-v32-reap-sglang.yaml) is unchanged in shape (8 B200s split 4+4). The only edit was removing the two engine-side parser flags now owned by SMG.
- Dynamo Frontend's `--router-mode kv --router-kv-events --router-reset-states` configuration is unchanged. SMG does **not** do KV-aware routing.
- HiSparse, IndexCache, TurboQuant, and SMC-SD remain configured at the SGLang worker level. SMG never sees them.
- SMG runs CPU-only (no `nvidia.com/gpu` requests), pinned to the same node as the Dynamo Frontend so the SMG → Frontend hop stays intra-node.

## Why `policy: round_robin` and not `cache_aware`

SMG's `cache_aware` policy and Dynamo Frontend's KV router both maintain prefix trees. Putting both in the path makes them disagree on hot prefixes — Dynamo sees the *real* SGLang KV cache state via ZMQ kv-events, SMG would see only its routing-decision history. Result: more cross-worker shuffles than either alone. With `round_robin` and a single Dynamo Frontend backend, SMG is a pure pass-through for routing decisions; Dynamo is the single owner of "which worker gets this prefix".

If/when this cluster grows to multiple Dynamo Frontends in front of multiple SGLang fleets, `policy: power_of_two` across Dynamo Frontends becomes appropriate — but each Dynamo Frontend still owns prefix routing within its own fleet.

## Pulling in SMG (and other addon) updates

Renovate at [`.github/renovate.json5`](../../../../.github/renovate.json5) handles upstream sync for **every** addon under `gitops/apps/` and `gitops/optional/`, plus the SMG image tag in this `values.yaml` and the sibling Postgres image tag in `addons/smg-postgres/postgres.yaml`:

| Update type | Behaviour |
|---|---|
| Patch + minor | Renovate opens a PR and auto-merges after CI (`platformAutomerge: true`). |
| Major | Renovate opens a PR with `needs-human-review` label. SMG / Dynamo platform / kube-prometheus-stack / GPU operator all carry contract changes across majors that should be read deliberately, not auto-merged. |

To bump manually outside Renovate, edit `targetRevision` in `gitops/apps/70-smg.yaml` and the matching `tag:` in this `values.yaml` together. Both pin sites are kept consistent by Renovate.

Renovate's native `argocd` manager auto-discovers any new ArgoCD `Application` you drop into `gitops/apps/` — no config edit needed for new charts. Image tags inside hand-rolled values files do need a `customManagers` entry following the SMG / Postgres pattern in `renovate.json5`.

## Caveats

### MCP tools.json mounting

`router.mcp.enabled: true` is set, but the current SMG chart (v1.4.1) does not template `extraVolumes` / `extraVolumeMounts` on the router Deployment. That means the MCP `configPath: /etc/smg/mcp/tools.json` cannot be backed by a ConfigMap from this addon alone. Two production paths once MCP tools are needed:

1. Build a custom OCI image extending `ghcr.io/lightseekorg/smg:1.4.1` with the desired `tools.json` baked in at the configured path. Override `global.image` in this values.yaml to point at it.
2. Wait for an SMG chart release that adds `router.extraVolumes` (issue worth filing upstream); then drop a ConfigMap mount in this values.yaml.

Until one of those lands, SMG starts with the MCP subsystem enabled but no tools registered, which is harmless — `tool_calls` from the underlying model still flow through normally; only MCP-driven tool execution is unavailable.

### Auth

`auth.apiKey` is empty by default. Production deployments should set it via an `ExternalSecret` mirror similar to `addons/smg-secrets/hf-token-mirror.yaml`, then add a `valueFrom.secretKeyRef` entry to `router.extraEnv` exposing it as `SMG_API_KEY`. Left disabled here so the addon doesn't accidentally lock out the existing internal callers that hit Dynamo Frontend directly.

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

A 200 with a `chat.completion` response means the full chain (SMG → Dynamo Frontend → SGLang decode worker via Dynamo's KV router) is live. The end-to-end script [`tests/smg-roundtrip.sh`](../../../../tests/smg-roundtrip.sh) attributes failures per layer (SMG `/health`, Dynamo Frontend `/health`, then the round-trip).

To verify each opt-in capability:

```bash
# Tracing: confirm spans show up in the Otel collector
kubectl -n observability logs -l app.kubernetes.io/name=opentelemetry-collector | grep smg

# History: confirm Postgres has rows
kubectl -n smg exec smg-postgres-0 -- psql -U smg -d smg -c '\dt'

# Grafana dashboard: confirm the ConfigMap is picked up
kubectl -n monitoring get cm -l grafana_dashboard=1 | grep smg
```
