# SMG × Dynamo × HiSparse — Integration Boundary

This doc is the *single source of truth* for what each layer is responsible for in the production inference path on the B200 A4 deployment. If you're about to add prefix awareness, KV state, or model-selection logic, find the right layer below before touching code.

## The chain

```
                                                                 ┌─────────────────────┐
                                                                 │ HiSparse / IndexCache│
                                                                 │   / TurboQuant 2.5b │
                                                                 │  (kernel level)     │
                                                                 └─────────▲───────────┘
                                                                           │
client    ───►   SMG router   ───►  Dynamo Frontend  ───►  SGLang prefill  │
(OpenAI)         (CPU only)         (KV-aware route)        (4× B200)      │
                                                                          │
                                                                ┌─────────┴───────────┐
                                                                │ SGLang decode       │
                                                                │ (4× B200)           │
                                                                │ + HiSparse + SMC-SD │
                                                                └─────────────────────┘
                                              ▲
                                              │
                       Dynamo's KV router consumes ZMQ kv-events from
                       prefill and decode workers (port 5557) to know
                       *exactly* which worker has which prefix in its
                       SGLang KV cache. This is the only prefix tree
                       in the system.
```

The boundary is not arbitrary — it follows directly from the LightSeek SMG team's PyTorch blog observation: "these approaches compose. You can run SMG in front of vLLM managed by llm-d, or in front of TensorRT-LLM with Dynamo handling GPU orchestration. The boundaries are clean because the responsibilities are different."

## Responsibility split

| Layer | Owns | Does not own |
|---|---|---|
| **SMG** (`addons/smg/`) | Client-facing OpenAI-compatible HTTP surface, model selection (`/v1/models`), tokenization (for token-counting + parser invocation), tool-call + reasoning parsing (`router.toolCallParser`, `router.reasoningParser`), retry, circuit-breaker, request timeouts, server-sent observability (Prometheus + OTel traces to `dynamo-collector.observability:4317` + Grafana dashboard), chat history / audit log (sibling Postgres in `smg` namespace), MCP tool registry | Prefix routing, KV state, GPU orchestration, kernel selection |
| **Dynamo Frontend** (`examples/deepseek-v32-reap-sglang.yaml` Frontend service) | KV-aware routing across prefill/decode workers (`--router-mode kv --router-kv-events`), disaggregation orchestration (handing prefill output to decode), per-request scheduling | Client TLS termination, model catalog, history |
| **SGLang prefill / decode workers** | Kernel-level inference: forward pass, KV cache management, speculative decoding (SMC-SD), HiSparse top-k selection (decode only), IndexCache / TurboQuant for the DSA indexer and dense KV | Routing decisions across workers (Dynamo owns that), chat-template structured-output parsing (SMG owns that — the `--tool-call-parser` / `--reasoning-parser` engine flags are intentionally absent) |
| **HiSparse** (kernel level inside SGLang decode) | Top-k attention with hierarchical sparse pattern (`top_k=2048, device_buffer_size=6144, host_to_device_ratio=10`), no-radix decode | Anything above the kernel boundary |

## Why SMG is on `policy: round_robin` here

SMG's `cache_aware` policy maintains its own prefix tree from observed routing decisions. Dynamo Frontend's KV router maintains a prefix tree from *actual* SGLang KV-cache state via the ZMQ kv-events stream. With both turned on you get two trees that diverge — Dynamo sees evictions immediately; SMG sees them only when a request misses. The miss rate goes up, not down.

The composition pattern is **gateway above, router below** — not "prefix-aware everywhere". SMG `cache_aware` is the right mode when SMG is the *only* prefix-aware layer (e.g. fronting a fleet of `vLLM` workers without llm-d). When the engine layer already owns KV awareness via its own router (Dynamo, llm-d), SMG should be `round_robin` over the engine's frontends.

When the cluster grows to >1 Dynamo Frontend (multi-region, multi-fleet), upgrade SMG to `power_of_two` *across Dynamo Frontends* — each Frontend still owns prefix routing within its fleet.

## Why HiSparse is decode-only and Dynamo is unchanged

`--enable-hisparse` requires `--disable-radix-cache` (HiSparse pre-empts the radix tree's prefix slots). Dynamo's KV-aware router does not look at the radix tree — it consumes the ZMQ kv-events stream which is emitted regardless of HiSparse / radix state. So Dynamo's routing remains correct over a HiSparse-enabled decode fleet without any change. SMG sits one level above; it never sees HiSparse at all.

This is why the user's stated invariant (*"Dynamo retains KV routing using SGLang HiSparse underneath"*) holds without further integration work: nothing about adding SMG in front changes the Dynamo↔SGLang contract.

## What changed in `examples/deepseek-v32-reap-sglang.yaml`

The 4-GPU prefill + 4-GPU decode disaggregation is unchanged. The only edit is the **removal** of two engine-side parser flags that have moved to the SMG router:

```diff
-            - --tool-call-parser
-            - deepseekv32
-            - --reasoning-parser
-            - deepseek-v3
```

Removed in *both* the prefill and decode worker arg lists. SMG now owns chat-template structured output (`router.toolCallParser: deepseek32`, `router.reasoningParser: deepseek_v31` in `addons/smg/values.yaml`), and SGLang returns raw model text. Two parsers attempting the same job would either double-parse or disagree, so the responsibility moves up the stack to the gateway — matching the "gateway above, router below" pattern that makes SMG composable in the first place.

## What did *not* change

- 4-GPU prefill + 4-GPU decode shape: zero changes. SMG runs CPU-only (no `nvidia.com/gpu` requests); the disaggregated split is preserved as-is on the single B200 A4 node.
- Dynamo Frontend args (`--router-mode kv --router-kv-events --router-reset-states`): unchanged.
- HiSparse (`--enable-hisparse --hisparse-config '{"top_k":2048,...}'`), IndexCache (`--nsa-indexer-mode indexcache`), TurboQuant (`--enable-turboquant-dense-kv-cache`), SMC-SD (`--speculative-algorithm SMC`): unchanged.
- Image (`ghcr.io/ai-blaise/optimization-playground-sglang-runtime:reap-nvfp4`): unchanged.

## Verifying the chain end-to-end

After ArgoCD has synced the SMG app and the Dynamo platform + DynamoGraphDeployment are healthy:

```bash
./tests/smg-roundtrip.sh
```

The script port-forwards SMG's ClusterIP service, sends an OpenAI-compatible chat request, and asserts a 200 with a `chat.completion` response. It also checks SMG's `/health` and the Dynamo Frontend's `/health` separately to attribute failures to the right layer.

## Memory / GPU footprint (B200 A4 single node)

| Pod | CPU | RAM | GPUs | Storage |
|---|---|---|---|---|
| SMG router | 1–4 | 2–4 GiB | 0 | — |
| SMG history Postgres | 0.25–2 | 0.5–4 GiB | 0 | 20 GiB PVC |
| Dynamo Frontend | (operator default) | (operator default) | 0 | — |
| SGLang prefill | (operator default) | 120 GiB shm | 4× B200 | — |
| SGLang decode | (operator default) | 120 GiB shm | 4× B200 | — |

The 8 B200 GPUs remain split 4+4 prefill/decode. The CPU-only sidecars (SMG router + sibling Postgres) add at most a 6-CPU / 8-GiB footprint on the same node. No GPU contention with the SGLang workers.
