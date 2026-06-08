# Proposal: Disaggregated Topology Readiness

**Author:** Jie Hao
**Date:** 2026-03-24 (updated 2026-05-22)
**Status:** Approved — See tracking issue [DEP-0014 (#7787)](https://github.com/ai-dynamo/dynamo/issues/7787). PR 1, PR 2a–d landed; PR 3 in progress.

## Problem

The frontend has no awareness of disaggregated serving topology. `GET /v1/models` lists any model with at least one worker, regardless of whether its topology is complete. The `check_ready()` function gating request handlers is a no-op.

This causes decode-only workers to receive aggregated requests and crash when their required counterparts (prefill, encode) are not yet available. The `/health` endpoint is **not in scope** — it reflects frontend process readiness, not model-level serving readiness.

## Proposed Solution: Worker-Declared Dependencies

Each worker declares its **`worker_type`** and **dependencies** at registration time, derived from existing CLI flags (no new flags needed). The frontend builds a per-model dependency view and checks readiness. Readiness is surfaced through four mechanisms: `/v1/models` filtering (Mechanism 1) and per-model request gating with HTTP 503 + diagnostic body (Mechanism 2) — both in initial scope; per-namespace dispatch gating (Mechanism 3) and a `GET /v1/models/{model}/readiness` detail endpoint (Mechanism 4) — both deferred to follow-up PRs after Phase 4.

### Terminology

- **`ModelType`** — Which endpoints a model exposes (`Chat`, `Completions`, `Embedding`, `Images`, `Audios`, `Videos`, `TensorBased`). The legacy `ModelType::Prefill` bit is **removed in this proposal** (Phase 3) and replaced by the orthogonal `worker_type`. An additional `ModelType.Empty` classattr is exposed in Phase 3 for prefill / encode workers that register with no OpenAI surface.
- **`worker_type`** — Which processing stage a worker handles. A **new first-class type** on `ModelDeploymentCard`, defined in `lib/llm/src/worker_type.rs`. A plain enum with four variants:
  - `Prefill`
  - `Decode`
  - `Encode`
  - `Aggregated`

  Each worker has exactly one role; values are **not combinable**. Registration validates that `worker_type` matches one of these four variants; `None` is rejected once Phase 3's strict mode lands. Promoted from the existing informal `WORKER_TYPE_PREFILL`/`WORKER_TYPE_DECODE` string constants used for metrics in `lib/llm/src/worker_monitor.rs`.
- **`needs`** — Peer roles this worker depends on, in **disjunctive normal form**: `Vec<Vec<WorkerType>>`. The outer Vec is a disjunction (alternatives); each inner Vec is a conjunction (AND-set). A worker's dependencies are satisfied if at least one inner AND-set is fully present among ready peers in the same namespace. Encode workers express their canonical dependency as `[[Prefill, Decode], [Aggregated]]` — satisfied either by a P+D pair or by a single Aggregated peer.
- **Namespace (for readiness grouping)** — The fully-composed Dynamo namespace string, i.e. `DYN_NAMESPACE` optionally suffixed with `-{DYN_NAMESPACE_WORKER_SUFFIX}`. The suffix is composed on the Python side in `components/src/dynamo/common/utils/namespace.py::get_worker_namespace()` and exists specifically "to support multiple sets of workers for the same model" (e.g., rolling updates). By the time a worker registers, `mcid.namespace` is the already-composed string (e.g., `dynamo-v1`).
- **WorkerSet key (ws_key)** — A Rust-side storage key: **`{namespace}:{model_type}:{worker_type}`** (see `worker_set_key` in `lib/llm/src/discovery/watcher.rs`). `model_type` renders as `|`-joined lowercase tokens from `ModelType::as_vec()` (e.g. `chat|completions`, `embedding`). `worker_type` renders as the canonical lowercase name (`prefill`, `decode`, `encode`, `aggregated`). This generalizes today's `{ns}` / `{ns}:prefill` split and lets every distinct `(namespace, model_type, worker_type)` combination get its own WorkerSet bucket. Critically, encode workers (`ModelType::empty()`) no longer collide with decode in the same namespace.
- **WorkerSets are per-Model.** From `lib/llm/src/discovery/model.rs:32`, each `Model` owns its own `worker_sets: DashMap<String, Arc<WorkerSet>>`. Composite WorkerSet identity is therefore `(model_name, ws_key)` — two different models never share a WorkerSet even if everything else matches.

`ModelType` and `worker_type` are **fully orthogonal**. `ModelType` answers "what OpenAI endpoints does this worker expose?" — `worker_type` answers "what processing stage does it handle?". A prefill worker exposes no OpenAI endpoints (`ModelType.Empty`) and has `worker_type=Prefill`; a decode worker for an LLM exposes `Chat | Completions` and has `worker_type=Decode`. Aggregated workers expose the OpenAI endpoints they serve and `worker_type=Aggregated`.

### Worker registration

Workers derive `{worker_type, needs}` from existing CLI flags at startup — no new flags required. **Encode workers also register** (today they skip registration; this proposal flips that in Phase 3 — see the PR-breakdown correction below). Derivation is uniform across backends:

| Worker configuration | `worker_type` | `needs` (DNF) |
|---|---|---|
| (default, no disagg flags) | `Aggregated` | `[]` (self-sufficient) |
| `--route-to-encoder` | `Aggregated` | `[[Encode]]` |
| `--disaggregation-mode prefill` | `Prefill` | `[[Decode]]` |
| `--disaggregation-mode prefill --route-to-encoder` | `Prefill` | `[[Decode, Encode]]` |
| `--disaggregation-mode decode` | `Decode` | `[[Prefill]]` |
| `--disaggregation-mode decode --route-to-encoder` | `Decode` | `[[Prefill, Encode]]` |
| `--disaggregation-mode encode` / `--multimodal-encode-worker` | `Encode` | `[[Prefill, Decode], [Aggregated]]` |
| `--disaggregation-mode encode --route-to-encoder` | (rejected at args validation — nonsensical) | — |

General rule: `needs = [base_needs(mode) ++ (Encode if --route-to-encoder else nothing)]`. The Encode row is the only place the disjunction has > 1 alternative: an encode worker is satisfied by either a P+D pair or a single Aggregated peer.

### Readiness check (live-computed, no cache)

Readiness is a **pure function of current WorkerSet state**. There is no separate bookkeeping struct to maintain and no scale-down clearing hook. Precedent: `Model::has_prefill()` at `lib/llm/src/discovery/model.rs:104` already iterates WorkerSets live in the request hot path; readiness follows the same pattern.

Readiness is checked **per-namespace**, not per-model, because the WorkerSet architecture only pairs prefill and decode within the same namespace. A model-wide check would incorrectly report "ready" when a decode exists in `dynamo-old` and a prefill exists in `dynamo-new`. Rolling updates use different `DYN_NAMESPACE_WORKER_SUFFIX` values (yielding distinct namespaces) — readiness naturally treats each as an independent bucket.

For each model, for a given namespace:

```text
present  = empty set of WorkerType
for ws in model.worker_sets where ws.namespace == target_namespace:
    if ws.worker_count() > 0:
        present.insert(ws.worker_type)

namespace_ready = true
for ws in model.worker_sets where ws.namespace == target_namespace:
    # satisfied if at least one inner AND-set in needs is fully in present
    satisfied = ws.needs.is_empty() ||
                ws.needs.any(|and_set| and_set.all(|wt| present.contains(wt)))
    if not satisfied:
        namespace_ready = false
        break

model_ready = any namespace is ready
```

Implementation lives in `lib/llm/src/discovery/model.rs` as methods on `Model`:

- `is_workers_ready(namespace: &str) -> bool`
- `first_ready_workers() -> Option<String>` (returns the namespace name)
- `has_ready_workers() -> bool`
- `missing_worker_types(namespace) -> Vec<WorkerType>` (for the 503 diagnostic — names roles that, if added, would tip at least one alternative AND-set into the present set)

Cost: O(worker_sets in model), typically 1–3 per model. Cheaper than an existing per-registration string allocation; no caching is justified at admission time. Mechanism 3 introduces memoization on the *dispatch* path for the same predicate.

Scale-down requires no new code. When a WorkerSet's `worker_count()` drops to zero, the next readiness call simply doesn't add its `worker_type` to `present` — if a peer worker depended on that type, the namespace flips to not-ready automatically. When a worker rejoins and the count returns to > 0, readiness recomputes correctly. The existing `remove_worker_set` teardown in `watcher.rs:413` still fires for its own reasons (engine cleanup, prefill router deactivation) but is independent of readiness computation.

### Mechanism 1: `/v1/models` filtering

`Model::is_displayable()` is extended with a topology check. A model only appears in `GET /v1/models` when at least one namespace is ready. Clients polling `/v1/models` to discover available models will not see a model until it is actually routable.

```json
GET /v1/models
{
  "object": "list",
  "data": [
    {
      "id": "llama-3.1-70b",
      "object": "model",
      "created": 1711929600,
      "owned_by": "nvidia"
    }
  ]
}
```

A model with only decode workers (missing prefill) will not appear in this list.

### Mechanism 2: Per-model request gating

A new `check_topology_ready(&State, &ModelName)` function in `lib/llm/src/http/service/openai.rs` performs per-model topology checks. It is called in every inference handler **after** the existing process-level `check_ready()` (activated separately by [PR #8590](https://github.com/ai-dynamo/dynamo/pull/8590)). The two checks are deliberately separate — process readiness is stateless and model-independent; topology readiness requires the model name and consults the live WorkerSet state. When a request arrives for a model whose topology is not ready in any namespace, the frontend returns HTTP 503 with a diagnostic body naming the missing worker types:

```json
HTTP 503
{
  "message": "Model `llama-3.1-70b` is registered but no namespace has a complete worker topology. At least one prefill/decode/encode role required by a registered worker is missing. Check worker startup logs for the affected namespace.",
  "type": "Service Unavailable",
  "code": 503
}
```

This gates all request handlers: chat completions, completions, embeddings, images, audio, videos, responses (when an explicit model is set), and the per-model `GET /v1/models/{model}`. The error body is the primary diagnostic surface for operators in the initial rollout — a dedicated detail endpoint (Mechanism 4) is deferred to a follow-up PR.

### Mechanism 3: Per-namespace dispatch gating (deferred to follow-up)

Mechanisms 1 and 2 answer the *binary* question "is there at least one namespace that can fully serve this model?". Once admission is satisfied, request dispatch treats every worker registered under the model name as an interchangeable member of the same routing pool, regardless of namespace. In any deployment with **two or more WorkerSets for the same model** — rolling updates, blue/green cutovers, capacity expansion — this is unsafe.

#### Scenario

`Qwen3-32B` running in namespace `ns-old` (P+D fully ready). Operator applies a new revision; `ns-new` starts. Decode pods in `ns-new` become Ready in ~30 s; prefill pods take ~2 min. During the ~90-second window:

| Namespace | Decode | Prefill | Topology |
|---|---|---|---|
| `ns-old` | ✅ ready | ✅ ready | complete |
| `ns-new` | ✅ ready | ❌ not yet | **partial** |

`check_topology_ready` admits the request (`ns-old` is complete). Dispatch picks among **all** registered decode instances, including `ns-new.backend.generate`. A share of traffic lands on the partial namespace's decode worker, which has no namespace-local prefill peer. Depending on backend and `kv_role` configuration this either silently falls back to local prefill (wrong topology executed), hangs waiting for a KV transfer that never arrives, or 500s. Blue/green cutovers exhibit the same shape.

#### Proposed solution

**Filter dispatch candidates by per-namespace topology.** Before `PushRouter` / `KvPushRouter` selects an instance, restrict the candidate set to instances whose namespace is fully ready. Partial namespaces remain visible in discovery (the MDC stays in etcd; `missing_worker_types` still computes for operator diagnostics) but contribute no instances to dispatch.

Behaviour after the fix, replaying the scenario:

- All traffic routes to `ns-old`.
- `ns-new` decode receives zero traffic until `ns-new` prefill registers.
- The instant `ns-new` becomes complete, KV-affinity can place follow-up requests on `ns-new` for warm-cache reasons.

#### Implementation: materialized ready-pool view in `ModelWatcher`

A pre-filtered "ready-instance pool" is materialized in the watcher's data flow and published to routers. Routers receive only ready-namespace instances and apply no filter of their own. The pool is recomputed on registration / instance-count-change events; dispatch is O(1) per request on a snapshot read.

This is the right shape for steady-state production load (thousands of req/s with churn every few minutes): the filter is computed once per registration event, not once per request. A lazy variant ("memoize the predicate, build the filtered pool at dispatch time") was considered and rejected because it does O(candidates) work *per request* even when no state has changed since the last request — at high QPS that's millions of redundant filter operations per second.

Touches:

- `lib/llm/src/discovery/watcher.rs` — extend the existing instance-publication pipeline with the per-namespace readiness filter
- `lib/llm/src/discovery/model.rs` — the existing `is_workers_ready` predicate becomes the filter input
- `lib/llm/src/discovery/model_manager.rs` — pool-storage seams (`add_*_model` helpers)
- `lib/kv-router/src/` — KV-affinity scoring reads the filtered candidate set
- `lib/runtime/src/pipeline/network/egress/push_router.rs` — `PushRouter` consumes a pre-filtered view

The filter applies at every dispatch site:

- **Decode router** — filter candidate decode instances by namespace readiness.
- **Prefill router** invoked from a decode worker (KV-transfer target lookup) — decode in `ns-X` should call prefill in `ns-X`; cross-namespace fallback is rejected at the filter.
- **KV-affinity** — candidates feed scoring after the filter, not before.
- **Request migration** (`migration_limit`) — migration target picker must apply the same filter.

No protocol changes. No MDC schema changes. No CLI flag changes.

#### Alternatives considered (Mechanism 3)

- **Lazy filter at dispatch with predicate memoization.** Per dispatch, iterate the candidate set and probe a cached `is_workers_ready` predicate. Cost is amortized-O(1) per *probe* but O(candidates) per *dispatch* — at high QPS this is per-request redundant work between churn events. The materialized-pool approach above is strictly less work in steady state.
- **Hide partial namespaces from etcd discovery entirely.** Workers unregister themselves when their namespace becomes partial. Rejected: a worker has no way to observe its peers' liveness without polling etcd or receiving notifications, and self-unregistration races against the worker that just died. Frontend-side filtering is strictly more reliable.
- **Per-namespace `model_name` aliasing** (`Qwen3-32B@ns-old`, `Qwen3-32B@ns-new`). Rejected: violates the OpenAI-compatible API contract.
- **Trust `--enforce-disagg=true` to fail-closed.** `--enforce-disagg` is a request-time routing policy, not a readiness input. It can't distinguish "no prefill exists anywhere" from "no prefill exists in this request's chosen namespace".

#### Open questions (Mechanism 3)

1. **Last-namespace-partial behavior.** If the last ready namespace becomes partial mid-request, the dispatch pool empties. Need to confirm the failure surface is a clean 503, not a stuck request waiting for instances.
2. **KV-affinity cold-cache when readiness flips.** A request whose previous turn was served by `ns-old` carries KV blocks in `ns-old`. If `ns-old` becomes partial and `ns-new` is now the only ready namespace, KV affinity is moot and the request takes a cold-cache penalty. Correct trade-off, should be explicit in operator docs.

### Mechanism 4: Readiness detail endpoint (deferred to follow-up)

A `GET /v1/models/{model}/readiness` endpoint exposes per-namespace topology detail for debugging, monitoring, and operator tooling. It supplements the 503 error body from Mechanism 2 (which is concise by design) with structured, queryable detail across all namespaces — including ones that are *not* the namespace dispatch would have admitted.

```json
GET /v1/models/llama-3.1-70b/readiness

{
  "model": "llama-3.1-70b",
  "ready": false,
  "reason": "no namespace has complete topology",
  "namespaces": {
    "ns-old": {
      "ready": true,
      "roles": {
        "decode":  {"workers": 2, "needs": [["prefill"]]},
        "prefill": {"workers": 1, "needs": [["decode"]]}
      },
      "present": ["decode", "prefill"],
      "missing_worker_types": []
    },
    "ns-new": {
      "ready": false,
      "reason": "missing worker types: prefill",
      "roles": {
        "decode": {"workers": 2, "needs": [["prefill"]]}
      },
      "present": ["decode"],
      "missing_worker_types": ["prefill"]
    }
  }
}
```

The endpoint is read-only and unauthenticated like the rest of the `/v1/*` surface. The response shape preserves the DNF form of `needs` (a list of inner AND-sets) so external tooling can reason about the satisfaction logic without re-deriving it.

#### Why deferred

The 503 body from Mechanism 2 already names the missing role(s) on the affected namespace, which is enough for the common "why is my request failing?" operator question. The detail endpoint is valuable for:

- **Dashboards** — Grafana / operator UIs that want a periodic snapshot of topology readiness across all models without round-tripping through 503 responses.
- **Cross-namespace visibility** — diagnosing rolling-update scenarios where one namespace is ready and another is partial; the 503 only fires when *every* namespace is partial.
- **CI / integration tests** — a structured assertion target for "expect this DGD to reach readiness within N seconds."

None of these are blockers for the initial Phase 3 rollout. The endpoint can be added without protocol churn once Mechanism 3 lands, because both mechanisms share the same per-namespace readiness computation in `Model::is_workers_ready`.

#### Scope

Land in a separate observability PR after PR 3 and Mechanism 3. Touches:

- `lib/llm/src/http/service/service_v2.rs` — register the new route.
- `lib/llm/src/http/service/openai.rs` — handler that calls `Model::namespace_topology()` (new method on `Model` returning the full structured view) and serializes it.
- `lib/llm/src/discovery/model.rs` — `namespace_topology()` accessor wrapping the data the existing `is_workers_ready` / `missing_worker_types` methods already compute internally.

No MDC schema changes. No CLI flag changes. No client-protocol changes beyond the new GET path.

#### Open questions (Mechanism 4)

1. **Namespace ID visibility.** Internal namespace strings (e.g. `dyn-{hash}-{revision}`) may be noisy for operators. Whether to expose them as-is, alias them, or expose only operator-meaningful labels needs an operator-side review.
2. **Caching policy.** The computation is cheap (O(workersets in model)) but called from external dashboards at potentially high frequency. Whether to add HTTP-level caching headers or rate-limit needs a sizing decision.
3. **Authentication.** Inherits the existing `/v1/*` auth posture today (none). If the broader frontend auth story changes, this endpoint follows it.

### Relationship to `--enforce-disagg`

`--enforce-disagg` is a **frontend runtime routing policy** (`components/src/dynamo/frontend/frontend_args.py`, plumbed via `RouterConfig.enforce_disagg` in `lib/llm/src/discovery/watcher.rs`). It controls whether the frontend is willing to route to a decode-only WorkerSet when prefill is unavailable at request time. It is **not** a readiness input:

- Decode workers across all backends register `needs = [[Prefill]]` (or `[[Prefill, Encode]]` with `--route-to-encoder`). No per-backend branching.
- For trtllm and sglang, the decode path has no aggregated fallback (`handler_base.py:1001` raises `ValueError("Disaggregated params are required for decode mode")` in trtllm; sglang behaves equivalently). `--enforce-disagg=false` with these backends is de facto misconfiguration — fallback only exists in vLLM, and is being deprecated there.
- Readiness does not attempt to validate backend-vs-flag compatibility. Once Phase 4 enforcement is on, a model is hidden / 503'd until prefill exists, so the "decode receives aggregated traffic" window narrows to the transient case where prefill registered and then died — the only scenario where `--enforce-disagg` still meaningfully differs in behavior (and only for vLLM).

This keeps the readiness computation a pure function of registered `{worker_type, needs}`, independent of frontend policy.

## Design Principles

1. **One model = one topology per namespace.** Operators should deploy either aggregated-style or disaggregated-style workers for a given `(model, namespace)`, not both. The system does not enforce this at registration time (the new ws_key `{ns}:{mt}:{wt}` would technically permit mixed WorkerSets to coexist), but request routing behavior under mixed deployments is undefined. This is an operator-facing design intent.
2. **All workers of the same `worker_type` share the same config.** Checksum validation enforces this.
3. **Workers know their dependencies at startup** from existing CLI flags.
4. **`Aggregated` has no P/D dependency but may depend on `Encode`** when `--route-to-encoder` is set. Encode is optional at deployment time, required at runtime if configured.
5. **`Aggregated` is a distinct enum variant.** It is not a bitflag alias for `Prefill | Decode`. The "E-PD" deployment (aggregated PD worker + encode worker) is expressed in the DNF form of Encode's `needs`: `[[Prefill, Decode], [Aggregated]]` — satisfied by either alternative.
6. **`ModelType` ⊥ `worker_type`.** Endpoints exposed (ModelType) and processing stage (worker_type) are independent dimensions. Prefill / encode workers register with `ModelType.Empty`.
7. **Readiness is namespace-scoped.** The WorkerSet architecture only pairs prefill and decode within the same namespace (via `oneshot`-based prefill router activation). Readiness must reflect actual routability, not just `worker_type` presence across the model.
8. **Rolling updates are namespace-scoped.** New versions take a different `DYN_NAMESPACE_WORKER_SUFFIX`, which yields a distinct namespace and an independent readiness computation. Cross-namespace routing of dispatch traffic during partial readiness is addressed by Mechanism 3 (deferred follow-up).
9. **Readiness is live-computed.** Answered from current WorkerSet state on every call; no cache, no scale-down clearing hook. Dispatch-time memoization is introduced in Mechanism 3.

## Related Work

No other OpenAI-compatible server implements equivalent topology-aware readiness at the API layer:
- **vLLM** runs prefill and decode as independent servers, coordinated by an example proxy script that polls `/v1/models` on each until both respond.
- **SGLang** uses separate servers coordinated over gRPC; no topology-aware readiness at the API layer.
- **GAIE** has no equivalent.

The problem is Dynamo-specific because Dynamo is the gateway that owns the multi-stage routing decision. Hence the solution lives in the Dynamo frontend.

## Changes Required

This section describes **end-state changes** — the state of the codebase after all four phases have landed. See [Implementation Phases](#implementation-phases) for the order in which these land. Mechanism 3 and Mechanism 4 are deferred follow-ups and **not** covered by the four-phase plan; they are scoped separately above.

**Type changes:**
- `lib/llm/src/worker_type.rs` (new) — Define `WorkerType` as a plain enum (`Prefill`, `Decode`, `Encode`, `Aggregated`). Display/FromStr canonicalize the lowercase names. Promotes the informal `WORKER_TYPE_PREFILL`/`WORKER_TYPE_DECODE` string constants in `lib/llm/src/worker_monitor.rs`.
- `lib/llm/src/model_card.rs` — Add `worker_type: Option<WorkerType>` and `needs: Vec<Vec<WorkerType>>` fields to `ModelDeploymentCard` (both `#[serde(default)]` initially; the `Option` is removed at the Python-binding boundary in Phase 3 when strict mode lands).
- `lib/llm/src/model_type.rs` — **Remove `ModelType::Prefill`** variant (and `supports_prefill()`, `as_vec()`/`units()` branches, `ALL_MODEL_TYPES` entry, `is_model_type_list_empty` branch). The `parse_endpoint_types` parser no longer accepts `"prefill"` as a token.
- `lib/bindings/python/rust/lib.rs` — Add `WorkerType` pyclass mirroring the Rust enum, and `#[classattr] const Empty: ModelType` for prefill/encode registrations that have no OpenAI surface.
- `components/src/dynamo/common/constants.py` — Ensure `ENCODE` exists on `DisaggregationMode` (already present).

**Worker registration (Python):**
- `components/src/dynamo/vllm/args.py`, `worker_factory.py`, `main.py` — Derive `worker_type`/`needs` from `disaggregation_mode` + `route_to_encoder`. Make `_create_multimodal_encode_worker` register via `register_vllm_model` (now landing in Phase 3 — see PR-breakdown correction below). Delete the `ModelType.Prefill` registration site at `worker_factory.py:596-602` and replace `ModelType.Prefill` with `ModelType.Empty` on prefill/encode call sites. Rewrite the `model_type != ModelType.Prefill` branch at `main.py:652` to test `worker_type != WorkerType.Prefill`. The omni paths (`vllm/omni/main.py`, `omni/stage_router.py`) register as `Aggregated, needs=[]`.
- `components/src/dynamo/trtllm/args.py`, `workers/llm_worker.py` — Same derivation. Lift the `if disaggregation_mode != DisaggregationMode.ENCODE` guard at `llm_worker.py:619` so encode workers also register (lands in Phase 3). The trtllm diffusion workers (`workers/image_diffusion_worker.py`, `workers/video_diffusion_worker.py`) register as `Aggregated, needs=[]`. **Unify trtllm's local `DisaggregationMode` enum onto `components/src/dynamo/common/constants.py`** (trtllm currently uses `"prefill_and_decode"` for aggregated vs common's `"aggregated"`); landed as a follow-up if the CLI-compat risk is acceptable.
- `components/src/dynamo/sglang/args.py`, `init_multimodal.py`, `init_llm.py` — Same derivation. Make `init_multimodal_encode_worker` register (Phase 3). Replace `output_type=ModelType.Prefill` at `init_llm.py:279` with `output_type=ModelType.Empty`. The sglang embedding (`init_embedding.py`), DLLM and diffusion paths (`init_diffusion.py`, plus the `register_image_diffusion_model` / `register_video_generation_model` helpers in `register.py`) register as `Aggregated, needs=[]`. The LoRA registration path in `request_handlers/handler_base.py` picks `(model_type, worker_type, needs)` per serving mode the same way the base-model registration does.
- `components/src/dynamo/global_router/__main__.py:105` — Replace `model_type=ModelType.Prefill` with `model_type=ModelType.Empty, worker_type=WorkerType.Prefill, needs=[[WorkerType.Decode]]`.
- Component-name divergence (sglang uses `encoder` vs vLLM `encode`) is **out of scope**; readiness is driven by `worker_type = Encode` on the card, not by the routing component string.

**Rust registration & discovery:**
- `lib/llm/src/discovery/watcher.rs` — **Rewrite** `worker_set_key`: signature becomes `(namespace: &str, model_type: ModelType, worker_type: WorkerType) -> String`, body produces `{namespace}:{model_type}:{worker_type}`. All call sites pass the new arguments. The `do_worker_set_registration` dispatch short-circuits on `worker_type == Prefill / Encode` before the model_type-based pipeline branches, so prefill and encode workers register without an engine.
- `lib/llm/src/discovery/watcher.rs:101, 120, 428, 997` + `:1154-1196` tests + `lib/llm/src/entrypoint/input/endpoint.rs:74` — **Flip** `supports_prefill()` / `ModelType::Prefill` checks to `worker_type == Some(WorkerType::Prefill)`.
- `lib/llm/src/discovery/model.rs` — Add live-compute readiness methods: `is_workers_ready(&str)`, `first_ready_workers()`, `has_ready_workers()`, `missing_worker_types(&str)`. Each iterates `self.worker_sets` filtered by namespace, builds the `present` set from `worker_type` values where `worker_count() > 0`, and checks DNF satisfaction. Extend `is_displayable()` to require at least one ready namespace. Rename/refactor `list_prefill_models()` (new name TBD in implementation).
- `lib/llm/src/discovery/model_manager.rs` — `add_*_model` helpers stamp `WorkerType::Aggregated` (or `Prefill + needs=[[Decode]]` for `add_prefill_model`) on the synthetic in-process MDC. Expose readiness summary to `check_topology_ready`.
- `lib/bindings/python/src/dynamo/_core.pyi` — Update `register_model()` signature; remove `ModelType.Prefill` export; add `ModelType.Empty` and `WorkerType`.
- `lib/llm/src/worker_monitor.rs` — Replace string constants with `WorkerType` values.
- `lib/bindings/c/src/lib.rs` — PrefillRouter discovery and decode-worker filtering switch to `card.worker_type == Some(WorkerType::Prefill)`.

**Frontend request gating & readiness:**
- `lib/llm/src/http/service/openai.rs` — Add a new `check_topology_ready(&State, &ModelName) -> Result<(), ErrorResponse>` that calls `Model::has_ready_workers()`. On failure, the 503 error body identifies the model and the missing-role condition (`ErrorMessage::service_unavailable_with_body`). Call it in every inference handler (chat, completions, embeddings, images, audio, videos, responses, Anthropic, per-model GET) immediately after the existing `check_ready()` guard (which [PR #8590](https://github.com/ai-dynamo/dynamo/pull/8590) activates for process readiness). Also extend `list_models_openai` to filter by readiness (unconditional).
- **Dependency:** this work assumes [PR #8590](https://github.com/ai-dynamo/dynamo/pull/8590) has landed; `check_topology_ready()` **composes with** its `check_ready()`, does not replace it.
- **Not in initial scope:** the `/v1/models/{model}/readiness` detail route (Mechanism 4) and per-namespace dispatch filtering (Mechanism 3) — both deferred to follow-up PRs.

## Implementation Phases

**This section is the semantic milestone view for design review.** Each phase is a coherent chunk of change that reviewers can reason about independently. The actual PR breakdown is in the next section ([PR Breakdown](#pr-breakdown)) — phases and PRs do not map 1:1.

### Phase 1 — Type plumbing with temporary compat shim

Introduces the `WorkerType` enum and new `ModelDeploymentCard` fields. A temporary compat shim in `Model::ws_role_and_needs` reads missing `worker_type` as `Aggregated` with no needs — this exists **only** to let the PR 1 → PR 2 rollout happen without a fleet-wide restart. Nothing observes the new fields yet.

**Design target: every worker must register an explicit `worker_type` and `needs`. No implicit defaults.** The shim is scaffolding, not contract — it comes out in Phase 3 when readiness goes live.

### Phase 2 — Worker-side derivation (prefill / decode / aggregated paths)

All backends start populating `worker_type`/`needs` on their cards for prefill, decode, and aggregated paths, derived uniformly from existing CLI flags. trtllm's local `DisaggregationMode` merges into the common enum (or defers if CLI risk too high). Prefill workers continue to register with `ModelType.Prefill` to keep existing Rust consumers working — the dual-track period. Encode-worker registration is **not** in this phase; it lands in Phase 3 alongside the strict-mode flip. Cards in etcd now carry correct topology metadata for the LLM path, but the frontend still ignores it for routing decisions.

### Phase 3 — Discovery switch-over, encode registration, strictness, `ModelType::Prefill` removal

The frontend stops consulting `ModelType::Prefill` and starts consulting `worker_type`. `worker_set_key` migrates to the new `{ns}:{mt}:{wt}` format. Live-compute readiness methods are added to `Model`. **The Phase 1 compat shim (empty `worker_type` → `Aggregated`) is removed**, enforcing the design target: empty `worker_type` now means "not ready." `register_model` becomes strict at the Python-binding boundary and rejects `None` `worker_type` with a `ValueError`. `ModelType.Empty` is added to the Python binding for prefill / encode registration.

Encode workers across all three backends start registering with `worker_type = Encode` and `needs = [[Prefill, Decode], [Aggregated]]`. The `if disaggregation_mode != ENCODE` guard in trtllm and the corresponding "internal-only" treatment in vLLM / sglang come out. Encode workers register with `ModelType.Empty` so the frontend doesn't try to build an OpenAI pipeline over them.

In-process engine paths (`entrypoint/input/endpoint.rs`) and any remaining test fixtures that built cards with `ModelDeploymentCard::default()` are updated to pass explicit values. Finally, `ModelType::Prefill` is removed from the Rust enum and the 4 Python call sites. End state: `worker_type` is the source of truth everywhere, strictly; the readiness predicate is wired into both admission (`check_topology_ready`) and listing (`list_models_openai`) — Phase 4 enforcement is folded into Phase 3 (see below).

### Phase 4 — Enforcement (Mechanisms 1 + 2), folded into Phase 3

All of Phase 4 ships as part of PR 3:

- HTTP 503 gating via `check_topology_ready()` is unconditional.
- `list_models_openai` filtering by readiness is unconditional.

### Addressing the cross-namespace dispatch gap (Mechanism 3 + 4)

After Phase 4 ships, the residual gap is Mechanism 3: dispatch still treats every worker under a model name as one routing pool, regardless of per-namespace topology, so rolling-update / blue-green / capacity-expansion deployments can route to a partial namespace. The plan is three sequential pieces of work, each shipping value standalone:

#### Step 1 — Ship PR 3 as-is

Mechanisms 1 + 2 via the `Model::has_ready_workers()` predicate, unconditional. Closes the original DEP scope: decode-only crashes fixed today. The dispatch gap (Mechanism 3) remains open but is benign for the single-WorkerSet deployments that are the common case at this stage.

#### Step 2 — Pool + dispatch + rewire (one bundled PR)

A single PR that:

1. **Materializes the ready-instance pool in `ModelWatcher`.** A pre-filtered per-namespace view, recomputed on registration / instance-count-change events. The current `Model::is_workers_ready` predicate becomes the seed of the pool, not the production query path.
2. **Wires `PushRouter` / `KvPushRouter` / KV-affinity / request migration to consume the pool.** Fixes the cross-namespace partial-readiness dispatch gap (Mechanism 3): incomplete namespaces no longer receive dispatched traffic.
3. **Rewires Mechanisms 1 and 2 to read from the same pool.** `list_models_openai` and `check_topology_ready` both query "does the pool have any instance for this model?" instead of the standalone predicate. `Model::has_ready_workers()` either becomes a thin facade over the pool query or is deleted with call sites updated directly. Single source of truth: a model is "ready" iff there is at least one entry in its pool.

Sub-changes (1)–(3) share infrastructure and ship together. Splitting them would leave the codebase in a worse intermediate state: a pool that nothing useful consumes (after only (1)), or two parallel computations of "ready" that have to be kept in sync (after only (1)+(2)).

Why this is the right shape: the pool replaces a per-request O(candidates) filter pass with a per-event O(candidates) recomputation. At production QPS with churn every few minutes, this is millions of redundant filter operations per second avoided. The unification with Mechanisms 1+2 means no parallel readiness state to drift.

#### Step 3 — `/v1/models/{model}/readiness` endpoint (Mechanism 4)

Sits on top of Step 2. The endpoint reads the same per-namespace pool view Step 2 established, plus the underlying cards for the `missing_worker_types` diagnostic. Operator-facing structured topology detail for dashboards, cross-namespace visibility, and CI assertions. Scope is one new route, one accessor on `Model`, and no protocol changes.

## PR Breakdown

**This section is the execution plan.** Phase 2 is split into a shared plumbing PR + one per backend; Phase 3 is the largest single PR and folds in the encode-worker registration (originally drafted in Phase 2 — see correction note in PR 2). All should land in Release X; within-phase PRs can be reviewed and merged independently (see "PR dependencies" below).

The pattern is expand-and-contract: (1) add net-new, (2) dual-track old+new for prefill/decode/aggregated, (3) switch reads to new and register encode, (4) enforce.

### PR 1 — Net-new types and method stubs

- `lib/llm/src/worker_type.rs` (new) — `WorkerType` plain enum with four variants: `Prefill`, `Decode`, `Encode`, `Aggregated`. Display/FromStr canonicalize lowercase. Custom serde (canonical lowercase strings for human-readable formats, raw enum-int for binary).
- `lib/llm/src/model_card.rs` — `worker_type: Option<WorkerType>` and `needs: Vec<Vec<WorkerType>>` fields with `#[serde(default)]`.
- `lib/llm/src/discovery/model.rs` — Add the live-compute readiness methods (`is_workers_ready`, `first_ready_workers`, `has_ready_workers`, `missing_worker_types`). Includes a **temporary compat shim** in `ws_role_and_needs` that reads empty `worker_type` as `Aggregated` with no needs — removed in PR 3.
- Tests: serde round-trip + strict wire-format assertions, Display/FromStr, and readiness-math unit tests covering E-P-D, E-PD, scale-down, and cross-namespace isolation. **Note:** Phase 1 agent draft used a `WorkerType` bitflag with `Aggregated = Prefill | Decode` alias; the final design is a plain enum + DNF `needs`. PR 1 lands the plain-enum version.

**Observable effect:** none. No new HTTP route, no wiring changes. Compat shim lets pre-PR-2 workers continue to behave as before; enforced strictness arrives in PR 3.

### PR 2 — Dual-track parallel path (split into one platform PR + three backend PRs)

Split across four PRs so each backend owner reviews their own surface. **PR 2a is a hard prerequisite** for the backend PRs (they need the bindings + signature changes 2a introduces). **PR 2b / 2c / 2d are independent** of each other and can land in any order — each backend's cards are read independently by the frontend, and compat-shim handling in the Model layer covers any backend that hasn't shipped yet.

**No shared derivation helper.** Each backend constructs `worker_type` and `needs` literally at its `register_model` call sites. The `(disaggregation_mode, route_to_encoder) → (worker_type, needs)` table is duplicated per backend rather than centralized — this aligns with the design target ("every worker registers explicit `worker_type` / `needs`") and keeps the topology decision visible at the call site.

**Correction (2026-05-22) — encode-worker registration moved to PR 3.** Earlier drafts scoped encode-worker registration to PR 2b/2c/2d alongside the prefill/decode/aggregated changes. During PR 2 execution this was deferred: PR 2b/2c/2d cover only prefill/decode/aggregated (and embedding / omni / diffusion registrations that already existed but lacked `worker_type`). Encode-worker registration lands in PR 3 alongside the `worker_set_key` migration that gives encode its own bucket via `worker_type` rather than colliding with decode. Without that migration, encode workers would have collided with decode in the same namespace.

#### PR 2a — Bindings plumbing + checksum fix

- `lib/bindings/python/rust/lib.rs` — Add `WorkerType` pyclass (regular pyo3 enum with the four variants `Prefill` / `Decode` / `Encode` / `Aggregated`, `__str__`, `__eq__`). Extend `register_model` with optional `worker_type: Option<WorkerType>` / `needs: Option<Vec<Vec<WorkerType>>>` params; validate non-`None` values are well-formed on the card in both the TensorBased fast path and the LLM `LocalModel::attach` path.
- `lib/bindings/python/src/dynamo/_core.pyi` — Mirror the new class and the extended `register_model` signature.
- `lib/llm/src/local_model.rs` — Extend `LocalModel::attach` signature with `worker_type: Option<WorkerType>` / `needs: Vec<Vec<WorkerType>>` arguments; existing in-process callers pass `None` / empty `Vec` until PR 3 updates them.
- `lib/llm/src/model_card.rs` — **Add `worker_type` and `needs` to `mdcsum()`.** Prerequisite for the backend PRs: once any backend starts setting these fields, the checksum must reflect them so a rolling update that changes only topology metadata forces drain-and-redeploy rather than silently joining a stale WorkerSet. Landing this in 2a (before any backend populates the fields) means the checksum change is a no-op for existing deployments — all cards still hash with `None` / empty.
- Tests: binding smoke tests covering the four enum variants, equality, and the `register_model` round-trip; mdcsum coverage of the new fields.

**Observable effect:** none. No backend uses the new API yet.

#### PR 2b — vLLM (prefill / decode / aggregated / embedding / omni)

- `components/src/dynamo/vllm/worker_factory.py` — In each `register_vllm_model` call site, pass `worker_type` and `needs` literally based on the worker's role. Decode: `worker_type=Decode, needs=[[Prefill]]` (append `Encode` when `--route-to-encoder` is set). Prefill: `worker_type=Prefill, needs=[[Decode]]` (append `Encode` similarly). Aggregated: `worker_type=Aggregated, needs=[]`. Embedding worker: `worker_type=Aggregated, needs=[]`.
- `components/src/dynamo/vllm/omni/main.py` and `omni/stage_router.py` — `worker_type=Aggregated, needs=[]` (omni workers serve the full pipeline behind one endpoint; no prefill/decode split visible to the frontend).
- `_create_multimodal_encode_worker` left untouched in 2b — encode-worker registration moves to PR 3.
- Dual-track: keep the existing `ModelType.Prefill` registration on prefill workers so Rust-side consumers (pre-PR-3) keep working.
- Tests: per-call-site assertions on the literal `worker_type` / `needs` passed in each branch (`--disaggregation-mode` and `--route-to-encoder` combinations).

#### PR 2c — trtllm (prefill / decode / aggregated / diffusion)

- `components/src/dynamo/trtllm/workers/llm_worker.py` — Same pattern: literal `worker_type` and `needs` at each `register_model` call site, no helper. The `if config.disaggregation_mode != DisaggregationMode.ENCODE` guard at line 619 stays in 2c — lifting it lands in PR 3.
- `components/src/dynamo/trtllm/workers/image_diffusion_worker.py` and `video_diffusion_worker.py` — `worker_type=Aggregated, needs=[]` (diffusion has no prefill/decode split).
- **trtllm `DisaggregationMode` unification** onto `components/src/dynamo/common/constants.py` if still in scope (trtllm currently uses `"prefill_and_decode"` for aggregated vs common's `"aggregated"`). If the CLI-compat risk is deemed too large for this DEP, defer and document as a separate follow-up.
- Tests: per-call-site assertions on the literal values passed.

#### PR 2d — sglang (prefill / decode / aggregated / embedding / diffusion / LoRA)

- `components/src/dynamo/sglang/init_llm.py` — Same pattern: literal `worker_type` and `needs` at each call site.
- `components/src/dynamo/sglang/init_embedding.py` — `worker_type=Aggregated, needs=[]`.
- `components/src/dynamo/sglang/init_diffusion.py` — Same. The `register_image_diffusion_model` and `register_video_generation_model` helpers in `register.py` hard-code `worker_type=Aggregated, needs=[]` internally (diffusion is always Aggregated, no kwarg threading needed at the helper boundary).
- `components/src/dynamo/sglang/request_handlers/handler_base.py` — LoRA registration in `LoraMixin.load_lora` picks `(model_type, worker_type, needs)` per serving mode the same way the base-model registration does.
- `init_multimodal.py:init_multimodal_encode_worker` is left untouched in 2d — encode-worker registration moves to PR 3.
- Tests: per-call-site assertions on the literal values passed; LoRA tests assert the gate pins `(PREFILL → (Empty ModelType, Prefill worker_type))`.

**Observable effect of PRs 2b/2c/2d individually:** cards from that backend start carrying `worker_type`/`needs` for prefill/decode/aggregated paths. Other backends keep registering with empty values — the compat shim in `Model::ws_role_and_needs` handles them as Aggregated. Mixed-backend deployments work unchanged until PR 3's strictness lands.

### PR 3 — Switch frontend reads to `worker_type`, encode registration, strictness, `ModelType::Prefill` removal

- `lib/llm/src/discovery/watcher.rs` — Rewrite `worker_set_key` to `{ns}:{mt}:{wt}`. All call sites updated. Dispatch in `do_worker_set_registration` short-circuits on `worker_type == Prefill / Encode` before the model_type-based pipeline branches.
- `lib/llm/src/discovery/watcher.rs:101, 120, 428, 997` + `:1154-1196` tests, plus `endpoint.rs:74` — Flip `supports_prefill()` / `ModelType::Prefill` checks to `worker_type == Some(WorkerType::Prefill)`.
- `lib/llm/src/discovery/model.rs` — **Remove the compat shim** (empty `worker_type` no longer silently maps to `Aggregated`). Drop the `readiness_missing_worker_type_field_treated_as_aggregated` test. `is_workers_ready` etc. treat a card with no declared role as misconfigured and refuse to vouch for the namespace.
- `lib/bindings/python/rust/lib.rs` — Strengthen `register_model` validation: reject `None` `worker_type` with a `ValueError` (Phase-3 strict mode). Prefill/Encode workers must use `ModelInput::Tokens`. Add `#[classattr] const Empty: ModelType` so Python prefill/encode call sites can express "no OpenAI surface" without conflating with `ModelType.Prefill`.
- **Encode-worker registration across backends:**
  - vLLM `_create_multimodal_encode_worker` — start calling `register_model` with `worker_type=Encode, model_type=ModelType.Empty, needs=[[Prefill, Decode], [Aggregated]]`.
  - trtllm `init_llm_worker` — lift the `if config.disaggregation_mode != DisaggregationMode.ENCODE` guard; encode workers register with the same `(worker_type, needs)` shape.
  - sglang `init_multimodal_encode_worker` — same.
  - Decode / aggregated paths in all three backends re-add `Encode` to `needs` when `--route-to-encoder` is set (Phase 2 deferred this for vLLM; restored in PR 3).
- **`ModelType.Prefill` removal:**
  - `lib/llm/src/model_type.rs` — remove the `Prefill` bit; `parse_endpoint_types` no longer accepts `"prefill"`.
  - All Python call sites (`vllm/worker_factory.py`, `vllm/main.py`, `sglang/init_llm.py`, `sglang/init_multimodal.py`, `sglang/request_handlers/handler_base.py` LoRA, `global_router/__main__.py`) switch from `ModelType.Prefill` → `ModelType.Empty`.
- `lib/llm/src/entrypoint/input/endpoint.rs` — Update the two in-process `attach()` call sites to pass real values (`Aggregated` for chat / Text engines, `Prefill+needs=[[Decode]]` for the prefill variant of InProcessTokens, `Aggregated` for the regular variant).
- `lib/llm/src/discovery/model_manager.rs` — `add_*_model` helpers stamp `Aggregated` on the synthetic in-process MDC; `add_prefill_model` stamps `Prefill + needs=[[Decode]]`.
- `lib/llm/tests/http_metrics.rs` + any other test fixtures using `ModelDeploymentCard::default()` — set explicit `worker_type`/`needs`.
- `lib/llm/src/http/service/openai.rs` — Add new `check_topology_ready(&State, &ModelName)` function wired to `Model::has_ready_workers()`. Call it in each inference handler (chat, completions, embeddings, images, audio, videos, responses, Anthropic, per-model GET) after the existing `check_ready()` guard from [PR #8590](https://github.com/ai-dynamo/dynamo/pull/8590). Extend `list_models_openai` to filter by readiness (unconditional).
- `lib/bindings/c/src/lib.rs` — PrefillRouter discovery and decode-worker filtering switch to `card.worker_type == Some(WorkerType::Prefill)`.
- **Deprecate** any remaining `ModelType::Prefill` references with `#[deprecated(note = "Use WorkerType::Prefill; removed in PR 4")]`.
- Tests: readiness math for E-P-D, E-PD, P-D, aggregated-alone, encode-alone, cross-namespace isolation, scale-down; plus tests that `None` `worker_type` rejection is enforced at `register_model`.

**Observable effect:** HTTP 503 gating and `/v1/models` filtering are both live unconditionally. `ModelType::Prefill` is vestigial — still on cards and in the enum if any consumer is still around, but no Rust consumer reads it. Compile-time deprecation warnings catch any accidental new callers.

### PR 4 — Remove `ModelType::Prefill` (final cleanup)

> **Deferred (revised):** PR 3 now **retains** `ModelType::Prefill` as a cross-version
> compatibility marker (dual-emitted by new prefill workers so an old frontend still
> detects them; see PR 5). Its removal therefore moves to **PR 5**. The non-`Prefill`
> cleanups below (worker_monitor string constants, `list_prefill_models()` rename) are
> independent and can still land here.

- `lib/llm/src/model_type.rs` — Confirm `Prefill` variant removal complete; remove any deprecated stubs left in PR 3 for compile-time bridging.
- `lib/bindings/python/rust/lib.rs` — Confirm `ModelType.Prefill` export removed; `ModelType.Empty` is the canonical "no OpenAI surface" value.
- `lib/llm/src/discovery/model_manager.rs` — Rename/refactor `list_prefill_models()` (new name TBD in implementation).
- Any straggler Python call sites still referencing `ModelType.Prefill` are flipped to `ModelType.Empty`.
- `lib/llm/src/worker_monitor.rs` — Replace string constants with `WorkerType` values.
- Tests updated.

**Observable effect:** none beyond the public API break (called out in release notes).

### PR 5 — Remove the cross-version compatibility layer (Release X+2)

PR 3 ships a **bidirectional old↔new compatibility layer** so that, for one release
cycle, a rolling upgrade can mix an old frontend with a new worker (and vice versa)
without hiding models or breaking disaggregated routing. That layer is intentionally
temporary and must be removed once the compatibility window closes — **two releases
after it ships (Release X+2)** — by which point every deployment is expected to be on
the new `worker_type`-aware path. Removing it restores the original strict Phase 3
design target (a card with no declared `worker_type` is misconfigured → not-ready).

Remove, in this order:

- **`ModelType::Prefill` dual-emit marker** — the retained `1 << 4` bit and its
  `supports_prefill()` accessor + `as_vec()`/`units()` entries (`lib/llm/src/model_type.rs`),
  the `ModelType.Prefill` Python classattr + `.pyi` stub, and the relaxed
  prefill-surface check in `register_model` (`lib/bindings/python/rust/lib.rs`).
- **Backend dual-emit** — flip prefill registration sites back from `ModelType.Prefill`
  to `ModelType.Empty`: vLLM `worker_factory.py` (+ LoRA `handlers.py`), sglang
  `init_llm.py` (+ LoRA `handler_base.py`), trtllm `llm_worker.py`, `global_router`,
  and backend-common `resolve_model_type`.
- **Frontend legacy shim** — `effective_worker_type()` and the legacy-card bypass branch
  in `Model::is_workers_ready` (`lib/llm/src/discovery/{watcher,model}.rs`), plus
  `warn_legacy_readiness_once`. Restore strict gating + the strict-mode tests; drop the
  compat-matrix tests that assert legacy-card behavior.

This is the deferred form of the original "PR 4 — remove `ModelType::Prefill`": because
PR 3 keeps the bit as the marker, its removal lands here.

**Observable effect:** cross-version mixing is no longer supported — old and new
frontends/workers must match (called out in release notes). Missing `worker_type` is
again rejected loudly rather than silently treated as Aggregated.

### Follow-up PRs (Step 2 + Step 3 of the dispatch-gap plan)

Both deferred to a separate execution track after PR 4. Scoped in [Addressing the cross-namespace dispatch gap](#addressing-the-cross-namespace-dispatch-gap-mechanism-3--4).

- **Step 2 — Pool + dispatch + rewire** (production-blocking for any deployment with rolling updates / blue-green / multi-WorkerSet capacity expansion). Closes Mechanism 3 and unifies Mechanisms 1+2 onto the same pool.
- **Step 3 — `/v1/models/{model}/readiness` endpoint** (Mechanism 4 — operator observability; sits on top of Step 2's pool view).

### PR dependencies

- **PR 1** has no dependencies; can land any time.
- **PR 2a** depends on PR 1 (types must exist).
- **PR 2b / 2c / 2d** depend on PR 2a (shared helper + bindings) but are **independent of each other** — they can land in any order. Each backend's cards are read independently; other backends continue operating on the compat shim until they land their own PR.
- **PR 3** depends on **all of** PR 2a + 2b + 2c + 2d (every backend must be dual-tracking before consumers switch to reading `worker_type`) and on external [PR #8590](https://github.com/ai-dynamo/dynamo/pull/8590) (which activates `check_ready()` for process readiness; our new `check_topology_ready()` composes with it).
- **PR 4** depends on PR 3 (consumers must be off `ModelType::Prefill` before removal). Note: the `ModelType::Prefill` removal itself is deferred to **PR 5** (PR 3 retains it as the compat marker); only the independent cleanups remain in PR 4.
- **PR 5** depends on PR 3 and lands **two releases later (Release X+2)** — it removes the cross-version compatibility layer (legacy shim + `ModelType::Prefill` dual-emit) after the upgrade window closes.
- **Step 2 PR (pool + dispatch + rewire)** depends on PR 3 (uses the live-compute readiness predicate that PR 3 finalizes).
- **Step 3 PR (Mechanism 4 endpoint)** depends on Step 2 PR (shares the per-namespace readiness machinery).

All Phase 1–4 PRs should land in Release X. **PR 5 (compat-layer removal) lands in Release X+2**, giving operators a full two-release window to upgrade old↔new before cross-version mixing support is dropped. Step 2 / Step 3 of the dispatch-gap plan are post-Release-X.

## Alternatives Considered

1. **`--serving-mode` frontend flag** — Global, not per-model. Doesn't scale to E/P/D or future topologies.
2. **Static dependency table** — Dependencies are deployment-time decisions, not type-level. A prefill worker may or may not need encode depending on `--route-to-encoder`.
3. **Explicit `Decode` `ModelType` variant** — Doesn't generalize beyond P/D, and conflates "what endpoints are served" with "what processing stage runs here." The orthogonal `WorkerType` enum subsumes this.
4. **String values for `worker_type` / `needs`** — Rejected in favor of a typed enum for compile-time safety, IDE support, and cheaper serialization. Lowercase strings are still the canonical wire representation in the MDC and on the ws_key.
5. **Cached per-namespace `WorkerTypeInfo` with register/clear hooks** — Initially considered. Rejected in favor of live computation, matching existing precedent (`Model::has_prefill()` at `model.rs:104`). Removes scale-down complexity entirely. (Memoization for dispatch-path performance returns in Mechanism 3, but only as a materialized view inside the watcher, not as registration-driven bookkeeping.)
6. **Bitflag `WorkerType` with `Aggregated = Prefill | Decode` alias** — Initial design. Lets readiness math be a single bitwise AND (`required & present == required`) and lets an E-PD deployment satisfy `Encode.needs = Prefill | Decode` "for free." Rejected in favor of an enum + DNF `needs` design because the bitflag alias conflates two distinct meanings (a worker's role vs a set of capabilities) and forces nonsensical encodings for "needs (P AND D) OR Aggregated." DNF expresses the disjunction explicitly and keeps `WorkerType` as a tagged role, not an arithmetic bag.
7. **OR-valued `needs` via a sentinel `AnyOf(Prefill, Decode, Aggregated)`** — Would let Encode express "I need any consumer" more precisely than a flat list. Rejected as a special-case construct once DNF made the disjunction first-class.
8. **Delaying `ModelType::Prefill` removal to a follow-up DEP** — Considered. Folded into Phase 3 (final removal in PR 4) to avoid a vestigial-code intermediate state where the Rust consumers have migrated but the enum variant still exists.
9. **Dedicated `GET /v1/models/{model}/readiness` detail endpoint, in initial scope** — Considered. Moved to Mechanism 4 (deferred follow-up) to keep the initial DEP scope bounded. The 503 error body from Mechanism 2 carries enough diagnostic detail (missing-role names per affected namespace) for the common "why is my request failing?" question; dashboards, cross-namespace visibility, and CI assertion targets motivate the dedicated endpoint and justify a separate observability PR.

## Open Questions

1. **Mechanism 3 last-namespace-partial behavior** — when the last ready namespace becomes partial mid-request, the dispatch pool empties. Need to confirm the failure surface is a clean 503, not a stuck request waiting for instances.
2. **Mechanism 3 KV-affinity cold-cache when readiness flips** — a request whose previous turn was served by `ns-old` carries KV blocks in `ns-old`. If `ns-old` becomes partial and `ns-new` is now the only ready namespace, KV affinity is moot and the request takes a cold-cache penalty. Correct trade-off, should be explicit in operator docs.
3. **Mechanism 4 namespace ID visibility** — internal namespace strings may be noisy for operators. Whether to expose them as-is, alias them, or expose only operator-meaningful labels needs an operator-side review.
4. **Mechanism 4 caching policy** — the computation is cheap (O(workersets in model)) but called from external dashboards at potentially high frequency. Whether to add HTTP-level caching headers or rate-limit needs a sizing decision.
