<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DEP: Native Gateway API Inference Extension (GAIE) support in Dynamo — including vanilla workers without the Dynamo runtime

> **Area:** gateway · k8s · dgdr · router  **Status:** draft — *proposal + implementation
> tracker*. Checkboxes below track delivery; `[x]` done & validated, `[~]` primitive exists
> (not yet packaged), `[ ]` not started.

## Summary

Dynamo already ships a token-aware, KV-cache-aware router. This DEP proposes making Dynamo a
**first-class Endpoint Picker (EPP) for the Kubernetes Gateway API Inference Extension (GAIE)**,
so a standard cluster gateway can delegate the *worker/pool selection decision* to Dynamo via an
`InferencePool` — for **vanilla `vllm serve` fleets** and for **Dynamo-managed workers**
(vLLM / SGLang / TRT-LLM) alike — **without requiring adoption of the full Dynamo control plane
on day one**.

## Motivation

Many teams already run a Kubernetes gateway (Envoy / Gateway API) and operate vLLM or SGLang.
For them, the smallest production change that adds value is to **keep the gateway and the engines
and insert Dynamo only where it improves the routing decision**. Dynamo's router is token-aware
and KV-cache-aware — materially better than load- or string-based endpoint picking — but today
wiring it behind a gateway is an assembly of control-plane parts (platform install, gateway/CRDs,
EPP image, graph, route, config), and several silent failure modes (version skew, missing EPP
image, unattached `HTTPRoute`, block-size mismatch) are hard to diagnose.

This DEP makes the **gateway-owned routing mode a tested, supported, boringly-easy path**, and
stages the deeper Dynamo layers (Planner, GlobalRouter, KVBM, DGDR auto-planning) as **opt-in
expansions** adopted *after* the routing wedge proves value.

## Relationship to DEP #10321 (Runtime-Free Router and Gateway On-Ramp)

This DEP is the concrete **gateway on-ramp** slice of the umbrella DEP
[#10321 — *Runtime-Free Router and Gateway On-Ramp*](https://github.com/ai-dynamo/dynamo/issues/10321).
It shares #10321's thesis — *adopt Dynamo's KV/load-aware routing for vLLM, SGLang and TRT-LLM
without committing to the full Dynamo runtime, keeping the GAIE/EPP adapter thin and the routing
logic in Rust.* PR #8783's native Rust EPP already realizes #10321's "move routing out of the
Go/CGO adapter into Rust"; this DEP extends it to **vanilla (non-Dynamo-runtime) workers** and
per-pod KV events.

The difference is depth: #10321's end state is a **runtime-free** router (decoupled from
`DistributedRuntime`, exposing `POST /route` / `POST /route_and_reserve` and a standalone HTTP
push-router). This DEP's external mode runs the native EPP on a *minimal in-process runtime* (the
inert `mem` discovery backend) — runtime-**light**, not runtime-**free**. This on-ramp can migrate
onto #10321's runtime-free router API; those items are tracked under "Runtime-free router (#10321)"
in the roadmap below.

## Goals / Non-Goals

**Primary objective.** Make **standalone vanilla `vllm serve` deployments** first-class in the
Dynamo EPP/router: a team already running stock vLLM behind a Kubernetes gateway gets Dynamo's
KV/load-aware routing — *and* P/D disaggregation — by adding only an `InferencePool` and the EPP,
**without adopting the Dynamo runtime, the Dynamo worker, or the Dynamo control plane**.

**Goals**
- Dynamo EPP is a drop-in `InferencePool.endpointPickerRef` for any GAIE-compatible gateway,
  serving **standalone vanilla `vllm serve` fleets** and Dynamo-managed workers from one binary.
- Token-aware **precise KV** routing for vanilla `vllm serve` and Dynamo workers.
- **Runtime-less P/D disaggregation for vanilla vLLM**: the EPP selects the prefill and decode
  workers (KV-aware) and emits the prefill endpoint; a decode-side P/D routing sidecar executes
  vLLM's KV-transfer handshake — no Dynamo worker required.
- A single, version-matched, published EPP artifact — no user builds the EPP.
- Intent-driven setup via DGDR (`features.inferenceGateway`) that generates the EPP config,
  `InferencePool`, sidecar, and `HTTPRoute`.
- Routing decisions and failure modes visible from `kubectl`.

**Non-Goals**
- Requiring the Dynamo control plane / Planner / runtime to get routing value.
- Replacing the customer's gateway.
- **Runtime-less P/D disaggregation for SGLang or TRT-LLM.** Their KV transfer is engine-coupled
  (a bootstrap-room rendezvous; an opaque context→generation handshake) and is not drivable by a
  generic external prefill-endpoint header, so disaggregation for those engines is done by the
  Dynamo worker/runtime. Vanilla-vLLM disagg is in scope precisely because vLLM's
  `kv_transfer_params` handshake *is* externally orchestrable from a sidecar.
- Owning, shipping, or forking the P/D routing sidecar, or re-implementing any engine's
  KV-transfer protocol inside the EPP.
- Cross-region request migration or global KV continuity.
- Hand-authored EPP plugin graphs or pool grids on the golden path.

## Ownership boundaries (the product contract)

| Layer | Owns |
|---|---|
| Customer / platform **gateway** | auth, tenant policy, rate limits, retries, billing, top-level observability |
| **Dynamo EPP / router** | token-aware worker/pool selection, P/D choice, routing decision traces |
| **Engines** (vLLM / SGLang / TRT-LLM) | inference; unchanged |
| **Planner / GlobalRouter / KVBM / DGDR auto-planning** | *expansion layers*, adopted after the wedge |

## Proposal

**Dual-mode EPP** — one GAIE-compliant binary; output is always the
`x-gateway-destination-endpoint` header:
- *external mode* (`DYN_EPP_EXTERNAL=true`): discover endpoints from the referenced
  `InferencePool` (selector + `targetPorts`) via a k8s pod reflector (Ready-filtered); the worker
  (or a tokenizer sidecar) tokenizes; consume the worker's **native vLLM ZMQ KV-cache events** for
  precise prefix routing.
- *dynamo mode*: Dynamo discovery + `ModelDeploymentCard` + in-EPP `OpenAIPreprocessor` + event
  plane KV.

**Tokenization** — three interchangeable backends behind one `DYN_EPP_TOKENIZE_URL`: a `transformers`
sidecar (no GPU), a proxy to a real vLLM `/tokenize` (exact parity), and a **Dynamo-processor**
sidecar (`DYN_TOKENIZER_ONLY` mode of the EPP binary; builds `OpenAIPreprocessor` offline). Pick
two of {exact-parity, no-GPU, no-worker-hop}.

**Precise KV-aware routing** — per-pod ZMQ listeners feed the `KvRouter` radix index, keyed by
`hash_pod_name(pod)`; coexists with predict-on-route (approximate) routing.

**GIE-packaged Dynamo workers** — each pod = a Dynamo frontend (per-pod OpenAI HTTP) + an engine
worker emitting **vLLM-format KV events** on a reachable port. The vLLM, SGLang, and TRT-LLM Dynamo
backends all emit the **same vLLM ZMQ wire format**, so one `zmq_listener` consumes every engine
and **no per-engine translation shim is required**.

**Routing profiles** — expose the router as named profiles instead of raw env vars:

| Profile | Meaning | Primitive today | Status |
|---|---|---|---|
| `optimized_baseline` | token-aware load/KV for one pool | `DYN_EPP_PREFIX_MODE`, `DYN_OVERLAP_SCORE_WEIGHT`, predict-on-route | [~] |
| `precise_kv` | live KV-event routing + block-size validation + cache health | `DYN_EPP_KV_EVENTS` + per-pod ZMQ listeners + `DYN_KV_CACHE_BLOCK_SIZE` | [~] validated |
| `conditional_pd` | prefill/decode + visible decode-only vs prefill-decode decision | prefill router + `DYN_ENFORCE_DISAGG`/`DYN_DECODE_FALLBACK` (plumbing) | [ ] no decision metric |
| `flow_control` | gateway priority/fairness/queueing | `nvext` priority_jump (partial) | [ ] |
| `workload_autoscaling` | DGDSA/Planner/HPA from router+frontend metrics | — | [ ] |

## Implementation — this branch stack

**MR1 (this DEP) targets `main`** ([PR #10336](https://github.com/ai-dynamo/dynamo/pull/10336)) and can be reviewed independently. The **EPP enablement MR — PR #8783 — is now merged into `main`**, so the implementation commits below (**MR2–MR6**) have been **rebased onto `main`** and review/merge there directly (MR2→MR3→MR4→MR5→MR6, each stacked on the previous). MR4 was reconciled against main's now-merged `ext-proc` Endpoint Picker — which independently grew its own external-mode/reflector routing — so MR4 is now the genuine delta on top of it (tokenizer sidecar + per-pod KV-event consumption + numeric target-port support). The full stack compiles (`cargo check`) and is clippy-clean against current `main`. Branches
`mkhadkevich/gaie-N-…`:

- [x] **gaie-2 (MR2, [PR #10337](https://github.com/ai-dynamo/dynamo/pull/10337))** `fix(kv-router): admit externally-registered workers into the scheduler candidate set` — externally-registered workers reached only the slot tracker, not the selector's candidate map → `NoEndpoints`/503; merge a default config for them at select time.
- [x] **gaie-3 (MR3, [PR #10338](https://github.com/ai-dynamo/dynamo/pull/10338))** `feat(llm): per-worker ZMQ KV-event source registration on KvRouter` — `register_worker_kv_events()` feeds each pod's native vLLM KV events into the router index.
- [x] **gaie-4 (MR4, [PR #10339](https://github.com/ai-dynamo/dynamo/pull/10339))** `feat(ext-proc): native EPP for vanilla vLLM` — external/GIE mode (InferencePool discovery + Ready filter), sidecar tokenization, per-pod KV-event consumption.
- [x] **gaie-5 (MR5, [PR #10340](https://github.com/ai-dynamo/dynamo/pull/10340))** `docs(ext-proc): vanilla-vLLM InferencePool example`.
- [x] **gaie-6 (MR6, [PR #10341](https://github.com/ai-dynamo/dynamo/pull/10341))** `docs(ext-proc): GIE-packaged Dynamo worker examples (vLLM + SGLang)`.
- [x] **gaie-epp-replica-sync (MR7, [PR #10388](https://github.com/ai-dynamo/dynamo/pull/10388))** — runtime-free cross-replica in-flight-load sync (EndpointSlice + ZMQ) for multi-replica EPP parity, `/metrics`, K8s-native disagg prefill discovery via Dynamo role labels, OSL plumbing, and **`x-prefiller-host-port` emission for runtime-less vanilla-vLLM P/D disaggregation** via a decode-side routing sidecar, plus the `vanilla-vllm-disagg/` example. Branches off gaie-4.

## Validation (live)

- [x] Vanilla `vllm serve` ← native EPP via a Gateway-API gateway — requests HTTP 200, EPP-directed, load-balanced, **no Dynamo control plane and no Planner**.
- [x] Precise KV on vanilla vLLM — KV-events-only (predict-off) cache affinity, selector logit 0.8 → 6.1.
- [x] High concurrency (N=300, concurrency=48, 4 prefixes) — ~430–450 req/s, p95 120–140 ms, 0 failures; load distributes across workers; tokenizer backends return byte-identical tokens.
- [x] Dynamo **vLLM** worker (GIE-packaged) ← native EPP — HTTP 200, precise KV (logit 0.8 → 6.1).
- [x] Dynamo **SGLang** worker (GIE-packaged) ← native EPP — HTTP 200, precise KV from SGLang's own KV events (predict-off logit 6.25); no translation shim.
- [x] **Scale-up scorer risk** — if the KV scorer always picks the warm replica, scale-up does nothing; confirmed the scorer **blends KV overlap with live load under concurrency** so new replicas receive traffic.
- [x] **Runtime-less vanilla-vLLM P/D disaggregation on real GPUs** — stock `vllm serve` prefill + decode (NixlConnector), EPP selects decode+prefill and emits `x-prefiller-host-port`, a decode-side routing sidecar runs the handshake: HTTP 200 with real NIXL KV transfer (decode-pod `KV Transfer metrics: Num successful transfers>0`), **no Dynamo runtime**. Scaled to 2 prefill + 2 decode (EPP discovers new workers at runtime); comprehensive suite — 9/9 functional+edge (short/long-multiblock/repeat-prefix/chat/unicode/stop-seq/streaming) + clean 4xx on malformed/unknown-model + **200/200 under concurrency 48** — load balanced across decode workers, prefill KV-aware selected, 0 prefill-routing failures, 0 aggregated fallbacks.

## Roadmap / tracker

### P0++ — make the current path trustworthy
- [ ] Publish **version-matched** operator + EPP + runtime + chart-default artifacts per release (version skew is the top silent failure).
- [ ] Compatibility checks surfaced as loud DGD/DGDR status conditions.
- [~] One tested gateway-owned quickstart with no placeholder tags (example manifests exist with real `:1.2.0` tags).
- [~] Validate `HTTPRoute` attachment / Gateway readiness (done manually + endpoint Ready filter).
- [ ] Replace raw `EndpointPickerConfig` examples with profile-based examples.
- [~] Routing decision logs/metrics incl. conditional P/D (decision *logs* exist; no metrics).
- [x] Document ownership boundaries (above).

### P0 — collapse manual wiring (the DGDR front door)
> The operator already has the primitives (`type: epp`, `EPPConfig`, generated `InferencePool`,
> `frontendSidecar`, EPP→Rust-router FFI). The work is packaging.
- [ ] `features.inferenceGateway` on DGDR/DGD (there is no `features.kvRouter` today).
- [ ] Operator generates EPP config + `InferencePool` + frontend sidecar + `HTTPRoute` + mesh glue from intent.
- [ ] `routingProfile` presets over the router env vars.
- [ ] Pool labels + `dyn-pool-filter` for single-cluster multi-pool routing (today: one pool per EPP).
- [ ] Route-explanation surface: eligible pools, rejected pools, selected pool, selected endpoint, reason.

### P1 — make multi-pool & Planner feel native
- [ ] Profiler sweeps TP/pool shapes → emits ISL/SLA grids.
- [ ] DGDR records a `ResolvedPlan` (backend, topology, pool set, evidence, Planner mode, warnings, next action).
- [ ] Planner consumes gateway/EPP metrics → live pool headroom → DGD/DGDSA scale → EPP sees new endpoints.
- [ ] Replay/simulation for routing-policy changes before rollout.
- [ ] Extend the pool contract to cluster/DC routing (session pinning, capacity leases, local rejection).

### Runtime-free router (DEP #10321) — deeper shared architecture
- [ ] Decouple the KvRouter / EPP from `DistributedRuntime` (true runtime-free core), replacing today's inert `mem`-backend shim.
- [ ] Sidecar router API: `POST /route` (query-only) and `POST /route_and_reserve` (route + reservation), per [#10321](https://github.com/ai-dynamo/dynamo/issues/10321).
- [ ] Standalone HTTP push-router mode (OpenAI-compatible; router owns worker selection, forwarding, response relay), per #10321.

## Failure modes → `kubectl` status conditions

"Easy" means the user sees the problem from `kubectl describe`, not EPP logs. Several *causes*
are de-risked; none are surfaced as conditions yet.

| Condition | Cause de-risked? | Surfaced as status? |
|---|---|---|
| route selected no eligible endpoints | [x] gaie-2 (external-worker admission) | [ ] |
| worker not Ready / dead endpoint routed | [x] gaie-4 `pod_is_ready` filter | [ ] |
| KV events missing · block-size mismatch | [~] config + validated | [ ] |
| version skew (operator/EPP/runtime) | [ ] | [ ] |
| gateway provider missing / Gateway not Programmed | [ ] | [ ] |
| HTTPRoute not attached / rejected | [~] cross-ns attach proven | [ ] |
| EPP config invalid · EPP image missing/incompatible | [ ] | [ ] |
| worker sidecar not in direct-router mode | [~] GIE-packaged pods run direct | [ ] |
| prefill profile set but no prefill workers | [ ] | [ ] |
- [ ] Implement the above as DGD/DGDR `status.conditions`.

## Conditional P/D

The disaggregation **decision must be legible**. Dynamo has the router; surface the decision:
- [ ] `disagg_decision_total{decision_type="decode-only|prefill-decode", reason=…}`.
- [ ] Per-request routing trace: non-cached tokens, cached-prefix tokens, selected prefill pool, selected decode pool, fallback reason.
- [ ] Policy threshold over non-cached tokens / prompt length / expected output / queue state / P-D saturation.
- [ ] First-class `conditional_pd` profile name (not just `DYN_ENFORCE_DISAGG`/`DYN_DECODE_FALLBACK`).

> Already present in the EPP: disagg profile runs prefill then decode; `DYN_ENFORCE_DISAGG=false`
> falls back to aggregated decode on prefill failure; `=true` rejects when prefill workers are
> missing; the decode scorer sets `x-dynamo-routing-mode` headers. The work is metrics + thresholds
> + profile, not core routing.

## Highest-leverage next steps

1. **P0 `features.inferenceGateway` + `routingProfile` on DGDR** — collapses most of the manual wiring.
2. **Conditional-P/D decision metric + trace.**
3. **Version-matched published EPP image** — removes the "users build the EPP" anti-pattern.
4. **Failure modes → status conditions.**
5. Upstream the kv-router external-worker fix (gaie-2) into the EPP base.
