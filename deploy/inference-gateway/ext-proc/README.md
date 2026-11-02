<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo EPP

This directory contains the native Rust Envoy `ext_proc` Endpoint Picker (EPP) for the Gateway
API Inference Extension. For build, test, and image workflows, see [`DEVEL.md`](DEVEL.md).
User-facing setup is in the published
[Gateway API documentation](../../../docs/fern/pages/kubernetes/kv-aware-routing/gateway-api.mdx).

This file records how the Runtime EPP (`DYN_EPP_MODE=dynamo`, `src/epp.rs`) selects workers,
and where its selection still differs from the frontend's.

## How the Runtime EPP selects

The Runtime EPP picks workers but never sends requests. The gateway forwards each request to the
frontend sidecar, which runs `--router-mode direct` and sends it to the workers the EPP chose.

The EPP selects through the same planning interface as the frontend. It builds an
`EppRouteRequest`, its implementation of `RouteRequest`, from what it parses out of the request
body and headers. It then keeps each booking instead of dispatching it:

| Stage | Selection | Booking | Released by |
|---|---|---|---|
| Prefill | `StageRouter::schedule` on the `PrefillRouter` | `RoutePlan::into_booking`, kept in `prefill_bookings` | The first of the prefill-complete or request-complete callbacks |
| Decode | `RoutingHost::plan_route` on a decode host built over the decode `KvRouter` | `RoutePlan::detach_booking`, kept under the reservation id | `mark_prefill_completed`, then `free`, by reservation id from the response callbacks |

With no prefill target, or on any prefill selection error, the request routes aggregated.

### Planned

These changes are planned for the near term, as part of
[#15457](https://github.com/ai-dynamo/dynamo/issues/15457). Neither is implemented yet.

- **Decode through `StageRouter`.** Decode will select through a decode `StageRouter`, as
  prefill does, instead of calling `RoutingHost::plan_route` on the decode host directly. Every
  stage the EPP picks for then uses the same `schedule` interface.
- **Encode stage.** An encode router will implement `StageRouter`, so the EPP can also pick the
  encode worker for a multimodal request. Today the EPP picks only prefill and decode workers.

## Routing parity with the frontend

The frontend reads every selection input from its `PreprocessedRequest`. The Runtime EPP supplies
only some of them. Each row below is one `RouteRequest` input. For what each input means, see
[What selection reads from a request](../../../lib/llm/src/kv_router/README.md#what-selection-reads-from-a-request).

| Input | `RouteRequest` method | Runtime EPP | Status |
|---|---|---|---|
| Request id, cancellation | `engine_context` | The booking id: `epp-prefill/<reservation id>` for prefill, the reservation id for decode. Never cancelled; dropping the pick retracts a queued selection. | Supplied |
| Prompt tokens | `routing_tokens` | Tokens from the EPP's own tokenizer | Supplied |
| Per-block multimodal info | `routing_tokens` | Not supplied | **Gap.** Image requests route on placeholder tokens, so cache matching ignores image content. [#17503](https://github.com/ai-dynamo/dynamo/pull/17503) closes this for images when frontend media decoding is off. |
| Cache namespace, priority jump, strict priority, routing constraints, allowed workers | `routing_hints` | From `nvext` and the request headers. Allowed workers are the gateway's subset hint. | Supplied |
| Worker pins (`backend_instance_id`, `prefill_worker_id`, `decode_worker_id`, DP ranks) | `routing_hints` | Not read | **Gap.** The EPP ignores a client's pin and sets the `x-dynamo-*` worker headers itself. Whether the EPP should honor a pin is undecided. |
| LoRA name | `routing_hints` | Not read | **Gap.** LoRA requests hash as base-model prompts, so cache matching does not separate adapters. |
| Expected output length | `routing_hints` | Not read | **Gap.** Decode load estimates for EPP picks do not use the hint. |
| Scoring and accounting overrides | `router_config_override` | Decode only: `decode_router_config_override`. Keep it in step with `build_decode_router_override` in `lib/llm/src/kv_router/prefill_router/mod.rs`. | Supplied |
| Policy class | `policy_class` | Header `x-dynamo-meta-policy-class`, as on the frontend | Supplied |
| Agent session context | `agent_context` | Not read | **Gap.** No session-aware scoring or subagent affinity groups. |
| Session-affinity id | `session_affinity_id` | Not read; the decode host has no affinity coordinator | **Gap.** No sticky sessions. The standalone EPP supports `x-dynamo-session-id` affinity. |
| Excluded workers, parked classifier lifecycle | `migration_state` | Not supplied | Not applicable: Migration retries run in the frontend sidecar, after the pick. |
| Staged KV that only this decode can release | `staged_kv_cleanup` | Not supplied | Not applicable: only the frontend's decode leg after remote prefill carries staged KV. |
| Query-only selection | `is_query_only` | Not supplied | Not applicable: the EPP always books. |
| The frontend request (device-aware multimodal cache keys) | `preprocessed` | Not supplied | Not applicable: the EPP's prefill and decode routers are KV-routed. |

Closing these gaps is part of the Runtime EPP's production readiness,
[DEP #16595](https://github.com/ai-dynamo/dynamo/issues/16595).

### Gaps outside `RouteRequest`

- **A prefill target that is still activating.** `schedule` returns a migratable
  `WorkerUnavailable` error, and the EPP falls back to aggregated routing, as it does on any
  prefill error. The frontend fails closed instead. See the prefill router's
  [known gap](../../../lib/llm/src/kv_router/prefill_router/README.md#known-gap-errors-fall-back-to-aggregated).
- **No conditional disaggregation.** The EPP never calls `StageRouter::preview`, so every
  request with a prefill target takes remote prefill.
- **No request classifiers** ([#16591](https://github.com/ai-dynamo/dynamo/issues/16591)).
  `detach_booking` aborts any classifier lifecycle a decode plan started.
- **No recorded routing decisions.** The frontend records each KV routing decision at dispatch
  for approximate-cache learning. The EPP never dispatches, so it records none.

### Keeping this list current

When a change adds a `RouteRequest` input, add a row here and decide whether `EppRouteRequest`
supplies it. When a change closes a gap, update its row in the same pull request.
