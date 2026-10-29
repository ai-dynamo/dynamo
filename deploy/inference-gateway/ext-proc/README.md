<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime EPP routing and callback lifetime

The endpoint picker (EPP) selects workers and books router load. Envoy forwards the HTTP request and sends HTTP processing events back over the external-processing (`ext_proc`) gRPC stream. Runtime EPP uses Dynamo's in-process preprocessor, discovery clients, and KV routers. P denotes prefill, D denotes decode, and PD means execution split across those workers.

## Distinguish the implementations

| Path | Selection and execution ownership |
|---|---|
| Runtime EPP, `src/epp.rs` | Dynamo discovery, local preprocessing, a prefill router and a managed decode KV router; Envoy forwards to a Dynamo worker frontend |
| Standalone EPP, `src/epp_router.rs` and `src/selector.rs` | Separate selector adapter and raw-worker flow; do not assume runtime EPP's P/D behavior applies |
| Standalone selection service | Selection and reservation lifecycle over HTTP; no backend request stream |
| Integrated Dynamo frontend | Preprocessing, route planning, backend dispatch, and response-guard lifetime |

Runtime EPP's decode path calls `route_decode` to query a worker, resolves its
HTTP destination, and then calls `add_request` to register load. These are separate
operations. A successful decode query does not own a booking.

The standalone EPP's ordinary raw-worker path calls `select_and_reserve`. The separate sidecar binary wires an `UnavailablePdAdapter` in the current implementation; that limitation must not be attributed to the runtime EPP path, whose downstream Dynamo worker frontend handles P/D execution.

Sources: [runtime decode query and registration](src/epp.rs), [standalone EPP selection](src/epp_router.rs), and [sidecar adapter wiring](../sidecar/src/bin/dynamo-epp-sidecar.rs).

## Initialization

EPP initializes a Dynamo runtime, waits for discovery, obtains a model deployment card, and constructs `OpenAIPreprocessor`. It creates a local `ModelManager` for routing construction and shared endpoint watches, then builds D's managed KV router.

It constructs `PrefillRouter` with a one-shot endpoint receiver and the same manager. A discovery task finds a prefill-only endpoint in the active namespace and activates the router. Worker additions and removals within that endpoint are tracked continuously by the endpoint client and runtime-config watches. Discovery of a replacement namespace/endpoint is a separate rebinding task and remains a TODO here.

Sources: [initialization](src/epp.rs) and [prefill endpoint discovery](src/epp.rs). The [frontend/discovery guide](../../../lib/llm/src/README.md) explains what the manager and tokenizer provide.

## One pick, two reservation lifetimes

For a body-bearing runtime EPP request:

1. Parse and tokenize the request locally, resolve routing hints and policy class, and restrict workers using the allowed endpoint subset.
2. Mint a fresh reservation UUID. It is independent of the HTTP `x-request-id`.
3. Try to reserve P with request ID `epp-prefill/{reservation_id}`. For KV routing this enters P admission and books P. A P error causes aggregated fallback in this implementation.
4. Query D through admission with `update_states=false`. In PD mode, disable
   D prefill-token tracking and adjust its cache-credit override.
5. Resolve D's HTTP destination. If discovery cannot resolve it, fail the pick;
   the held P reservation is cleaned up when it leaves scope.
6. Call `add_request` with the reservation UUID to register D's load. Registration
   errors are logged; they do not cause this pick to fail.
7. Retain P's reservation in `prefill_bookings`, keyed by that UUID, and return
   the destination, worker/rank headers, eligible token injection, and
   `PickResult.reservation_id` to the ext_proc server.

With KV routing, P reservation and D query each enter admission sequentially.
P can remain booked while D waits. D's later registration does not repeat that
admission. The split leaves a race between selection and load registration and
is not an atomic P/D reservation transaction. Built-in non-KV P routing returns
an advisory, untracked reservation.

Sources: [pick and booking adoption](src/epp.rs), [P reservation contract](../../../lib/llm/src/kv_router/prefill_router/query.rs), and [D registration](../../../lib/llm/src/kv_router.rs).

Runtime EPP does not apply the integrated frontend's conditional D-preview policy here. It attempts P first and forwards a P pin on success. The worker-side prefill router respects that pin and executes the selected topology. See the [prefill guide](../../../lib/llm/src/kv_router/prefill_router/README.md).

## How Envoy events find the booking

Envoy does not know Dynamo's booking ID or construct picker callback arguments. It sends standard HTTP processing events on the ext_proc stream for that HTTP request. EPP's server owns one `RequestContext` for that stream.

After `pick()` succeeds, the server saves:

```rust
ctx.body_routed = true;
ctx.booking_id = result.reservation_id.clone();
```

Later response events arrive on the same stream, so the server reads the saved ID from the same context. There is no response-header lookup or requirement for the backend to echo the booking ID. Concurrent streams have separate contexts even when their HTTP request IDs are identical.

```text
HTTP request → ext_proc stream → RequestContext → saved booking_id
                                      ↑
                         response events on the same stream
```

Source: [saving the reservation ID](src/server.rs) and [retrieving it for the callback](src/server.rs).

## Current callbacks and their inputs

| Picker method | Actual server trigger | Inputs | Runtime EPP effect |
|---|---|---|---|
| `on_prefill_complete` | First nonempty response-body chunk after successful routing, once per stream | Booking ID | Release P; mark prefill complete in D's router accounting |
| `on_request_complete_with_usage` | Processing loop exits after successful routing, including closure/error paths | Booking ID and optional `ResponseUsage` | Release any remaining P; free D |
| `on_request_complete` | Default target of the usage-aware method when an implementation does not override it | Booking ID | Implementation-specific cleanup |

The plain and usage-aware completion methods are alternatives at the trait boundary, not two independent completion notifications. The server directly invokes the usage-aware method.

```rust
pub struct ResponseUsage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
}
```

EPP parses usage from backend JSON or reassembled server-sent event (SSE) data. Usage can be absent, and each field can be absent. These callbacks do not receive HTTP status, headers, raw response bodies, generated token IDs, timing, or an explicit success/error/disconnect reason. Implementations also have access to their own stored state through `&self`.

Sources: [callback interface](src/picker.rs), [body and usage parsing](src/server.rs), and [runtime callback effects](src/epp.rs).

### Prefill completion is inferred

Response headers alone do not trigger prefill completion. The first nonempty body chunk triggers it asynchronously so router bookkeeping does not delay forwarding that chunk. The server does not verify that those bytes contain a generated token. An SSE metadata event or an error body can satisfy the condition.

“Mark prefill complete on D” refers to router accounting. It clears prefill phase/load while retaining D's decode reservation. PD mode already disables D prompt-token tracking; aggregated mode does track that load. P's release is a separate action.

Source: [first-body trigger](src/server.rs).

### Completion follows the processing loop

The trait comment describes completion in terms of response end-of-stream or trailers. The actual invocation is after the processing loop returns. Response end-of-stream updates context state, but the cleanup hook is outside the loop. Use the implementation when reasoning about errors and disconnects.

The server uses the saved reservation ID, falling back to the request ID when the picker supplied no reservation. It passes any parsed usage to the callback. Runtime EPP logs usage but does not convert those final totals into incremental output-block updates.

Source: [completion callback invocation](src/server.rs).

## Other event points available to an extension

The server already handles request headers, body, and trailers; response headers, body, and trailers; and stream errors/closure. The request body being complete is the normal selection trigger. These event branches are possible places to add richer picker hooks, but the current trait has no generic callback-registration API.

Useful extensions include HTTP status inspection, parsed output progress, and completion with an explicit outcome. A true backend prefill-completion notification would require a backend signal; ordinary HTTP event arrival alone does not provide it.

Envoy configuration controls which events and bodies reach EPP. This server expects `FULL_DUPLEX_STREAMED` in both body directions because it needs the request body before it can return the routing header response. It does not require `send_body_without_waiting_for_header_response`: that flag applies to `STREAMED` mode, while this server requires `FULL_DUPLEX_STREAMED`. Configuring Envoy's event delivery does not define Dynamo callback arguments.

Sources: [event branches](src/server.rs) and [protocol requirements](src/server.rs).

## Cancellation and cleanup boundaries

While `pick()` is waiting, the server races selection against closure of its response channel. Dropping queued selection retracts the pending admission. The owned P reservation covers failures before adoption. D is queried without a booking and registered later by request ID. After successful registration, callbacks and request-lease expiry provide D cleanup. These paths do not establish that every failure after registration releases D immediately.

Prefill release removes the entry from `prefill_bookings` before releasing it, so duplicate or reordered phase/completion signals do not release the same stored P reservation twice. A terminal callback can release P even if no nonempty response body arrived.

Stream closure does not prove backend work stopped. The code explicitly leaves Envoy retries and backend work continuing after an ext_proc disconnect as an unresolved ownership issue. Do not claim the current callback protocol gives exact backend cancellation semantics.

Sources: [queued cancellation](src/server.rs), [P release helper](src/epp.rs), and [disconnect limitation](src/server.rs).

## Feature differences from the integrated frontend

The shared selection core supplies admission, selection, and booking modes, while runtime EPP composes D query and registration separately. Full frontend behavior still depends on the adapter and request lifetime:

| Area | Observed runtime EPP behavior in the current implementation |
|---|---|
| Model resolution | Initializes from a discovered card; request model is not checked through the full frontend catalog/alias path |
| Routing input | Carries cache namespace, priorities, policy class, allowed workers, and constraints; does not forward every frontend hint, session/agent field, expected output, or multimodal block input |
| Session affinity | Runtime EPP does not forward session/agent context into its D selection call |
| Conditional disaggregation | P-first reservation, rather than the integrated D-preview/policy path |
| Classifier lifecycle | Runtime EPP does not use the dispatching RoutingHost request-lifecycle path |
| Response progress | Phase callback and terminal usage only; no generated-token output-block or approximate-cache progression equivalent to the frontend request guard |
| KV-transfer hints | Normal dispatch installs `kv_hint`; the EPP result does not carry that hint through the same mechanism |
| Migration and retries | Does not expose the full frontend migration state or a populated fallback list |
| Worker address changes | Runtime discovery and the pod-address reflector can disagree; endpoint reconciliation is a TODO |
| Admission and booking | D query and registration are separate even within one EPP; best-effort replica sync also does not establish global atomic admission |
| Pool changes | Automatic namespace rebinding remains a TODO |

These are source observations, not a claim that every feature is required for every deployment. A shared frontend scheduling adapter should carry the applicable inputs and define how response events drive the remaining lifetime work. Reusing the chat-completions request/response shape alone does not supply these semantics.

Sources: [runtime EPP router state](src/epp.rs), [initialization limitations](src/epp.rs), [preprocessing limitations](src/epp.rs), [runtime selection inputs](src/epp.rs), [dispatch KV hint](../../../lib/llm/src/kv_router/routing_host/kv.rs), and [frontend output progression](../../../lib/llm/src/kv_router/routing_host/request_guard.rs).
