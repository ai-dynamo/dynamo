<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sidecar testing framework: focused prototype

Share assertions at the existing `LLMEngine` boundary. Keep pure conversion tests
inside each sidecar crate and native server fixtures beside each backend mocker.
This prototype covers seven scenarios, including four wire scenarios exercised
against both existing mockers. It does not validate real inference engines.

## Inputs and baseline

Source requirements are the [unit inventory](https://docs.google.com/document/d/1CsltmzPcSNNCFuZEJVcUtG7lMwKcdHT-5FtezRTOwxI/edit)
(R01–R32), [integration inventory](https://docs.google.com/document/d/1Eex29sWCf-g3HsTdHVhJWkNiyfGQJUy0mDLjy91-WpM/edit)
(C1–C13), and [allowed testing plan](https://docs.google.com/document/d/146cWxYabWmNAy-tx-sQ9i7JeQvDf-J_3aOg5kuk_a9E/edit).
All three were accessible. Inventory IDs below refer to behavior groups, not
test-function counts or a promise to implement every subcase.

Repository evidence was inspected at fetched main
`1dca16f9b74dcd938496ec6c79be5d227c372c93`. The prototype checkout is
`/home/juliendarve/projects/dynamo/.worktrees/sidecar-testkit-prototype`, branch
`jdarve/sidecar-testkit-prototype`. The original dirty checkout was preserved.
Neither excluded artifact, its discussion, nor its implementation branch was
opened. Main had no `#14879` commit marker and no `lib/sidecar/testkit` directory;
this is not proof against an unattributed cherry-pick. The allowed integration
inventory itself describes a different, pre-existing harness branch and includes
implementation summaries. Those summaries are not design evidence here: only its
behavior requirements and legacy source-case references are used. This provenance
overlap limits a claim of complete informational separation despite using an
independent checkout and implementation.

The inventories are not an accurate description of every current capability.
For example, current vLLM startup discovers model/server metadata and checks
health, and both vLLM and SGLang expose native KV-event sources. Current SGLang
gRPC output is incremental; the older inventory's cumulative-slicing prescription
does not apply to this protocol revision. Verify inventory status
claims against code before implementing a case. The allowed plan's suggestion
that premature EOF should become cancellation is also unsuitable: unexpected
truncation must remain a failure, separate from explicit cancellation.

## Production boundary and ownership

The real path is `Worker → LLMEngine sidecar → native gRPC → engine`, then back
through response conversion. Sidecars adapt tokenized requests, discover/consume
engine metadata, translate streams/errors, and release their client resources.
The shared worker owns discovery publication, request ingress, and process
lifecycle. Engines own token generation, scheduling, cache state, and derived
capacity. Tests should check authoritative metadata arrives intact, not repeat
engine calculations in an expected-value helper.

| Test level | Real components | Replaced components | What it establishes |
| --- | --- | --- | --- |
| Unit | Production parser/converter/error mapper | In-memory native values | Local transformation and validation; no socket |
| Wire integration, prototype | Sidecar constructor/start/generate/cleanup, native protobuf, Tonic/TCP, Mocker scheduler | Inference engine/model; request context supplied by existing test helper | Adapter and transport behavior against the implemented mock protocol |
| Process integration, future | Sidecar executable, Worker, discovery, endpoint client, wire fixture | Engine/model | Registration, external error propagation, CLI/env, SIGTERM composition |
| Native compatibility, future | Same driver with pinned engine and small cached model | Nothing at engine boundary | Agreement with real engine protocol and observable work release |

Relevant production sources: [`backend-common/src/engine.rs:194`](../../backend-common/src/engine.rs#L194),
[`backend-common/src/worker.rs:577`](../../backend-common/src/worker.rs#L577),
[`sidecar/vllm/src/engine.rs:201`](../vllm/src/engine.rs#L201),
[`sidecar/sglang/src/engine.rs:243`](../sglang/src/engine.rs#L243), and
[`sidecar/trtllm/src/engine.rs:192`](../trtllm/src/engine.rs#L192).
The existing [`backend-common` testing module](../../backend-common/src/testing.rs)
already provides request contexts and generic lifecycle conformance. Reuse its
context; do not introduce a second engine trait or claim a fake `LLMEngine`
tests a production sidecar. The prototype adds stronger, named assertions for its
selected contracts rather than running that broader conformance bundle as one test.

## Structure and extension rules

```text
lib/sidecar/{common,vllm,sglang,trtllm}/src/  private unit tests beside owners
lib/sidecar/testkit/
  src/lib.rs                              shared requests and scenario assertions
  README.md                               this design and coverage boundaries
lib/mocker/servers/{vllm,sglang}/tests/
  sidecar.rs                              native fixture + named scenario entry points
lib/mocker/servers/{vllm,sglang}/src/       backend-specific mock behavior
lib/sidecar/Dockerfile                     isolated CPU test stage
```

`dynamo-sidecar-testkit` is an unpublished helper crate consumed as a
dev-dependency. It depends on backend-common, Mocker metrics, Futures and Tokio;
it does not depend on sidecar implementations or native protobuf versions.
Scenario functions accept `&impl LLMEngine`, a request context where necessary,
and concrete expected data. `wait_idle` observes a scheduler metrics watch plus
the mocker's active-request count. No backend registry, universal native-message
enum, custom test runner, or speculative capability system is needed.

The native fixtures retain Tonic version differences (vLLM 0.14, SGLang 0.13),
startup discovery/health wiring, raw native request construction, and logprob
expectations. Requested top-logprobs=2 currently yields three vLLM candidates
versus two SGLang candidates; sharing that wire-specific constant would be wrong.
Shared assertions operate on Dynamo outputs after those differences are declared.

The existing [vLLM service](../../mocker/servers/vllm/src/server.rs#L80) and
[SGLang service](../../mocker/servers/sglang/src/server.rs#L102) run real CPU
Mocker scheduling, synthetic token/logprob generation, model discovery and request
release. They support one data-parallel rank and their respective handoff formats;
vLLM also publishes native KV events. Neither validates model computation, real
KV transfer, native engine health transitions or arbitrary wire corruption.
The only production mocker extension here is a shutdown method delegating to its
existing scheduler shutdown, so normal fixtures can await resource release.

To add a test, identify its inventory contract and owning boundary first. Add a
private unit test for a pure function. For an equivalent wire contract, add one
scenario function and one thin named invocation per supported backend. A new
native behavior belongs with that mocker and must have a documented limitation.
Split `src/lib.rs` by streaming/errors/cancellation when more scenarios justify it.
Do not parameterize unsupported backends into silently skipped green cases.

To add TensorRT-LLM, implement its scheduler-backed native service separately,
including model-info fallback, delta/terminal messages and targeted Abort, then
write a local fixture invoking applicable shared scenarios. Existing in-crate TRT
fake services are useful unit/wire evidence but are not a production Mocker server.
This PR neither implements nor validates a TensorRT-LLM mocker.

## Seven prototype scenarios

| Scenario | Inventory | Exact prototype scope and reason |
| --- | --- | --- |
| Typed native status mapping | R05 | Common mapper, its vLLM Tonic bridge, and SGLang mapper; preserve error category, RPC and peer message. Retain SGLang's different alias-code policy. Cheap, useful failure diagnostics. |
| Request field presence | R06 | Real vLLM/SGLang converters preserve IDs, selected sampling/stop/output fields and absent/zero/nondefault values. Pure tests need no public converter seam. TRT remains existing coverage. |
| Opaque numeric fidelity | R21 | Extend vLLM JSON tests for exact boundary values, negative inexact integers and incoming NaN/infinities. Demonstrates a backend-specific contract without forcing sharing. |
| Tokens, terminal, logprobs, usage | C4, related R09/R15 | Shared assertions run through both real adapters against existing Mocker. Compare ordered tokens and selected logprob values with a direct native RPC using the same request ID; require incremental chunks, one last terminal and exact usage. Top alternatives are checked for shape only. No scripted metadata-only/post-terminal injection in this prototype. |
| Native admission failure and recovery | C6 | Both mockers reject an oversized native output budget. Verify typed invalid-argument and native reason, no successful output, then a valid request. Handles vLLM eager and SGLang lazy RPC opening. No configured fake success/error oracle. |
| Explicit cancellation and release | C8, related R12 | After the first nonterminal token and observable active peer work, stop the actual request context; require one cancelled terminal, no subsequent output, zero scheduler work, then recovery. This is wire scope; concurrent-request isolation and Worker-driven Abort are deferred. |
| Consumer drop and release | C9 | Drop a live consumer without calling stop or Abort; require zero peer work and a subsequent successful request. Catches transport leaks that local cancelled-output assertions miss. |

These are seven behaviors even though unit tables and backend invocations produce
more test functions/executions. Existing streaming/cancellation assertions are
refined and shared where they overlap; existing handoff, KV-event, metadata and
explicit Abort evidence is retained. Moving assertions is not deleting coverage.

R06's zero sampling values test wire presence, not acceptance of those settings
by a real engine. Nondefault fields use distinct values to detect swapped mappings.

The native comparison detects lost/duplicated tokens and conversion mistakes;
it deliberately does not certify the mocker's token algorithm or sampling math.
The rejection case protects sidecar error propagation, not a universal real-engine
output limit. The tested limit is explicitly the mocker's admission policy.

## Mapping the remaining inventory

| Unit IDs | Placement / required additional capability |
| --- | --- |
| R01–R03 | Existing common/backend-local parser/config tests; metadata fixtures must follow each backend's current source of truth. |
| R04 | Common transport/backend-specific startup tests; inject connector futures and paused time only if a narrow production seam is justified. Retain actual socket deadline tests as integration. |
| R07–R08 | Backend-local admission/structured-output converters, with supported/rejected capability rows and exact native payloads. |
| R09–R11 | Existing response-state tests plus actual consumer tests. Script metadata-only, invalid terminal, early EOF and read status beside each mocker. Apply delta/cumulative rules from the pinned native protocol, not the older inventory. EOF categories differ by backend. |
| R12–R14 | Prototype covers only cancellation after a token; add open/read gates and two request identities for pending-open, cleanup, isolation and cancellation phases. |
| R15–R17 | Backend response fixtures for opt-in/logprob values, prompt/routed metadata timing and hidden-stop policy. Prototype checks basic shape, not all associations or stop policies. |
| R18 | SGLang/TRT targeted Abort and bounded control RPC fixtures; vLLM cancellation uses stream release. |
| R19–R20 | vLLM opaque prefill/decode and cache identity fixtures; no cache algorithm reproduction. Existing handoff integration remains separate. |
| R22 | Retain vLLM non-finite logprob normalization tests. |
| R23–R26 | SGLang discovery/readiness/role/metadata/bootstrap cases; controllable metadata/health replies, supplied host addresses. |
| R27 | SGLang rendezvous conversion and real prefill consumer; success and failure gates, not a claim of native KV transfer. |
| R28–R30 | SGLang LoRA/rank/trace forwarding and released wire tags. Pure wire acceptance differs from real adapter loading or rank scheduling. |
| R31–R32 | TRT metadata fallback and engine-reported cached usage; native fixtures, no derived-capacity oracle. |

| Integration IDs | Design destination / scope outside this prototype |
| --- | --- |
| C1–C3 | CPU process fixture with real Worker, isolated file discovery, local model metadata, endpoint client and controlled native readiness. Assert published metadata and bounded failure, including no registration on failure. |
| C4 | Prototype wire subset; add real Dynamo ingress and native request observation for richer option/metadata cases. |
| C5 | Real endpoint rejection and zero native submissions; local converter rejection alone cannot certify the ingress path. |
| C6 | Prototype native invalid-argument case; add unavailable-open fault and process-ingress observation. |
| C7 | Controlled native stream faults after acceptance: EOF, status, malformed terminal, actual connection loss. Require prefix preservation and failure, never false success. |
| C8–C9 | Prototype wire subsets; add concurrent isolation/pending headers, Worker cancellation, and independent real-engine active-work evidence. |
| C10 | Executable SIGTERM during startup and serving, discovery withdrawal, bounded exit, reaped children. Direct engine cleanup cannot replace this. |
| C11–C12 | Existing vLLM/SGLang handoff tests plus real PrefillRouter and two sidecars; then native-engine transfer and cancellation. TRT disaggregation is not promised. |
| C13 | Pinned real vLLM, SGLang and TRT aggregate profiles; reuse success/cancel/drop assertions with backend-specific expected fields. Small cached model and ordinarily one GPU; supported native handoff needs two engines and may need two GPUs. |

The inventories' legacy appendices map engine scheduling, model construction,
multimodal computation, cache movement and deployment matrices to their existing
owners. Their exclusion here neither deletes those tests nor asserts upstream
coverage. Future supported sidecar administration, KV relay, embeddings or
multimodal APIs need their own consumer contracts; implementation changes since
an inventory was written can move cases into scope.

## Fixtures, synchronization and diagnostics

Each wire test owns its bound loopback listener (`127.0.0.1:0`), mocker, real
sidecar, contexts and service task. Bind before constructing the sidecar; use the
actual discovery/health handshake to establish readiness. No model tokenizer,
etcd, NATS, environment mutation, shared namespace or fixed port is required.
The native reference request completes before its same-ID sidecar request.

Generation phases have five-second deadlines. Cancellation/drop synchronize on a
received first token and an active peer request, not a sleep. A deliberately slow
10,000-token mock generation leaves work to cancel; if it already completed the
test fails explicitly. This is not an exact scheduler-time assertion. Waiting for
idle uses metrics notifications and checks active routes plus running/waiting
counts. A 10 ms predicate recheck handles route removal after the final metrics
notification; correctness depends on the observed counts, not elapsed delay.
Timeout output includes all three counts. TCP tests use real Tokio time;
paused time is reserved for future pure connector/future tests.

Normal teardown cleans up the sidecar, stops the mock scheduler, signals the
native server and awaits its task within a bound. The owning guard stops its
resources on panic/timeout as well. Panic cleanup aborts the owned server task and
relies on the per-test Tokio runtime to drop remaining connection tasks; only
normal teardown explicitly joins them through the server. Failures identify backend/test name, operation phase, native
error text, observed output/usage, and scheduler counts. Future process fixtures
also need owned process groups, captured stderr/status and bounded kill-and-reap.

The prototype injects a real native validation failure through its input and
local cancellation/drop through real contexts/stream ownership. It does not add
a general fault DSL. Future header stalls, status/EOF, malformed responses and
health transitions need explicit gates/recorders beside the corresponding native
service. Do not claim a gRPC status models a TCP reset or that a transport drop
proves a real engine freed GPU work.

## Local and CI execution

From the repository root, select packages to avoid unrelated workspace feature
unification:

```sh
cargo test --locked -p dynamo-sidecar-common -p dynamo-vllm-sidecar -p dynamo-sglang-sidecar --lib
cargo test --locked -p dynamo-vllm-mocker -p dynamo-sglang-mocker --test sidecar
docker build -f lib/sidecar/Dockerfile --target sidecar-tests .
```

The commands also retain existing selected-package tests; they are not a claim
that every test in those crates is one of the seven new/refined scenarios.
The CPU Docker stage uses the existing plain Ubuntu builder with Rust, C/C++,
CMake, libclang, protoc and protobuf headers. It installs no CUDA toolkit or
inference engines and requires no GPU or model download. First-time compilation
still needs Rust crates and existing web assets. `dynamo-memory` pulls in `cudarc`
and `nixl-sys`; their dynamic-loading fallback is why a CUDA installation is not
required, not absence of those crates. Model-hub offline flags and a networkless
execution phase make accidental model dependencies visible.

The normal pre-merge Rust job already runs workspace tests. The isolated stage is
additionally wired into the trusted sidecar build workflow, so a failure fails
that job; sidecar and mocker/backend-common path filters trigger it. This provides
a separate CPU-environment check without a new test runner or automatic retries.
Other Rust dependency changes still receive ordinary Rust CI; isolated-stage
filter scope must not be described as every conceivable transitive change.

Actual validation results and mutation evidence are recorded in the PR's
Validation section after execution. A green CPU suite does not replace native
engine compatibility on engine/protocol bumps.

## Alternatives and limits

Putting every unit and integration test in a central testkit would require making
private converters public, hiding backend differences behind adapters, and
introducing dependency cycles. Keeping all tests separate preserves local access
but repeats shared assertions and lets their strength drift. The chosen split
shares only the proven common boundary and keeps fixture code near its protocol.

A new generic engine fake or `mockall` layer would bypass the production adapter;
existing scheduler-backed mockers give stronger integration evidence. Conversely,
forcing all malformed-wire unit cases through a scheduler wastes setup and makes
deterministic faults harder. Small native scripted peers remain appropriate when
needed. Snapshot/property testing and a coverage ratchet can be evaluated for a
specific uncovered risk; neither is required to validate these seven scenarios.

No inference-engine implementation, sampling correctness, model quality, native
memory release, KV transfer, distributed discovery, CLI serving lifecycle or
TensorRT-LLM mocker is certified by this prototype. The design keeps those gaps
explicit and does not use aggregate line coverage as a substitute for them.
