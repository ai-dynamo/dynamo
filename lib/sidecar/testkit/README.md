<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecar testing

Unit tests live beside the production code they exercise. They construct inputs, call the real parsing, conversion or state-management functions, and check the results without starting an inference engine. Common behavior is tested in the common crate; backend behavior is tested in the vLLM, SGLang or TensorRT-LLM crate. There is no shared unit scenario or backend-adapter layer. Shared CPU integration tests live in the testkit crate and connect real sidecars to local Mocker servers, which simulate engine responses without loading a model.

## File layout

```text
lib/sidecar/
├── common/src/
│   ├── args.rs                 # Inline tests: argument defaults and validation
│   ├── endpoint.rs             # Inline tests: endpoint parsing and normalization
│   ├── error.rs                # Inline tests: transport status mapping
│   ├── json.rs                 # Inline tests: shared JSON/protobuf conversion
│   ├── transport.rs            # Production policy and existing socket test
│   └── transport/tests.rs      # Retry/pool policy using paused time, no sockets
├── vllm/src/
│   ├── model.rs                # Inline tests: discovery metadata and configuration
│   ├── engine.rs               # Inline tests: worker configuration and local lifecycle
│   ├── lora.rs                 # Inline tests: adapter validation, identity and locking
│   ├── convert.rs              # Production conversion and child-module declarations
│   ├── convert/
│   │   ├── request_tests.rs    # Request fields, validation, routing and handoffs
│   │   └── response_tests.rs   # Stream conversion, logprobs, stops and usage
│   ├── test_fixtures.rs        # Native request, response and metadata builders
│   └── tests.rs                # Broader tests using a local fake gRPC server
├── sglang/src/
│   ├── client.rs               # Inline tests: discovery, status mapping and RPC deadlines
│   ├── engine.rs               # Inline tests: worker metadata, bootstrap and KV sources
│   ├── native_http.rs          # Inline tests: envelopes, tracing and local HTTP server
│   ├── protocol.rs             # Production conversion and child-module declarations
│   └── protocol/
│       ├── request_tests.rs    # Request fields, public refusals, routing and rendezvous
│       └── response_tests.rs   # Token/logprob conversion, stops, usage and errors
├── trtllm/src/tests/
│   ├── convert_request.rs     # OpenEngine request conversion and unsupported inputs
│   ├── convert_response.rs    # Token events, terminal usage and protocol errors
│   ├── disagg.rs              # KV sessions and prefill/decode conversion
│   ├── engine.rs              # Lifecycle, limits, abort and cancellation
│   └── e2e.rs                 # Ignored by default; requires live OpenEngine servers
└── testkit/
    ├── src/
    │   ├── lib.rs              # Bounded waits and public testkit exports
    │   ├── control.rs          # Per-request controls, observations and controller test
    │   ├── server.rs           # Local server lifetime, including abrupt shutdown
    │   ├── fixtures.rs         # Shared request construction and stream collection
    │   └── assert.rs           # Token, terminal, usage and error assertions
    ├── tests/
    │   ├── sidecar_mocker_integration.rs         # Direct sidecar-to-Mocker scenarios over real gRPC
    │   ├── router_sidecar_mocker_integration.rs  # Sidecar children, discovery, routing and shutdown
    │   └── support/
    │       ├── mod.rs         # Fixture contracts and scheduler-state waits
    │       ├── vllm.rs        # vLLM protocol, discovery, health and child-command adapter
    │       ├── sglang.rs      # SGLang protocol, discovery, health and child-command adapter
    │       ├── trtllm.rs      # OpenEngine wire adapter and sidecar child command
    │       └── process.rs     # Local discovery, worker processes and TCP routing
    └── README.md              # This guide
```

Small suites use an inline `#[cfg(test)] mod tests` in their production module.
The larger request and response suites are separate files, declared in
`vllm/src/convert.rs` and `sglang/src/protocol.rs`:

```rust
#[cfg(test)]
mod request_tests;
#[cfg(test)]
mod response_tests;
```

These remain child modules of their production module, so `use super::*` gives them access
to its private functions. A separate test file does not require making
production functions public.

Common's `transport/tests.rs` is registered once from `common/src/lib.rs` using
`#[path = "transport/tests.rs"] mod transport_tests`. The production transport
source is also included for a second Tonic version; registering these policy
tests at the crate root avoids running them twice. The existing socket test
inside `transport.rs` remains with each transport implementation.

The testkit library owns request controls, bounded waits, stream collection,
assertions and server lifetime. Concrete sidecars, Mockers and protocol libraries
are development dependencies used by the integration tests. Production sidecars
and Mockers do not depend on testkit. Backend fixtures live in `tests/support/`
and are local to the integration suite, rather than a public fixture API.

Each top-level Rust file in `testkit/tests/` builds a separate test executable.
The two CPU files separate direct engine calls from child-process startup,
discovery and shutdown, making each setup easier to follow and run independently.

## Adding a unit test

Add the test to its production module's `tests` child module, or to the existing
request/response test file. Use ordinary `#[test]` or `#[tokio::test]`
attributes. Call the actual production helper and assert the behavior being
protected. Keep setup local unless several tests need the same builder.

Reusable vLLM inputs belong in `vllm/src/test_fixtures.rs`, which is compiled
only for tests. It contains plain functions for requests, model/server metadata,
responses and cache handoffs. Both the isolated units and the broader
`vllm/src/tests.rs` suite use them. Helpers used by only one suite can stay in
that suite. Tests for another backend should use that backend's production
modules and native fixtures.

The broader `vllm/src/tests.rs` exercises connections, RPCs, discovery,
cancellation and administration against a local fake server. It remains
separate because these checks cover interactions across modules, while the
isolated tests call functions directly. Both are compiled into the library's
test binary.

SGLang keeps discovery and worker fixtures in the owning inline modules and
request/response fixtures in its protocol child modules. A private
`from_discovered` helper lets worker tests exercise real construction without
starting a server. Its retained gRPC and HTTP server tests cover the transport
boundary; isolated conversion tests do not establish native engine behavior.

## Running tests

From the repository root, run the common, vLLM, SGLang and TensorRT-LLM library tests, including their local-server tests and the testkit controller regression:

```sh
cargo test --locked -p dynamo-sidecar-common -p dynamo-vllm-sidecar \
  -p dynamo-sglang-sidecar -p dynamo-trtllm-sidecar -p dynamo-sidecar-testkit --lib
```

Run one request-conversion test by its full name:

```sh
cargo test --locked -p dynamo-vllm-sidecar --lib \
  convert::request_tests::canonical_priority_preserves_native_ordering -- --exact
```

The vLLM dependency enables common's `tonic-v14` feature. To cover that version
when running common alone, add `--features tonic-v14`. Building requires the
repository's normal Rust prerequisites; running these suites needs no GPU,
model download or inference-engine installation.

CI runs them through the ordinary `cargo test --locked --all-targets` step and
nightly Rust coverage. There are no per-test lane markers or custom unit runner.

## Integration tests

A Mocker is a CPU simulation of an inference engine. The tests run production
sidecar code over real sockets, but the Mocker supplies tokens and scheduler
state instead of loading model weights. Process tests additionally launch the
actual sidecar executable and use the production Worker, discovery and router.
They create their own local tokenizer files, file-backed discovery and TCP
connections; neither etcd nor NATS is required.

Each suite exercises a different request path:

- `sidecar_mocker_integration.rs` uses a backend fixture to call the production
  sidecar engine library, which sends native gRPC requests to a CPU Mocker.
- `router_sidecar_mocker_integration.rs` uses local discovery to find sidecar
  child processes and sends requests to them over TCP. Each sidecar calls a CPU
  Mocker over native gRPC. Handoff scenarios also use the production PrefillRouter.

The controller sits at the native protocol boundary. Each request ID has its
own plan and observations, so a test can hold or fail one request while proving
that another still completes. Wait for controller events rather than guessing
when work has started. A token checkpoint counts native responses containing
tokens, not individual tokens: a response may contain several tokens. Compare
the sidecar output with the controller's observed token vector.

The local test server owns a dedicated thread and runtime. Shutting it down
also drops accepted connections and RPC handlers, even when a test retains live
clients. That makes peer-loss tests deterministic.

### What runs where

| Suite | Scope | Execution |
| --- | --- | --- |
| `sidecar_mocker_integration.rs` | Shared streaming, errors, cancellation, cleanup, active work release, consumer drop, request/logprob fields and peer teardown for vLLM, SGLang and TensorRT-LLM; native rejection, malformed responses and shutdown during pending SGLang health checks | CPU, ordinary pre-merge Cargo tests |
| `router_sidecar_mocker_integration.rs` | All three backends: registration/error recovery, delayed/failed/interrupted startup, cancellation, SIGTERM and real PrefillRouter handoff; vLLM/SGLang model aliases and native health readiness; SGLang tokenizer/parser discovery, native tracing and changed-role startup | CPU, ordinary pre-merge Cargo tests |

A generic scenario is reusable code, not evidence that every backend runs it. vLLM, SGLang and TensorRT-LLM register the shared wire and process scenarios.

The alias and health-publication scenarios use `DiscoveryFixture` controls for vLLM and SGLang. vLLM keeps Control healthy while the fixture controls Inference readiness; SGLang uses its native HealthCheck RPC. Separate tokenizer discovery, parser settings inherited from engine metadata, changed engine roles and native trace headers remain SGLang-specific because vLLM has different contracts for those values.

TensorRT-LLM supplies its model identity through `--model-path`, discovers its context window through `Control.GetModelInfo`, and currently has no engine-discovered alias or native health gate. It implements `ProcessFixture`, not `DiscoveryFixture`. Its wire assertions check OpenEngine token events, output logprobs and terminal usage; prompt logprobs are unsupported. Registration uses Dynamo’s default KV block size because this sidecar does not yet publish KV-routing capacity.

CPU handoff checks opaque vLLM metadata, SGLang concurrent bootstrap coordination and TensorRT-LLM OpenEngine KV sessions, including their opaque attributes. CPU handoff cancellation holds the peers before Mocker admission; it proves transport cleanup and recovery, not native GPU transfer cleanup. Separate TensorRT-LLM backend-local tests cover deferred cancellation before dispatch, while awaiting decode headers, and until the first token. A Mocker cannot prove that a real engine accepts the serialized request, executes a structured-output constraint, releases its real scheduler work, or transfers GPU KV cache.

Existing tests in `lib/mocker/servers/{vllm,sglang,trtllm}/tests/sidecar.rs` retain distinct KV-event and handoff coverage. Backend-local socket tests in `vllm/src/tests.rs` retain broader media, LoRA, administrative and connection behavior. Python serving and fault-tolerance tests remain in place: passing this testkit does not establish complete parity with the legacy Python backend.

### Relationship to serving E2E tests

`tests/serve/test_sidecar.py` starts the frontend, production sidecar executable
and real engines. It checks HTTP serving, distinct prefill/decode workers and
KV-aware routing. These deployment checks remain separate from the CPU Mocker
tests.

The TensorRT-LLM aggregated case checks streaming output logprobs and token metadata, exact completion usage, and schema-constrained JSON. It also cancels native requests by explicit stop and consumer drop, disconnects an HTTP stream, and requires native executor completion before token-budget exhaustion, an empty queue, zero used KV blocks, and subsequent requests to complete. The test-only engine wrapper disables block reuse and collects native iteration/request statistics, so OpenEngine registry removal alone cannot satisfy cleanup. Serial submission timestamps and a unique active-to-complete scheduler ID attribute each cancellation. Prompt logprobs are unsupported.

The TensorRT-LLM disaggregated case requires distinct prefill/decode worker IDs, exact completion usage, selected logprobs aligned with every completion token (including the prefill-produced first token), and a fresh completed KV transfer. The engine wrapper records positive transfer bytes and elapsed transfer time from native request performance metrics. Greedy token IDs must match a request on the same decode engine using local prefill. A gated first decode response exercises cancellation after native transfer; both engines must release their KV blocks and a fresh routed transfer must succeed afterward. A one-token local request refreshes prefill statistics because idle transfer reaping emits no new snapshot. The TensorRT-LLM sidecar deliberately defers client cancellation until the first decode token confirms transfer completion, then stops generation and releases resources. Tests should verify that deferred-cancellation contract, without requiring interruption of an in-flight KV byte transfer. The GPU test holds the first native decode token until it observes the receiving sidecar’s structured cancellation event for that request. While holding the token, it requires an open native stream and fresh executor activity; after release, it requires early executor completion and resource cleanup. CPU tests additionally cover cancellation before dispatch and while response headers are pending. The ordinary launch scripts still own engine startup and teardown.

Run both GPU cases inside the TensorRT-LLM test image with the current Dynamo wheel and `dynamo-trtllm-sidecar` executable on `PATH`:

```sh
pytest tests/serve/test_sidecar.py -m 'sidecar and trtllm and gpu_1' --models-dir=/path/to/huggingface/hub
```

These cases use Qwen/Qwen3-0.6B and the existing post-merge and nightly sidecar jobs. They run sequentially; no profiled VRAM marker is claimed. Both disaggregated workers share one GPU.

The legacy Python backend suite is also distributed by behavior, including
`tests/serve/test_vllm.py`, `tests/fault_tolerance/cancellation/test_vllm.py` and
`tests/fault_tolerance/migration/test_vllm.py`.

### Adding an integration test

1. Choose the boundary being protected. Parsing and state transitions without
   I/O belong beside production code. Direct native RPC behavior belongs in
   `sidecar_mocker_integration.rs`; Worker/discovery or process lifetime belongs in
   `router_sidecar_mocker_integration.rs`.
2. For shared behavior, write a scenario accepting only its fixture type. Use `SidecarFixture` for the common engine lifecycle, `WireFixture` when a test must observe active scheduler work, and `ProcessFixture` when it launches a sidecar child. `DiscoveryFixture` adds alias and health controls for backends that support those contracts; protocol details stay in the backend's support file.
3. Register common baseline scenarios once in the small enrollment macro. All three backends invoke the same macro, producing a separate ordinary Tokio test for every scenario. The macro only declares tests; it does not run or select them dynamically. A new baseline is therefore included for every enrolled backend.
4. Put a genuine difference in a named fixture value, such as whether
   `generate()` waits for response headers. Keep checks meaningful for each
   backend: SGLang starts its lazy RPC when the response stream is polled.
   A check that only describes vLLM protobuf fields or errors belongs in an
   explicit vLLM test, not an optional callback or a no-op fixture method.
5. Give faulted requests distinct IDs. Await the received/checkpoint/dropped
   event, assert the observed prefix and terminal or typed error, and prove
   scheduler/route release. Where recovery is part of the contract, send a
   healthy request through the same engine. Use bounded waits and preserve the
   independent-request assertions when testing cancellation.

To add a backend, implement its native `Protocol` adapter and `SidecarFixture`
in `tests/support/`, then enroll it through the existing macro. Its adapter must
observe actual native requests/responses and scheduler state. Implement the
additional wire or process contract only when enrolling those scenarios, and
run them before claiming coverage. Backend-specific assertions stay next to
that backend's explicit tests.

### Running CPU integration tests

Build the sidecar binary and run the testkit together from the repository root:

```sh
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
  cargo test --locked -p dynamo-vllm-sidecar -p dynamo-sglang-sidecar \
    -p dynamo-trtllm-sidecar -p dynamo-sidecar-testkit
```

Use normal test parallelism. These suites need a Linux host with the repository's
Rust build prerequisites, permission to bind loopback sockets and spawn child
processes, and writable temporary storage. Running them needs no GPU, engine
installation, model download or external discovery service.

The workspace's ordinary `cargo test --locked --all-targets` builds all three sidecar executables through their executable integration targets. A testkit-only command can instead pick up an older binary from the build directory. Always build the vLLM, SGLang and TensorRT-LLM packages alongside testkit when validating source changes. After that build, you can select a suite or test by name:

```sh
cargo test --locked -p dynamo-sidecar-testkit --test sidecar_mocker_integration
cargo test --locked -p dynamo-sidecar-testkit --test router_sidecar_mocker_integration
```
