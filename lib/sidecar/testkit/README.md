<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecar testing

Unit tests live beside the production code they exercise. They construct inputs,
call the real parsing, conversion or state-management functions, and check the
results without starting an inference engine. Common behavior is tested in the
common crate; vLLM behavior is tested in the vLLM crate. There is no shared unit
scenario or backend-adapter layer.

## File layout

```text
lib/sidecar/
├── common/src/
│   ├── args.rs                 # Inline tests: argument defaults and validation
│   ├── endpoint.rs             # Inline tests: endpoint parsing and normalization
│   ├── error.rs                # Inline tests: transport status mapping
│   ├── transport.rs            # Production policy and existing socket test
│   └── transport/tests.rs      # Retry/pool policy using paused time, no sockets
├── vllm/src/
│   ├── model.rs                # Inline tests: discovery metadata and configuration
│   ├── engine.rs               # Inline tests: worker configuration and local lifecycle
│   ├── json.rs                 # Inline tests: JSON/protobuf conversion
│   ├── lora.rs                 # Inline tests: adapter validation, identity and locking
│   ├── convert.rs              # Production conversion and child-module declarations
│   ├── convert/
│   │   ├── request_tests.rs    # Request fields, validation, routing and handoffs
│   │   └── response_tests.rs   # Stream conversion, logprobs, stops and usage
│   ├── test_fixtures.rs        # Native request, response and metadata builders
│   └── tests.rs                # Broader tests using a local fake gRPC server
└── testkit/
    ├── src/
    │   ├── control.rs          # Per-request holds, faults and native observations
    │   ├── server.rs           # Local server lifetime, including abrupt shutdown
    │   ├── fixtures.rs         # Shared request construction and stream collection
    │   └── assert.rs           # Token, terminal, usage and error assertions
    ├── tests/
    │   ├── conformance.rs     # Direct sidecar-to-Mocker scenarios over real gRPC
    │   ├── cross_process.rs   # Sidecar children, discovery, routing and shutdown
    │   ├── native_engine.rs   # Real vLLM compatibility, cancellation and KV transfer
    │   └── support/
    │       ├── mod.rs         # Fixture contracts and scheduler-state waits
    │       ├── vllm.rs        # vLLM protocol, Mocker and child-command adapter
    │       ├── sglang.rs      # SGLang protocol and Mocker adapter
    │       └── process.rs     # Local discovery, worker processes and TCP routing
    ├── native_probe.py        # Observes completed native NIXL transfers
    └── README.md              # This guide
```

Small suites use an inline `#[cfg(test)] mod tests` in their production module.
The larger request and response suites are separate files, declared in
`vllm/src/convert.rs`:

```rust
#[cfg(test)]
mod request_tests;
#[cfg(test)]
mod response_tests;
```

These are still child modules of `convert`, so `use super::*` gives them access
to its private functions. A separate test file does not require making
production functions public.

Common's `transport/tests.rs` is registered once from `common/src/lib.rs` using
`#[path = "transport/tests.rs"] mod transport_tests`. The production transport
source is also included for a second Tonic version; registering these policy
tests at the crate root avoids running them twice. The existing socket test
inside `transport.rs` remains with each transport implementation.

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

## Running tests

From the repository root, run all common and vLLM library tests, including their
local-server tests:

```sh
cargo test --locked -p dynamo-sidecar-common -p dynamo-vllm-sidecar --lib
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

```mermaid
flowchart TD
    U[Backend-local unit tests] --> C[Production conversion and lifecycle helpers]
    W[conformance.rs: shared scenarios and backend assertions] --> F[Backend fixture]
    F --> E[Production sidecar engine]
    E -->|native gRPC| G[Per-request fault and observation controller]
    G --> M[CPU Mocker scheduler]
    P[cross_process.rs] --> D[Local discovery and TCP router]
    D --> B[Sidecar child: production Worker and engine]
    B -->|native gRPC| G
    P --> R[Real PrefillRouter]
    R --> B
    N[pytest native launcher] --> T[native_engine.rs]
    N --> V[Real vLLM processes and model weights]
    T --> E2[Production sidecar engine]
    E2 -->|native gRPC| V
    V --> K[GPU scheduler and NIXL KV transfer]
```

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
| `conformance.rs` | Four common scenarios registered for both vLLM and SGLang; active cancellation, consumer drop and peer teardown initially enrolled for vLLM; vLLM request/logprob fields, admission rejection and malformed response checks | CPU, ordinary pre-merge Cargo tests |
| `cross_process.rs` | Registration and error recovery, readiness, startup failure/interruption, request isolation, SIGTERM withdrawal/drain; vLLM prefill/decode handoff through the real router | CPU, ordinary pre-merge Cargo tests |
| `native_engine.rs` | Real logprobs and structured output, native scheduler cancellation/drop, completed KV transfer between engines | GPU, post-merge and nightly via pytest |

A generic scenario is reusable code, not evidence that every backend runs it.
SGLang currently registers the four baseline scenarios. Its process and active
work enrollment requires implementing and exercising the additional fixture
contract. TensorRT-LLM is not enrolled here.

The native tests retain their own purpose: a Mocker cannot prove that real vLLM
accepts the serialized request, executes a structured-output constraint, releases
its real scheduler work, or transfers GPU KV cache through NIXL. CPU handoff
checks the routing and opaque metadata contract. Native handoff separately
requires completed transfer bytes and compares output with local inference.
Neither check establishes migration or cancellation during an actual transfer.

Existing tests in `lib/mocker/servers/{vllm,sglang}/tests/sidecar.rs` retain distinct
KV-event and handoff coverage. Backend-local socket tests in `vllm/src/tests.rs`
retain broader media, LoRA, administrative and connection behavior. Python
serving and fault-tolerance tests remain in place: passing this testkit does not
establish complete parity with the legacy Python backend.

### Adding an integration test

1. Choose the boundary being protected. Parsing and state transitions without
   I/O belong beside production code. Direct native RPC behavior belongs in
   `conformance.rs`; Worker/discovery or process lifetime belongs in
   `cross_process.rs`. Use `native_engine.rs` only when the assertion requires
   an actual inference engine or GPU state.
2. For shared behavior, write a scenario accepting only its fixture type. Use
   `SidecarFixture` for the common engine lifecycle, `WireFixture` when a test
   must observe active scheduler work, and `ProcessFixture` when it launches a
   sidecar child. Keep protocol messages in the backend's support file.
3. Register common baseline scenarios once in the small enrollment macro. Both
   backends invoke the same macro, producing a separate ordinary Tokio test for
   every scenario. The macro only declares tests; it does not run or select them
   dynamically. A new baseline is therefore included for every enrolled backend.
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
  cargo test --locked -p dynamo-vllm-sidecar -p dynamo-sidecar-testkit
```

Use normal test parallelism. These suites need a Linux host with the repository's
Rust build prerequisites, permission to bind loopback sockets and spawn child
processes, and writable temporary storage. Running them needs no GPU, engine
installation, model download or external discovery service.

The workspace's ordinary `cargo test --locked --all-targets` builds the vLLM
executable through its executable integration target. A testkit-only command
can instead pick up an older binary from the build directory. Always build the
vLLM package alongside testkit when validating source changes. After that build,
you can select a suite or test by name:

```sh
cargo test --locked -p dynamo-sidecar-testkit --test conformance
cargo test --locked -p dynamo-sidecar-testkit --test cross_process
```

### Running native GPU integration tests

Use the repository's vLLM test image and its pinned engine version. The launcher
checks that the Python vLLM package and bundled `vllm-rs` versions agree. Model
weights for `Qwen/Qwen3-0.6B` must already be cached. Compatibility and cancellation
need one GPU; handoff is scheduled on two GPUs. The launcher assigns each engine
its GPU and dynamically allocated ports.

Build the native Rust test executable on the same platform as the test image:

```sh
cargo test --locked -p dynamo-sidecar-testkit --features native-tests \
  --test native_engine --no-run --message-format=json > /tmp/sidecar-native-build.jsonl
export DYNAMO_SIDECAR_NATIVE_TEST="$(jq -r \
  'select(.reason == "compiler-artifact" and .profile.test == true and .target.name == "native_engine") | .executable // empty' \
  /tmp/sidecar-native-build.jsonl)"
python3 -m pytest tests/sidecar/test_native_integration.py -v
```

For a one-GPU host, add `-k 'not handoff'`. Set `SIDECAR_NATIVE_MODEL_PATH` to an
existing local model directory when needed. CI builds/uploads the executable in
`shared-sidecar-tests.yml`, and the pytest job downloads it before starting the
engines. The `native-tests` Cargo feature only enables this explicit GPU target;
it is not needed for CPU tests. Do not interpret a pre-merge CPU pass as native
GPU validation.
