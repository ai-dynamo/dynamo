<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecar testkit

The testkit holds the shared CPU integration tests for the sidecars. They connect real sidecars to local Mocker servers, which simulate engine responses without loading a model.

The testkit library owns request controls, bounded waits, stream collection,
assertions and server lifetime. Concrete sidecars, Mockers and protocol libraries
are development dependencies used by the integration tests. Production sidecars
and Mockers do not depend on testkit. Backend fixtures live in `tests/support/`
and are local to the integration suite, rather than a public fixture API.

Each top-level Rust file in `testkit/tests/` builds a separate test executable.
The two CPU files separate direct engine calls from child-process startup,
discovery and shutdown, making each setup easier to follow and run independently.

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
  HTTP error scenarios additionally run the real frontend: native invalid-argument
  errors become HTTP 400, and failures after a token reaches the client become SSE
  errors without a successful finish. Both paths verify subsequent recovery.

The SGLang fixture is one Rust-server listener. `Fixture::start_node` starts the
listeners of one node, which share a controller and report one
`DeploymentInfo.node_listeners` table; each listener has its own Mocker
scheduler, KV-event publisher, Abort log, request log (`received`), health
control and optional HTTP peer. A single-listener fixture reports a one-entry
table.

The SGLang fixture owns its optional HTTP peer, discovery metadata and shutdown.
The peer imitates SGLang's Rust server: it serves no `/abort_request`, omits
empty `output_ids`, and its requests are aborted with the `sglang.api.v1` Abort
RPC, which the gRPC fixture records. The HTTP cancellation regressions use that
peer and the production sidecar engine. They hold an admitted request before
headers or during streaming, then check targeted abort before disconnect,
decode's drain before abort, another request's isolation and recovery. A native
HTTP endpoint that never becomes ready leaves the engine on gRPC.

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
| `sidecar_mocker_integration.rs` | Shared streaming, errors, cancellation, cleanup, active work release, consumer drop, request/logprob fields and peer teardown for vLLM, SGLang and TensorRT-LLM; native rejection, malformed responses and shutdown during pending SGLang health checks; vLLM engine loss from its health stream; SGLang with one sidecar per node: Abort only on the owning listener, by that dispatch's own rid, for gRPC and native requests, and engine loss naming the lost listener's DP rank | CPU, ordinary pre-merge Cargo tests |
| `router_sidecar_mocker_integration.rs` | All three backends: registration/error recovery, delayed/interrupted startup, cancellation, SIGTERM, frontend HTTP errors/recovery and real PrefillRouter handoff; all three plus LPX: request migration after engine failure (before or after the first token) and SIGTERM; vLLM/SGLang model alias publication and native-health readiness; SGLang tokenizer/parser discovery and changed-role startup; vLLM withdrawal and process exit on engine loss; SGLang with one sidecar per node: the prefill rank and port through a real PrefillRouter under round-robin over two nodes (with a pinned rank), and withdrawal and process exit when one listener is lost | CPU, ordinary pre-merge Cargo tests |

A generic scenario is reusable code, not evidence that every backend runs it.
vLLM, SGLang and TensorRT-LLM register the shared wire and process scenarios. TensorRT-LLM uses the existing OpenEngine Mocker through a controlled inference service. Its process tests supply a local model path and verify the context limit returned by OpenEngine.

The alias and health-publication scenarios use `DiscoveryFixture` controls for vLLM and SGLang. TensorRT-LLM is excluded from these scenarios because its sidecar uses a configured model path and does not discover served aliases or gate publication on native health. vLLM keeps Control healthy while the fixture controls Inference readiness; SGLang uses its native HealthCheck RPC. Separate tokenizer discovery, parser settings inherited from engine metadata and changed engine roles remain SGLang-specific because vLLM has different contracts for those values.

CPU handoff checks opaque vLLM metadata, SGLang concurrent bootstrap coordination, and TensorRT-LLM OpenEngine KV session forwarding. CPU handoff cancellation holds the peers before Mocker admission; it proves transport cleanup and recovery, not native scheduler or transfer cleanup. A Mocker cannot prove that a real engine accepts the serialized request, executes a structured-output constraint, releases its real scheduler work, or transfers GPU KV cache.

### Adding an integration test

1. Choose the suite. Direct native RPC behavior belongs in
   `sidecar_mocker_integration.rs`; Worker/discovery or process lifetime belongs in
   `router_sidecar_mocker_integration.rs`.
2. For shared behavior, write a scenario accepting only its fixture type. Use
   `SidecarFixture` for the common engine lifecycle, `WireFixture` when a test
   must observe active scheduler work, and `ProcessFixture` when it launches a
   sidecar child. Its alias and readiness controls keep protocol messages in the
   backend's support file, so another backend can enroll without copying the
   scenario body.
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
  cargo test --locked -p dynamo-vllm-sidecar -p dynamo-sglang-sidecar \
    -p dynamo-trtllm-sidecar -p dynamo-lpx-sidecar \
    -p dynamo-sidecar-testkit
```

Use normal test parallelism. These suites need a Linux host with the repository's
Rust build prerequisites, permission to bind loopback sockets and spawn child
processes, and writable temporary storage. Running them needs no GPU, engine
installation, model download or external discovery service.

The workspace's ordinary `cargo test --locked --all-targets` builds the sidecar executables through their executable integration targets. A testkit-only command can instead pick up an older binary from the build directory. Always build all four sidecar packages alongside testkit when validating source changes. After that build, you can select a suite or test by name:

```sh
cargo test --locked -p dynamo-sidecar-testkit --test sidecar_mocker_integration
cargo test --locked -p dynamo-sidecar-testkit --test router_sidecar_mocker_integration
```
