<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecar testing

Unit tests live beside the production code they exercise. They construct inputs,
call the real parsing, conversion or state-management functions, and check the
results without starting an inference engine. Common behavior is tested in the
common crate; backend behavior is tested in the vLLM or SGLang crate. There is
no shared unit scenario or backend-adapter layer. Shared CPU integration tests live
in the testkit crate and connect real sidecars to local Mocker servers, which
simulate engine responses without loading a model.

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
└── testkit/
    ├── src/
    │   ├── lib.rs              # Bounded waits and public testkit exports
    │   ├── control.rs          # Per-request controls, observations and controller test
    │   ├── server.rs           # Local server lifetime, including abrupt shutdown
    │   ├── fixtures.rs         # Shared request construction and stream collection
    │   └── assert.rs           # Token, terminal, usage and error assertions
    ├── tests/
    │   ├── sidecar_mocker_integration.rs     # Direct sidecar-to-Mocker scenarios over real gRPC
    │   ├── router_sidecar_mocker_integration.rs   # Sidecar children, discovery, routing and shutdown
    │   ├── kv_events.rs       # Disaggregated rank streams and Worker index recovery
    │   ├── native_engine.rs   # Real vLLM/SGLang compatibility, cancellation and KV transfer
    │   └── support/
    │       ├── mod.rs         # Fixture contracts and scheduler-state waits
    │       ├── vllm.rs        # vLLM protocol, Mocker and child-command adapter
    │       ├── sglang.rs      # SGLang protocol, discovery, health and child-command adapter
    │       └── process.rs     # Local discovery, worker processes and TCP routing
    ├── native_probe.py        # Native NIXL transfer and NVTX observations
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
The CPU files separate direct engine calls, child-process/HTTP lifecycle,
and KV-event recovery, making each setup easier to follow and run independently.
The separate `native_engine` target uses a Cargo feature to keep GPU execution
out of ordinary CPU test runs while allowing a compile-only check.

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

From the repository root, run all common, vLLM and SGLang library tests, including
their local-server tests and the testkit controller regression:

```sh
cargo test --locked -p dynamo-sidecar-common -p dynamo-vllm-sidecar \
  -p dynamo-sglang-sidecar -p dynamo-sidecar-testkit --lib
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

- `sidecar_mocker_integration.rs` uses a backend fixture to call the production sidecar engine
  library, which sends native gRPC requests to a CPU Mocker.
- `router_sidecar_mocker_integration.rs` uses local discovery to find sidecar child processes and
  sends requests to them over TCP. Each sidecar calls a CPU Mocker over native
  gRPC. HTTP scenarios start the production frontend service; handoff scenarios
  use the production PrefillRouter.
- `kv_events.rs` publishes native MessagePack/ZMQ events for two ranks each of
  prefill and decode. It checks event-plane relay, Worker snapshots and buffered
  eviction recovery, plus the router indexer's interpretation of those events.
- `native_engine.rs` calls the production sidecar engine library against real
  vLLM or SGLang engines over native gRPC. The pytest launcher starts those engines
  and the Rust test executable. Single-GPU cases share a device; two-GPU handoff
  variants assign one physical GPU to each engine.

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
| `sidecar_mocker_integration.rs` | Shared streaming, errors, cancellation, cleanup, active work release, consumer drop, request/logprob fields and peer teardown for vLLM and SGLang; native rejection and malformed response checks; SGLang HTTP discovery, opaque payloads, cancellation and isolation | CPU, ordinary pre-merge Cargo tests |
| `router_sidecar_mocker_integration.rs` | Both backends: discovery, readiness, failures, public HTTP disconnects, migration, engine restart, templates, tracing, JSONL, health/live, SIGINT/SIGTERM and PrefillRouter handoff; SGLang opaque HTTP and discovery specifics | CPU, ordinary pre-merge Cargo tests |
| `kv_events.rs` | Shared two-rank prefill/decode KV relay, prefix matches, snapshots, selective eviction and per-source recovery cursors | CPU, ordinary pre-merge Cargo tests |
| `native_engine.rs` | Real logprobs, structured output, cancellation/drop, transfer cleanup and LoRA; vLLM controls, raw/preprocessed images, encoder handoff and NVTX emission; SGLang native HTTP and managed launch | GPU, post-merge and nightly via pytest |

A generic scenario is reusable code, not evidence that every backend runs it.
Both vLLM and SGLang register the shared wire and process scenarios.
TensorRT-LLM is not enrolled here.

The native tests retain their own purpose: a Mocker cannot prove that a real engine
accepts the serialized request, executes a structured-output constraint, releases
its real scheduler work, or transfers GPU KV cache. CPU handoff checks
opaque vLLM metadata and SGLang concurrent bootstrap coordination. Native handoff
separately requires completed transfer bytes and checks decode output length and token usage.
SGLang uses its native transfer metrics; vLLM uses the NIXL probe.
vLLM cancels decode before submission and requires completed transfer, idle
schedulers and a fresh successful handoff.
Both backends also cancel after the native NIXL transfer has been accepted while
engine completion is held by the probe, then require scheduler cleanup and a
fresh successful handoff. NIXL may already have finished copying the bytes;
this does not establish cancellation while packets are physically in flight.
CPU handoff cancellation holds the peers before Mocker admission; it proves
transport cleanup and recovery, not native scheduler or transfer cleanup.

The encoder cases use the existing ECExampleConnector shared-file transport.
They require a real encoder tensor to be saved and loaded by the consumer;
E+P+D separately requires NIXL transfer of the resulting KV cache. The image
case compares raw and preprocessed-media generation. Its NVTX check uses
`VLLM_USE_V2_MODEL_RUNNER=0`: vLLM 0.30's default V2 runner with
`--enforce-eager` accepted the tracing flag but emitted no layerwise ranges.
The observer calls the real NVTX API; it does not capture an exported profiler trace.

Existing tests in `lib/mocker/servers/{vllm,sglang}/tests/sidecar.rs` retain distinct
KV-event and handoff coverage. Backend-local socket tests in `vllm/src/tests.rs`
retain broader media, LoRA, administrative and connection behavior. Python
serving and fault-tolerance tests remain in place: passing this testkit does not
establish complete parity with the legacy Python backend.

### Relationship to serving E2E tests

`tests/serve/test_sidecar.py` starts the frontend, production sidecar executable
and real engines. It checks HTTP serving, distinct prefill/decode workers and
KV-aware routing. A two-GPU headless case adds two logical localhost nodes,
with a native vLLM follower and one Rust sidecar on the leader; it does not prove
physical multi-host networking. The native integration suite calls the Rust engine adapter
directly, so it does not replace those deployment checks.

The native suite adds detailed assertions beyond the existing sidecar E2E tests:
token/logprob correspondence and structured JSON output, scheduler cleanup and
recovery after explicit cancellation or consumer drop, and completed NIXL
transfer bytes. Successful handoff overlaps with E2E split serving, but the
native test also inspects the handoff metadata and decode output contract.
Testing the same container does not make these assertions equivalent.

The legacy Python backend suite is also distributed by behavior, including
`tests/serve/test_vllm.py`, `tests/fault_tolerance/cancellation/test_vllm.py` and
`tests/fault_tolerance/migration/test_vllm.py`.

### Adding an integration test

1. Choose the boundary being protected. Parsing and state transitions without
   I/O belong beside production code. Direct native RPC behavior belongs in
   `sidecar_mocker_integration.rs`; Worker/discovery or process lifetime belongs in
   `router_sidecar_mocker_integration.rs`. Use `native_engine.rs` only when the assertion requires
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
  cargo test --locked -p dynamo-vllm-sidecar -p dynamo-sglang-sidecar \
    -p dynamo-sidecar-testkit
```

Use normal test parallelism. These suites need a Linux host with the repository's
Rust build prerequisites, permission to bind loopback sockets and spawn child
processes, and writable temporary storage. Running them needs no GPU, engine
installation, model download or external discovery service.

The workspace's ordinary `cargo test --locked --all-targets` builds both sidecar
executables through their executable integration targets. A testkit-only command
can instead pick up an older binary from the build directory. Always build the
vLLM and SGLang packages alongside testkit when validating source changes. After that build,
you can select a suite or test by name:

```sh
cargo test --locked -p dynamo-sidecar-testkit --test sidecar_mocker_integration
cargo test --locked -p dynamo-sidecar-testkit --test router_sidecar_mocker_integration
cargo test --locked -p dynamo-sidecar-testkit --test kv_events
```

### Running native GPU integration tests

The GPU suite runs after sidecar E2E in the same one- or two-GPU sidecar test container
for each backend, in post-merge and nightly. The two suites have separate pytest
steps, timeouts and results, and share the image, GPU assignment and pytest setup.
Legacy backend jobs remain separate. The suite uses the `predownload_models`
fixture to prepare `Qwen/Qwen3-0.6B` and, for LoRA cases,
`codelion/Qwen3-0.6B-accuracy-recovery-lora` before starting an engine. The launcher checks
that the Python vLLM package and bundled `vllm-rs` versions agree, or that SGLang
matches its pinned version. vLLM has eight single-GPU cases and SGLang has six;
each also has a two-GPU handoff variant. The vLLM image and encoder cases need
`Qwen/Qwen3-VL-2B-Instruct`. Encoder/consumer pairs and encoder/prefill/decode
triples share one physical GPU while retaining independent native engines.
Single-GPU handoff starts two independent engines on the same assigned device,
with separate caches and ports; only the two-GPU variant tests distinct devices.
CI runs the cases sequentially in the dedicated native step after E2E finishes.

Build the native Rust test executable on the same platform as the test image:

```sh
cargo test --locked -p dynamo-sidecar-testkit --features native-tests \
  --test native_engine --no-run --message-format=json > /tmp/sidecar-native-build.jsonl
export DYNAMO_SIDECAR_NATIVE_TEST="$(jq -r \
  'select(.reason == "compiler-artifact" and .profile.test == true and .target.name == "native_engine") | .executable // empty' \
  /tmp/sidecar-native-build.jsonl)"
python3 -m pytest tests/sidecar/test_native_integration.py -m 'sglang and gpu_1' -v
```

Select `-m 'vllm and gpu_1'` in the vLLM image. Select `gpu_2` instead to run
distinct-device handoff with two visible GPUs. Set `CUDA_VISIBLE_DEVICES` to
select the allocated devices. Set
`SIDECAR_NATIVE_MODEL_PATH` to an existing local model directory when needed.
For offline runs, also pass
`--models-dir /path/to/hf_cache` with a populated cache to skip downloads.
LoRA cases require the adapter snapshot to be cached as well; they use the real
trained adapter to verify that selection changes native token scores. vLLM loads
and unloads it through the sidecar control API; SGLang preloads it at startup.
CI builds/uploads the executable in `shared-build-sidecar-tests.yml`, and the
sidecar jobs download it alongside the production sidecar binary.
This CPU build produces a Rust test executable, not a separate runtime image. The
`native-tests` Cargo feature enables this explicit GPU target. Pre-merge compiles
it with `--no-run` on CPU alongside the ordinary CPU test execution; only
post-merge and nightly execute it against real engines. A pre-merge CPU pass
does not establish native GPU behavior.

Both `post-merge-ci.yml` and `nightly-ci.yml` run the native step through
`shared-test.yml` in the existing one- and two-GPU vLLM and SGLang sidecar jobs. Each
backend depends on its own image build. The E2E selection excludes `sidecar_native`;
the following native step selects only those tests
and runs them sequentially with retries disabled. Both steps contribute to the
job's final result. The native step selects the GPU marker matching its runner.

The launcher in `tests/sidecar/test_native_integration.py` starts and cleans up
its engine processes inside the existing container after the E2E step finishes.
There is no additional native GPU job or test-container startup.

### Kubernetes lifecycle

`tests/deploy/test_sidecar_lifecycle.py` uses the existing deployment harness and
the shipped aggregate manifests (currently vLLM 0.28.0 and SGLang 0.5.17).
It checks native init-container readiness,
engine-only restart with the Rust sidecar still alive, recovered inference, and
owned-pod deletion. It requires a configured cluster and accessible images:

```sh
python3 -m pytest tests/deploy/test_sidecar_lifecycle.py \
  --namespace sidecar-tests --image YOUR_SIDECAR_IMAGE \
  --frontend-image YOUR_FRONTEND_IMAGE -v
```

These cases are excluded from ordinary sidecar GPU lanes. Collection alone does
not establish Kubernetes lifecycle behavior or absence of node-level orphan processes.
