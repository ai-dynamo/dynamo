<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecars: Agent Guide

A sidecar is a small Rust process that runs next to an inference engine. The
engine (vLLM, SGLang, TensorRT-LLM or LPX) runs on its own and exposes a native
gRPC API. The sidecar registers that engine with Dynamo and translates requests
and responses between Dynamo and the engine.

Read the repository [AGENTS.md](../../AGENTS.md) first for build setup. The full
testing reference is [testkit/README.md](testkit/README.md); this file is the
short version.

## Layout

| Path | Contents |
|------|----------|
| `common/` | Code shared by all sidecars: arguments, endpoints, errors, JSON/protobuf conversion, transport retries |
| `vllm/`, `sglang/`, `trtllm/`, `lpx/` | One crate per engine. Most also have `launch/*.sh` scripts and `deploy/` manifests |
| `testkit/` | Shared CPU integration tests that run real sidecars against fake engines |
| `SUPPORT_NOTES.md` | User-visible limits; update it when you add or remove one |

Each engine crate uses the same folders: `src/` for code and unit tests,
`deploy/` for Kubernetes manifests, `launch/` for the scripts that end-to-end
tests start, and `proto/` where the crate vendors protobuf files. We are
investigating removing `tests/`; do not add anything to it. Do not add new
folders.

## Writing sidecar code

- **Let the engine own engine values.** Prefer values the engine reports, such
  as block size, context limit, parsers and model aliases. Some backends still
  accept operator overrides, but do not add new calculations in the sidecar,
  where the copy goes stale when the engine changes. If the engine does not
  expose a value, add it to the engine's API.
- **Shared behavior goes in `common/`; engine behavior stays in its crate.** Do
  not add a shared abstraction for a single backend.
- **Refuse unsupported features clearly.** Return a typed error and note the
  limit in `SUPPORT_NOTES.md`.
- Keep `README.md` files that carry an agent note minimal, as the note says.
  Update `testkit/README.md` when the test layout changes.

## Testing philosophy

- **Test at the lowest level that can prove the behavior.** A unit test is
  cheaper than an integration test, and an integration test is cheaper than
  an end-to-end test.
- **Test production code.** Tests run the real sidecar code; integration tests
  fake only the engine. Production crates never depend on testkit.
- **Keep the backends at parity.** Every backend should support the same
  features and pass the same tests. That is why integration tests are shared:
  one scenario checks the same behavior on every backend enrolled in it.
- **Write a scenario once; run it on every backend.** A shared scenario is an
  async function generic over one fixture trait, such as
  `async fn streaming<F: SidecarFixture>()`. A macro, or a thin per-backend
  `#[tokio::test]` wrapper, runs it as one ordinary Tokio test per backend.
  Put a real backend difference
  in a named fixture value, not in an `if backend` branch, an optional
  callback or a no-op method. A check that only makes sense for one backend is
  an explicit test for that backend.

## Where a new test goes

| Test | What it is | Use it when | Where it lives |
|------|------------|-------------|----------------|
| Unit | Calls a real function directly, usually without an engine or sockets | The behavior needs no I/O: parsing, conversion, state changes | Beside the production code |
| Integration | Runs real sidecar code over real sockets against a **Mocker**, a CPU fake engine that returns tokens and scheduler state without loading a model. No GPU or model download | The behavior is between the sidecar and its engine, or about the sidecar process (startup, discovery, shutdown) | `testkit/tests/` |
| End-to-end | Runs the real frontend, sidecar binary and engine on a GPU | Only a real engine can prove it, such as KV transfer or real scheduler cleanup | `tests/serve/test_sidecar.py`, outside `lib/sidecar` |

### Unit tests

Add the test to the production module's `#[cfg(test)] mod tests`, or to its
existing `*_tests.rs` child module. Child modules reach private functions
through `use super::*`; do not make code public to test it.

### Integration tests

To add a shared testkit scenario:

1. Pick the smallest fixture trait from `testkit/tests/support/mod.rs`:
   `SidecarFixture` for the engine lifecycle, `WireFixture` to observe
   scheduler work, `ProcessFixture` to launch a sidecar child process
   (`DiscoveryFixture` and `HandoffFixture` extend it).
2. Write the scenario as an async function generic over that trait.
3. Enroll it for each backend: in `enroll_baseline!`
   (`sidecar_mocker_integration.rs`) or `migration_tests!`
   (`router_sidecar_mocker_integration.rs`), or with a per-backend
   `#[tokio::test]` wrapper like the others in the router suite.
4. Keep native protocol details in `tests/support/<backend>.rs`, not in the
   scenario.

To add a backend, implement its `Protocol` adapter and fixture in
`tests/support/<backend>.rs`, then enroll it. The existing macros need
`SidecarFixture`, `WireFixture` and `ProcessFixture`.

### End-to-end tests

End-to-end tests live outside `lib/sidecar`, in `tests/serve/test_sidecar.py`.
They use the same config-and-payload format as the other serve tests
(`tests/serve/test_vllm.py` and others). Inside `lib/sidecar`, they use only
the `launch/*.sh` scripts they start.

- **Config.** Each entry in `sidecar_configs` is a `SidecarConfig`, which
  extends `EngineConfig` from `tests/utils/engine_process.py`. It names the
  launch script (`directory`, `script_name`, `script_args`), the `model`, the
  `env`, the pytest `marks` (backend, `gpu_1`/`gpu_2`, `timeout`, `post_merge`,
  KV-cache size), the worker health-check count, and `gpu_env`/`port_env` for
  variables the script reads.
- **Payloads.** `request_payloads` is a list of `BasePayload` objects from
  `tests/utils/payloads.py`. A payload is an HTTP request to the frontend plus
  its expectations: response text, status code, finish reason, token counts.
  Its `bind_environment`, `before_request` and `validate` hooks add checks on
  engine metrics. Build payloads with the helpers in
  `tests/utils/payload_builder.py`, such as `http_cancellation_payloads`,
  `kv_routing_payload` and `engine_failure_payload`.
- **Run.** `run_serve_deployment` starts the script once, sends each payload
  in order, validates it and tears the deployment down.

To add end-to-end coverage:

1. Prefer adding a payload to an existing config's `request_payloads`. Each
   new config costs minutes of engine startup.
2. Reuse a builder in `payload_builder.py`, or add one there if other serve
   tests could use it.
3. Add a new config only for a new deployment shape (a new launch script or
   topology).
4. Send requests through frontend HTTP and check engine side effects with the
   metrics checkers (`VllmMetricsChecker`, `SGLangMetricsChecker`). Direct
   native-RPC clients belong in integration tests.

These tests carry the `sidecar` marker, which PR CI excludes; post-merge and
nightly CI run them. Before writing them, read [tests/AGENTS.md](../../tests/AGENTS.md),
[.ai/pytest-guidelines.md](../../.ai/pytest-guidelines.md) and
[.ai/test-model-size-guardrails.md](../../.ai/test-model-size-guardrails.md);
for launch-script changes, read
[.ai/bash-launch-guidelines.md](../../.ai/bash-launch-guidelines.md).
