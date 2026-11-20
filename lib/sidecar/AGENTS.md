<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Sidecars: Agent Guide

A sidecar is a Rust process that runs next to an inference engine (vLLM,
SGLang, TensorRT-LLM or LPX) and translates between Dynamo and the engine's
native API. Anything that must be specialized to an engine belongs in its
sidecar, so the frontend and router can stay engine-agnostic.

## Layout

| Path | Contents |
|------|----------|
| `common/` | Code shared by all sidecars |
| `vllm/`, `sglang/`, `trtllm/`, `lpx/` | One crate per engine |
| `testkit/` | Shared CPU integration tests: real sidecars against fake engines |
| `SUPPORT_NOTES.md` | User-visible limits; update it when you add or remove one |

Each engine crate uses the same folders: `src/` for code, `deploy/` for
Kubernetes manifests, `launch/` for launch scripts, and `proto/` where the
crate vendors protobuf files. Do not add new folders.

## Writing sidecar code

- **Let the engine own engine values.** Prefer values the engine reports, such
  as block size, context limit, parsers and model aliases. Do not add new
  calculations in the sidecar, where the copy goes stale when the engine
  changes. If the engine does not expose a value, add it to the engine's API.
- **Shared behavior goes in `common/`; engine behavior stays in its crate.**

## Testing

- **Test production code.** Integration tests fake only the engine.
  Production crates never depend on testkit.
- **Keep the backends at parity.** Every backend should support the same
  features and pass the same tests. That is why integration tests are shared:
  one scenario checks the same behavior on every backend enrolled in it.

| Test | What it is | Use it when | Where it lives |
|------|------------|-------------|----------------|
| Unit | Calls a real function directly, without an engine or sockets | The behavior needs no I/O: parsing, conversion, state changes | Beside the production code |
| Integration | Runs real sidecar code over real sockets against a **Mocker**, a CPU fake engine. No GPU or model | The behavior is between the sidecar and its engine, or about the sidecar process (startup, discovery, shutdown) | `testkit/tests/` |
| End-to-end | Runs the real frontend, sidecar and engine on a GPU | Only a real engine can prove it, such as KV transfer | `tests/serve/test_sidecar.py`, outside `lib/sidecar` |

### Unit tests

Add the test to the module's `#[cfg(test)] mod tests` or its existing
`*_tests.rs` child module. Reach private functions through `use super::*`; do
not make code public to test it.

### Integration tests

1. Pick the smallest fixture trait in `testkit/tests/support/mod.rs`:
   `SidecarFixture` for the engine lifecycle, `WireFixture` to observe
   scheduler work, `ProcessFixture` to launch a sidecar process.
2. Write the scenario once, as an async function generic over that trait, such
   as `async fn streaming<F: SidecarFixture>()`.
3. Enroll it for each backend: in `enroll_baseline!`
   (`sidecar_mocker_integration.rs`), or with a per-backend `#[tokio::test]`
   wrapper (`router_sidecar_mocker_integration.rs`).
4. Keep native protocol details in `tests/support/<backend>.rs`. Put a real
   backend difference in a named fixture value, not an `if backend` branch. A
   check that applies to one backend only is an explicit test for it.

To add a backend, implement its `Protocol` adapter and fixture in
`tests/support/<backend>.rs`, then enroll it.

### End-to-end tests

End-to-end tests live outside `lib/sidecar`, in `tests/serve/test_sidecar.py`.
They use the same config-and-payload format as the other serve tests:

- **Config.** Each `sidecar_configs` entry is a `SidecarConfig` (an
  `EngineConfig`) that names a `launch/*.sh` script, the model, the
  environment and the pytest marks.
- **Payloads.** Each config's `request_payloads` are `BasePayload` objects: an
  HTTP request to the frontend and its expected result. Build them with the
  helpers in `tests/utils/payload_builder.py`.

`run_serve_deployment` starts the script once, then sends and validates each
payload in order. Prefer adding a payload to an existing config: each new
config costs minutes of engine startup. Add a config only for a new launch
script or topology. Check engine side effects with the metrics checkers
(`VllmMetricsChecker`, `SGLangMetricsChecker`); direct native-RPC clients
belong in integration tests.
