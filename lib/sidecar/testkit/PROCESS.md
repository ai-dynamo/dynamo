<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM process and handoff integration

The final CPU process suite collected and executed six tests successfully:
**6 passed, 0 failed, 0 ignored**, in 22.47 seconds. This execution used the working changes
on base `cdcd721e72fdfd521c93f35c7f91745b1f7dd01f`, native protocol 0.3.0,
and the actual rebuilt `dynamo-vllm-sidecar` executable. Pinned vLLM is 0.29.0;
these CPU results do not establish real-engine compatibility, GPU work release,
or KV transfer. Native-engine validation is reported separately in [NATIVE.md](NATIVE.md):
cancellation passed, while actual KV transfer hit a pinned-engine blocker.

## Production boundaries

`tests/support/process.rs` owns child process groups, bounded exit and kill/reap
fallback, captured logs, temporary model/tokenizer metadata, a unique discovery
namespace, and dynamic ports. It uses production file discovery, TCP request
transport and ZMQ events without external NATS, etcd, model downloads or GPUs.
A controlled TCP gate observes real connection attempts before allowing native
startup. Each scenario has a 60-second outer deadline and bounded phase waits.

The native peers reuse the wire suite's fixture and request controls. Tests call
the published Worker endpoint. Handoff tests instantiate the real Dynamo
`PrefillRouter`, which discovers and dispatches to two actual sidecar processes.
They bypass the HTTP frontend because HTTP parsing and text generation belong
to the preserved E2E suite.

The five common startup, registration, ingress, cancellation and shutdown
scenario bodies are generic over `ProcessFixture`, which extends the wire
suite's `SidecarFixture`. The vLLM process profile supplies the executable,
native endpoint, supported request options and exact discovered capacity
assertions. Common lifecycle assertions remain in reusable scenario bodies.
Only the handoff case inspects vLLM protobuf fields directly: its native
prefill/decode metadata is a backend-specific contract.

All new fixtures instantiate vLLM only. Shared runtime production regressions
run once in the runtime crate. New SGLang and TensorRT-LLM activation is deferred;
existing tests and E2E allocation remain unchanged.

## Coverage and ownership

| Collected test | DEP requirement | Observable guarantee |
| --- | --- | --- |
| `vllm_registration_and_errors_recover_through_worker_ingress` | C1/C4/C5/C6/C7 | Exact model, role, context, DP and capacity metadata; tokenizer/formatter publication; absent unsupported parser names; nondefault sampling and logprob options survive real Worker ingress with exact native request, output and usage. Unsupported `n=2` yields a typed invalid-argument cause without native submission. Native opening failure, midstream failure and missing terminal preserve their typed error and exact prefix; the same child serves a valid request after each failure. |
| `vllm_delayed_startup_publishes_only_after_native_readiness` | C2 | An observed native connection attempt remains gated while discovery has no model or serving endpoint. Releasing the gate allows readiness and successful endpoint generation. Uses the environment endpoint option; other scenarios use CLI wiring. |
| `vllm_failed_and_interrupted_startup_leave_no_registration` | C3/C10 | Refused connection, accepted connection without RPC response, and SIGTERM during startup all exit within bounds without publishing a usable endpoint. |
| `vllm_worker_cancel_and_consumer_drop_release_only_the_target` | C8/C9 | Explicit cancellation during native opening or after a token reaches the correct remote request. Pure consumer stream drop releases remote work without the caller explicitly cancelling. An independently accepted request continues with exact tokens/usage, and the next request succeeds. Active Mocker scheduler work is observed before cancellation and drains afterward. |
| `vllm_sigterm_withdraws_worker_and_releases_active_native_request` | C10 | Actual endpoint withdrawal, router exclusion and refusal of new requests precede native cleanup and process exit during the configured grace period. An active native request is released, scheduler work drains, the child exits successfully, and the independent native peer remains usable. |
| `vllm_prefill_router_preserves_handoff_failure_and_cancellation` | C11/C12 CPU | The real PrefillRouter uses advertised prefill/decode roles; preserves correlated opaque handoff fields and normalizes `remote_port`; returns decode tokens/usage without leaking prefill tokens; prevents decode submission after failed prefill; cancels after both peers accept but before decode produces its first token; drains both peers and serves a following handoff. |

Existing conversion/configuration assertions remain in their unit owners. Native
connection-loss and malformed-terminal faults remain in the CPU wire suite.
The existing manually chained vLLM Mocker handoff and distinct vLLM adapter
handoff assertions are retained. Shared Worker lifecycle and adapter semantics
remain in `dynamo-backend-common`, credited once. [COVERAGE.md](COVERAGE.md)
contains the broader preservation mapping; [DEVIATIONS.md](DEVIATIONS.md) records
changes from the read-only DEP.

## Failures reproduced and corrected

| Failure | Classification | Correction and regression |
| --- | --- | --- |
| Cancelling while native response headers were held timed out; independently reproduced against the frozen pre-fix binary in 10.35 seconds | Production defect | TCP response setup waited for the response prologue before forwarding cancellation. The runtime now observes Stop, Kill and provider drop during that wait, forwards the correct control and releases setup. `test_response_stream_cancellation_before_prologue` checks all three controls, including that Stop never becomes Kill. The process cancellation scenario proves native handler release and independent-request survival. |
| The first correction delivered cancellation but the caller received CannotConnect, temporarily excluding the healthy worker from routing | Production defect | The addressed router now returns an empty stream with the original cancelled context when the local response provider fails. Existing remote migration-error classification stays unchanged. Both process cancellation and cancelled-handoff recovery exercise this correction. |
| PR3's initial typed Cancelled setup error failed four existing Python cancellation cases on both CI architectures | Stack regression | Existing callers expect `generate()` to return an empty stream after local cancellation. The corrected adapter preserves that contract; both process scenarios now require successful stream setup instead of accepting an exception. The existing Python assertions remain unchanged. |
| Shutdown waited for every persistent model card to disappear | Harness defect | The actual Worker contract withdraws serving endpoints. The test now reads authoritative discovery, observes exclusion in the existing router, rejects new requests and verifies withdrawal precedes cleanup/exit. It does not infer worker liveness from persistent metadata. |
| A setup error was expected only inside an already-open stream | Harness defect | Accept the real pre-stream error return and assert its exact semantic type through the runtime's existing cause chain. Stream failures retain exact prefix and typed terminal assertions. |
| Immediate retry selected no worker after an injected connection failure | Harness defect | Await the router's actual availability through its existing five-second inhibition period; preserve fault detection and assert successful reuse. |
| DEP parser metadata and startup assumptions did not match the chosen base | Refreshed support evidence | vLLM now uses Control metadata and health discovery. Parser flags remain unsupported and are rejected before discovery by retained isolated tests; process metadata must not advertise them. |

The initial full execution reported 1 passed/5 failed; after correcting the
first harness defects and TCP forwarding, it reported 3 passed/3 failed. No
failed assertion was converted to a skip. The final run above includes the
necessary production fixes and all six enabled scenarios.

## Validation commands and remaining work

From the repository root, with the Rust/protobuf build prerequisites installed:

```bash
cargo build --locked -p dynamo-vllm-sidecar --bin dynamo-vllm-sidecar
cargo test --locked -p dynamo-sidecar-testkit --features process-tests --test cross_process -- --list
cargo test --locked -p dynamo-sidecar-testkit --features process-tests --test cross_process -- --test-threads=1 --nocapture
cargo clippy --locked -p dynamo-sidecar-testkit --features process-tests --test cross_process -- -D warnings
```

The fixture finds the sidecar beside the test target's `debug` directory.
Set `DYNAMO_VLLM_SIDECAR` to an absolute executable path when using separate
build directories. Use the same `CARGO_TARGET_DIR` and protobuf compiler for
building the binary and tests.

| Check | Executed result |
| --- | --- |
| Final actual sidecar build | Passed after the cancellation compatibility correction; 8.85-second incremental build |
| Process target compilation and execution | Corrected source compiled; all 6 cases collected/executed in the isolated CPU container, 0 ignored; 22.48-second execution |
| Focused runtime cancellation regression | Passed; 1 test covers Stop/Kill/provider drop, 0 ignored; 0.01 seconds |
| Broader TCP server tests | 45 collected; initial parallel execution 40 passed/5 failed due to existing TLS tests mutating process environment concurrently; immediate serial execution 45 passed/0 failed/0 ignored in 0.26 seconds. This used the TCP fix before the additional addressed-router guard. |
| Process source formatting and whitespace | Passed |
| Targeted process Clippy | Corrected cancellation source passed with `-D warnings`; 1 minute 40 seconds |
| Isolated CPU container | Initial implementation: 54 collected/executed (9 conformance, 2 Mocker, 2 common transport, 35 retained vLLM socket, 6 process), all passed. After the cancellation contract correction: all 6 strengthened process cases passed again in 22.48 seconds. Both containers disabled external networking; zero ignored. |
| Current-head CI | Pending; tracked by the stack's validation report |
| Pinned native-engine compatibility/cancellation/handoff | Cancellation passed; native handoff executed and failed due to the upstream float conversion described in NATIVE.md. CPU Mocker evidence is not credited as native transfer. |

The `cross_process` target requires the `process-tests` feature for the
framework's post-merge CPU lane. Default CPU unit/wire tests remain pre-merge;
existing E2E tests and GPU lanes are preserved. An intended lane or a collected
test is not execution evidence; pending validation is not complete.

## Executed source fingerprints

SHA-256 values below identify the corrected process-suite container execution. Paths
are relative to the repository root.

```text
87177d30aee8ee69aec64daf85324870d6082b00dee0665f4330867019800e53  lib/sidecar/testkit/tests/cross_process.rs
a09e20217dba7583626b4e0702b396e2b9d9508619ef99645923f0b3805883e1  lib/sidecar/testkit/tests/process/cancellation.rs
bd2408a8936d8e8583a3735b273081bfb0bf12e91718d360e8e71a2a3f14bdef  lib/sidecar/testkit/tests/process/handoff.rs
57c429a8b156e251c93edf9aac99c581bd4b983d395616a31343b01494c8e962  lib/sidecar/testkit/tests/process/lifecycle.rs
3585d65ef42242a9734d0363308ae81079a95a36a5b4411c1c8db26558ef896b  lib/sidecar/testkit/tests/process/vllm.rs
a59f2306bf505425958bc7352dcb80e4ea555075cf030a835eb6dc971a73d701  lib/sidecar/testkit/tests/support/mod.rs
429494ab4610aea74a0848f61f0009549590115e83ae1fab0c596d7bac640de0  lib/sidecar/testkit/tests/support/process.rs
13cae2df5700e15832ee48f11263da1d55f386c15d33670d993eb15ec2e9940b  lib/sidecar/testkit/tests/support/vllm.rs
b43c87b0e5d9377f0fd7acc866ab2e69a6617b9c2048e0f9f6881aefe2a628a0  lib/runtime/src/pipeline/network/tcp/server.rs
7094b8642cb38f98535884ddffe0eccb08d14d9c7711cc54a932e409173381f4  lib/runtime/src/pipeline/network/egress/addressed_router.rs
```

Executed artifacts:

```text
7a24d288a1455f8f992de3cb8814ebe4c0d070554866795271dae8c8ab545ef0  dynamo-vllm-sidecar
00c79f5557eeb38e89e21480834c6421d421c34836e717a8c6e69d5b04b842bd  dynamo-sidecar-testkit-cross_process
```
