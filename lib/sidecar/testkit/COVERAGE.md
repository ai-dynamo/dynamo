<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Coverage and execution ledger

All five tabs of the sidecar testing DEP remain read-only. The user-approved
stack is [#14879](https://github.com/ai-dynamo/dynamo/pull/14879) (DIS-2941,
shared CPU foundation), [#15089](https://github.com/ai-dynamo/dynamo/pull/15089)
(DIS-2942, isolated units), and
[#15091](https://github.com/ai-dynamo/dynamo/pull/15091) (DIS-2943, additional
vLLM integration). #15088 is superseded. The foundation is refreshed onto main
`4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`; pins remain vLLM 0.29.0 and
`vllm-proto` 0.3.0. Shared Rust dependencies follow the refreshed base.

## Coverage at the foundation and unit boundaries

The foundation's four families remain instantiated for both vLLM and SGLang.
New backend units are vLLM-only; common production tests run once. Counts below
describe retained scope; refreshed execution and historical evidence are
separated below.

| Requirement / retained assertions | Owning layer | Disposition at #15089 |
| --- | --- | --- |
| R09/C4 tokens, terminal placement, length reason, usage and ignored post-terminal replay | #14879 shared streaming scenario in [conformance.rs](tests/conformance.rs) | Preserve both backend enrollments; retain nondefault model/connection setup and native observations. |
| R11/C6/C7 opening failure, premature EOF, stream read error, exact delivered prefix and remote release | #14879 shared failure scenario | Preserve vLLM `Unknown` versus SGLang `EngineShutdown` EOF expectations and typed injected errors. |
| R12/R14/C8 cancellation before native submission, while opening and during read; request A cancellation leaves B live | #14879 shared cancellation scenario | Preserve both backend enrollments, independent request controls, cancelled terminal/usage and successful completion of B. |
| R13 active cleanup, cancelled terminal/usage, remote route release and repeated cleanup | #14879 shared cleanup scenario; vLLM isolated worker units | Move only vLLM's no-I/O before-start/idempotent-cleanup subsection after replacement validation. Preserve SGLang's original subsection and both active-stream checks. |
| R01–R32 endpoint/configuration, conversion, validation and state transitions | #15089 [isolated units](UNITS.md) | Full matrix and original-to-new assertion mapping retained; no new SGLang/TRT units. |
| Native request mapping, gRPC rank metadata, discovery/health, connection pool, decode cancellation, opaque handoff, media/Encode, LoRA and RL | Existing [vLLM library tests](../vllm/src/tests.rs) | Preserve transport/runtime assertions. Pure definitions and exact request-field assertions move only to their mapped isolated replacements. |
| vLLM Mocker streaming/logprobs/usage, opaque prefill/decode, explicit cancellation of active scheduler work, KV relay/indexer | Existing [vLLM Mocker suite](../../mocker/servers/vllm/tests/sidecar.rs) | Retain all four cases at this boundary; replacement deletions belong to #15091. |
| SGLang Mocker incremental tokens/logprobs/usage, opaque prefill/decode, Abort release, two-request cancellation isolation and shutdown | Existing [SGLang Mocker suite](../../mocker/servers/sglang/tests/sidecar.rs) | Retain all four cases and their distinct assertions. |
| Existing SGLang/TensorRT-LLM production tests and E2E | Existing owning suites/workflows | Preserve coverage and scheduling; no new backend fixture migration or activation. |

## Additional integration owned by #15091

These obligations remain explicit upcoming work at this unit boundary. They are
not credited as new #14879 scenarios or completed #15089 coverage.

| Obligation | Intended owner and preservation constraint |
| --- | --- |
| Exact native text, selected/top/prompt logprob values and nondefault request fields | vLLM wire additions, reusing the shared foundation wherever contracts match. Preserve distinct existing streaming assertions before any replacement. |
| Native oversized-output rejection, malformed terminal, recovery on the same engine | vLLM native adapter and wire scenarios; typed errors, exact prefix and no false successful terminal remain required. |
| Explicit cancellation and pure consumer drop while Mocker scheduler work is active | CPU wire integration; observe active work before interruption and release afterward. A paused response alone is insufficient. |
| Post-cleanup admission and peer termination with live clients | Wire lifecycle additions and bounded server ownership; do not infer detached-handler cleanup from dropping the outer server task. |
| Real executable startup, discovery/metadata, Worker ingress, shutdown and PrefillRouter handoff | CPU process integration through production owners; shared scenario code where contracts match, new fixtures instantiated only for vLLM. |
| Actual engine-work release, native compatibility, KV transfer and transfer-time cancellation | Direct native integration. CPU Mocker output and opaque handoff forwarding cannot establish real engine release or KV transfer. |

The five potential wire replacements are
`grpc_request_errors_are_propagated`, `cancellation_drops_the_remote_stream`,
`cancellation_interrupts_pending_response_headers`,
`sidecar_streams_mocker_tokens_logprobs_and_usage`, and
`dropping_sidecar_stream_cancels_mocker_work` (which performs explicit stop).
They remain at #15089. #15091 must map and validate their distinct assertions
before removing them; moving this work between PRs is not a coverage reduction.

## Validation and historical evidence

On 2026-09-22, foundation `286d6fd5` collected and passed all eight conformance
cases and executed all eight retained Mocker cases, with GPUs hidden. Targeted
Clippy, formatting and commit hooks passed. The restacked unit boundary passed
62 isolated cases in a network-isolated CPU container, all eight conformance
cases and all 102 common/vLLM library cases; overlapping selections are not
added together. See [UNITS.md](UNITS.md) for commands and accounting. Refreshed
trusted CI is pending. Run the retained wire boundary independently of the
isolated unit runner:

```sh
cargo metadata --locked --format-version 1 >/dev/null
cargo test --locked -p dynamo-sidecar-testkit --test conformance -- --list
cargo test --locked -p dynamo-sidecar-testkit --test conformance
cargo test --locked -p dynamo-vllm-mocker --test sidecar -- --list
cargo test --locked -p dynamo-vllm-mocker --test sidecar
cargo test --locked -p dynamo-sglang-mocker --test sidecar -- --list
cargo test --locked -p dynamo-sglang-mocker --test sidecar
cargo clippy --locked -p dynamo-sidecar-testkit --all-targets --no-deps -- -D warnings
```

Use Rust 1.96.1, protoc 30.2, matching `PROTOC`/`PROTOC_INCLUDE`, and an external
`CARGO_TARGET_DIR`. CPU execution may set `CUDA_VISIBLE_DEVICES=` and
`NVIDIA_VISIBLE_DEVICES=void`; the suites need no engine, model download or
external discovery service. Those environment settings alone do not prove
network-isolated container execution. Unit commands and their historical
failure/mutation evidence are recorded in [UNITS.md](UNITS.md).

Historical #14879 at `d486872d103b3229101773189f76df9d1c1ec1fc` reported eight
shared conformance and eight retained Mocker cases passing. The superseded
#15088 at `73192d69351b51763a018edc75b7ec275d9fa68a` passed nine vLLM conformance
and two remaining vLLM Mocker cases in an isolated CPU container, plus 60 vLLM
library cases. Those histories have different scope and bases; neither is the
new foundation's collection or execution result. Historical #15089's final
62 isolated cases and broader 110-case run are detailed in UNITS.md.

The pinned native handoff failure, blocked transfer-time cancellation and
unexecuted two-GPU success topology remain obligations of #15091. No native
success follows from the refreshed CPU foundation or unit tests.
