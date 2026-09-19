<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Coverage and execution ledger

Source: all five tabs of the read-only sidecar testing DEP. The rollout uses
Dynamo base `cdcd721e72fdfd521c93f35c7f91745b1f7dd01f`, vLLM 0.29.0 and
`vllm-proto` 0.3.0. Reference #14879 was inspected at
`d486872d103b3229101773189f76df9d1c1ec1fc`; this stack neither modifies it nor
uses its branch as a base. DIS-2941 owns this wire increment; DIS-2942 owns units;
DIS-2943 owns process and native compatibility integration.

## Wire coverage (DIS-2941)

| DEP IDs / prior assertions | Owning boundary and new case | Disposition / execution |
| --- | --- | --- |
| R09/C4; old `sidecar_streams_mocker_tokens_logprobs_and_usage` | CPU wire `streaming::vllm_tokens_terminal_logprobs_and_usage` | Adds full native token/text/selected and alternative logprob values, prompt logprob values, exact terminal and all usage fields, nondefault request fields, and ignored post-terminal replay. Passed in the 9-case conformance run (see evidence below). |
| R11/C6/C7; old `grpc_request_errors_are_propagated` | CPU wire `errors::vllm_open_failure_early_eof_and_read_failure` | Typed open/read Unavailable, early EOF Unknown, exact emitted prefix, no false success, remote drop, subsequent healthy request on same engine. Passed in the 9-case conformance run (see evidence below). |
| C6 native rejection | CPU wire `errors::vllm_native_rejection_recovers_on_same_engine` | Real Mocker admission rejects oversized output; InvalidArgument and exact native reason survive; following request succeeds. Passed in the 9-case conformance run (see evidence below). |
| C7 malformed native terminal | CPU wire `errors::vllm_malformed_terminal_fails_then_recovers` | Scripted token then invalid native enum; exact prefix, protocol Unknown, no successful completion, recovery. Passed in the 9-case conformance run (see evidence below). |
| R12/R14/C8; old `cancellation_interrupts_pending_response_headers`, `cancellation_drops_the_remote_stream` | CPU wire `cancellation::vllm_cancellation_before_open_during_open_and_during_read` | Before-open no submission, held headers drop, independently paused concurrent streams on a two-connection pool; cancellation A leaves B pending and B completes; same-engine healthy follow-up. Passed in the 9-case conformance run (see evidence below). |
| C8; old `dropping_sidecar_stream_cancels_mocker_work` (actually explicit stop) | CPU wire `cancellation::vllm_explicit_cancel_releases_active_scheduler_work` | Observe nonterminal output and active scheduler first; exactly cancelled terminal with partial usage, route release and zero running/waiting scheduler work, healthy follow-up. Passed in the 9-case conformance run (see evidence below). |
| C9 | CPU wire `cancellation::vllm_consumer_drop_releases_active_scheduler_work` | Drop live consumer without stop or Abort; observe active scheduler first, then remote drop and zero running/waiting scheduler work, healthy follow-up. Passed in the 9-case conformance run (see evidence below). |
| R13 direct cleanup | CPU wire `lifecycle::vllm_cleanup_during_read_and_post_cleanup_admission` | Live-stream cancellation and post-cleanup generation cannot submit native work; PR2 moves before-start error and repeatable cleanup into isolated coverage. Passed in the 9-case conformance run (see evidence below). |
| C7 peer termination; framework bounded teardown | CPU wire `lifecycle::vllm_teardown_terminates_handlers_with_clients_alive` | Abruptly terminate dedicated native server runtime while real engine and stream objects remain alive; join runtime, observe handler and scheduler release, exact prefix plus typed failure. Passed in the 9-case conformance run (see evidence below). |
| Native request cache namespace, structured outputs, metadata/KV sources, engine config; connection reuse; opaque prefill/decode; EPD; RL/LoRA | Existing vLLM socket tests | Retained. Broad old aggregate test has distinct assertions beyond shared C4. |
| Mocker prefill/decode opaque handoff and KV relay/indexer | Existing `lib/mocker/servers/vllm/tests/sidecar.rs` | Retained until PR3 demonstrates equivalent replacements. |
| SGLang/TensorRT-LLM fixtures, migration, registrations | Existing owning suites only | New rollout deferred by current user instruction. Existing source and CI unaffected. |
| Actual engine compute release / KV transfer | PR3 explicit native target | Cannot be established by CPU Mocker scheduler or fake handoff. |
| Existing E2E allocation | Root Python suites/workflows | Unchanged per current rollout. |

## Execution evidence

The working-tree conformance run collected and executed 9 cases: 9 passed,
0 failed, 0 ignored, 0 filtered, in 1.81 seconds. The retained Mocker handoff and
KV-relay/indexer suite executed 2 cases: both passed in 0.68 seconds.

```sh
cargo test --locked -p dynamo-sidecar-testkit --test conformance
cargo test --locked -p dynamo-vllm-mocker --test sidecar
cargo clippy --locked -p dynamo-sidecar-testkit --test conformance --no-deps -- -D warnings
```

Clippy passed. Host builds used Rust 1.96.1, protoc 30.2 and an external Cargo
cache. These initial runs include coordinated uncommitted changes from the stack;
per-commit isolated validation is recorded below when executed. Dependency
inspection found only standard C/C++ runtime libraries, not CUDA/NIXL. It does
not substitute for isolated-container execution. Required CI for the final
stack heads is not yet complete.

The first run passed 8/9. The failing assertion expected `CannotConnect` after
abrupt HTTP/2 shutdown; native transport actually reports `Unknown`. The corrected
harness still requires the exact prefix, one typed error, RPC context, no success
terminal, handler drop and scheduler release. No production fix was needed.

## Replacement accounting

After replacement execution, remove only these five old definitions:
`grpc_request_errors_are_propagated`, `cancellation_drops_the_remote_stream`,
`cancellation_interrupts_pending_response_headers`,
`sidecar_streams_mocker_tokens_logprobs_and_usage`, and
`dropping_sidecar_stream_cancels_mocker_work` (which actually performed explicit
stop). The mapping above retains their assertions, including one-token chunks,
three top candidates, prompt metadata, exact prefix and cancelled terminal.
Richer aggregate mapping, native metadata, decode cancellation, handoff,
multimodal, LoRA, RL and KV-relay cases remain. Pure unit relocation is a separate
increment and is not counted as coverage deletion.

### Independent PR1 snapshot

The snapshot contains only the wire increment on `cdcd721e` (no unit, process,
native or production fixes from later PRs). Its retained vLLM library suite
collected and executed 60 cases: 60 passed, zero ignored, in 0.89 seconds.
The exported conformance (9) and retained Mocker (2) cases also passed inside
the CPU Dockerfile image with `docker run --rm --network none`; no model cache,
GPU device or host network was mounted. Execution took 1.80 and 0.67 seconds.

The exporter initially included Cargo's non-test helper executable as a duplicate
entry. Target-name and test-profile selection fixes that harness defect; exported
manifests reject missing or duplicate targets. No production behavior changed.
