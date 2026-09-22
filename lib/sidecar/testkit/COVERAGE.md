<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Coverage and execution ledger

The five DEP tabs remain read-only. The user-approved stack is #14879
(DIS-2941: shared vLLM/SGLang CPU foundation), #15089 (DIS-2942: common/vLLM
isolated units), then #15091 (DIS-2943: additional vLLM wire, process and native
integration). #15088 is superseded. The foundation is refreshed onto main
`4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`; vLLM 0.29.0 and protocol 0.3.0 remain pinned.
Updated shared Rust dependencies require validation of each new head.

## Retained foundation and isolated units

#14879 owns four shared families for both vLLM and SGLang: streaming/terminal
usage/replay; opening/EOF/read failure and delivered prefix; cancellation before
submission/during opening/during read with independent-request survival; and
cleanup. #15091's nine vLLM wire scenarios incorporate those four vLLM families,
not an additional duplicate enrollment. Its four SGLang foundation cases remain.

#15089 owns the full R01–R32 mapping and pure assertion relocations in
[UNITS.md](UNITS.md). Shared production units run once; new backend units are
vLLM-only. The no-I/O before-start/idempotent-cleanup subsection moves only for
vLLM; SGLang retains its original subsection. Both backends retain active-stream
cleanup, cancelled terminal/usage and remote release at the wire boundary.

Both Mocker suites retain all four cases through #15089. At the integration
boundary, the two distinct vLLM manual-handoff/KV-relay cases and all four SGLang
Mocker cases remain. SGLang's incremental logprobs/usage, prefill/decode,
Abort release, request isolation and shutdown assertions are not replaced by
vLLM tests. Existing TensorRT-LLM, shared-production and E2E ownership remains.

## vLLM wire coverage owned by #15091

The following maps all distinct assertions from the prior wire increment to
its new integration owner. All nine cases were collected and passed in the
refreshed integration candidate's CPU container, with zero ignored. The source
is based on #15089 `b3ab1638`; no final integration commit is attributed here.
Older vLLM results remain separately labeled as historical.

| DEP IDs / prior assertions | Owning boundary and new case | Disposition / execution |
| --- | --- | --- |
| R09/C4; old `sidecar_streams_mocker_tokens_logprobs_and_usage` | CPU wire `streaming::vllm_tokens_terminal_logprobs_and_usage` | Adds full native token/text/selected and alternative logprob values, prompt logprob values, exact terminal and all usage fields, nondefault request fields, and ignored post-terminal replay. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| R11/C6/C7 | CPU wire `errors::vllm_open_failure_early_eof_and_read_failure` | Typed open/read Unavailable, early EOF Unknown, exact emitted prefix, no false success, remote drop, subsequent healthy request on same engine. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| C6 native rejection; old `grpc_request_errors_are_propagated` | CPU wire `errors::vllm_native_rejection_recovers_on_same_engine` | Real Mocker admission rejects oversized output before a response stream opens; InvalidArgument and exact native reason survive; following request succeeds. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| C7 malformed native terminal | CPU wire `errors::vllm_malformed_terminal_fails_then_recovers` | Scripted token then invalid native enum; exact prefix, protocol Unknown, no successful completion, recovery. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| R12/R14/C8; old `cancellation_interrupts_pending_response_headers`, `cancellation_drops_the_remote_stream` | CPU wire `cancellation::vllm_cancellation_before_open_during_open_and_during_read` | Before-open no submission, held headers drop, independently paused concurrent streams on a two-connection pool; cancellation A leaves B pending and B completes; same-engine healthy follow-up. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| C8; old `dropping_sidecar_stream_cancels_mocker_work` (actually explicit stop) | CPU wire `cancellation::vllm_explicit_cancel_releases_active_scheduler_work` | Observe nonterminal output and active scheduler first; exactly cancelled terminal with partial usage, route release and zero running/waiting scheduler work, healthy follow-up. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| C9 | CPU wire `cancellation::vllm_consumer_drop_releases_active_scheduler_work` | Drop live consumer without stop or Abort; observe active scheduler first, then remote drop and zero running/waiting scheduler work, healthy follow-up. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| R13 direct cleanup | CPU wire `lifecycle::vllm_cleanup_during_read_and_post_cleanup_admission` | Live-stream cancellation and post-cleanup generation cannot submit native work; #15089 moves only vLLM before-start error and repeatable cleanup into isolated coverage; SGLang retains its foundation checks. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| C7 peer termination; framework bounded teardown | CPU wire `lifecycle::vllm_teardown_terminates_handlers_with_clients_alive` | Abruptly terminate dedicated native server runtime while real engine and stream objects remain alive; join runtime, observe handler and scheduler release, exact prefix plus typed failure. Historical vLLM run passed; refreshed #15091 candidate also collected and passed, 0 ignored. |
| Native request cache namespace, structured outputs, metadata/KV sources, engine config; connection reuse; opaque prefill/decode; EPD; RL/LoRA | Existing vLLM socket tests | Retained. Broad old aggregate test has distinct assertions beyond shared C4. |
| Mocker prefill/decode opaque handoff and KV relay/indexer | Existing `lib/mocker/servers/vllm/tests/sidecar.rs` | Retained: opaque manual handoff and KV relay/indexer have distinct assertions. |
| SGLang foundation and Mocker suites; TensorRT-LLM owning suites | Existing owners | Retain the four SGLang conformance cases and all four SGLang Mocker cases; no new backend activation or migration. |
| Actual engine compute release / KV transfer | #15091 explicit native target | Cannot be established by CPU Mocker scheduler or fake handoff. |
| Existing E2E allocation | Root Python suites/workflows | Unchanged per current rollout. |

## Replacement accounting

These five replacements belong only to #15091. Preserve and validate every
listed assertion before removing its original definition; shared-family
refactoring must also retain the existing SGLang assertions.

| Original definition | Replacement boundary and distinct assertions |
| --- | --- |
| `grpc_request_errors_are_propagated` | Native rejection wire scenario: `generate()` returns a setup error before opening a response stream, preserving native RPC/code/reason and typed category; handler release and recovery also pass. Unit status tables do not replace transport propagation. |
| `cancellation_drops_the_remote_stream` | Wire cancellation: remote context/drop, exact delivered prefix and cancelled terminal/usage. |
| `cancellation_interrupts_pending_response_headers` | Held-header wire cancellation: vLLM `generate()` itself stays pending before headers, cancellation completes without manually releasing them, and the native handler is dropped. SGLang retains its lazy-stream contract. |
| `sidecar_streams_mocker_tokens_logprobs_and_usage` | vLLM streaming wire: exact native token/text values, one-token native delta assertions where specific to vLLM, three candidate entries, selected/top values, prompt metadata, terminal and usage; generic shared assertions remain chunk-size independent. |
| `dropping_sidecar_stream_cancels_mocker_work` | Explicit-stop wire scenario: this old test called Stop; retain that behavior and add independently observed active scheduler drain. Pure consumer drop is a distinct added case. |

Richer aggregate mapping, native metadata, decode first-token cancellation,
media, LoRA, RL, opaque handoff and KV-relay assertions remain. Pure unit
relocation is recorded in UNITS.md and is not deletion of coverage.

## Process and native boundaries

[PROCESS.md](PROCESS.md) preserves all six Worker/real-sidecar/PrefillRouter
scenarios, runtime regression assertions and failure classifications.
[NATIVE.md](NATIVE.md) preserves real-engine compatibility, cancellation and
positive-transfer requirements. CPU scheduling or opaque handoff forwarding
cannot establish real GPU-work release or actual NIXL transfer.
The pinned native handoff defect, transfer-time cancellation and required
two-GPU success topology remain blocked; #15091 is not merge-ready on CPU
results alone. [SUPPORT.md](SUPPORT.md) retains all 162 capability IDs and both
unnumbered restrictions; [DEVIATIONS.md](DEVIATIONS.md) records departures.

## Validation commands

The runner and CPU-container commands are in [README.md](README.md). Direct
wire commands, including retained SGLang coverage, are:

```sh
cargo test --locked -p dynamo-sidecar-testkit --test conformance -- --list
cargo test --locked -p dynamo-sidecar-testkit --test conformance
cargo test --locked -p dynamo-vllm-mocker --test sidecar
cargo test --locked -p dynamo-sglang-mocker --test sidecar
cargo clippy --locked -p dynamo-sidecar-testkit --test conformance --no-deps -- -D warnings
```

Refreshed #14879 at `286d6fd5bbfe10efb9c929edca6c3185032b84cc` passed eight
shared and eight retained Mocker cases locally, and Clippy passed. Its exact
trusted copy activated full PR run 35791357457 and Pre Merge 35791350230;
the initial CI snapshot was still pending. Refreshed #15089 at
`b3ab1638513e255828acddc40f045d069ed6bc33` passed 62 isolated cases in the
CPU container, 102 total common/vLLM library cases, eight conformance cases and
eight retained Mocker cases. Formatting, Clippy, pre-commit and ownership checks
passed. The 62 isolated cases are part of the 102-library selection, not an
additional set. UNITS.md owns the detailed fresh unit evidence.

## Refreshed integration execution

The #15091 candidate based on `b3ab1638513e255828acddc40f045d069ed6bc33`,
including the strengthened native-rejection setup and held-header cancellation
assertions above, passed the isolated CPU artifact run. This is candidate-source
evidence on refreshed main `4a0547f8`, not a final integration commit or CI result.

| Collected and executed selection | Passed | Ignored |
| --- | ---: | ---: |
| Common/vLLM isolated units | 11 + 51 = 62 | 0 |
| vLLM/SGLang shared conformance | 9 + 4 = 13 | 0 |
| Retained vLLM/SGLang Mocker cases | 2 + 4 = 6 | 0 |
| Retained common/vLLM library wire cases | 2 + 35 = 37 | 0 |
| Actual-sidecar process integration | 6 | 0 |
| Complete CPU selection | **124** | **0** |

The 56 wire cases are the three middle rows. Unlike the overlapping historical
library selections, these runner selections are disjoint. The separate
`--framework vllm --level all --list` artifact collection returned 116 cases,
excluding the eight retained SGLang cases; it is not a second execution.

```bash
python3 lib/sidecar/testkit/run.py --framework all --level all --export "$artifacts"
docker build -f lib/sidecar/testkit/CPU.Dockerfile -t sidecar-cpu "$artifacts"
docker run --rm --network none sidecar-cpu --framework all --level all
python3 lib/sidecar/testkit/run.py --framework vllm --level all --artifacts "$artifacts" --list
cargo clippy --locked -p dynamo-sidecar-testkit --features process-tests,native-tests --all-targets --no-deps -- -D warnings
```

The runner listed each selection before executing it and rejected failures,
ignored cases or count mismatches. Recorded artifacts are
`restack/integration-container.log`, `restack/integration-vllm-collection.log`,
and `restack/integration-clippy.log` in the separate execution report. Clippy
passed in 1 minute 6 seconds; formatting passed. [PROCESS.md](PROCESS.md)
records the additional runtime checks. Refreshed native execution collected
three cases: compatibility and cancellation passed; handoff failed on the
pinned upstream defect, with zero decode tokens and no completed transfer.
[NATIVE.md](NATIVE.md) records the result and unexecuted two-GPU/transfer-time
cancellation requirements. Final current-head CI remains pending.

## Historical execution on the superseded stack

Historical #15088 at `73192d69351b51763a018edc75b7ec275d9fa68a` used main
`cdcd721e`. Its nine vLLM conformance and two remaining vLLM Mocker cases passed
inside a CPU container with `--network none`, no GPU/model cache or engine
installation. The retained vLLM library suite passed 60 cases. These counts
exclude the newly retained foundation SGLang cases and are not the refreshed
stack's collection. Historical #15089 and #15091 results are labeled in their
layer reports; old green CI covers only the recorded old heads.

The initial historical wire run passed 8/9. The failed harness expectation
assumed `CannotConnect` after abrupt HTTP/2 shutdown; actual transport returned
`Unknown`. Corrected assertions still required exact prefix, one typed error,
RPC context, no successful terminal, handler drop and scheduler release. No
production fix was needed for this expectation. Historical corrected conformance
and Mocker runs passed 9/9 in 1.81 seconds and 2/2 in 0.68 seconds, respectively.

The historical exporter initially selected a non-test helper executable as a
duplicate. Selecting Cargo's test profile and exact target corrected that
harness defect; manifests reject missing or duplicate targets. Source inspection
and intended CI allocation never count as execution.
