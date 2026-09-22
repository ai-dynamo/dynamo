<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Direct native integration (DIS-2943)

These tests address native compatibility gaps that CPU peers cannot establish:
logprob and structured-output acceptance, actual engine work release after
cancellation or consumer drop, and actual KV transfer between native engines. They supply preprocessed token requests directly
to the Rust sidecar engine. There is no Dynamo frontend or new E2E deployment.
The existing E2E tests and their CI allocation are unchanged.

This is #15091's vLLM-only native increment above #14879 and #15089, refreshed
onto main `4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`. The existing SGLang wire foundation
and Mocker cases remain; no new SGLang native activation is added. Native
execution on the refreshed stack is recorded below; current-head CI remains pending.

The preceding #15089 boundary at `b3ab1638` passed its isolated unit, full
common/vLLM library and retained foundation/Mocker suites, as recorded in
[UNITS.md](UNITS.md). Those results do not execute this integration layer.

The launcher requires both vLLM and its bundled `vllm-rs` to report 0.29.0. Dynamo
uses the published `vllm-proto` 0.3.0. Missing binaries, pins, metrics, telemetry or
GPU assignments fail rather than skip. Qwen/Qwen3-0.6B uses an 8192-token context,
eager execution, two scheduler slots and an explicit 1119388000-byte KV cap.
The cancellation case reserves one GPU; handoff reserves two in CI. Both use
unique ports and bounded owned-process cleanup through the existing ManagedProcess.

```sh
python3 lib/sidecar/testkit/run.py --level native --export "$native_artifacts"
DYNAMO_SIDECAR_NATIVE_TEST="$native_artifacts/native_engine" \
  python3 -m pytest -vv tests/sidecar/test_native_integration.py
```

Choose `native_artifacts` on the configured external build storage.
For an existing cache use the repository's `--models-dir` pytest option.
`SIDECAR_NATIVE_MODEL_PATH` may select a local snapshot of that same model, and
`SIDECAR_NATIVE_GPUS` may select assigned GPU IDs. The launcher otherwise respects
`CUDA_VISIBLE_DEVICES`. `native` is excluded from `all`; directly executing the
Rust target without the launcher cannot establish its required resources.

| Case | Required observation | Boundary |
| --- | --- | --- |
| `vllm_native_logprobs_and_structured_output_are_compatible` | Requested selected/top and prompt logprobs align with tokens; schema-constrained output parses as the required JSON value; exact terminal/usage | Direct sidecar + one pinned native engine |
| `vllm_cancellation_and_consumer_drop_release_native_work` | Successful initial request, nonterminal tokens and running scheduler work before interruption; Cancelled+EOF for explicit stop; zero running and waiting work after stop and pure drop; fresh request after each | Direct sidecar + one native engine, authoritative native Prometheus scheduler gauges |
| `vllm_handoff_transfers_native_kv` | Tokenless prefill terminal, nonempty native engine/block identities, opaque handoff consumed by decode, exact output length and usage, and positive completed NIXL transfer bytes | Direct sidecar + two native engines, actual NIXL completion telemetry |

The pinned Rust frontend does not export the NIXL transfer-completion statistic.
A test-only Python worker extension observes `NixlKVConnectorStats.record_transfer`
after calling the real implementation, and records `totalBytes` and `descCount`.
It changes no request, output, scheduler policy or transfer result. Every case
starts fresh engines and a fresh observation file. Successful generation alone
never satisfies the transfer assertion.

## Refreshed execution

The integration candidate based on unit head `b3ab1638` and refreshed main
`4a0547f8` collected and executed all three launcher cases on 2026-09-22:
**two passed, one failed**, with no skips or xfails. Compatibility passed its
logprob, schema-constrained JSON, terminal and usage assertions. Cancellation
passed explicit stop and pure consumer drop, including active-to-idle native
scheduler observations and successful follow-up requests.

Handoff failed the exact output assertion: zero decode tokens instead of eight.
The decode engine again logged `TypeError: 'float' object cannot be interpreted
as an integer` at `range(remote_pp_size)`; the transfer observation file was
empty. This reproduces the pinned upstream blocker below on the refreshed
Dynamo source. Neither generation nor completed NIXL transfer is credited.

The run used `dynamo-vllm-029-core-test:extras` (image ID
`sha256:c2788078325eccd59a6179792df190b411a1054a392efa88922183f9151cab34`),
vLLM/vllm-rs 0.29.0, protocol 0.3.0 and the same Qwen snapshot named below.
The launcher ran all cases through `tests/utils/profile_pytest.py
--no-find-min-vram`, with `DYNAMO_SIDECAR_NATIVE_TEST` pointing at the freshly
exported binary, a read-only model cache and `SIDECAR_NATIVE_GPUS=0,0`.
Pytest took 233.86 seconds; the profiled wall time was 245.6 seconds and the
combined run peaked at 6.3 GiB on one RTX 6000 Ada. This combined measurement
is not a new per-case VRAM budget. The separate execution report retains
`restack/native-container.log`, engine logs and the empty transfer probe.

Two native engines shared one GPU for the diagnostic handoff reproduction.
The required two-GPU success topology and cancellation during actual native
transfer remain unexecuted and blocked. No final integration-head CI or
post-merge/native workflow success is claimed by this local run.

## Historical execution and remaining blockers

The following executions used the superseded #15091 implementation ending at
`9bcc0773`, on base `cdcd721e`. The engine/protocol pins are unchanged, but
these results do not certify the refreshed Dynamo runtime. Preserve them as
revision-specific evidence; the refreshed execution is recorded above.

All three Python cases collected in the pinned vLLM test image. The final native
compatibility case passed in 50.11 seconds (61.5-second profiled wall time,
3.5 GiB peak). Requested output/prompt logprobs, schema-conforming parsed JSON,
terminal reason and actual prompt/output usage all passed. The native
cancellation case passed locally with real GPU inference: explicit cancellation
and pure consumer drop each reached idle scheduler gauges and a successful
follow-up request. Single-pass NVML profiling observed a 3.5 GiB peak; the case
remains an integration test despite the generic profiler's E2E recommendation.
The run used vLLM 0.29.0, bundled vllm-rs 0.29.0, Qwen snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`, and an RTX 6000 Ada.

Native handoff executed and failed on the pinned engine: prefill returned valid
metadata, but decode produced zero tokens and no completed transfer. vLLM's
native protobuf `Struct` conversion restores numeric values as JSON floats;
`remote_pp_size` therefore reaches the NIXL worker as `1.0`. Its
`range(remote_pp_size)` raises `TypeError`, and the scheduler aborts KV loading.
This is an upstream-engine compatibility blocker, not successful handoff coverage.
The test remains failing, without a skip, xfail or relaxed output/transfer assertion.
#15091 cannot merge with this pinned-engine failure unresolved.

Local handoff used two engine processes on the workstation's one GPU (explicit
`SIDECAR_NATIVE_GPUS=0,0`). CI still reserves two GPUs; the required successful two-GPU
topology has not executed. The failed local run took 182.78 seconds and peaked at
6.3 GiB. Refreshed execution reproduced the same blocker as recorded above;
current-head CI remains pending. Historical pre-merge CI did not run this
post-merge/native lane.

The initial launcher failures were harness setup defects: a minimal local image
lacked pytest-benchmark; the profiling wrapper requires its own CLI rather than
pytest `--profile`; and ManagedProcess needed an explicit display name to avoid
constructing a log path from an absolute executable path. They occurred before
native inference and required no production change.

Native handoff cancellation (C12) also remains blocked. A cancelled CPU handoff
does not establish actual NIXL work release. The current pinned engine cannot
complete the prerequisite KV load, and response headers alone would not prove
that a transfer was still active. No headers-only check is credited as equivalent.

The first C13 compatibility execution failed a harness assertion that native
output candidate IDs are unique. Pinned vLLM intentionally returns the selected
token plus its top-k list, which can repeat that token. The corrected assertion
requires exactly the selected entry plus two ordered native candidates, including
the greedy selected-token duplicate and equal logprob values. Prompt logprobs
retain their map semantics. No production change or reduced semantic check was
needed. The corrected final native execution passed as recorded above.

After correcting the candidate expectation, C13 passed logprob checks but its
JSON-schema request produced unconstrained text. The native frontend had
automatically enabled Qwen3 reasoning. Pinned vLLM delays grammar constraints
until reasoning ends; the test's preprocessed token prompt supplied no such
boundary. The compatibility launcher now explicitly selects
`--reasoning-parser none` for this plain structured-output contract. The exact
JSON assertion remains intact; other native scenarios keep their configuration.

With reasoning disabled, the repeated-comma synthetic prompt used for logprob
alignment led to schema-permitted whitespace until its token limit. The
structured-output request now uses a meaningful chat-tokenized question that
does not contain the expected answer; the schema alone requires `{"ok":true}`.
The fixture explicitly requests a token vector (`return_dict=False`) from the
pinned tokenizer and checks usage against that vector's length. These are test
input/setup corrections, not sidecar behavior changes or relaxed assertions.
