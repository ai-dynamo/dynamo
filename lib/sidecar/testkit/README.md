<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared sidecar test framework

## Goal and scope

Build a Rust-only, CPU-only testing framework for the vLLM and SGLang sidecars.
Share scenarios and assertions where the contract is common and sharing stays
simple. Reuse synchronization, server lifetime management and request builders.
Keep native integration details in small framework adapters. Add future tests to these
boundaries instead of creating another independent fake server for each test.

The initial scope is four scenario families: streaming, failures, cancellation,
and cleanup. Each runs against both frameworks, giving eight registered tests.
TensorRT-LLM remains outside this increment because it has no corresponding
Mocker server.

The shared foundation is [#14879](https://github.com/ai-dynamo/dynamo/pull/14879).
This draft is a separate alternative to
[#15089](https://github.com/ai-dynamo/dynamo/pull/15089), on the same foundation
base, with unit tests beside the production code they exercise. The original
PR remains unchanged. Additional vLLM wire, process and native integration
remain in [#15091](https://github.com/ai-dynamo/dynamo/pull/15091).
The unit suite covers common code and vLLM; SGLang units remain follow-up work.
Existing shared integration scenarios, reusable fixtures, SGLang wire cases and
E2E allocation remain in place.

The testing strategy has two distinct execution paths. Pure unit tests call
conversion or parsing functions directly. Tests of actual sidecar generation and
lifecycle use a real localhost connection to a CPU-only Mocker. These are Rust
integration tests that serve the same fast pre-merge testing goal. They need no
inference-engine installation, model
download, GPU device, Python process, container, or external discovery service.
Building still requires the repository's ordinary Rust workspace prerequisites.

## Architecture and ownership

```text
Shared scenario
    |
    v
Real vLLM or SGLang sidecar
    | localhost gRPC
    v
Framework test adapter: observe requests and control responses
    |
    v
Existing native Mocker service in lib/mocker/servers/{vllm,sglang}
    |
    v
Existing scheduler and synthetic generation in lib/mocker
```

The Mockers remain in their existing crates. They own normal simulated engine
behavior, including native generation responses. The testkit controls when those
responses are delivered and introduces deliberately abnormal behavior. The real
sidecar performs request conversion, connection handling, response conversion,
cancellation, and cleanup through its normal public API.

The testkit library has no direct concrete sidecar, Mocker, protobuf, or tonic dependency.
The integration tests depend on those crates through `dev-dependencies`. Normal
production sidecar and Mocker builds do not depend on the testkit. The vLLM
sidecar uses a dev-dependency to share common input builders in its unit tests.

| Location | Responsibility |
|---|---|
| `src/server.rs` | Bind an available localhost port, own the server task, await bounded shutdown, and abort on drop if explicit teardown did not finish. |
| `src/control.rs` | Per-request plans, persistent observations, explicit pause/release coordination, and native-response interception. |
| `src/fixtures.rs` | Construct ordinary `PreprocessedRequest` values and collect actual sidecar outputs. |
| `src/assert.rs` | Assert exact token preservation, terminal placement, usage, and typed errors. |
| `src/lib.rs` | Export the helpers and provide labeled, bounded waits. |
| `tests/support/mod.rs` | Define the fixture interface and configuration shared by the two adapters. |
| `tests/support/{vllm,sglang}.rs` | Start each existing Mocker service, construct its real sidecar, delegate RPCs, and interpret native messages. |
| `tests/conformance.rs` | Define the four shared scenarios and enroll each backend once, generating its four tests. |

Wire adapters belong to the central integration suite. Isolated units are
ordinary `#[cfg(test)]` child modules beside their production owners, keeping
private converters accessible. Small tests are inline; the larger transport and
vLLM conversion modules use adjacent test files. Inputs, production calls and
assertions live together, with no unit source-group macros or backend adapters.
The only retained unit macro, `sidecar_test!` in `src/lanes.rs`, records each
case's CI lane. Common fixtures remain a test-only testkit dependency; reusable
native fixtures remain in `tests/support/fixtures/vllm.rs` for both layers.

## Request controls and observations

A controller belongs to one test fixture. Each request ID has its own handle,
plan, native request, native response history, token observation, progress flags,
and release signal. There is no global state shared between tests. Register a
handle before submitting the request so even cancellation before submission can
be checked. Request IDs must be unique within that controller.

Normal forwarding is the default. A request plan can fail or hold RPC opening.
It can also select a stream checkpoint independently of the action performed
there: the Nth native response containing output tokens, or a terminal response.
Actions continue the stream, close it, return an injected error, or replay the
first token response. The latter checks that the sidecar ignores data after
completion. A checkpoint can pause until the test explicitly releases it.

The initial API supports one stream checkpoint per request. It does not introduce
a general scripting language. Extend the plan representation when a concrete
test needs multiple interventions on the same request.

`Received`, `Checkpoint`, and `Dropped` are persistent progress flags. A wait can
observe an event that happened before the wait began. Request A's events and
release signal cannot advance request B. Waits carry a label and a ten-second
failure bound; ordering uses notifications rather than sleeps.

The `Protocol` trait is implemented on a locally owned adapter type with native
request, response, and error types. This keeps framework-specific fields and
transport-library versions out of the shared controller. The adapter constructs
native errors and interprets token fields; it does not duplicate the sidecar's
conversion code or the Mocker's generation algorithm. Full native messages remain
available for future assertions about fields beyond token IDs.

Source responses are recorded before deliberate stream alteration. Expected
tokens come from those Mocker responses, not a fixed synthetic token sequence.
Injected post-terminal replay is excluded from that expected sequence. Both
adapters accumulate the native token deltas emitted by their pinned protocols.
Paused-stream checks compare the accumulated sidecar prefix with those native
tokens without assuming one token per response. The alternate-model scenario
checks discovered model identity and vLLM's native model selector; SGLang's
tokenized generation RPC has no model selector.

## Four scenarios that exercise the foundation

| Scenario | Behavior protected | Infrastructure exercised |
|---|---|---|
| Streaming (R09) | Exact tokens, one final length response, correct usage, and ignored data after completion. | Default forwarding and terminal replay; configurable model and connection count; native observations and shared assertions. |
| Failures (R11) | Opening failure, premature EOF, and read failure preserve delivered tokens and report a typed error. | Opening control, response checkpoint, explicit release, and framework-specific error mapping. |
| Cancellation (R12) | Cancellation before submission, while opening, and while waiting for another response. | Independently controlled requests A and B: pause A after two token responses and B after one, cancel A, verify B remains pending, then release B to normal completion. |
| Cleanup (R13) | Generation before startup fails; repeated cleanup succeeds; cleanup cancels an active stream. | Separate construction/startup, unsubmitted-request observation, remote stream release, and explicit server teardown. |

The two-request cancellation case also checks a focused part of R14: cancelling
one request must not terminate another. It does not claim full concurrency or
stress coverage. Different prompt lengths and output budgets exercise the shared
request and assertion helpers without requiring identical native tokens across
frameworks.

Premature EOF remains a typed error in this branch: `Unknown` for vLLM and
`EngineShutdown` for SGLang. These tests preserve the sidecar's production error
contract.

## Adding the rest of the suite

| Future test | Where to add it | What to reuse or extend |
|---|---|---|
| Endpoint/configuration parsing and request conversion | The owning common or backend production module | Common inputs, plain setup functions and direct production conversion; no server. |
| Shared stream, cancellation, or lifecycle behavior | A new scenario in the central integration suite | Both existing fixtures, per-request controls, and output assertions. |
| Native malformed responses, logprob metadata, or handoff fields | Framework-specific tests in the central suite, or pure conversion tests | Native message observation and adapter-specific response overrides; keep exact wire fields visible. |
| Discovery, readiness, or model metadata | Framework-specific service tests | Shared server lifetime; add controlled native discovery/health handlers when their tests are introduced. |
| Connection deadlines, resets, GOAWAY, or malformed frames | Dedicated transport tests | Server/connection controls below the normal gRPC handler; a returned status is not a TCP reset or GOAWAY. |
| CLI flags, environment wiring, or signals | Separate executable tests | Process lifetime helpers and relevant request/assertion helpers. |
| Protobuf field compatibility | Direct encoding tests | Native protocol fixtures; no server or Mocker. |

Use the existing `LLMEngine` interface for real sidecars. Keep supported behavior
differences explicit in adapter expectations or scenario parameters. Add a new
shared interface only when concrete consumers need it; avoid a large capability
trait whose unused operations are implemented as no-ops or skipped tests.

The existing `backend-common::testing::run_conformance` suite remains a separate
future integration step. It checks additional invariants such as metrics, KV
event sources, and concurrent generation, and does not replace deliberate fault
injection. This increment reuses its `mock_context` helper.

Use paused Tokio time for future deadline tests when their I/O scheduling is
controlled. These four socket scenarios use explicit events and bounded real
time. They do not test elapsed deadlines or rely on shortened sleeps.

## Running and validating

```bash
cargo test --locked -p dynamo-sidecar-testkit --test conformance
cargo clippy --locked -p dynamo-sidecar-testkit --all-targets --no-deps -- -D warnings
```

The crate is a workspace member, so the existing pre-merge workspace Rust test
job discovers all eight cases without a feature flag or separate CI job. Tests
start their servers inside the Rust test process on OS-assigned ports. GPU-free
execution can also be checked with `CUDA_VISIBLE_DEVICES=` and
`NVIDIA_VISIBLE_DEVICES=void`.

Retain the existing Mocker `tests/sidecar.rs` suites when migrating these four
scenarios. They cover logprobs, scheduler cancellation, and prefill/decode handoff
that these shared scenarios do not replace. This harness adds coverage without
removing those suites.

For this increment, acceptance requires eight shared cases passing, including
two-request cancellation isolation, the existing Mocker sidecar integration
tests passing, formatting and Clippy passing, and no production sidecar/Mocker
behavior changes. When native protocol APIs change, update the adapters and
rerun both the shared cases and the retained Mocker integration suites.

## Limits of the evidence

The four scenarios prove the common stream/lifecycle path and the exercised
request isolation. They do not prove future fault mechanisms before tests use
them. Native response history is retained for the fixture's lifetime; this is
intended for bounded correctness tests, not long-running load generators.

Cancellation checks observe the server-side RPC being dropped and the Mocker's
registered response routes being released. The fast simulated scheduler may
already have completed, so those checks do not prove interruption of active
scheduler work. Existing integration tests retain that coverage.

A Mocker can share a protocol misunderstanding with a sidecar. Real-engine
compatibility, model inference, and actual KV-cache transfer remain separate
integration/nightly concerns. Their results cannot be inferred from this CPU-only
suite.

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Isolated units beside production

This alternative preserves the 81 isolated cases from #15089: 11 common and
70 vLLM cases. The ten formerly shared vLLM scenarios now live in the vLLM
modules that own their production calls and assertions. No assertion is removed
by this relocation, and no SGLang unit coverage is claimed.

Common argument, endpoint and error units are inline in `common/src`; transport
units live in `common/src/transport/tests.rs`, registered once by the common
crate root to avoid duplicate collection through the two Tonic implementations.
vLLM model, worker, JSON and LoRA units are inline; request/response cases and
candidate checks live in `vllm/src/convert/{request_tests,response_tests}.rs`.
The former `tests/unit/` tree is removed.

`src/lanes.rs` retains the small lane macro unchanged, included by both crate
roots. Every governed unit declares its earliest CI lane with `sidecar_test!`:
`pre_merge`, `post_merge` or `nightly`. Later lanes include earlier lanes. All
current units are `pre_merge`; assigning a future test to `nightly` excludes it
from pre-merge and post-merge execution. Suite, backend and lane remain
independent runner selections:

```sh
python3 lib/sidecar/testkit/run.py --suite unit --framework vllm --lane pre-merge --list
python3 lib/sidecar/testkit/run.py --suite unit --framework vllm --lane pre-merge
```

Common fixtures, including `minimal_request`, remain in `src/fixtures.rs`.
Native builders remain in `tests/support/fixtures/vllm.rs`, outside unit sources,
so integration tests can use them too. The existing four wire families and both
retained Mocker suites remain. Only vLLM's before-start/repeated-cleanup wire
subsection has its isolated replacement; SGLang's checks remain.

Fresh validation preserved all 81 scenario/lane pairs and passed all 81
exported-runner units, 121 common/vLLM library tests, eight testkit conformance
and both four-case Mocker suites. All 81 units also passed in a CPU container
with networking disabled; targeted Clippy, formatting and pre-commit passed.
The runs at `47ae1fb270` belong to the original shared-unit layout and remain
historical evidence. These results do not claim GitHub CI, a full Dynamo
workspace run or native-engine/GPU execution.
[UNITS.md](UNITS.md) records the source layout, assertion mapping, lane semantics
and inventory/export commands. [COVERAGE.md](COVERAGE.md) records retained
integration coverage and revision-specific execution;
[DEVIATIONS.md](DEVIATIONS.md) records departures from the read-only DEP.
