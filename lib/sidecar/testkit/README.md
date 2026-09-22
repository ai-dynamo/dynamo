<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared CPU integration-test harness for sidecars

## Goal and scope

Build a Rust-only, CPU-only testing framework for the vLLM and SGLang sidecars.
Share test scenarios, synchronization, server lifetime management, request
construction, and assertions wherever the sidecar contract is the same. Keep
native protocol details in small framework adapters. Add future tests to these
boundaries instead of creating another independent fake server for each test.

The foundation has four scenario families: streaming, failures, cancellation,
and cleanup. Each remains covered for both vLLM and SGLang. The integration
increment extends vLLM's existing families and adds distinct vLLM cases without
registering the foundation's vLLM cases a second time. Its refreshed conformance
suite collected and passed nine vLLM and four retained SGLang cases.
No new SGLang or TensorRT-LLM scenario or native-engine activation is added.

The stack is [#14879](https://github.com/ai-dynamo/dynamo/pull/14879) (this
foundation), [#15089](https://github.com/ai-dynamo/dynamo/pull/15089) (isolated
units), then [#15091](https://github.com/ai-dynamo/dynamo/pull/15091) (additional
vLLM wire coverage and process/native integration). The later increments reuse
this harness; new backend coverage is vLLM-only. Existing SGLang cases and E2E
allocation remain in place.

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
The integration tests depend on those crates through `dev-dependencies`. No
production sidecar or Mocker depends on the testkit. This prevents testing
infrastructure from becoming part of their normal dependency graph.

| Location | Responsibility |
|---|---|
| `src/server.rs` | Bind an available localhost port, contain server connections/handlers in an owned runtime, and bound explicit shutdown and fallback cleanup. |
| `src/control.rs` | Per-request plans, persistent observations, explicit pause/release coordination, and native-response interception. |
| `src/fixtures.rs` | Construct ordinary `PreprocessedRequest` values and collect actual sidecar outputs. |
| `src/assert.rs` | Assert exact token preservation, terminal placement, usage, and typed errors. |
| `src/lib.rs` | Export the helpers and provide labeled, bounded waits. |
| `tests/support/mod.rs` | Define the fixture interface and configuration shared by the two adapters. |
| `tests/support/{vllm,sglang}.rs` | Start each existing Mocker service, construct its real sidecar, delegate RPCs, and interpret native messages. |
| `tests/conformance.rs` and `tests/conformance/` | Enroll retained SGLang and extended vLLM scenarios without duplicating the four foundation vLLM cases. |

Framework adapters are shared within the central integration suite. They are not
public fixture APIs for other crates. Pure tests beside the sidecar implementation
can use generic testkit helpers as a development dependency where useful; they do
not need to construct a Mocker.

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
| Cleanup (R13) | Active-stream cleanup remains covered for both backends. SGLang retains before-start and repeated-cleanup checks; the vLLM no-I/O subsection moves to its isolated worker unit. | Separate construction/startup, unsubmitted-request observation, remote stream release, and explicit server teardown. |

The two-request cancellation case also checks a focused part of R14: cancelling
one request must not terminate another. It does not claim full concurrency or
stress coverage. Different prompt lengths and output budgets exercise the shared
request and assertion helpers without requiring identical native tokens across
frameworks.

Premature EOF remains a typed error in this branch: `Unknown` for vLLM and
`EngineShutdown` for SGLang. These tests preserve the sidecar's production error
contract.

## Choosing the owning layer

| Behavior | Owning layer | What to reuse or extend |
|---|---|---|
| Endpoint/configuration parsing and request conversion | A test module beside the implementation | Existing value builders or assertions where useful; no server. |
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

The stack is refreshed onto main `4a0547f8ba2675f14d50e48d6aec53b1bc3cf3e3`; pinned vLLM 0.29.0 and
`vllm-proto` 0.3.0 are unchanged. [UNITS.md](UNITS.md),
[COVERAGE.md](COVERAGE.md), [PROCESS.md](PROCESS.md) and [NATIVE.md](NATIVE.md)
record owners, commands and limits. [SUPPORT.md](SUPPORT.md) preserves the full
capability mapping; [DEVIATIONS.md](DEVIATIONS.md) records changes separately
from the read-only DEP. #15088 is superseded by the user-approved stack.

```bash
python3 lib/sidecar/testkit/run.py --level unit --list
python3 lib/sidecar/testkit/run.py --level pre-merge
python3 lib/sidecar/testkit/run.py --level integration
```

`unit` selects common code once and vLLM's isolated modules. `wire` includes the
retained SGLang foundation and Mocker coverage alongside vLLM's wire coverage.
`process` launches actual vLLM sidecars; `integration` combines wire and process.
`pre-merge` combines units and wire. `all` selects CPU layers; native engine
execution requires its separate launcher. No new SGLang units/process/native
fixtures are implied by retaining its existing wire coverage.

Export and execute the same CPU selection without engines, GPU devices, model
caches or external networking. Place the artifact directory outside the source
checkout on the configured build storage:

```bash
python3 lib/sidecar/testkit/run.py --level integration --export "$artifacts"
docker build -f lib/sidecar/testkit/CPU.Dockerfile -t sidecar-cpu "$artifacts"
docker run --rm --network none sidecar-cpu --level integration
```

Use Rust 1.96.1, protoc 30.2, matching `PROTOC`/`PROTOC_INCLUDE`, and an external
`CARGO_TARGET_DIR`. The runner collects selected cases before execution and
rejects empty, missing or duplicate targets, failures, ignored cases and count
mismatches. Refreshed #14879 at `286d6fd5` passed eight shared and eight retained Mocker
cases plus Clippy. #15089 at `b3ab1638513e255828acddc40f045d069ed6bc33`
passed 62 isolated cases in the CPU container, 102 total common/vLLM library
cases, eight shared conformance cases and eight retained Mocker cases; formatting,
Clippy, pre-commit and ownership checks passed. These selections overlap and
must not be added.

The #15091 integration candidate based on `b3ab1638` passed all 124 selected
cases in the isolated CPU container: 62 units, 56 wire cases and six process
cases, with zero ignored. The wire selection includes nine vLLM/four SGLang
conformance cases, two vLLM/four SGLang Mocker cases, and two common/35 vLLM
retained library cases. The vLLM-only selector separately collected 116 cases;
that collection is not another execution. All-target Clippy with both integration
features and formatting passed. [COVERAGE.md](COVERAGE.md) records commands and
[PROCESS.md](PROCESS.md) records the refreshed runtime regressions. Native
compatibility and cancellation passed; handoff reproduced its upstream blocker,
as recorded in [NATIVE.md](NATIVE.md). Final current-head CI remains pending.
Older CPU/native/CI results remain labeled historical.

Default CPU unit/wire coverage remains pre-merge. New process/native CI follows
the post-merge/nightly allocation; existing E2E allocation is preserved.

## Limits of the evidence

The four scenarios prove the common stream/lifecycle path and the exercised
request isolation. They do not prove future fault mechanisms before tests use
them. Native response history is retained for the fixture's lifetime; this is
intended for bounded correctness tests, not long-running load generators.

Cancellation checks observe the server-side RPC being dropped and the Mocker's
registered response routes being released. The fast simulated scheduler may
already have completed, so those checks do not prove interruption of active
scheduler work. The additional vLLM scheduler-active cases and retained SGLang Mocker tests
address their distinct CPU scheduler assertions; actual engine release remains
separate native coverage.

A Mocker can share a protocol misunderstanding with a sidecar. Real-engine
compatibility, model inference, and actual KV-cache transfer remain separate
integration/nightly concerns. Their results cannot be inferred from this CPU-only
suite.
