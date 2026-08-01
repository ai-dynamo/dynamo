# Harness Compatibility Lab

This is a discovery harness for live coding-agent compatibility. It is deliberately separate from `tests/frontend`: pytest remains the destination for stable regressions, not the place to discover a changing client protocol.

## Objective

Run current Codex and Claude Code against the same isolated Dynamo deployment, exercise real coding workflows, and retain enough evidence to answer three questions for every scenario:

1. Did the harness actually reach the intended behavior?
2. Did Dynamo receive, normalize, and return the expected protocol shape?
3. If not, is the divergence in the harness, transport, frontend, backend, or model behavior?

An inconclusive model decision is `not reached`, never a Dynamo pass.

## Run on any VM

Requirements: Python 3, the selected current `codex` or `claude` binary on `PATH`, and a Dynamo URL that serves `/v1/models`, `/v1/responses`, and `/v1/messages`. The runner creates a timestamped artifact directory under `/tmp/dynamo-harness-compat` unless `--artifacts` is supplied.

```bash
cd <dynamo-checkout>
python3 tools/harness_compat/live_scenario.py \
  --harness codex --scenario inject_agent_message \
  --model MiniMaxAI/MiniMax-M2 \
  --endpoint-url http://127.0.0.1:8000
```

Run the stable daily set directly against the same endpoint:

```bash
python3 tools/harness_compat/nightly.py \
  --model MiniMaxAI/MiniMax-M2 \
  --endpoint-url http://127.0.0.1:8000
```

For a loopback-only endpoint on another machine, replace `--endpoint-url` with `--remote-http-port <port>`. Add `--remote-run-root <run-dir>` only when the deployment exposes the standard run logs and you want them copied into every artifact. `nightly.py --dry-run` prints the exact individual invocations.

Use `SCENARIOS.md` to choose a workflow, and `python3 tools/harness_compat/codex_driver.py --help` or `python3 tools/harness_compat/claude_driver.py --help` to list its exact `--scenario` names. No venv, package install, or deployment-specific configuration is required for a direct endpoint run.

## Add a case

1. Add one focused branch to `codex_driver.py`, `claude_driver.py`, or `claude_interactive_driver.py`, and add its string to that driver's `--scenario` choices.
2. Return one content-free reach signal; `live_scenario.py` maps it to `pass` or `not_reached` and always retains the sanitized evidence.
3. Add the purpose and wire expectation to `SCENARIOS.md`, then run it twice through `live_scenario.py`.
4. Record the artifact-backed result in `FINDINGS.md`. Promote only deterministic cases to `CORE_CASES` in `nightly.py`; review any new protocol discriminator before accepting it in `protocol_baseline.json`.

## Topology

```text
local disposable coding repo
  ├─ Codex app-server controller ─┐
  └─ Claude Code controller ──────┼─> capture-only proxy ─> SSH tunnel ─> Dynamo frontend ─> TP4 MiniMax worker
                                 │          │                         │
                                 │          └─ sanitized wire record   └─ request trace + frontend/worker logs
                                 └─ harness transcript and tool results
```

The normal proxy mode is transparent: it neither injects headers nor rewrites request bodies. The only request fields retained are a fixed header allowlist and structural JSON fingerprints; request content and credentials stay out of artifacts. Fault-injection mode is explicit, one-shot, and endpoint-shaped so error handling can be exercised without claiming that Dynamo produced the injected error.

## Process

1. Provision one aggregated TP4 MiniMax worker on TRY-67676, bound to loopback, with the Anthropic endpoint and Dynamo request tracing enabled.
2. Start a local SSH tunnel plus capture-only proxy on dynamically chosen loopback ports.
3. Run an ordinary coding task first. Confirm a successful tool loop before any induced edge case.
4. Run one scenario at a time with a native controller:
   - Codex uses app-server JSON-RPC for deterministic compaction, steering, and interruption.
   - Claude Code uses its native interactive transport; stream-json is used where it exposes the required observation without replacing the harness.
5. Collect the six evidence streams below into one timestamped artifact directory.
6. Triage the first differing boundary and make the smallest Dynamo fix only after reproducing it.
7. Re-run the full original scenario, then promote only a deterministic reduced form into the nightly suite.

`launch_try67676.sh` takes `MODEL_NAME` as the primary served name and an optional comma- or whitespace-separated `MODEL_ALIASES` list. The launcher waits for the primary name, while Dynamo registers every alias against the same worker set. Use this for native harness child features that select a fixed secondary model name.

The MCP scenarios start `fixture_mcp_server.py` only from each run's isolated client configuration. Its fixed-result tools exercise normal, error, progress, elicitation, and client-roots paths without recording request content or adding an external dependency.

## Evidence contract

Each run directory contains:

- `scenario.json`: parameters, model, timing window, and a one-way client-session digest without credentials or plain session IDs.
- `fault.json`: the requested proxy fault mode and ordinal, with no request content.
- `harness.jsonl`: controller requests, lifecycle notifications, and exit status.
- `wire.jsonl`: request method/path, allowed headers, JSON item/content discriminators, response status, SSE event names/timestamps, tool names, and terminal stop reasons. It never retains prompt text, model text, or tool arguments.
- `frontend.log` and `worker.log`: remote server logs.
- `request-trace.jsonl`: cumulative Dynamo request trace. The artifact analyzer scopes it back to the client-session digest (including descendant parent links), falling back to the run window only for old artifacts.
- `result.json`: `pass`, `not_reached`, `harness_failure`, `dynamo_failure`, or `inconclusive`, with the first divergent boundary.

`summarize_run.py <artifact-dir>` emits a content-free request-shape fingerprint suitable for comparing a future native-harness run against its accepted baseline.

The canonical debugging order is:

```text
harness transcript → raw ingress → Dynamo normalization → backend request → stream/error egress → raw response → harness transcript
```

## Promotion rule

Discovery runs always use current installed harness binaries and are allowed to surface new wire shapes. A nightly test gets only the minimum stable assertion that caught a confirmed Dynamo regression:

- pure parsing or header mapping → unit/protocol test;
- frontend translation or stream contract → focused frontend E2E test;
- client-controlled lifecycle behavior → this native lab’s nightly subset;
- unknown new client item/event type → retained capture fixture plus a tracked compatibility finding.

This avoids freezing the full coding-agent behavior into a brittle pytest suite while retaining a reproducible detector for upstream drift.

## Preliminary nightly subset

This is the first proposed subset, not an enabled CI job. It deliberately keeps a complete pass within a bounded TP4 allocation and moves model-directed expansion into the discovery lane.

| Layer | Candidate | Why it is in the first nightly subset | Budget |
| --- | --- | --- | --- |
| Dynamo unit | `cargo test -p dynamo-llm test_nvcreate_response_ --lib` | Locks the confirmed Codex `agent_message.encrypted_content` translation regression without asking a model to choose a delegation graph. | local CPU |
| Codex | `steer_after_tool` | Proves a real tool turn can be steered, the terminated lifecycle is valid, and a later turn remains usable. | 180 s/turn |
| Codex | `compact` | Covers explicit native compaction followed by a tool-using turn. | 180 s/turn |
| Codex | `structured_output` | Detects drift in native `text.format` emission. | 180 s/turn |
| Codex | `inject_agent_message` | Exercises the persisted `agent_message.encrypted_content` handoff shape without relying on model-directed subagent expansion. | 180 s/turn |
| Codex | `tool_failure` | Proves a native failed command result reaches the agent and the same turn recovers. | 180 s/turn |
| Claude Code | `compact` | Covers a native compact boundary and the next Messages tool turn. | 420 s/run |
| Claude Code | `structured_output` | Detects `output_config.format`/Messages translation drift. | 420 s/run |
| Claude Code | `resume` | Covers persisted Messages history across a fresh CLI process. | 420 s/run |
| Claude Code | `tool_failure` | Proves a failed Messages `tool_result` is accepted before recovery. | 420 s/run |
| Both | one rotated 4xx/5xx/overload fault plus one rotated SSE truncation case | Exercises client error and reconnect behavior without imposing an artificial common retry policy. | 420 s/run |

Run the Codex subagent workflow weekly in discovery with an explicit turn cap, not as an initial nightly pass gate. The model can validly expand a one-child request into a deep graph; the parser regression itself is deterministic and already guarded by the unit test. Claude Agent, automatic compaction, and interactive terminal cancel/steer remain discovery-only until they have reproducible native reach signals.

Run the live subset with `nightly.py --remote-http-port 20585 --remote-run-root /data/harness-compat-lab/runs/<deployment-run>`. It runs nine core cases plus the same endpoint-native status and one early/mid-stream SSE truncation through both Codex and Claude Code daily (13 invocations total). Add `--include-weekly` for the four stable P1 lifecycle sentinels: Codex inline review, detached review, stale-lifecycle recovery, and Claude input close. `--fault-status 429` selects a specific status rotation member; `--dry-run` prints the exact invocations. Run `goal_lifecycle` weekly through `live_scenario.py` until it has a clean serialized canary.

After every successful native case, `nightly.py` compares its content-free protocol discriminator set to `protocol_baseline.json`. New headers, request fields, item/content types, advertised tool names/types, or output-format discriminators fail the run and are printed as a compact drift record; update the baseline only after triage.

## Error campaign contract

The proxy can inject exactly one selected client, authorization, conflict, rate-limit, server, or overload HTTP status on a selected request. The response uses the native envelope for the requested endpoint: OpenAI-style `error` for Responses and Anthropic-style top-level `type: error` for Messages. Record the client disposition rather than imposing one retry policy across harnesses:

- terminal failure with a native lifecycle error;
- recovery after a surfaced lifecycle error; or
- transparent retry followed by a successful terminal result.

Each result must show the injected HTTP status/error discriminator, native terminal status, request count, and whether the next normal request completed. A hang, malformed SSE terminal, or polluted follow-up turn is a compatibility failure.
