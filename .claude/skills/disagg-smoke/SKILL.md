---
name: disagg-smoke
description: KVBM disagg smokes — golden two-request flow (R1 cold + R2 warm), audit-equiv (legacy vs unified), and hub-side layout-mismatch rejection smoke. Golden modes use KVBM_DISAGG_LEADER / KVBM_BLOCK_LAYOUT from env; audit-equiv runs the smoke twice and diffs per-side kvbm_audit streams; incompat-smoke posts crafted P2P register payloads to a hub-only bringup and asserts each mismatch class is rejected with the expected error.
---

# Skill: KVBM Disagg Smoke

End-to-end smokes against a live `kvbm-hub` (+ Prefill + Decode topology, for the goldens). Owns its scripts; depends on `disagg-bringup` for launch helpers.

## Skill assets

| File | Purpose |
|---|---|
| `two-request-smoke.sh` | **Golden.** Drive R1 + R2 against a single leader. Mints an experiment dir, brings up hub + Prefill + Decode (via `disagg-bringup` scripts), runs the two prompts, prints a validation grep digest, renders `trace.html` via `disagg-trace`. Inherits `KVBM_DISAGG_LEADER` and `KVBM_BLOCK_LAYOUT` from env. |
| `audit-equiv.sh` | **Golden.** Run the smoke twice (`KVBM_DISAGG_LEADER=legacy` then `=unified`), then run `audit_diff` on per-side `kvbm_audit` streams to assert behavioral equivalence. |
| `incompat-smoke.sh` | **Failure-scenario.** Hub-only (no vLLM, no GPU). Brings up `kvbm_hub` on alt ports (28337/21337, velo disabled), POSTs crafted P2P + CD register payloads via curl, asserts each layout-mismatch class returns `400` with the expected error substring. Six scenarios: baseline-accept, cross-mode, universal canonical mismatch, operational NHD vs HND, operational page_size mismatch, CD without P2P. ~2s end-to-end. Exits 0 on all-pass, 1 on any failure with the failing scenario named on stderr. Run when the connector/hub gate is touched or as a fast regression check before merging changes to `layout_compat`. |

## Running via a subagent (recommended)

The smoke produces three multi-MB log files (`hub.log`, `prefill.log`,
`decode.log`) plus a `trace.html`. Pulling them into the primary
context burns tokens fast — especially when chasing a specific marker
across all three. **Delegate the run + log analysis to a
general-purpose subagent** and ask for a structured short summary;
the agent does the heavy reading and returns only the answer.

Use `sonnet` for the default subagent model — the workflow is
well-defined (run a script, grep specific events, return a verdict).
`haiku` works for pure pass/fail status checks. Save `opus` for
genuinely ambiguous failures (rare).

### Recommended subagent prompt template

```
Run the KVBM disagg two-request smoke and answer <SPECIFIC QUESTION>.
Do NOT dump full logs back — return under 300 words.

Steps:
1. From /home/ryan/repos/dynamo/.claude/worktrees/page-size (or
   whatever the user's worktree is), tear down any stale processes:
     pkill -f vllm.entrypoints.openai 2>/dev/null
     pkill -9 -f kvbm_hub 2>/dev/null
     sleep 3
2. (Optional, if bindings/hub need rebuild) Source env.sh and run
   the appropriate cargo/maturin commands. Report any build errors.
3. Run the smoke with KVBM_REPO set to the worktree:
     KVBM_REPO=<worktree-path> [KVBM_DISAGG_LEADER=unified] \
       bash <worktree>/.claude/skills/disagg-smoke/two-request-smoke.sh
   Capture stdout — note the EXP=<path> line for the experiment dir.
4. Inspect <experiment-dir>/{hub,prefill,decode}.log for the
   specific markers <USER QUESTION> asks about. Note any ERROR
   lines (ignoring UCX warnings + kernel_config noise).
5. Report back:
   - Verdict (1 line) — answer to the specific question.
   - 3-5 supporting evidence lines (event names + counts, NOT full
     log text).
   - Experiment-dir path for follow-up inspection.
   - Any unexpected errors.

Constraints:
- Do NOT cat full log files.
- Quote at most 2 short matched lines per evidence point.
- If the smoke fails to bring up, return the failure mode and exit.
```

### Asymmetric-TP / stamped-path discrimination markers

When asking "did the smoke exercise the stamped (AB-4+) path or the
legacy fallback?", the subagent grep cheatsheet:

| Marker | Path |
|---|---|
| `session_pull_rdma_start` / `session_pull_rdma_done` | AB-5 wiring — `VeloSession::pull` routed through `rdma_pull_with_opts` (regardless of stamped vs legacy inside). |
| `cd_bound_ensure_started` with `num_sequence_hashes>0` | Live CD pull happened (not zero-passthrough). |
| `"asymmetric pull"` in tracing | `SpmdParallelWorkers::dispatch_asymmetric_pull` fired — only triggers when `local_tp != remote_tp`. Symmetric TP=1↔TP=1 will NOT emit this. |
| `"no remote {:?} handle for (instance="` (ERROR) | Stamped path tried but `remote_handles_rank` was empty — pre-fix regression marker. |
| `"Metadata already imported"` (ERROR) | `ensure_remote_metadata` race surfaced — concurrent imports overlapped. |
| Pull succeeds + ZERO of the above ERRORs + symmetric topology | Stamped path with degenerate single-shard plan. Legacy fallback is also consistent with this. |

For a strict legacy-vs-stamped discriminator in a SYMMETRIC topology
(the live smoke is TP=1 both sides), the connector leader's
`init.rs:433` gate is the source of truth: it stamps only when
`reference_config.num_heads.is_some()`. The subagent can confirm by
grepping the prefill log for the rust-side initialisation tracing
(or by checking that the connector exported its template). A more
robust approach is to add a `tracing::info!` in
`rdma_pull_with_opts` legacy-fallback vs stamped branches and grep
for the marker — see the inline comments in
`lib/kvbm-engine/src/leader/instance.rs` around line 1186 and 1196.

## Modes

### Single-leader (default)

```bash
SKILL=/home/ryan/repos/dynamo/.claude/skills/disagg-smoke
bash "$SKILL/two-request-smoke.sh"                                       # operational (default)
KVBM_DISAGG_LEADER=unified bash "$SKILL/two-request-smoke.sh"           # unified leader, operational layout
KVBM_BLOCK_LAYOUT=universal bash "$SKILL/two-request-smoke.sh"          # universal layout
KVBM_DISAGG_LEADER=unified KVBM_BLOCK_LAYOUT=universal bash "$SKILL/two-request-smoke.sh"  # both
```

Output goes to `/tmp/kvbm-experiments/<ts>-two-request/`:
- `hub.log` / `prefill.log` / `decode.log`
- `trace.html` (rendered if `disagg-trace` is available)

### Universal-mode block layout

`KVBM_BLOCK_LAYOUT=universal` activates Universal G2 layout (mode-dominant G2 pinning, fused permute kernels). The value is injected into `kv_connector_extra_config.default.block_layout` in the JSON config rather than relying on env-var propagation — vLLM's EngineCore subprocess does not inherit the parent environment, so the env-var path is silently dropped before the connector runs.

The smoke verifies the requested layout is actually active: after bringup it queries the hub's describe endpoint for both instances and fails loudly on a mode mismatch (exit 1). This assertion would have failed on every pre-fix smoke run that claimed Universal mode.

### Audit-equivalence

```bash
bash "$SKILL/audit-equiv.sh"
```

Runs the smoke twice (legacy then unified). Per-side `audit_diff --normalize-request-ids --filter-prefixes session_factory_active_gauge` asserts the leader-emitted audit signatures match. Exit 0 on equivalence, 1 on divergence.

Prerequisite: `cargo build -p kvbm-connector --bin audit_diff` (the binary lives at `target/debug/audit_diff`).

### Layout-mismatch (incompat-smoke)

```bash
bash "$SKILL/incompat-smoke.sh"
```

Hub-only failure-scenario smoke. Builds + spawns a fresh `kvbm_hub` per scenario on alt ports (`28337`/`21337`, velo disabled). Posts crafted `RegisterRequest` JSON via curl and asserts each mismatch class produces the documented rejection.

Six scenarios:

| # | Setup | Expected |
|---|---|---|
| 1 | Universal baseline registers | HTTP 200 |
| 2 | After universal baseline: operational candidate | HTTP 400, body mentions `mode` |
| 3 | Universal baseline + universal candidate with `num_heads_total` 64 → 48 | HTTP 400, body mentions `canonical` or `head` |
| 4 | Operational NHD baseline + operational HND candidate | HTTP 400, body mentions an operational/NHD/HND term |
| 5 | Operational NHD both sides + `page_size` 16 → 32 | HTTP 400, body mentions `page_size` or `per_worker_config` |
| 6 | CD-only register (no P2P feature) | HTTP 400, body mentions `p2p` or `disagg` |

Each scenario starts a fresh hub so the baseline state is deterministic; total runtime ~2 s. Exit 0 on all-pass, 1 if any scenario produces an unexpected status or missing error keyword (the failing scenario is named on stderr).

Run when the connector or hub gate is touched, before merging changes to `layout_compat`, or as a fast pre-flight before the goldens. Comprehensive Rust-side coverage of the gate lives in `lib/kvbm-hub/tests/cd_layout_compat.rs` (25+ cases) and remains the canonical reference; `incompat-smoke.sh` is the operator-facing one-command version.

Prerequisite: `cargo build -p kvbm-hub --bin kvbm_hub` (no other components needed — no vLLM, no GPU).

## Prerequisites

- `/disagg-bringup` prerequisites (canonical venv, fresh bindings, kernels lib, hub binary).
- `audit_diff` binary built (only required for `audit-equiv` mode).

## Validation expectations (R1 + R2)

- **R1 prefill**: cold cache, ~0/N hit rates; G1→G2 register events fire after forward pass.
- **R1 decode**: full pull pipeline (`worker_pull_chunk_start → worker_session_pull_call → worker_session_pull_returned → worker_g2_to_g1_done`).
- **R2 decode** policy_decision: `matched_tokens >= 48` (3 blocks of warm cache from R1).
- **R2 prefill**: `cd_bound_ensure_started` with `num_sequence_hashes=3`, then `ensure_started_async_onboard` (NOT `ensure_started_zero_passthrough`).
- **R2 prefill** pull-from-decode events fire (`session_pull_request → session_pull_send → session_pull_rdma_*`).
- **R2 decode** pull-from-prefill events fire (the net-new blocks).
- All three logs: zero ERROR lines (excluding the standard UCX / kernel_config noise).

## See also

- `/disagg-bringup` — owns launch-prefill / launch-decode / start-hub / new-experiment scripts.
- `/disagg-teardown` — cleanup between runs.
- `/disagg-trace` — render `trace.html` from an experiment dir.
- `/disagg-hub-curl` — exercise hub HTTP endpoints during a run.
- `/kvbm-maturin-dev` — rebuild bindings (release by default; use a release kvbm build for smokes).
