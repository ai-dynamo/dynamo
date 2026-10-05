# Top-level resume pointer (written 2026-10-03 ~18:50 PDT, before a context compaction)

Read this first after any compaction or restart. For full detail, see:
- the worklog: `<worklog>`;
- `CONTRACT.md` (A1–A18);
- `facts/STATE.md`.

## Running background work (do not message running agents; wait for notifications)

| What | ID | Notes |
|---|---|---|
| Phase 2 workflow | `<workflow-run>` | Tier A is 31/33 done (dualmap-s3 and stickybounded-s3 resuming after the CPU cluster reboot). Then: mid audit, then Tier B/C (M2, M1-v2 per A17, M1-noaff per A15, the AIS league per A16 via `runs/phase2-ais/`), then select+test (finalists, one-shot test, A14 SLA transfer, A5 lag, robustness), then final audit, then report. Resume with `Workflow({scriptPath: …/learned-routing-phase2-<workflow-run>.js, resumeFromRunId: "<workflow-run>"})`. |
| App PR monitor | n/a | Auto-fix is on for #15449, #15450, #15453; events arrive as `<ci-monitor-event>`. Policy: close low-signal threads, rerun flaky CI, minimal obvious fixes only, no features. |

## Top-level duties still owed

1. **Phase 2 report → live finalist runs (A13.2).**
   - Lane ready: `facts/live.json` and `live_smoke.json`; code under `WT/benchmarks/learned_routing/live/`.
   - Run on the GPU cluster 8×H100: default@defaults (repeat it for live noise), the val-best baseline, M1, M0 and
     round_robin, on Mooncake, FAST25 and sessions cells (open and closed). AgentX has no live
     loadgen.
   - The GPU cluster lock is no longer required (A13 addendum). Validation only; never feed selection.
2. **Paper (backloaded, `facts/PAPER_REQUESTS.md` R1 + R2).**
   - One batched pass at the very end, from the partial drafts under `WT/notes/learned-routing/paper/`.
   - R1: a background section on routing primitives, the API, the cost functions under the
     abstraction, and the major campaign caveats.
   - Fill in the results, then run the `/audit` publication pass.
   - Sanitized for public use (A18). Avoid many permission prompts: no tlmgr (packages are already
     installed) and a single workflow.
3. **Public sync** only via `CR/publish/publish.sh sync`, then the scan, then push
   `rupei/learned-routing-public`.
   - Apply the queued fixes in `CR/publish/TODO.md` first.
   - Never push `rupei/learned-routing`, `-ais`, `-lag` or `internal/DO-NOT-PUSH-*`.
4. **Draft PRs not watched by the app:** aisimulate#378 and dynamo#15589. Check them at milestones;
   after each merge or close, their worktrees join the cleanup batch.
5. **End-of-campaign batched cleanup.** The operator approves it once, at the end. Candidates:
   - in the worklog;
   - `CR/CLEANUP.md`;
   - `CR/publish/REPORT.md`, the issues section;
   - the session scratchpad (about 3.9 GB of docs-verification scratch);
   - `/tmp/learned-routing-lit` (now copied to `CR/literature`);
   - remote CPU cluster scratch and the uv-cache-aarch64 directory;
   - the sm-docs and aisimulate-sm worktrees;
   - the lag, ais and public worktrees, once they're done;
   - the internal backup ref.
6. **Sync the notes mirror at milestones** (gate done; next: end of tier B, test, report). Use the
   sanitized pipeline.

## Operator preferences learned this session (also in memory)

- **Decisions:** the operator delegates them ("I'll leave the decisions to you"). Come back only for genuine
  judgment calls.
- **Deletions:** backload them to one end batch unless forced (memory:
  standing-housekeeping-authorization). Remote unrelated holds and artifacts may be cleaned as we go.
- **Public content:** nothing internal (memory: no-internal-infra-in-public).
- **Git:** no co-author trailers on upstream-bound commits; DCO sign-off. Restacks, rewording and
  `git gc` are covered by a standing go.
- **Messaging:** don't message running workflow agents; put late facts into the next stage's prompt
  or a contract amendment.
- **Permission prompts:** minimize them. Each Workflow launch and each out-of-folder or network action
  can prompt, so batch work.
- **AIS:** allowed as an input now (A16). The headline league stays router-observable.

## Interim numbers (validation k0-2, selection split, not final)

| Policy | Clipped log-ratio vs default@defaults |
|---|---|
| M1 | 0.164 |
| ramjet | 0.142 |
| llm-d-precise-prefix | 0.141 |
| two-tier | 0.136 |
| lmetric / M0 | 0.133 |
| sticky-hard | 0.116 |
| llm-d-optimized-baseline | 0.105 |
| chwbl | 0.029 |

- M1 vs the best heuristic: +0.022, below the MDE of 0.038.
- M1 restarts: s1 0.1615, s2 0.1636, s3 0.1578, so the initialization barely matters.

## Update 2026-10-04 09:35 PDT

- The phase2-mid audit flagged the unconstrained M1 for gaming. The A11.2 sign-constrained re-run is now M1 (m1c-s1 g20, val k0-2 0.166; lead over ramjet +0.024, still below the MDE).
- Operator decision on the escalation: no size-aware retune. CONTRACT A19 pre-registers a secondary token-weighted goodput. When Select+Test reports, check that it applied A19; if it didn't, run the rescore as a top-level follow-up before REPORT.
- Operator 2026-10-04 ~09:40 PDT: no GitHub support purge of the leaked SHAs (it was infra names and paths, not internal code). Dropped from owed items.
- Operator 2026-10-04 ~09:55 PDT: aisimulate#378 and dynamo#15589 are marked READY (both CI-green, mergeable, blocked on review). Both are bound in this session with app Auto-fix on. Policy for review comments, mostly bot comments (CodeRabbit etc.): wait a while for them to land; fix the obvious ones; reply to and resolve (decline) the low-signal ones; leave ambiguous ones that need the operator's call OPEN and list them for the operator. No new features. The stack PRs #15449/#15450/#15453 are no longer bound in this session (get_status 09:50); I asked the operator whether to re-bind.

## Update 2026-10-05 02:00 PDT

- Phase 2 (`<workflow-run>`) COMPLETE. Headline: M1-v2 beats tuned ramjet on test, +0.0316, p 0.0017, scoped. REPORT is at `CR/report/REPORT.md`.
- Phase 3 is RUNNING as workflow `<workflow-run>` (script `.../workflows/scripts/learned-routing-phase3-<workflow-run>.js`). Two tracks run in parallel: (1) live prep, then live relays on the GPU cluster, then live analysis, then the live audit; (2) the paper draft. Then paper-final, the publication audit, and publish (only `rupei/learned-routing-public`). CONTRACT A20.
- Waiting on the operator: (a) a faithful-LMetric/SMetric equal-budget follow-up; (b) whether to fix the `lmetric` port on #15450 to add the paper's queued-prefill term.
- Owed, top-level: check aisimulate#378 and dynamo#15589 at milestones (both are bound in the app with Auto-fix on); end-of-campaign batched cleanup.

## Update 2026-10-05 09:30 PDT: cleanup approved (operator)

- **Operator:** "ok 3 also just standing approval on it, do it when you feel like it can be done, just make sure the major campaign logs / notes are saved to that remote branch as we talked about, without any exposing internal info".
- **Order:**
  1. Wait for the paper follow-up `<workflow-run>` to finish (it pushes the public branch).
  2. Archive pass: make sure `rupei/learned-routing-public` carries the major logs and notes (STATE, CONTRACT, DEVIATIONS, UPSTREAM_FOLLOWUPS, audits, REPORT, LIVE, the paper, literature notes, plus a sanitized copy of the worklog worklog), only through `publish.sh` with a fail-closed scan.
  3. Inventory: CLEANUP.md plus the resume list plus remote scratch, each with a keep/delete call.
  4. Delete, top-level, by exact path, and log every path and size in the worklog.
  5. Independent verification.
- **Keep:** the campaign root's facts, runs records, cache and anything the freeze pins; the main campaign worktree; every local branch (remove worktrees only).

## Update 2026-10-05 ~11:55 PDT: gist (operator)

- **Operator:** "maybe make that a github gist if that's sensible". On the follow-up questions they chose a SECRET gist, and to KEEP rupei/learned-routing-public as the artifact (code, raw records, LaTeX sources).
- **Gist plan.** About 20-25 flat files taken from the SANITIZED publish output, never from the internal sources:
  - an index README, REPORT.md, LIVE.md, the paper PDF and REPRODUCE.md;
  - a sanitized campaign log built from the worklog worklog plus STATE.md;
  - CONTRACT.md, DEVIATIONS.md, UPSTREAM_FOLLOWUPS.md, LESSONS.md and BIBLIOGRAPHY.md;
  - key result JSON (HEADLINE_TEST, finalists, test_results, robustness, live_results).
  REPORT image links are rewritten to the branch's raw URLs. The scan is fail-closed before `gh gist create` (secret). Record the gist URL in publish/REPORT.md and the worklog.
- **Order:** after the fix round `<workflow-run>`, which pushes the branch. Then the gist, then the cleanup.
- Operator 2026-10-05 ~12:00 PDT: the gist LINKS to the paper PDF and figures on the branch (GitHub renders PDFs inline); no PDF copy in the gist.
