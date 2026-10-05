# Learned routing in AISim: worklog

**Goal:** learn a classical discrete-choice routing function for Qwen3-32B (vLLM 0.24.0, H100 SXM,
TP2, aggregated) using offline AISim/DynoSim replay. It must beat every heuristic router on held-out
workloads and held-out worker counts, and must not depend on AIS at runtime.

**Pointers**
- Plan: `<repo>/notes/learned-routing/PLAN.md`; the operator decisions are at its bottom.
- Campaign root: `<campaign-root>`.
- Worktree: `<worktree>`, local branch `rupei/learned-routing`,
  based on `origin/rupei/router-policy-aic-ttft`.
- Literature sidecar: `/tmp/learned-routing-lit/`; its lessons go to
  `<campaign root>/literature/LESSONS.md`.

### 2026-10-02 13:02 PDT — Campaign launch

**Operator decisions:**
- Objective: goodput at loadgen-defined loads, open-loop or closed-loop. No capacity-at-SLA search.
- The learned model gets no AIS-derived features.
- AgentX: simulate 128K context; if that fails, token-scale the plays to fit 32K.
- No time cap.
- If the pilot gate fails twice, stop with a diagnostic.
- Bug fixes and plumbing go on the local campaign branch only; push nothing.
- CPU-cluster spillover is allowed. Use the cluster runbook for it.

**Scouting facts:**
- The stack (#15450 → #15453) wires builtin-catalog policies into offline replay; main rejects them.
- Replay sets `affinity_target` to None, so sticky sessions have to be emulated by a policy.
- The default picker breaks ties with unseeded `fastrand`.
- AgentX corpus is HF `semianalysisai/cc-traces-weka-062126@23f152f6` (1.85 GB, SHA-256 `29b6a19e…`).
  82 plays fit 128K.
- Workstation `<workstation>`: 24 cores, 125 GB RAM, 1.2 TB free, idle at launch.

**Rules:**
- Agents never delete files.
- Running agents are never messaged.
- Processes are stopped only by exact PID.
- Commits are DCO-signed with no co-author trailer.

**State:** launching the campaign Workflow and the literature sidecar (run IDs below).

### 2026-10-02 13:20 PDT — Workflows launched

- **Phase 1** (setup → build → calibrate → pilot → gate, with audits at each checkpoint):
  - run `<workflow-run>`
  - script: `<agent-config>`
  - transcripts: `.../subagents/workflows/<workflow-run>/`
  - resume: re-invoke the Workflow tool with `{scriptPath, resumeFromRunId: "<workflow-run>"}`
- **Literature sidecar**: run `<workflow-run>`. PDFs go to `/tmp/learned-routing-lit/pdfs/`, notes to
  `/tmp/learned-routing-lit/notes/`, and ranked lessons to `CR/literature/LESSONS.md`.
- **Shared contract**: `<campaign-root>/CONTRACT.md`.
- **Next step:** when the gate decides, the top-level session reads `LESSONS.md` and the gate output,
  sends a push notification, and starts phase 2 (full tuning → test → report) if the gate says escalate.

### 2026-10-02 13:35 PDT — Disk cleanup (operator-authorized list only)

- **Result:** Used went from 545G to 163G (−382G).
- **Removed with `cargo clean`:**

  | Target dir | Freed |
  |---|---|
  | `<repo>/target` | 250.9 GiB |
  | `<repo>/lib/bindings/python/target` | 77.9 GiB |
  | `<aisimulate-checkout>/target` | 28.8 GiB |
  | worktree `kv-cache-hit-15443` | 24.8 GiB |
  | worktree `offline-replay-optimization-4fabcb` | 22.1 GiB |

- **`uv cache prune`:** freed only 19.4 MiB; the cache is still 14 GiB.
- **Untouched:** the `learned-routing` worktree, `_core.abi3.so`, every `.venv` and `.git`, and the
  <prior-campaign> data.
- **Side effect:** the next `cargo build` or `maturin develop` in the main dynamo checkout and in
  aisimulate will be a full rebuild.

### 2026-10-02 14:10 PDT — Literature sidecar done (<workflow-run>)

- **Output:** 67 papers across 4 angles, with PDFs and notes under `/tmp/learned-routing-lit/`. The
  15 ranked lessons are in `CR/literature/LESSONS.md`.
- **Verification:** an adversarial pass checked every lesson against its PDF page. No lesson was
  deleted; 11 were corrected and 2 non-supporting citations removed.
- **Lessons that change actions:**
  - Replay has no global KV tier, so SMetric's gains don't transfer. Under that condition LMetric loses
    36% to a tuned linear score on agentic traffic, so include a cache-heavy multi-start.
  - The context term should be a column-restricted LCL.
  - The duplication-invariant rank feature is #{j: x_j < x_i}/N.
  - The headline test needs 6–8 trace segments.
  - The concentration flag is min(2/N, 1/N+0.25).
- Phase-1 stages from calibration on read `LESSONS.md` automatically.
- **Setup:** passed its audit after the fixer replaced plain repeats with CRN replicates (contract
  amendment A1, commit `<commit-02>`). The builders have been running since about 14:00.

### 2026-10-02 14:30 PDT — Operator amendment A2, written into CONTRACT.md and PLAN.md

- **Headline:** against the best baseline on the same footing (one config per policy, tuned on pooled
  train data, chosen on val).
- **SLA:** no TTFT SLO. "Good" = mean ITL ≤ I AND E2E ≤ S × E0(ISL, OSL), where E0 is the
  uncontended, no-reuse, AIS-timed latency, used for evaluation only.
- **Goodput:** windowed (LR-01).
- **Baselines:** evaluated as implemented on the branch; no paper-faithful variants or paper
  reproduction, only a brief sanity check.
- **Pickup:** running agents are not messaged. Integration, the build audit and calibration read the
  contract at the start of their stage.

### 2026-10-02 14:35 PDT — Operator guidance for the gate

- If I'm confident the gate result is sound, launch phase 2 without waiting for the operator.
- Come back to the operator for any genuine judgment call (ambiguous gate evidence, scope trade-offs,
  results that contradict the plan's assumptions).

### 2026-10-02 16:20 PDT — Amendment A3: our own AgentX lowering

- **Proposed by the operator.** Lower AgentX ourselves to Agentic Mooncake v2 so plays can recycle,
  with Poisson play arrivals in open loop and native lanes over many copies in closed loop.
- **Why:** native Weka can't hold N = 16/32 workers at steady state (timestamp mode starts every root at
  t = 0; lanes make one pass with no recycling).
- **Guards:**
  - copies get disjoint hash ranges and unique IDs, and draw only from their own split's play pool;
  - a parity check against native Weka on a single play.
- **Mechanics:** a build+verify sidecar workflow reports in `facts/agentx_lowered.json`. Calibration
  takes AgentX last and falls back to native Weka (N ≤ 8) if the sidecar fails.
- **Run:** AgentX lowering sidecar workflow `<workflow-run>` launched 16:25 PDT (build → 2-lens adversarial verify, up to 3 rounds).

### 2026-10-02 16:35 PDT — Amendment A4: trace extension by duplication allowed

- **What:** the operator approved synthetic extension by duplication for AgentX and other families,
  wherever a longer replay is needed.
- **Rules:** disjoint hash ranges and IDs per copy, duplication only within a split's own pool,
  provenance recorded, and duplicates not counted as independent segments.

### 2026-10-02 16:45 PDT — Amendment A5

- **Staleness:** a test-stage robustness check only. A replay router-state lag patch, then the
  finalists re-scored at lag 0/50/200 ms. Training stays on fresh state.
- **Worker-set extrapolation:** counts only; workers stay identical.

### 2026-10-02 16:55 PDT — Amendment A6: straight-main lane

- **What the operator authorized:** pushing production-ready, generic, permanent fixes found during the
  campaign directly to main via `straight-main`.
- **Where:** only in dedicated clean worktrees off `origin/main` (`sm-dynamo`, `aisimulate-sm`), never in
  the main checkout, the campaign WT, or the aisimulate checkout's `codex/replay-small-fixes` branch.
- **Process:** campaign agents flag candidates in `UPSTREAM_FOLLOWUPS.md`. A prep-and-review workflow
  ports them and checks them independently; the top-level session runs `straight-main`.
- **First batch:**
  - #1: the `seed` parameter on the default cost function;
  - #4: the plugin default picker ignoring the request temperature override;
  - #6: synthetic per-request session IDs passed only in closed loop (AISim side).
- **Run:** straight-main prep workflow `<workflow-run>` launched 17:00 PDT (candidates #1, #4, #6; port → 2-lens review → 1 fix round → re-review). The top-level session runs straight-main on candidates that come back ready.

### 2026-10-02 17:35 PDT — Straight-main lane results (`<workflow-run>`)

- **#6 (aisimulate):** commit `1bf36cfb`. The direct push to main was REJECTED (ruleset: 3 required
  status checks). Pushed instead as branch `rupei/replay-withhold-synthetic-session-ids`; the draft PR
  is pending `gh` re-auth.
- **#1 (seed):** held on local `rupei/sm-dynamo` (`fdfbf834ad`). Not useful on main, so it's a #15450
  stack follow-up.
- **#4 (temperature override):** intentional per #14785; only a stale docs example remains.
- **Worktrees:** `<other-worktree>` and `<aisimulate-checkout>`
  remain, and are cleanup candidates later.

### 2026-10-02 17:20 PDT — Status and operator decisions

**Campaign status:**
- Build audit round 0's 4 majors are all fixed. In round 1, leakage and harness/splits passed; the
  goodput lens found a drain-window major, which was fixed in commit `<commit-12>`. Waiting on the
  round-2 goodput re-audit.
- The AgentX lowering sidecar is OK: it uses AISim's own WekaImporter via a Rust CLI, parity holds
  84/84, and the isolation audit passed. The parity-fidelity audit is pending.

**Operator decisions:**
- (a) Push the seed commit onto `rupei/router-policy-ports` (#15450).
- (b) Open a draft PR fixing the stale override examples in `router-examples.md`.

**Run:** prep-and-review workflow `<workflow-run>`. The top-level session pushes after review.
The operator was asked to re-run `gh auth login` so the draft PRs can be opened.

### 2026-10-02 17:45 PDT — Phase 1 resumed; side tasks

- **Phase 1 `<workflow-run>`** stopped at the build audit: the round-2 goodput lens had a major (an E0
  off-by-one plus a zero-tolerance knife-edge at S×scale = 1; AgentX 11–12% of default rows sit
  exactly at E0).
  - Changed the script's build-audit maxRounds from 2 to 3 and resumed from cache as task `wjcrcfr07`.
  - That runs the fixer for round 2, re-audits, then continues to calibrate, pilot and gate.
- **AgentX lowering sidecar `<workflow-run>`:** the parity-fidelity r0 major (the generator can't
  express AgentX `think_mult`/`osl_mult` cells, 5 of 22) is in its fix round.
- **Docs fix:** branch `rupei/docs-router-override-examples` pushed (`dc5a37d714`, `782f5ad9bc`; review
  passed, minors addressed). Draft PR pending `gh` re-auth.
- **Stack-seed:** the first attempt declined because it couldn't see the operator's answer. Rerunning
  with the answer quoted verbatim (`<workflow-run>`); the top-level session pushes
  `rupei/router-policy-ports` after review.

### 2026-10-02 18:05 PDT — gh re-authenticated; draft PRs opened

- **ai-dynamo/aisimulate#378** (draft): fix(replay): withhold synthesized trace session IDs from
  placement. Head `rupei/replay-withhold-synthetic-session-ids`. Complements aisimulate#177.
- **ai-dynamo/dynamo#15589** (draft): docs(router): correct per-request router override examples. Head
  `rupei/docs-router-override-examples`.
- Both are bound to the session's PR tracking. CI is not being polled.

### 2026-10-02 18:25 PDT — Seed pushed onto #15450

- `rupei/router-policy-ports` went from 49d940ef5d to bc5e9f7c8d (fast-forward; operator-approved).
  - `590806646b` adds the optional seed to `dynamo-default-cost-fn`.
  - `bc5e9f7c8d` adds a docs cross-reference.
- **Evidence:** the review passed. Default replay is byte-identical to the base on the 4 canonical rows.
  A seeded YAML policy gives identical routing across processes; unseeded, two runs routed 2,468
  requests differently.
- **Not done:** the operator restacks `rupei/router-policy-aic-ttft` themselves.
- **Cleanup candidates** (reproducible; not deleted): `stack-ports/{target,.scratch,.venv}` and the
  stack-ports worktree; `sm-dynamo` (holds `fdfbf834ad`, now superseded); `sm-docs` once #15589
  merges; `aisimulate-sm` once #378 merges.
- **Unrelated:** git warns that the shared repo has too many unreachable loose objects (a `gc.log` sits
  in the worktree metadata). It needs a `git prune` decision from the operator; not touched.

### 2026-10-02 18:55 PDT — Stack housekeeping, restack and PR monitoring

- **Wording:** `rupei/router-policy-ports` docs commit message reworded, `bc5e9f7c8d` → `2827da7ce0`
  (same tree, signed). Pushed with force-with-lease under the operator's standing authorization (now a
  memory).
- **Cleanup:**
  - removed the `stack-ports` and `sm-dynamo` worktrees, plus local branch `rupei/sm-dynamo` (same
    patch-id as `590806646b`);
  - `cargo clean` on `aisimulate-sm`, freeing about 20 GB.
- **Git housekeeping:** `git gc --prune=1.day.ago` in dynamo took loose objects 8,593 → 65 and packs
  41 → 2 (839 → 737 MiB). Stale worktree `gc.log` files removed.
- **Restack:** `stack-aic` has an unpushed merge `2619682a56` (ports into aic-ttft); builtin tests pass.
  The layered main → api → ports → aic merge is running as workflow `<workflow-run>` with conflict
  resolution, parity against main, and adversarial review. The top-level session pushes after review.
  - Conflicts against main: ports in `offline_replay_bench.rs`; aic in that file plus
    `selector/mod.rs` and the mocker's `kv_router/mod.rs`.
- **PR monitoring:** the app's auto-fix and comment monitor is on for #15449, #15450 and #15453.
  Operator rules:
  - close low-signal comments;
  - rerun flaky CI;
  - make only minimal obvious fixes, with no new features or behavior changes unless really warranted.

### 2026-10-02 19:05 PDT — Stack review comments arrived (app auto-fix events)

- **Volume:** #15449 had 7 comments, #15450 had 26 plus the base-conflict verdict, #15453 had 14.
- **Triage:** read-only triage and skeptic workflow `<workflow-run>`, categorizing each item as fix,
  close or flag under the operator's policy. Fixes are applied after the restack workflow
  `<workflow-run>` finishes, so the two don't collide on the branches.
- **Plan:** push restack merges and minimal fixes together, then reply to and resolve threads. Flag API
  design calls to the operator (the public `SchedulingRequest` field, the CPU-cost question).
- **2026-10-02 19:15 PDT, operator decision on #15453:** the source-breaking public
  `SchedulingRequest.modeled_prefill_backlog_ms` field is accepted. Close those threads with a reply
  saying so; no refactor.

### 2026-10-02 19:25 PDT — AgentX lowering sidecar complete (`<workflow-run>`)

- **Result:** both audit lenses PASS in round 1 (after the round-0 transform fix, commit `<commit-13>`).
  `facts/agentx_lowered.json` status is ok.
- **Lowering:** the AISim WekaImporter via a Rust CLI. Graph digests match for 82/82 plays, and replay
  parity holds 84/84 for the base and each of 5 transforms.
- **Loads found:**
  - closed loop is stationary at 2–6 lanes per worker;
  - the open-loop cliff sits between 0.002 and 0.0025 plays/s per worker, so there is no stationary
    open-loop L3 and L3 must use closed lanes.
- **Calibration must wire:** `cells.py`, which accepts agentic_lanes for weka only, and `replicates.py`,
  which rejects agentic_mooncake (replicate k = GenSpec seed k).
- **Scratch:** 4.0 GiB in `traces/agentx_lowered/gen`.
- **2026-10-02 19:30 PDT:** replied on #15453 to the CPU-cost question (the default path is untouched; the 1,024/4,096-worker benchmark is deferred): https://github.com/ai-dynamo/dynamo/pull/15453#issuecomment-5963889136

### 2026-10-02 19:40 PDT — Operator decisions before stepping away

- **Confirmed bugs in ported policies:** fix them if the fix fits without adding much core-router API
  change. Otherwise leave the code and push a loud TODO/NOTE comment at the site.
- **Replay device-tier overlap fix**, if confirmed and default replay stays byte-identical: put it on
  the stack (#15450) AND cherry-pick it onto the campaign branch before phase-2 baseline tuning.
- **Genuine design or judgment threads:** leave them open, with no reply, and summarize them for the
  operator on return.
- **Already decided:** the `SchedulingRequest` source break is accepted (close those threads), and the
  CPU-cost reply is posted.

### 2026-10-02 19:55 PDT — Stack review triage results (`<workflow-run>`)

- **Already resolved:** almost all threads the app relayed were resolved earlier by the author.
  - #15449: 4/4 resolved.
  - #15450: 18/18 resolved. The device-tier fix is already at head with a test; CHWBL matches KubeAI
    #528; the llm-d zero penalty matches upstream.
  - #15453: 8/9 resolved.
- **Closed now:** the #15453 thread PRRT_kwDOOCjvss6n2XpI (test-assertion nit), with a reply and resolve.
- **Pending fix:** #15450's `RouterPolicy::load` snapshot race (bench-only; parse the retained bytes),
  applied after the restack.
- **A7 corrected:** the device-tier fix is already in the campaign base.
- **Note:** the restacked heads need fresh `/ok to test` if copy-pr-bot doesn't auto-trigger.

### 2026-10-02 20:30 PDT — Stack restacked onto main and pushed (`<workflow-run>`, review PASS)

- **api** b80ecf6fa4 → 2c41b73ce6, a clean merge plus 3 test call sites adapted to the capacity
  lookup.
- **ports** 2827da7ce0 → 299620d275, reconciling `offline_replay_bench.rs`. Main's
  `--router-queue-threshold` and ports' `--router-policy-config` now compose.
- **aic** 2be8d9c43a → 7578d04869, keeping both the RequestSnapshot backlog field and main's
  raw-cached Cell, and the opt-in fill after main's token_seq move.
- **Parity:** all layers have replay digests equal to origin/main 2fde30bbb5 on 6 rows, each with 2
  processes.
- **State:** all three PRs are MERGEABLE (blocked only on review and checks).
- **Next:** `RouterPolicy::load` snapshot-race fix on ports, then merge up into aic (`<workflow-run>`).
- **Scratch:** the stack-api, stack-ports and stack-aic worktrees remain; clean up after the bench fix.

### 2026-10-02 20:55 PDT — Bench race fix pushed; stack worktrees cleaned (`<workflow-run>`, review PASS)

- **Fix:** ports 299620d275 → 15d891da1f, `fix(bench)`: parse the retained policy bytes and seed
  `policy_config_cache`. aic 7578d04869 → c4dcfcd709 (merge).
  - Race probe: 3,387 of 20,000 loads mismatched before, 0 of 20,000 after.
  - Digests are unchanged with and without a policy file.
- **Reply:** posted on resolved thread 4149911252 (discussion r4171157926).
- **Cleanup:**
  - removed the stack-api, stack-ports and stack-aic worktrees (39 GB; branches equal origin);
  - removed about 11 GB of finished workflow scratch from the session scratchpad;
  - kept the small evidence directories.

  Disk used is now 231G.

### 2026-10-02 21:00 PDT — Cleanup policy: backload deletions

- **Operator request:** batch all cleanup into one end-of-campaign pass for approval. No more ad hoc
  deletions unless forced, e.g. by disk pressure.
- **Running list of end-of-campaign cleanup candidates:**
  - `<other-worktree>` (once #15589 merges or closes);
  - `<aisimulate-checkout>` (once aisimulate#378 merges or closes);
  - `CR/traces/agentx_lowered/gen` (4.0 GiB) and other campaign scratch, after the report;
  - remaining small evidence dirs in the session scratchpad;
  - `/tmp/learned-routing-lit` (keep until the report is written);
  - the campaign worktree's `target/` and `.venv`, after the campaign.

### 2026-10-03 01:20 PDT — Repo notes mirror (operator request; contract amendment A8)

- **Location:** campaign branch `notes/learned-routing/`.
- **Mirror:** `sync_from_campaign.sh` copies curated small files from CR (2.4 MB; 52 files; per-file
  cap 1 MiB).
- **Docs:** `README.md` and `REPRODUCE.md` are being written and verified by workflow
  `<workflow-run>`.
- **Publishing:** push `rupei/learned-routing` to origin with an explicit refspec (the branch has no
  upstream by design); no PR. Re-sync at the gate, at phase-2 checkpoints and at the end.
- **This worklog** stays the operator diary.
- **2026-10-03 01:40 PDT:** notes mirror commit <commit-17> pushed as `origin/rupei/learned-routing` (8.3 MB, 124 files). CPU-cluster lane workflow `<workflow-run>` launched; contract amendment A9 says to use CPU cluster for any batch projected over about 1 h local.
- **2026-10-03 01:50 PDT:** operator OKs multiple concurrent CPU-cluster nodes, mixed node types allowed (A9 addendum; parity smoke required once per OS/glibc image).

### 2026-10-03 02:20 PDT — Repo notes: first docs pass and a follow-up

- **Docs pass** (`<workflow-run>`): README and REPRODUCE written and verified. Hashes, regeneration,
  CLIs and the derivation chain all check out. Two majors were found:
  - the AgentX dependency was overstated (5 of 26 cells need non-cap300 manifests);
  - off-host inputs were missing from git.
- **Mirror extended:** calibration driver scripts, all six AgentX base manifests,
  `write_engine_config.py`, the cells_validation files, and literature notes and bibliography. The cap
  is now 5 MiB per file with a 25 MiB budget; the mirror is 15 MB.
- **Literature persisted:** copied from `/tmp/learned-routing-lit` to `CR/literature/` (147 MB,
  including PDFs). The `/tmp` copy is now a cleanup candidate.
- **Pending fix:** `campaign/audits/build/` is gitignored by the `Build/` rule, so the next commit uses
  `git add -f`.
- **Incident:** the writer agent put the operator's email in a crates.io User-Agent header (one
  request). Reported to the operator; later prompts forbid personal identifiers in network requests.
- **Running:** docs fix and verify workflow `<workflow-run>`.

### 2026-10-03 02:45 PDT — Notes pushed; calibration done; pilot starting

- **Calibration:** fix r0 (session warm-up) finished at 01:57 and re-froze the cells as `lr-cells-v5`
  (test `7b998b8e…`, sessions SLA I 46.05 ms / S 2.118). Re-audit r1 PASSED at 02:33. Phase 1 has
  moved on to the pilot.
- **Notes:** verified README and REPRODUCE plus a mirror (19 MB) pushed: `origin/rupei/learned-routing`
  at <commit-20> (snapshot <commit-19>).
  - The archival script copies are committed verbatim, skipping the Python formatters and whitespace
    hooks for that path.
  - `audits/build` was force-added past `.gitignore`.
- **Cleanup candidates:** about 3.9 GB of docs-workflow regeneration scratch in the session scratchpad,
  and `/tmp/learned-routing-lit` (now copied to `CR/literature`).

### 2026-10-03 02:51 PDT — CPU-cluster lane ready (`<workflow-run>`; verify PASS)

- **Parity:** remote replays are bit-exact with local (6,312 records; 48.9M per-request rows), plus an
  independent re-check on 4 node images.
- **Throughput:** EPYC 9654P nodes run about 2.5x the workstation, EPYC 7702P about 1.2–1.4x; the 5
  measured nodes together are about 8x.
- **Remote lr-train:** works, with numerics pinned for bit-exact CMA-ES.
- **Allocations:** all 19 jobs recorded and none held.
- **Hard deadline:** CPU cluster maintenance reboot at <cutoff>; jobs must end by then (contract amendment
  A10).
- **Remote scratch:** 7.5 GB at `<cpu-cluster>:<scratch>`,
  added to the cleanup list.

### 2026-10-03 05:30 PDT — PHASE 1 COMPLETE: gate ESCALATE; phase 2 launched

**Pilot validation** (14 cells, fresh replicates k3–10), clipped log-ratio vs default:

| Policy | Score | Notes |
|---|---|---|
| M1 tuned | **+0.165** | ratio 1.19, 13/14 cells better, segment Δ +0.147 (SE 0.044) |
| llm-d-precise-prefix | +0.135 | |
| ramjet | +0.135 | |
| lmetric | +0.133 | |
| M0 tuned | +0.128 | |
| round_robin | −0.577 | |

- **Errors:** 0 across 56,552 replays.
- **Headline risk:** M1 vs the best heuristic is +0.030, below the MDE of 0.038. The lead comes mostly
  from one Mooncake segment, and M1 trails on all 3 AgentX segments. M1 also exceeds the
  worker-share cap on 5 Mooncake val cells.

**Gate:** facts/gate.json full_plan.
- Tier A: 10 tuned baselines plus M1, 3 restarts each, B = 400 per run.
- Tier B: M2 rank 2.
- Tier C: conditional ablations.
- Projected about 40 h wall on the CPU cluster plus the workstation.

**Amendment A11** (top-level, from verified lessons):
- M1 multi-start (θ0; a behavior-clone ML fit to llm-d-precise-prefix on train; cache-heavy);
- a concentration gaming test, with sign-constrained re-runs if it's gaming;
- the N=6 test labeled selection-exposed;
- the lag patch in a separate worktree (`rupei/learned-routing-lag`).

**Phase 2:** workflow `<workflow-run>`.
- Stages: Prep plus the lag build, then relay-orchestrated Tier A, mid audit, Tier B/C, select and
  test, final audit, report.
- Script: `<agent-config>`
- Resume with `{scriptPath, resumeFromRunId: "<workflow-run>"}`.

**Constraint:** CPU cluster reboots <cutoff>; jobs must end before it, then resume afterwards.
**Notification:** push notification sent to the operator.
- **2026-10-03 05:40 PDT:** notes mirror synced at the gate (pilot.json, gate.json, pilot histories and best configs; 20 MB) and pushed to `origin/rupei/learned-routing` at <commit-21>. The sync now skips duplicate training `results.jsonl` (cache-ingested). The push notification wasn't delivered (Remote Control inactive).
- **2026-10-03 05:50 PDT:** operator OK'd the M1 heuristic-fit init with disclosure. Amendment A12 adds 'M1-default-init' (s1 only) as a test finalist and a report row.

### 2026-10-03 06:00 PDT — Paper backbone drafting track started

- **Operator request:** a LaTeX and PDF write-up, paper-ish, for team circulation. Use the `/audit` skill
  at the final pass.
- **Location:** `notes/learned-routing/paper/` on the campaign branch, alongside README, REPRODUCE and
  `campaign/`, as the operator asked.
- **Workflow:** `<workflow-run>`: skeleton (TinyTeX latexmk), 6 parallel section writers, integrate
  and build, then fact-check against the campaign facts, the code and the literature PDFs.
- **Results:** sections are `\pending{}` until phase 2 completes; pilot numbers are labeled
  preliminary.
- **Next:** at the end, an `/audit` publication-ready pass, then commit and push the sources plus the
  PDF.
- **2026-10-03 06:10 PDT:** operator paper request R1 (routing-primitives background section, cost functions under the abstraction, major campaign caveats) recorded in `facts/PAPER_REQUESTS.md`; it runs as a restructuring pass after the backbone.

### 2026-10-03 06:25 PDT — Live GPU validation track (contract amendment A13)

- **Operator request:** live GPU runs to check that the relative results hold, with AIPerf as loadgen
  and semantic equivalence to offline replay.
- **Live-lane prep:** workflow `<workflow-run>`.
  - Build: an AIPerf input generator and the A2 scoring adapter; a Dynamo+vLLM 0.24 Qwen3-32B TP2 H100
    deployment recipe with catalog router policies.
  - An offline per-request loadgen semantic-equivalence audit.
  - One 8×H100 GPU smoke (default router, one Mooncake cell) compared against offline replay.
  - An independent audit.
- **GPU etiquette:** the GPU cluster cooperative lock was free at launch (the last hold was released
  2026-09-28). One owned 8-GPU request at a time; jobs and cancel commands go in `facts/live.json`.
- **Later:** live finalist runs once phase 2 writes `facts/finalists.json`. Report paired deltas and
  Kendall τ, sim vs live. Validation only, never selection.

### 2026-10-03 06:40 PDT — Remote cluster cleanup (operator: nothing else uses GPU cluster or CPU cluster)

- **Request:** remove unrelated holds and artifacts on the GPU cluster and CPU cluster. The GPU cluster lock is no longer needed
  (A13 addendum; the `<prior-campaign>` memory is updated).
- **Workflow `<workflow-run>`:** inventory, skeptic, execute.
  - Auto: cancel stale non-campaign jobs, and delete disposable artifacts unrelated to Dynamo/AISim.
  - Protected: campaign jobs and paths per the facts and job records, everything Dynamo/AISim,
    dotfiles, credentials, toolchains and HF weights.
  - Unrelated research data, such as the other campaign's outputs, gets listed for one operator
    confirmation, not deleted.
- **2026-10-03 06:55 PDT:** amendment A14 pre-registers an SLA-transfer secondary analysis on test (scale sweep, ITL-only, E2E-only, length-scaled TTFT, absolute E2E; constants from train only), before `finalists.json` exists.
- **2026-10-03 07:10 PDT:** amendment A15 adds a mandatory session-affinity ablation (M1-noaff with θ₆ pinned at 0, plus a post-hoc θ₆=0 check). Operator intuition plus pilot evidence (θ₆ small); tested under lag as well.

### 2026-10-03 07:45 PDT — AIS-informed features now allowed (amendment A16)

- **Operator:** retracted the no-AIS constraint; AIS may now inform features (estimated prefill time,
  remaining in-progress prefill, and similar).
- **Design:** feature set v3 = v1 + `est_prefill_ms`, `prefill_backlog_ms` (the PREFILL_TIME input),
  `est_ttft_ms`, `est_decode_step_ms`, `itl_externality_ms`, plus context sources (min est TTFT, mean
  decode step).
- **AIS league:** M1-ais and M2-ais, against tuned M0-ais (AIS prefill-load model) and tuned
  llm-d-optimized-baseline (modeled TTFT). Everything is normalized to the same non-AIS default
  reference.
- **Headline:** stays the router-observable league (pre-registered). The AIS league is a pre-registered
  secondary comparison.
- **Workflow `<workflow-run>`:** build v3 in an isolated worktree `rupei/learned-routing-ais` with its
  own build and bundle, verify (estimates, leakage, live info parity, build parity), and plan the arms.
  Phase-2 tier B/C relays schedule them once `facts/ais_features.json` is ok.
- **2026-10-03 08:00 PDT:** operator delegated decisions; amendment A17 makes M1-v2 a mandatory tier-B arm (inits include LMetric's representable point), sets a symmetric headline learned-arm selection on fresh val k3-10, and fixes the compute cut order.

### 2026-10-03 10:20 PDT — Remote cluster cleanup done (`<workflow-run>`)

- **Inventory:** 116 items. GPU cluster has no jobs. CPU cluster has only the 12 protected `lr-train-p2a-*` campaign
  jobs, so there was nothing stale to cancel.
- **Deleted:** 3 empty CPU cluster slurm stubs (402 B: unrelated jobs).
- **No unrelated research data found.**
- **Left for the end batch:**
  - `<cpu-cluster>:<scratch>/uv-cache-aarch64` (205 MB of Python wheels, unrelated and
    disposable);
  - the `slurm-<job>.out` stub (a CPU cluster storage benchmark).
- **Left alone:**
  - a GPU-cluster shared-storage `.bashrc` owned by another user inside the project dir (not ours to touch);
  - 34 Dynamo-related July–August slurm stubs, reclassified as protected.
- **Record:** `facts/cluster_cleanup.json`.
- **2026-10-03 10:30 PDT:** operator stopped the paper workflow `<workflow-run>` (too many permission prompts). The paper is backloaded to the end as one batched pass (R1, results, /audit); see `facts/PAPER_REQUESTS.md` R2.

### 2026-10-03 10:45 PDT — AIS features (v3) ready (`<workflow-run>`; verify PASS)

- **Build:** isolated worktree `rupei/learned-routing-ais` (<commit-ais-01>, <commit-ais-02>, <commit-ais-03>), build
  555ca082.
- **v3 = v1 + 5 AIS features:**
  - own prefill: exact AIS through a new research input `REQUEST_PREFILL_TIME`, or a closed-form
    fallback (6.6% mean error);
  - backlog: the PREFILL_TIME input;
  - estimated TTFT;
  - decode step and ITL externality: AIS-calibrated closed form, 2.3% mean error.
- **Verification:**
  - 65,805 decisions recomputed independently, all matching;
  - OSL changes 0 decisions;
  - existing policies are byte-identical to the tier-A build (36/36);
  - CPU cluster parity holds on 3 node types.
- **AIS host load model:** changes default's routing but not its goodput beyond noise (ratio 1.0006 ±
  0.017).
- **Informational, train only:** a hand-written min-estimated-TTFT rule scores +10.7% over default.
- **Arms:** `runs/phase2-ais/MANIFEST.json`, 12 runs (M1-ais, M0-ais, llm-d-modeled × 3, then M2-ais ×
  3). A second orchestrator instance yields to tier A.
- **Live info-parity caveat:** a live router lacks per-worker own-prefill from AIS until follow-up #15;
  live v3 would use the closed form.

### 2026-10-03 11:00 PDT — Live lane READY (`<workflow-run>`; audit PASS)

- **Loadgen:** AIPerf 0.13.0, upstream. `gen_aiperf_inputs.py` is exactly equivalent to replay for
  Mooncake, FAST25 and synthetic sessions, in open and closed loop. Checked on 13 cells:
  - arrivals bit-identical;
  - ISL/OSL exact;
  - prefix LCP structure identical at engine blocks;
  - sessions and think times;
  - closed-loop handoff;
  - windows.

  Stub-server runtime check: 45,462 requests, goodput moved at most 0.06%.
- **AgentX:** no faithful live loadgen (needs a dedicated driver). Live validation excludes AgentX.
- **GPU smoke on the GPU cluster** (<node> job <job>, plus <job> to verify a lifecycle fix): default router,
  full Mooncake cell `mooncake-w2-base-n4-open-L2` k0 (4,002 requests).
  - Live goodput 3.4454 vs replay 3.4535 (ratio 0.9976).
  - Prefix-hit rate 0.3599 vs 0.3589.
  - Live TTFT is higher and live ITL lower than AIS; the two offset each other in E2E.
- **Cost:** 6.19 GPU-hours in total; nothing held now.
- **Fixed during the smoke:** 4 deploy bugs, including one where the check-phase frontend survived and
  served the cell (DEVIATIONS).
- **Code:** commits under `live/` in WT.
- **Next:** live finalist runs after `facts/finalists.json`. Repeat default@defaults for live noise;
  cover closed-loop and sessions cells.
- **Decision:** don't build a live AgentX driver now. Revisit if AgentX turns out to decide the
  headline.

### 2026-10-03 11:20 PDT — PUBLIC EXPOSURE FIX: campaign branch deleted from origin

- **Operator rule:** no internal NVIDIA infra in anything public. Saved as a memory and as amendment
  A18.
- **What leaked:** `origin/rupei/learned-routing` (head <commit-21>) had internal identifiers in 83
  changed files (cluster, node, partition and account names, internal paths) and 2 commit messages.
- **Action:** deleted from origin at about 11:15. Local branch intact; reference backup at
  `rupei/learned-routing-public-backup-<commit-21>`. GitHub may keep the commits reachable by SHA for a
  while; a support purge is optional.
- **Scanned clean:** all PR bodies and comments (#378, #15589, #15449/50/53), the stack branches (the
  only hits are main's own content), the docs branch, the aisimulate branch.
- **Next:** build a sanitized publish pipeline (`public-publish` workflow) and republish a clean
  squashed history.

### 2026-10-03 16:15 PDT — Sanitized public branch pushed (`<workflow-run>`; independent scan PASS)

- **Pushed:** `origin/rupei/learned-routing-public` at 8a8bd0fa9a. 5 clean commits on public base
  2be8d9c43a: plugins, harness and workloads, Slurm CPU runner, live lane, notes.
- **Scans:** the fail-closed scan is CLEAN (432 files, 5 commits, 44 internal commits checked). My
  independent grep also found 0 hits. Router and bindings code is byte-identical to the internal tuning
  build.
- **Tests:** 17/17 functional-equivalence cases pass (site values via local site.env only), plus cargo
  127 and pytest.
- **Pipeline (local only):** `CR/publish/` (denylist, scan.py, sanitize.py, publish.sh, overlays,
  site/). All future syncs go through `publish.sh sync`. Report at `CR/publish/REPORT.md`; minor queued
  fixes at `CR/publish/TODO.md`.
- **Backup ref renamed** to `internal/DO-NOT-PUSH-leaked-<commit-21>`, upstream unset. It's an
  end-batch cleanup candidate.
- **Note:** the AIS and lag branches (`rupei/learned-routing-ais`, `-lag`) are local only; they get
  published through the same pipeline later.
- **2026-10-03 18:50 PDT:** wrote `CR/facts/TOP_LEVEL_RESUME.md` (running work, owed top-level duties, preferences, interim numbers) ahead of a context compaction the operator asked about.
- **2026-10-04 09:35 PDT:** operator decision on the phase2-mid escalation. The constrained M1 still concentrates short requests on one worker and loses some long-prompt goodput. Decision: no size-aware retune; keep the request-count headline and the LR-13 flags. Pre-registered CONTRACT A19 (secondary token-weighted goodput `good_tokens_rps_window`, run through the headline pipeline, never used for selection) and added a HEADLINE_TEST addendum, both before any test evaluation. Status: tier B/C 7/30 at 07:41; the CPU cluster round ends ~11:15-11:50.
- **2026-10-04 09:40 PDT:** operator declined the GitHub support purge of the leaked commits. They exposed internal infrastructure names and paths, not internal code. The item is closed.
- **2026-10-04 09:55 PDT:** marked aisimulate#378 and dynamo#15589 ready for review (operator). Both are CI-green. Bound both in the app with Auto-fix on. Comment policy: fix the obvious ones, decline the low-signal ones, leave the ambiguous ones open for the operator.
- **2026-10-04 ~16:30 PDT:** aisimulate#378: CodeRabbit posted one minor comment (the synthetic_session_id doc comment overstated what is withheld). Fixed in 2b1c2f4f (doc-only, cargo fmt check clean), pushed as a fast-forward, replied, and resolved the thread. dynamo#15589: the CodeRabbit summary plus an APPROVED from dynamo-review-agent; no threads; still REVIEW_REQUIRED (code owner). Nothing needs the operator's call.
- **2026-10-05 01:55 PDT, phase 2 complete** (`<workflow-run>`; 30 agents, ~44 h). Headline on test: M1-v2 vs tuned ramjet (val-best), segment mean +0.0316 (95% CI [0.015, 0.050]), 10/12 segments ahead, one-sided exact Wilcoxon p 0.0017. That is below the 0.038 MDE and the 0.060 robustness spread, so only the sign is claimed. M1-v2 beats all 11 tuned baselines individually. The sign holds under lag 10/50/200 ms and every timing perturbation; the policy ranking is stable (τ 0.80-0.95).
  - Caveats: A19 token-weighted goodput is -0.009 (10/12 ahead, one outlier segment); TTFT p90 for 32K-64K prompts is about 14% worse than ramjet; concentration flags hold on every learned arm; significance is lost at SLO scales 0.5-0.75.
  - Final-audit escalation: faithful LMetric (with the paper's queued-prefill term, which our port omits) is untuned yet beats all tuned ported baselines on val, and default + faithful LMetric gets 53-66% of M1-v2's margin. Headline scoped to "beats the branch's ported heuristics, each tuned with an equal budget". Operator to decide on the follow-up and the port fix.
  - REPORT at `CR/report/REPORT.md`. CONTRACT A20 written for phase 3.
- **2026-10-05 02:00 PDT:** launched phase 3 workflow `<workflow-run>`: live finalist runs on the GPU cluster, the paper in parallel, then the publication audit and the sanitized public sync (A20).
- **2026-10-05 08:10 PDT, lmetric port fixed (operator):** `bba12df941` on #15450 adds queued prefill to the P-token (regression test, clippy, a 2-lens review on paper fidelity and regressions). Merged into #15453 at `33a53229af`. PR #15450 body updated with the formula and the re-run replay rows: mean TTFT 263→180 / 291→190 / 194→139 ms, with ramjet and two-tier reproducing exactly. New worktree `.claude/worktrees/router-policy-ports` (with its release and debug target dirs) added to the cleanup batch. CONTRACT A21 tells the paper and report stages to disclose that the frozen campaign used the pre-fix port.
- **2026-10-05 ~08:30 PDT, CI reruns:** aisimulate#378 Full CI (run 37243346425): the macOS wheel build failed with a crates.io index connect timeout (infrastructure), rerun as attempt 2. dynamo#15449 PR run 37086387586: rust-gpu cancelled at its 30-minute limit in KVBM disk-transfer tests, after network retry warnings, on code the PR doesn't touch (infrastructure), rerun.
- **2026-10-05 ~08:45 PDT:** phase 3 complete. Public branch at `d9e92375dc`, with notes, REPORT, LIVE and the 112-page paper; scans clean. Launched follow-up `<workflow-run>`: author "Rudy Pei", the A21 lmetric disclosure, the remaining pending markers, then rebuild, re-sync and an independent check.
- **2026-10-05 09:30 PDT:** operator gave standing approval for the end-of-campaign cleanup, on the condition that the major logs and notes are archived, sanitized, to the public branch first. Plan: wait for the follow-up push, archive through publish.sh, inventory, delete top-level, verify.
- **2026-10-05 ~09:45 PDT, paper restructure (operator):** use /audit, and keep the main text's narrative clean, moving technical detail that adds no insight into footnotes and the appendix. Route: source editing as a publication-ready pass. Claims and novelty checks are reused because the claims are unchanged; each move gets a meaning-preservation check; then an editorial and visual review of every page. Plan: (1) read-only diagnosis (a structure planner plus an isolated introduction reader), adjudicated by me; (2) per-file writers with exclusive file ownership, rebuild, checks, and sanitized re-sync. Starts after follow-up <workflow-run> finishes; cleanup comes after the restructure.
- **2026-10-05 ~10:05 PDT, restructure diagnosis launched:** workflow `<workflow-run>` (structure planner writing `<scratch>/restructure/plan/EDIT_PLAN.md`). Its fresh-reader agent got a placeholder instead of the introduction (my error), so I'll ignore that output. A separate isolated fresh reader was launched with the real introduction text, rendered from the PDF, with no paths and no other context. Both are read-only.
- **2026-10-05 ~10:40 PDT:** restructure plan adjudicated: `<scratch>/restructure/plan/EDIT_PLAN.md` (main text ~97 → ~50 pp, 14 appendices, 5 writers plus an integrator) and DECISIONS.md (main-text figures regenerated, title box neutral, R1 split accepted, provenance appendix public, all fresh-reader blockers accepted). Launching the writers and integrator, with read-only review afterwards.
- **2026-10-05 ~10:45 PDT, restructure execution launched:** workflow `<workflow-run>` (journal `<agent-config>`). Five writers W1-W5 own disjoint files and build private copies under `<scratch>/restructure/build-W*`. Then integrator W6 builds in place, regenerates tables/figures, runs the fact-preservation diff and label grep, and commits in the WT. Then three read-only reviews: meaning, editorial and visual (every page). After that I adjudicate, then fixes, a fresh-reader recheck of the introduction, and the sanitized re-sync. Cleanup comes after.
- **2026-10-05 ~11:40 PDT, restructure round 1 returned:** WT `<commit-51>`; main text 97 → 63 pp (125 total); build clean; fact diff clean. Reviews: editorial 1 major/11 minor; visual 1 major/8; meaning 1 major (pre-existing 'no baseline still improving' overstatement)/9. Adjudication `<scratch>/restructure/ADJUDICATION_R1.md`: all majors accepted; the abstract lmetric footnote reversed (intro + body only); 50-pp target dropped. Launching the fix round.
- **2026-10-05 ~11:45 PDT, fix round launched:** workflow `<workflow-run>`. Four fixers with disjoint files (FD owns the generators and preamble), then an integrator (in-place build, fact diff vs <commit-50> and <commit-51>, commit). Verification: a meaning recheck, an isolated fresh reader on the revised intro, and every-page visual. A second fixer runs if blockers or majors appear. Then publish: README A21 note, then publish.sh sync, scan and push of the public branch only.
- **2026-10-05 ~11:55 PDT, operator:** a SECRET GitHub gist as the readable front door (about 20-25 flat sanitized files); keep `rupei/learned-routing-public` as the code/data/sources artifact. To do after the fix round and before the cleanup.
- **2026-10-05 ~12:00 PDT:** operator OK'd linking the paper PDF and figures from the gist to the branch, with no binary copy in the gist.
- **2026-10-05 ~13:45 PDT, fix round returned:** WT `<commit-53>` (127 pp; main text pp. 1-63; build clean; numbers unchanged). Verification: meaning pass, fresh intro pass (2 majors fixed in round 2), visual (2 majors fixed). Public branch pushed `72a9f93dc1` (800 files, every scan clean). Leftover minors adjudicated in `<scratch>/restructure/ADJUDICATION_R2.md`, including M1 (an AIS-vs-AISim misattribution regression). Launching a final polish, then re-sync, gist and cleanup inventory.
- **2026-10-05 ~13:50 PDT, final workflow launched:** `<workflow-run>`: polish (adjudicated minors) → spot verify → sanitized re-sync (mirror now includes a sanitized copy of this worklog as campaign/WORKLOG.md) → secret gist front door (flat sanitized files from the pushed branch; PDF and figures linked) → read-only cleanup inventory (`<scratch>/cleanup/MANIFEST.md`). After it finishes I run the deletions myself, under the operator's standing approval.
