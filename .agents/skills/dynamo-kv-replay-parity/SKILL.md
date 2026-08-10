---
name: dynamo-kv-replay-parity
description: Runs deterministic byte-parity and paired performance campaigns for Dynamo offline KV-aware replay across native-G1 vLLM and SGLang configurations, including forced preemption and disaggregated handoff lifecycles. It is used when validating replay refactors, routing changes, scheduler-event changes, or performance-sensitive offline simulation changes against a baseline revision.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - offline-replay
    - kv-router
    - parity
    - performance
---

# Dynamo KV replay parity

Compare two revisions of Dynamo offline KV-aware replay using deterministic virtual-time
reports and statistically paired real-time measurements. Use the existing replay and
benchmark harnesses; extend them only when a required signal is unavailable.

This campaign intentionally excludes round-robin routing. It also does not require a
same-timestamp event-only progress scenario. Validate those concerns with focused tests
outside this skill when a change specifically affects them.

## Required inputs

Resolve and record before running anything:

- immutable baseline and candidate commit SHAs;
- Rust toolchain, build profile, flags, and host characteristics;
- each artifact's exact Cargo features from the relevant manifests;
- Mooncake trace path, SHA-256 checksum, and deterministic slice rule;
- engine, topology, concurrency, worker counts, block sizes, and native G1 capacities;
- the canonical-report exclusion allowlist; and
- the performance run-order seed, CPU placement, timing scope, and invalidation rules.

Use the project root `.venv/bin/python` for Python analysis and `uv pip` for any approved
installation. Do not compare against moving branches or reuse a binary after changing its
checkout. Do not describe the configuration as "all features"; record explicit feature
names. The current campaign is native-G1-only; do not add removed KVBM replay features or
runtime arguments to make a historical command line build.

## Stage 1: Pin revisions and artifacts

1. Confirm the baseline is an ancestor or otherwise document the comparison relationship.
2. Create isolated checkouts for both revisions using the same host and toolchain.
3. Apply any temporary determinism correction identically to both revisions. Keep it out
   of the measured semantic delta and record its patch checksum.
4. For the current `dynamo-bench` harness on Linux, build one release artifact per
   revision with exactly `replay-bench` and no default features.
   `--canonical-reports-jsonl` requires `replay-bench`, which also selects the seeded
   router.
5. Reuse that artifact across all native-G1 correctness and performance rows. Do not
   build separate engine or production-routing artifacts. If the named manifest feature
   or runtime contract no longer exists, stop and report that this protocol needs
   updating rather than guessing a replacement matrix.
6. Copy the artifacts to a temporary campaign directory. Record exact features, binary
   SHA-256, binary size, and `.text` size.
7. Return each checkout to its original branch after extracting the binaries.

Never build while collecting performance samples.

## Campaign concurrency

Treat one engine/topology/frozen-configuration comparison row as the unit of node
placement. When scheduler capacity permits, allocate multiple nodes and assign independent
rows to them. The nodes do not need to be homogeneous because no individual comparison
crosses nodes. Keep the baseline, candidate, all determinism repetitions, and lifecycle
evidence for one row on the same node. Use the same prebuilt revision artifacts and inputs
throughout the campaign, and record the node characteristics and CPU placement for every
row.

Native replay is normally single-core: pin each process to one physical core after
confirming that assumption during preflight. Keep the CPU-set size, NUMA placement, and
affinity identical between baseline and candidate for a row.

For correctness, run the baseline and candidate repetitions for a row concurrently when
its node has sufficient resources. Pin concurrent processes to disjoint CPU sets, give each
process separate output paths, and ensure they do not share mutable state. Keep every
repetition in a separate process even when several repetitions run at the same time.

For performance, keep the entire row's warmups and 60-pair measurement series on
its assigned node. Execute only one performance invocation at a time on that node with a
fixed CPU placement, and keep each randomized baseline/candidate pair adjacent. Different
performance rows may run concurrently on different nodes, but do not pool one row's pairs
across heterogeneous nodes. Do not run another replay, build, profiler, or unrelated
workload concurrently on a node collecting performance samples.

Multi-node and within-node correctness parallelism are campaign throughput optimizations;
they must not change workload concurrency or weaken performance isolation.

## Stage 2: Establish deterministic reports

Require the harness to control every known entropy source:

- assign stable request UUIDs during workload creation;
- use an owned seeded RNG for temperature sampling and equal-score tie selection;
- sort routing candidates by `(worker_id, dp_rank)` before deterministic selection;
- use stable request ordinals and same-time event sequence numbers;
- recursively sort JSON object keys;
- sort only explicitly unordered collections such as per-request records;
- preserve semantically ordered event and lifecycle arrays;
- exclude exactly `/summary/wall_time_ms`, `/summary/processed_tokens_per_s`, and
  `/summary/processed_output_tokens_per_s`.

This exclusion list is closed: include every other field, and stop for review before adding
another exclusion. Make semantic fields such as request UUIDs deterministic rather than
dropping them. Require parity output metadata to report `replay_bench: true`; otherwise the
seeded routing contract was not active.

Run baseline twice and candidate twice in separate processes. Each revision must produce
one unique canonical digest. This is an entropy-leak check, not a statistical trial. If a
revision is internally unstable, stop and diagnose it; do not increase repetitions and
average the outputs.

## Stage 3: Qualify a long interaction-heavy corpus

Prefer a fixed contiguous 5,000-request Mooncake window over many parity repetitions.
Preserve arrival order and prefix locality. Record the starting offset, request count, and
checksum. Do not randomly sample rows, duplicate a shorter trace, or silently claim a
5,000-request campaign when fewer usable requests exist.

The committed `lib/bench/testdata/mooncake_trace_1000.jsonl` fixture is suitable for a
quick harness preflight, not the authoritative long-corpus campaign.

Qualify and freeze one configuration per comparison row, or per explicitly named
configuration family when rows genuinely share every relevant parameter. For each frozen
configuration, prove that it exercised every applicable path:

- KV-overlap-sensitive routing;
- immediate and queued placement;
- a small, bounded number of preemptions at the block-capacity edge;
- disaggregated prefill/decode handoff;
- terminal cleanup.

Use coverage counters, lifecycle traces, or report evidence rather than inferring these
paths from successful completion. Target one to three preemptions per configuration. Zero
means the edge was not exercised; repeated preempt/re-admit cycling, a rapidly growing
preemption count, or failure to advance virtual time invalidates the fixture. Tune capacity
or concurrency minimally and identically for baseline and candidate within the row or
family, and back off rather than accepting a preemption flood. Never tune the revisions
separately.

### Start from qualified internal-polynomial seeds

For the canonical 5,000-row Mooncake window with the internal polynomial mocker, read
[internal-polynomial golden points](references/internal-polynomial-golden-points.md)
before searching for capacity edges. Treat those configurations and observed counters as
suggested starting points, not universal constants or substitutes for qualification on the
pinned baseline. Requalify one frozen configuration identically on both revisions.

Use the reference's expected preemption, retraction, queue, reuse, worker, and handoff
signals as drift detectors. Matching lifecycle counts do not waive an unstable canonical
digest; stop correctness and performance work for that row until both internal
determinism and cross-revision parity are established.

## Stage 4: Run byte parity

Run the 5,000-request corpus for this authoritative matrix:

| Engine semantics | Topology | Memory path | Routing |
| --- | --- | --- | --- |
| vLLM pass-end | Aggregated | Native G1 | KV-aware |
| vLLM pass-end | Disaggregated | Native G1 | KV-aware |
| SGLang pass-end | Aggregated | Native G1 | KV-aware |
| SGLang pass-end | Disaggregated | Native G1 | KV-aware |

KVBM G2-G4 replay was removed from the current mocker. It is not an unsupported row in
this matrix and must not be emulated with ignored flags. A campaign that needs historical
KVBM parity must pin a historical revision together with the matching historical skill
protocol. For each current row:

1. Produce canonical baseline and candidate outputs with the frozen configuration.
2. Compare their bytes or SHA-256 digests exactly.
3. Verify the coverage evidence independently of the digest.
4. Delete matching full reports and retain their digests.
5. Preserve full reports and a focused diff only when outputs disagree.

TensorRT-LLM can be additional smoke coverage, but it is not a substitute for either
authoritative engine semantic.

## Stage 5: Classify semantic differences

Byte mismatch is a review gate, not an instruction to preserve incorrect behavior. Allow
an intentional mismatch only when the candidate is demonstrably more faithful to the
specified engine, scheduler, or routing semantics.

For every proposed exception, record:

- the exact fields, requests, or lifecycle events that differ;
- the baseline behavior and why it is incorrect or less faithful;
- the candidate behavior and the semantic source of truth supporting it;
- why the difference is caused by the intended change rather than leaked entropy;
- a focused regression test that fails on the old behavior and passes on the correction;
- any downstream report or API compatibility impact; and
- the reviewer-visible disposition.

Use `PASS_WITH_SEMANTIC_EXCEPTIONS` only when every byte difference is covered by such a
record. Unexplained, incidental, or merely convenient differences fail. Do not patch the
candidate back to known-wrong behavior just to obtain identical bytes.

## Stage 6: Force rare lifecycles when needed

Use small deterministic fixtures only for required paths the long corpus cannot reliably
trigger:

- single-worker KV-router queueing;
- a preemption-edge fixture that targets one to three preemptions and then completes;
- scale-to-zero followed by scale-up and pending-work release; and
- backend-specific prefill/decode handoff ordering.

Assert bounded preemption, continued virtual-time progress, and the lifecycle itself. A
final-completion smoke test does not prove preemption or queueing occurred.

## Stage 7: Measure performance

Measure every supported row from Stage 4 with its frozen configuration. The authoritative
gate reuses the `replay-bench` artifacts from byte parity, so routing selection is seeded
and matched between revisions. Require timing metadata to report `replay_bench: true`.
Describe the result as seeded matched-routing replay-loop parity, not production-routing
performance. Production-routing timing is outside the default campaign because it requires
a separate build without `replay-bench`; add that campaign only when the user explicitly
requests it.

The primary metric is replay execution time. Start its timer after trace normalization,
workload construction, and engine/runtime preparation, immediately before
`prepared.run(...)`; stop it immediately after that call returns and before collector
finalization or report aggregation. Emit this value as `replay_execution_ms`. Record setup
and end-to-end time separately as diagnostics. If the harness does not expose this exact
timing boundary, extend it before running the campaign; do not substitute the existing
broader `wall_time_ms`.

Stage binaries and trace inputs on node-local storage and verify their checksums before
warmups. File transfer is campaign setup, not a sample. For each measured invocation, pass
`--iterations 1` and a unique node-local `--timings-jsonl` path. Do not pass
`--canonical-reports-jsonl`, `--report-json`, or another full-report output option in a
gated performance invocation. Run one fresh process per sample so lazy per-request capture,
report serialization, and in-process iteration state cannot contaminate the metric.

For each row:

1. Run five warmups per arm, alternating arms for ten warmups total.
2. Generate and persist a fixed-seed, balanced 60-pair schedule with 30 baseline-first and
   30 candidate-first pairs in randomized order.
3. Collect all 60 measured pairs. Keep the two invocations in each pair adjacent and
   compute `r_i = candidate_replay_execution_ms / baseline_replay_execution_ms` regardless
   of run order.
4. Treat the ratios as independent and identically distributed, or otherwise exchangeable,
   only when the campaign can justify that sampling assumption; pair adjacency does not
   establish it. Predeclare thresholds for serial-dependence diagnostics, including lag
   autocorrelation and pair-order trends against elapsed time and available temperature or
   frequency telemetry, and record their results.
5. If the diagnostics breach their thresholds or exchangeability cannot be justified,
   report `INCONCLUSIVE` or use a predeclared dependence-aware method. Do not apply the
   order-statistic gate.
6. Otherwise sort the 60 ratios. Conditional on the sampling assumption, use the 24th order
   statistic as the exact distribution-free one-sided 95% lower confidence bound for the
   population median ratio and the 37th order statistic as the corresponding upper bound.
   Do not use a Wald interval.
7. Pass when the upper bound is at most `1.05`.
8. Fail when the lower bound is greater than `1.05`.
9. Otherwise report `INCONCLUSIVE`; do not add samples adaptively and claim the original
   confidence level.

Never remove an observation because its value looks like an outlier. Retain every attempted
sample and its process status. A replay error, assertion failure, malformed timing record,
or other product failure fails or blocks the row; it is not a discardable sample. Only a
predeclared environmental condition, such as scheduler eviction, node-health failure,
affinity violation, or independently detected competing load, may invalidate a sample.
Invalidate both members of that pair, record the evidence and reason, and rerun the complete
pair with the same arm order. Never retry or replace a sample silently. A paired bootstrap
of log ratios may be reported as a secondary effect-size diagnostic, but it does not decide
the gate.

Also fail release binary or `.text` growth above 5% until the added footprint is explained
and narrowed.

### Investigate unacceptable overhead

When a statistically meaningful regression exceeds the accepted window:

1. Confirm baseline and candidate performed equivalent semantic work. Separate an accepted
   semantic correction from framework overhead when the correction intentionally adds work.
2. Inspect the diff and hot-path structure for obvious causes: event capture, request
   cloning, heap allocation, admission vectors, dynamic dispatch, lock traffic, or widened
   generic monomorphization.
3. If static analysis does not identify a convincing cause, rerun the representative
   failing configuration under Samply on a supported host. Profile baseline and candidate
   with equivalent release/debug-symbol settings and inputs.
4. Compare self-time, call stacks, allocation-heavy paths, and new monomorphized functions.
   Attribute the regression to specific code before optimizing or requesting a waiver.
5. Repeat the paired performance gate after any fix.

Prefer an available Samply workflow skill when one is installed. Do not profile concurrently
with builds or unrelated load.

## Stage 8: Decide and report

Use exactly one semantic result:

- `PASS`: all authoritative canonical outputs match;
- `PASS_WITH_SEMANTIC_EXCEPTIONS`: every mismatch is an evidenced improvement covered by
  a regression test; or
- `FAIL`: any unexplained mismatch, missing lifecycle evidence, or unstable revision.

Report performance independently as pass, fail, or inconclusive. A semantic exception does
not waive an unexplained performance regression.

The final report must include:

- revision SHAs and determinism-patch checksum, if any;
- trace checksum and 5,000-request slice specification;
- every artifact's exact feature manifest, binary checksum, and routing-mode assertion;
- each row's frozen configuration, node characteristics, CPU set, and NUMA placement;
- one row per engine/topology/memory path with digests and lifecycle evidence;
- the exact canonical exclusion allowlist;
- every semantic exception record;
- performance timing scopes, persisted arm-order schedule, all attempted samples and
  invalidations, paired ratios, sampling assumption, dependence diagnostics, and conditional
  order-statistic confidence bounds;
- binary and `.text` sizes;
- profiler findings for any investigated regression; and
- skipped or unsupported coverage without overstating the result.

## Stage 9: Clean up

Delete temporary full reports, traces, binaries, profiler captures, patches, and worktrees
after recording the required evidence. Retain full outputs only for unresolved mismatches
or performance investigations. Remove task-created Cargo targets when disk pressure matters,
but never delete unrelated caches or worktrees.
