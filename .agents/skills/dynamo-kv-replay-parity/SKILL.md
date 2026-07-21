---
name: dynamo-kv-replay-parity
description: Runs deterministic byte-parity and paired performance campaigns for Dynamo offline KV-aware replay across aggregated and disaggregated vLLM and SGLang configurations, including forced preemption and KVBM offload lifecycles. It is used when validating replay refactors, routing changes, scheduler-event changes, or performance-sensitive offline simulation changes against a baseline revision.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - offline-replay
    - kv-router
    - kvbm
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
- exact Cargo features from the relevant manifests;
- Mooncake trace path, SHA-256 checksum, and deterministic slice rule;
- engine, topology, concurrency, worker counts, block sizes, and KVBM capacities;
- the report fields excluded from canonical comparison.

Use the project root `.venv/bin/python` for Python analysis and `uv pip` for any approved
installation. Do not compare against moving branches or reuse a binary after changing its
checkout. Do not describe the configuration as "all features"; record explicit feature
names. Typical plumbing includes replay determinism plus KVBM offload, but feature names
can differ between `dynamo-mocker` and `dynamo-bench`.

## Stage 1: Pin revisions and artifacts

1. Confirm the baseline is an ancestor or otherwise document the comparison relationship.
2. Create isolated checkouts for both revisions using the same host and toolchain.
3. Apply any temporary determinism correction identically to both revisions. Keep it out
   of the measured semantic delta and record its patch checksum.
4. Build separate release binaries, copy them to a temporary campaign directory, and
   record binary plus `.text` sizes.
5. Return each checkout to its original branch after extracting the binary.

Never build while collecting performance samples.

## Stage 2: Establish deterministic reports

Require the harness to control every known entropy source:

- assign stable request UUIDs during workload creation;
- use an owned seeded RNG for temperature sampling and equal-score tie selection;
- sort routing candidates by `(worker_id, dp_rank)` before deterministic selection;
- use stable request ordinals and same-time event sequence numbers;
- recursively sort JSON object keys;
- sort only explicitly unordered collections such as per-request records;
- preserve semantically ordered event and lifecycle arrays;
- exclude only approved real-time fields such as wall time and derived throughput.

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

Tune one shared configuration until the campaign can prove that it exercised:

- KV-overlap-sensitive routing;
- immediate and queued placement;
- a small, bounded number of preemptions at the block-capacity edge;
- disaggregated prefill/decode handoff;
- terminal cleanup;
- G1 to G2 eviction; and
- G2 to G1 restoration.

Use coverage counters, lifecycle traces, or report evidence rather than inferring these
paths from successful completion. Target one to three preemptions per configuration. Zero
means the edge was not exercised; repeated preempt/re-admit cycling, a rapidly growing
preemption count, or failure to advance virtual time invalidates the fixture. Tune capacity
or concurrency minimally and identically for baseline and candidate, and back off rather
than accepting a preemption flood. Freeze the qualified configuration before the comparison;
never tune each revision separately.

## Stage 4: Run byte parity

Run the 5,000-request corpus for this authoritative matrix:

| Engine semantics | Topology | Routing |
| --- | --- | --- |
| vLLM pass-start | Aggregated | KV-aware |
| vLLM pass-start | Disaggregated | KV-aware |
| SGLang pass-end | Aggregated | KV-aware |
| SGLang pass-end | Disaggregated | KV-aware |

Enable KVBM offload on supported Linux hosts and require the lifecycle evidence from
Stage 3 for every applicable row. For each row:

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
specified engine, scheduler, routing, or KVBM semantics.

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
- scale-to-zero followed by scale-up and pending-work release;
- G1 to G2 eviction followed by G2 to G1 restoration; and
- backend-specific prefill/decode handoff ordering.

Assert bounded preemption, continued virtual-time progress, and the lifecycle itself. A
final-completion smoke test does not prove offload, preemption, queueing, or restoration
occurred.

## Stage 7: Measure performance

Measure the same four engine/topology rows with the frozen KVBM-enabled configuration and
authoritative production routing behavior. Run seeded matched-routing variants separately
as diagnostics; do not substitute them for production results.

For each row:

1. Run five warmups.
2. Collect 30 measured baseline/candidate pairs.
3. Randomize which arm runs first within each pair while keeping the pair adjacent.
4. Capture one machine-readable internal elapsed sample per process invocation.
5. With `.venv/bin/python` and SciPy, compute fixed-seed, one-sided 95% bootstrap upper
   and lower confidence bounds for the median candidate/baseline ratio.
6. Pass when the upper bound is at most `1.05`.
7. Fail when the lower bound is greater than `1.05`.
8. If the interval crosses `1.05`, collect 60 total pairs. A still-inconclusive result
   blocks a performance-parity claim.

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
- full frozen configuration and feature manifest;
- one row per engine/topology with digests and lifecycle evidence;
- every semantic exception record;
- paired performance ratios and confidence intervals;
- binary and `.text` sizes;
- profiler findings for any investigated regression; and
- skipped or unsupported coverage without overstating the result.

## Stage 9: Clean up

Delete temporary full reports, traces, binaries, profiler captures, patches, and worktrees
after recording the required evidence. Retain full outputs only for unresolved mismatches
or performance investigations. Remove task-created Cargo targets when disk pressure matters,
but never delete unrelated caches or worktrees.
