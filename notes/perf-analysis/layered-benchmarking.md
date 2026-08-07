<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Do we need layered benchmarking infrastructure?

> **Status: working draft — not published guidance.** Tracked here so the reasoning survives,
> not because it is reviewed. Published documentation lives under `docs/fern/pages/`.

Record of a design discussion. Verified against `origin/main` @ `cd6b63b0cc`.

## The question

If end-to-end benchmarking collects sufficiently detailed data points and segments, we can
likely identify issues from there. What does component-wise benchmarking add?

## Where the e2e-only position is right

Fine-grained e2e is the primary instrument, and most component benchmarks people build are
waste. If a stage breakdown reconciles against the end-to-end number, e2e already tells you
where the time went, and component benchmarking adds nothing.

The reconciliation test — does the sum of parts recover the whole — is the right gate on
whether anything more is needed. If it reconciles, stop. If it does not, the residual is the
finding.

**Default position: instrument e2e well, and build a component harness only when e2e
demonstrably cannot answer the question.**

## The one thing e2e structurally cannot do

Not a resolution problem. No number of additional spans fixes it.

**In a saturated system, only the bottleneck's capacity is observable.** Everything else is
idle-waiting, and its ceiling is invisible by construction.

The repository already records this, in
`.agents/skills/dynamo-frontend-benchmark/SKILL.md`:

> with mock workers, the Dynamo **frontend is rarely the bottleneck** — it's latency/IO-bound,
> sitting ~60–85% of its pinned cores with ~0 internal lock contention. Frontend micro-opts
> therefore show flat e2e throughput on this setup; their value is CPU-efficiency/headroom.

Against real GPUs it is worse: the GPU saturates first, so the frontend sits further from its
limit and is even less observable.

You cannot learn a component's ceiling from a run where something else is the ceiling. E2E
tells you the **current** bottleneck; component isolation tells you the **next** one. Without
it, every optimization outside the current bottleneck shows flat e2e, and "this change did
nothing" is indistinguishable from "this change bought headroom that matters once the GPU
bound moves."

## Four consequences

1. **Signal-to-noise.** E2E variance is dominated by the noisiest component. A 3% frontend
   regression is undetectable under 10% e2e run-to-run spread. Offline replay parity can
   resolve small differences *only because* it deleted everything noisy.
2. **Iteration cost.** An e2e GPU run costs minutes and a cluster; a mock-worker run costs
   seconds on one box. That changes which methods are available at all — bisection,
   fifty-config sweeps, running under a profiler.
3. **CI gating.** Gap 5 in [gaps.md](gaps.md) has no e2e solution. GPU e2e cannot run
   per-PR. A regression gate, if one ever exists, has to be component-level.
4. **Attribution versus explanation.** A span reading "routing took 12 ms" localizes but does
   not explain. Whether that is radix-tree work, lock contention, or allocation needs a
   profiler on the component, usually in isolation so the profile is not dominated by GPU
   wait.

## Where component benchmarking earns its bad reputation

The same failure mode as gap 1: **a component benchmark is a model of how the component is
used, and the model is rarely validated.** Optimize the frontend at 60k-token prompts when
production runs 500. Sweep a router at cache-hit rates the deployment never produces.

It also misses interaction effects, which is often where real failures live — queueing,
backpressure, cache interference between stages. Those appear only when the parts are
connected.

## Recommendation: two layers plus a rule

Not a pyramid built speculatively.

1. **E2E with fine-grained attribution** — primary, always, and the only source of headline
   numbers.
2. **Component harnesses, built narrowly** — only where one of these holds:
   - the component is plausibly the *next* bottleneck and e2e cannot show its ceiling;
   - the measurement is cheap enough to gate in CI;
   - e2e localized a cost but could not explain it.
3. **The rule that keeps it honest.** Every component benchmark must trace to an e2e
   observation that motivated it — a residual that would not reconcile, a saturation that
   could not be attributed, a ceiling that could not be seen. And it must record the input
   distribution it assumes, so drift from real traffic is checkable.

That rule is the actual answer. Layered infrastructure is not needed as a design principle.
What is needed is for e2e to be good enough to *prove* where it falls short, and then to build
exactly that much more.

---

## What the repository has today

The three tiers already exist. Nothing was designed that way; each layer appeared when
someone hit a wall e2e could not get past.

### Verified: recipe benchmarks capture no server-side data

Across all **33** `recipes/**/perf.yaml`:

| Setting | Occurrences |
| --- | --- |
| `DYN_REQUEST_TRACE` | 0 |
| `OTEL_EXPORT_ENABLED` | 0 |
| `DYN_FPM_TRACE` | 0 |
| Server-side metric collection | 0 — several explicitly pass `--no-server-metrics` |

Recipe perf jobs are **client-side only**. Request tracing, OTLP export, and Prometheus
scraping all exist in the runtime and are documented, but are wired into **no** benchmark.
Every published recipe number is an opaque-interval measurement with no server-side
decomposition.

This lands on §3.4 of [benchmarking-procedure.md](benchmarking-procedure.md): OTEL and
Prometheus are **launch-time** decisions. Miss them and the run is lost — and all 33 recipes
miss them by default. Concrete and fixable.

### Verified: the two frontend harnesses are alternatives, not a stack

`benchmarks/frontend/scripts/run_perf.sh` and
`.agents/skills/dynamo-frontend-benchmark/scripts/start.sh` both launch `dynamo.frontend` plus
`dynamo.mocker`, both defaulting to port **8000**. They cannot run concurrently — they would
collide on the port and on etcd worker registrations. Neither invokes the other's scripts, and
neither mentions the other in text.

Three pieces of the skill are topology-independent and compose with either harness:

| Script | Why it composes |
| --- | --- |
| `isolate.sh` / `unisolate.sh` | System-wide cgroup cpuset changes, unrelated to topology |
| `analyze_folded.py` | Operates on folded stacks from any profiler run |
| `extract_throughput.py` | Reads any AIPerf artifact directory |

So the duplication is specifically **topology management** — `start.sh` + `stop.sh` versus
`run_perf.sh` — roughly 300 lines that exist twice. "The whole harness is duplicated" is too
strong.

> **Authorship caveat.** An earlier version of this note attributed the skill to a specific
> person based on `git log -- <path>`. That is unsound: the skills tree has been moved at
> least twice, and a path-scoped log without `--follow` attributes the earliest commit to
> whoever performed the move, while squash-merges attribute it to whoever merged. Only two
> commits appeared for a 234-line SKILL.md with eleven scripts, which should have been the
> tell. **The attribution has been withdrawn**, along with the conclusion built on it — that
> the duplication was necessarily independent rather than deliberate. The findings above come
> from reading the files, not from git history.

### Micro-bench coverage

Ten criterion benches, mapped to the request path:

| Bench | Stage |
| --- | --- |
| `lib/llm/benches/tokenizer_simple.rs` | Preprocess — encode/decode, HF and tiktoken |
| `lib/llm/benches/tokenizer_dataset.rs` | Preprocess — sequential versus batched |
| `lib/kv-router/benches/tracking_hash.rs` | Route — block hashing |
| `lib/kv-router/benches/policy_queue.rs` | Route — scheduling queue, pop and drain |
| `lib/llm/benches/kv_router_bench.rs` | Route — **no bench functions found** |
| `lib/kvbm-logical/benches/block_manager.rs` | KV management — match active/inactive, bulk drop |
| `lib/runtime/benches/tcp_codec_perf.rs` | Transport — request/response encode and decode |
| `lib/llm/benches/transfer_context_v2.rs` | Transport — blocking, async, concurrent |
| `lib/runtime/benches/compute_pool_overhead.rs` | Runtime — compute pool, `block_in_place` |
| `lib/llm/benches/request_trace_finish_metadata.rs` | Postprocess — finish metadata, tool calls |

**Holes, against the request path:**

- **HTTP ingress** — no bench for axum/hyper request parsing or OpenAI deserialization.
- **SSE response relay** — nothing, despite streaming being the dominant response mode.
- **Dispatch/egress** — covered only indirectly through the TCP codec; no bench of the
  push-router send path.
- **Detokenization under streaming** — the encode side is well covered; per-token decode in a
  streaming response is not measured the way encode is.

Both major holes sit at the **client-facing edges** — where a closed-loop client's latency is
most sensitive, and exactly what mock workers cannot speak to, since the mocker replaces
generation.

`lib/llm/benches/kv_router_bench.rs` yielding no bench functions is worth confirming and
possibly deleting: it is compile-gated in CI, so it costs build time while implying coverage
that is not there. The `lib/kv-router/benches/` pair covers that ground.

---

## How each tier is run, and on what cadence

Nothing measures performance automatically. One tier executes in CI but discards its timings;
the other two run only when a person starts them.

### Tier 1 — End-to-end · **on demand only**

No workflow invokes AIPerf. Grepping `.github/workflows/` for `aiperf` returns nothing.

```bash
# Kubernetes — deploy the recipe first, then its benchmark job
deploy/utils/setup_benchmarking_resources.sh          # one-time: PVCs, HF token secret
kubectl apply -f <model>/<framework>/<mode>/deploy.yaml -n $NAMESPACE
kubectl apply -f <model>/<framework>/<mode>/perf.yaml  -n $NAMESPACE

# Local endpoint
aiperf profile --model <model> --url http://localhost:8000 \
  --endpoint-type chat --streaming --concurrency 10 --request-count 100
```

### Tier 2 — Subsystem · **on demand for timing; parts run functionally in CI**

`pre-merge.yml` exercises the replay path, but for **correctness**, not speed:

| Line | Command | What it checks |
| --- | --- | --- |
| 198 | `cargo test -p dynamo-mocker --features replay-bench,kvbm-offload replay_forces_g1_to_g2` | the G1 to G2 offload lifecycle fires |
| 199 | `cargo test -p dynamo-bench --test mooncake_trace --features mocker-kvbm-offload g2` | trace replay still works |
| 207-213 | `compile_bench <feature> --bench <target>` | the feature-gated entrypoints still build |

None produces a timing number. For measurement, run them yourself:

```bash
# Frontend sweep against mock workers
python benchmarks/frontend/scripts/sweep_runner.py --mode local --backend mocker \
  --concurrency 64 --isl 4000 --osl 500

# Offline replay of a trace
python -m dynamo.replay --trace-format mooncake --router-mode kv_router \
  --num-workers 4 --report-json report.json

# Interconnect ceiling
kubectl apply -f deploy/pre-deployment/nixl/nixlbench-deployment.yaml -n $NAMESPACE
```

### Tier 3 — Micro · **runs automatically every PR, with timing discarded**

`dynamo-pipeline.yml:169` runs `cargo test --locked --all-targets ...`. `--all-targets`
includes bench targets, so all ten criterion benches **execute on every pipeline run** — but
every `[[bench]]` sets `harness = false`, so criterion owns the binary and detects test mode:
each benchmark runs **once, for validation**, with no sampling, timing, or comparison.

That the benches really are invoked (not merely compiled) is confirmed by the workflow's own
comment, which explains that `--test-threads=1` had to move to an environment variable because
under `--all-targets` the flag "is also handed to the criterion `--bench` targets, which reject
it."

To get actual numbers:

```bash
cargo bench -p dynamo-kv-router                     # every bench in one crate
cargo bench -p dynamo-llm --bench tokenizer_simple  # one bench
cargo bench -p dynamo-kv-router -- --save-baseline before
# ... make a change ...
cargo bench -p dynamo-kv-router -- --baseline before
```

The last pair is criterion's built-in A/B: it reports a percentage change with a bootstrap
confidence interval and a p-value against the stored baseline.

### Why the Tier 3 timings are discarded

**Not a decision.** `--all-targets` entered this workflow in
`ci: fold container-validation-dynamo into pr, post-merge, and nightly` (#8525, 2026-04-23), a
CI consolidation where the flag means "compile and test everything." The benches were swept in
by that flag, and criterion's test mode is criterion's designed response to it.

The only later commit touching that line,
`ci: run dynamo_llm rust-gpu tests single-threaded to fix teardown SIGSEGV` (#11853), treats
the benches purely as an obstacle to keeping the **test** command green. Across both commits,
benchmark results were never the subject. The question of whether to gate on performance was
not answered — it was never posed.

The accidental upside is real and worth keeping: a bench that panics or stops compiling fails
the pipeline, so none of these can silently bit-rot.

### What actually blocks a regression gate

The instruments are finished; the environment for them does not exist.

- **No baseline is retained.** Criterion compares against `target/criterion/`, which is
  ephemeral in CI. Only sccache is configured — nothing persists criterion output between runs,
  so there is nothing to compare against.
- **The runners may not be able to produce a timing signal.** The workflow documents that
  "under 12x contention 3/3 parallel shards segfault." `lib/kv-router/benches/policy_queue.rs`
  already sets `.noise_threshold(0.03)`; if run-to-run variance on a shared runner exceeds 3%,
  the gate is not buildable on this infrastructure.
- **No threshold policy.** One bench sets a noise threshold; the other nine do not, and nothing
  defines what a breach should do to a PR.

**First step is a measurement, not a proposal:** run `cargo bench` twice on the same commit on
the same runner and compare. That answers whether a gate is possible here at all, and it is
cheap. Everything else is premature until it is answered.

### The pattern across tiers

| Level | A/B procedure available | Status |
| --- | --- | --- |
| Micro | Criterion change detection, bootstrap CIs | **Built in, never run** |
| Offline replay | 60 pairs, randomized, distribution-free interval | **Fully specified, no runbook** |
| Serving | — | **Does not exist** |

The most rigorous procedure sits at the level with the least at stake and goes unused. The
level where decisions are actually made — published serving numbers — has none. Someone
hand-built at replay level what criterion already provides one layer down.

## Open

- Confirm whether `kv_router_bench.rs` is dead and remove it if so.
- Decide whether recipe `perf.yaml` should enable server-side capture by default, or whether a
  separate investigation variant is the better shape (see the two-tier run idea in
  [benchmarking-procedure.md](benchmarking-procedure.md) §4.1).
- Record, per harness, what e2e could not answer and what input distribution it assumes.
  Currently zero harnesses state either.
- Run `cargo bench` twice on one commit on a CI runner and compare, to establish whether a
  micro-level regression gate is buildable on this infrastructure.
