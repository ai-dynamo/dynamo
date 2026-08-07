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

## Open

- Confirm whether `kv_router_bench.rs` is dead and remove it if so.
- Decide whether recipe `perf.yaml` should enable server-side capture by default, or whether a
  separate investigation variant is the better shape (see the two-tier run idea in
  [benchmarking-procedure.md](benchmarking-procedure.md) §4.1).
- Record, per harness, what e2e could not answer and what input distribution it assumes.
  Currently zero harnesses state either.
