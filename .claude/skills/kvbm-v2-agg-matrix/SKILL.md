---
name: kvbm-v2-agg-matrix
description: Run the v2-only agg determinism matrix — 4 cells over (G1 LW/FC) × (G2 Operational/Universal) on Qwen3-0.6B. Pins the new routing dimensions opened by Tier 1 FC auto-detect. Iterable — expect to extend axes and tighten assertions as we learn what catches real regressions.
user-invocable: true
disable-model-invocation: true
---

# Skill: KVBM v2 Agg Matrix

Runs `tests/kvbm_integration/test_determinism_agg_v2_matrix.py` against the local sandbox venv. The test parametrizes a 4-cell matrix:

| Spec id | G1 path | G2 layout |
|---|---|---|
| `v2-Qwen3-0.6B-inter-g2ope-g1lw` | LayerSeparate (per-layer) | Operational (inherits G1) |
| `v2-Qwen3-0.6B-inter-g2ope-g1fc` | FullyContiguous (cross-layer) | Operational (inherits G1) |
| `v2-Qwen3-0.6B-inter-g2uni-g1lw` | LayerSeparate (per-layer) | Universal (fused permute kernel) |
| `v2-Qwen3-0.6B-inter-g2uni-g1fc` | FullyContiguous (cross-layer) | Universal |

Each cell runs `base_test_determinism_with_cache_reset` plus three structural assertions:

1. **KVBM was actually exercised** — metrics-endpoint deltas show non-zero `kvbm_offload_blocks_d2h` during the run. Catches "test silently passes with KVBM disabled" regressions.
2. **G1 path matched the spec** — server-log grep for `[KVBM] prefer_cross_layer_blocks={True,False}` and `[KVBM] {Cross-layer KV cache,KV caches} registered`. Catches the JSON-config plumb breaking without the determinism numerics noticing.
3. **G2 utilization is surfaced** — printed per cell so we can ratchet `KVBM_MATRIX_CPU_BLOCKS` to a tighter minimum.

This skill is the follow-on to the disagg-smoke shell smokes. The shell smokes prove the *liveness* of one path end-to-end with one prompt; this skill proves *determinism* across the 4-cell routing matrix with many prompts and a cache-reset boundary.

## Prerequisites

- `.sandbox/` venv with `kvbm` importable: `/dynamo:kvbm:sandbox-venv` + `/dynamo:kvbm:maturin-dev` first (see the `kvbm-maturin-dev` skill). Build **release** (the maturin-dev default) — this 4-cell × ~40-iter matrix is the run a debug build hurts most.
- HF model cache populated with `Qwen/Qwen3-0.6B` (will download on first run).
- One GPU (the matrix is `gpu_1`-marked, no TP yet).

## First-time run (calibration)

```bash
# All 4 cells, default 40 iterations per cell. CONSERVATIVE — we do not
# yet have real G2 utilization data for this matrix; 40 is calibration.
# Each cell prints actual G2 utilization at the end; use those numbers
# to ratchet KVBM_MATRIX_MAX_ITERATIONS / KVBM_MATRIX_CPU_BLOCKS upward
# until the heaviest cell sits around 75% G2 utilization.
bash .claude/skills/kvbm-v2-agg-matrix/run.sh
```

After the first successful run, set `KVBM_MATRIX_MAX_ITERATIONS` to whatever pushes the heaviest cell's utilization to ~75% — typically a 2-5x multiple of 40 depending on prompt length and how aggressively G2 fills.

## Single cell (fast iteration on one routing combo)

```bash
# Force FC + Universal only.
bash .claude/skills/kvbm-v2-agg-matrix/run.sh -k g2uni-g1fc

# LW + Operational baseline — useful when chasing FC regressions to confirm
# the baseline still works.
bash .claude/skills/kvbm-v2-agg-matrix/run.sh -k g2ope-g1lw
```

The `-k` flag is passed straight to pytest, so any substring of the spec id works.

## Tune iteration counts

```bash
# Fast cycle — minimum signal, fastest turnaround. Structural assertions
# (KVBM exercised, G1 path matched spec) still fire; determinism signal
# is weaker because each prompt is recorded fewer times per phase. Use
# this while developing new connector code.
KVBM_MATRIX_MAX_ITERATIONS=10 bash .claude/skills/kvbm-v2-agg-matrix/run.sh

# Calibration (default) — 40 iterations. Surfaces routing regressions
# without yet knowing the right G2 sizing.
bash .claude/skills/kvbm-v2-agg-matrix/run.sh

# Heavier runs — set explicitly based on the utilization printout from
# your first run. Aim for ~75% G2 utilization on the heaviest cell.
KVBM_MATRIX_MAX_ITERATIONS=200 bash .claude/skills/kvbm-v2-agg-matrix/run.sh

# Nightly / pre-release — match the existing test_determinism_agg suite's
# full count. Long.
KVBM_MATRIX_MAX_ITERATIONS=1000 bash .claude/skills/kvbm-v2-agg-matrix/run.sh
```

**Why 40 by default?** Each iteration issues at minimum one Shakespeare prompt; control sequences fire every 10 iterations, random every 7. At 40 iterations each control prompt gets ~4 recorded issuances per phase and each random prompt ~5-6, giving multiple positional comparisons across the cache-reset boundary. Without real utilization numbers from a first run, 40 is the conservative calibration value — high enough to exercise routing and onboard paths, low enough to avoid overrunning unknown G2 capacity. Earlier drafts of this skill defaulted to 200 based on extrapolation from synthetic data; that was unsubstantiated.

## Tune the G2 sizing

The `KVBM_MATRIX_CPU_BLOCKS` env var (default `4000`) sets `cache.host.num_blocks` on every cell. Each run prints per-cell G2 utilization, e.g.:

```
[matrix] v2-Qwen3-0.6B-inter-g2uni-g1fc: G2 offload total = 1247 blocks (of 4000 configured) — utilization = 31.2%
```

If every cell reports utilization < 50%, lower `KVBM_MATRIX_CPU_BLOCKS` so the cell that hits the highest count is at ~80% utilization. Tighter sizing surfaces overflow / eviction bugs.

## Iteration loop

This skill is expected to evolve. When extending the matrix:

1. **New cell:** add a tuple to `_matrix_specs()` in `test_determinism_agg_v2_matrix.py`. Update the table in this SKILL.md.
2. **New axis (e.g. onboard_mode=intra):** crossing doubles the cell count. Default mode controlled by `KVBM_MATRIX_ONBOARD_MODE`; cross-and-test by running the skill twice with different env values.
3. **New structural assertion:** add to `_assert_kvbm_exercised` / `_assert_g1_path_matches_spec` / new helpers in the test module.
4. **Failing matrix cell:** the runner prints per-cell verdicts. Re-run the failing cell with `-v -s` for full server log + KVBM event stream.

## When to run this

- Before merging changes to `connector.py::prefer_cross_layer_blocks`, `dim_probe.py::select_fc_*`, `pending.rs::PendingLayoutMode`, `vllm/layout.rs::determine_*_kv_layout`, or any FC-related kvbm-physical code.
- After rebuilding bindings (`/dynamo:kvbm:maturin-dev`) when iterating on the connector seam.
- As a pre-merge gate for any PR touching G1 routing or G2 layout selection.

## What this skill does NOT cover (yet)

- **Conditional disagg / pull-from-decode flows.** Disagg adds prefill+decode as separate instances and exercises P2P pull; needs a 2-GPU topology and is the next iteration.
- **TP > 1.** All cells are TP=1. The `kv-router`/asymmetric-TP routing dimension is a separate axis.
- **MLA backends.** MLA's 3-dim fused K/V doesn't support FC; covered by the existing `test_determinism_agg.py` instead.
- **Hybrid models** (multi-backend or multi-group). KVBM v2 rejects these in both LW and FC; the matrix doesn't test rejection messages.
- **Perf regressions.** This skill checks correctness only. Use `kvbm-run-perf` for TTFT / throughput trending.

## Assets

| File | Purpose |
|---|---|
| `run.sh` | Entry point. Wraps `pytest` with sensible defaults, prints a 4-cell verdict matrix at the end. |
| `SKILL.md` | This file. |

## See also

- `/dynamo:kvbm:run-validation` — broader pytest scope coverage (v1+v2, all models, MLA-gated). The matrix lives separately to keep the routing-dimension cycle fast and iterable.
- `/dynamo:kvbm:maturin-dev` — rebuild kvbm-py3 bindings before running.
- `/dynamo:kvbm:decomposed-run` — three-shell flow for keeping vLLM hot during eval iteration; works with matrix spec ids too (export `KVBM_SPEC_ID=v2-Qwen3-0.6B-inter-g2uni-g1fc`).
- `.claude/skills/disagg-smoke/fc-mode-smoke.sh` — live disagg-CD wiring smoke for FC, complementary to this matrix.
