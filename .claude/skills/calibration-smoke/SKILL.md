---
name: calibration-smoke
description: End-to-end smoke for the kvbm-hub prefill-router calibration handler. Brings up hub + Qwen3-0.6B prefill + decode, then issues a CalibrationRequest sized for the small model, validates the analyzed response, exercises the cache + force-refresh paths, and audits per-request first-token uniqueness.
---

# calibration-smoke

End-to-end smoke for the prefill-router **calibration handler**
(`kvbm.prefill_router.calibrate`). Brings up the standard
prefill-router stack (hub with `--prefill-router`, one auto-wrapped
`kvbm.vllm.prefill` worker, one decode worker), then issues a
`CalibrationRequest` over the hub's HTTP proxy and validates the
fitted performance model that comes back.

Why this exists separately from `disagg-smoke/prefill-router-smoke.sh`:
that smoke exercises the **dispatch** handler with R1+R2 traffic.
Calibration is a sibling handler on the same worker — same registration
path (`detect.py::try_wrap_engine` now passes the calibrate lambda) but
a different code path (sweep + quadratic TTFT fit + cache). This smoke
confirms the calibrate handler is actually wired in the auto-wrap
production path and that the fitted model is sensible.

## What it asserts

Per HTTP call:

1. `POST /v1/features/prefill-router/calibrate/<instance>` with body
   `{"seq":[512,1024,2048,4096,8192], "osl": 16}` returns 200 + a
   `CalibrationResponse` with:
   - `from_cache: false` on the first call,
   - `results.performance_model` with finite `t2ft_linear`,
     `t2ft_quadratic`, `t2tl_intercept`, `t2tl_slope`, and both R² in
     `[0, 1]`,
   - `results.traces` length == 5 (one per requested ISL),
   - `results.n_opt > 0` and `results.n_att > 0`,
   - `resolved.seq == [512,1024,2048,4096,8192]` (no clamps under the
     8192 cap),
   - `defaults.max_seq_len == 8192`.
2. Same request re-issued → `from_cache: true`, body byte-for-byte
   identical, response in < 1 s.
3. Same request with `?force=true` → `from_cache: false`, fresh
   `first_token` set disjoint from the cached run's `first_token` set
   (proves the cache-buster counter is doing its job).
4. Inside any run, every `first_token` in `results.traces` is distinct
   (proves the cache-buster gives one unique first token per request,
   not per sweep).
5. Different-request cache miss: a second (un-forced) request with a
   modified sweep (halved ISLs and halved OSL) returns
   `from_cache: false` — proves the cache keys off the resolved
   request, not just instance identity. Then re-issuing that same
   modified request returns `from_cache: true` against the *new*
   snapshot — proves the cache slot was correctly overwritten.

## Sizing — why 8192 / OSL=16

The `spark-gb10` profile defaults `KVBM_MAX_MODEL_LEN=1024`, which is
too small to fit a meaningful sweep: the resolver requires ≥ 4
distinct ISLs for the quadratic TTFT fit, and the smallest sensible
ladder under a 1024 cap is `[256, 512]` — not enough.

The smoke **auto-sizes the hub's `--max-seq-len`** from the requested
sweep: floor of 8192, raised to `max(seq) + osl + 64` (rounded up to
the hub block size, default 16) when the requested top ISL would
otherwise blow the cap. With the default sweep `[512, 1024, 2048,
4096, 8192]` and OSL=16, the cap lands at `8272` so the top ISL fits
(the engine cap is on `ISL + OSL`, not `ISL` alone — naive
8192-on-8192 trips the resolver's drop rule).

Under the hood: the hub is the source of truth, so the smoke exports
`KVBM_HUB_MAX_SEQ_LEN` — kvbm-hub-bringup passes it as the hub's
`--max-seq-len`, and kvbmctl then renders the worker's
`--max-model-len` from the live hub. We **also** export
`KVBM_MAX_MODEL_LEN` to the same value so hardware-profiles.sh
doesn't reset it to its 1024 profile default and the response
validator's `defaults.max_seq_len` cross-check still works.

User overrides on either `KVBM_HUB_MAX_SEQ_LEN` or
`KVBM_MAX_MODEL_LEN` are honored; the smoke picks the larger of
{caller-set value, auto-computed minimum} so it never silently runs
with a cap below what the sweep needs.

OSL=16 is intentionally small: this is a smoke for *handler plumbing
+ analysis correctness*, not a real performance characterization. A
real run on a real model uses OSL=64 and a 32k+ ISL ceiling.

Single-stream throughout. Decode is idle during calibration (the
calibrate handler holds the calibrating-flag and the dispatch handler
returns `calibration_in_progress` for any prefill that arrives mid-run).

## Prerequisites

Same as `disagg-smoke/prefill-router-smoke.sh`:

- `.sandbox` venv built with the **current** kvbm wheel (the new
  `ok_with_payload` pyclass method + `calibrate_lambda` /
  `calibration_defaults` constructor kwargs require maturin develop on
  the post-calibration-handler commit).
- `target/debug/kvbm_hub` and `target/debug/kvbmctl` built.
- vLLM + Qwen3-0.6B weights cached in the venv.
- GPU available (set `KVBM_PREFILL_CUDA_VISIBLE_DEVICES`,
  `KVBM_DECODE_CUDA_VISIBLE_DEVICES`).

## Usage

```bash
# Spark / GB10 (default profile, default ISL sweep, default OSL=16):
bash .claude/skills/calibration-smoke/calibration-smoke.sh

# Custom ISL sweep / OSL (must satisfy the >=4 distinct ISLs rule):
CALIB_SEQ=512,1024,2048,4096,6144,8192 CALIB_OSL=8 \
  bash .claude/skills/calibration-smoke/calibration-smoke.sh

# Different model context cap (must be >= the largest ISL+OSL):
KVBM_MAX_MODEL_LEN=16384 CALIB_SEQ=1024,2048,4096,8192,12288 \
  bash .claude/skills/calibration-smoke/calibration-smoke.sh

# Larger box (overrides spark-gb10 defaults):
KVBM_HARDWARE_PROFILE=h100-a100 KVBM_MAX_MODEL_LEN=32768 \
  CALIB_SEQ=1024,2048,4096,8192,16384,24576 CALIB_OSL=64 \
  bash .claude/skills/calibration-smoke/calibration-smoke.sh
```

Logs land in `<experiment-root>/{hub,prefill,decode}.log` and the
calibration response bodies are dumped to
`<experiment-root>/calib-{first,cached,forced}.json` for inspection.

## Exit codes

- `0` — all assertions passed.
- `1` — hard failure with the log tail dumped to stderr.
- `2` — bad arguments / environment.
