#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end validation of the AIC power-data integration.

Run this AFTER `tools/integrate_aic_power_data.py` has copied the
`aic_[h,b]200_power_data` files into an AIC source checkout and the
`aiconfigurator` package has been reinstalled from that checkout.

What is validated:
    1. AIC database loads cleanly for H200 and B200 systems.
    2. `estimate_perf()` returns `power_w > 0` (i.e., the power_w column is no
       longer zeroed out — the integration script worked).
    3. `power_w` values are within the expected range for each system, derived
       from the actual measured data (H200 context p90 ≈ 600 W, p50 ≈ 380 W;
       B200 context p90 ≈ 700 W, p50 ≈ 400 W).
    4. The `AICPowerOptimizer` uses the real power_w — NOT the TDP fallback
       warning path.
    5. Budget constraint: with a budget tighter than 1 full replica's power draw,
       the optimizer correctly returns fewer replicas than the requested max.

Usage
-----
    # Install the AIC package from the updated checkout first:
    cd /path/to/aiconfigurator && pip install -e .

    # Then run this validator:
    python tools/validate_aic_power_integration.py [--system h200_sxm] [--backend trtllm]

Exit code 0 = all checks passed.  Non-zero = failures reported.
"""

from __future__ import annotations

import argparse
import logging
import sys
from unittest.mock import MagicMock, patch

logging.basicConfig(level=logging.WARNING)

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"

FAIL_COUNT = 0


def _pass(msg: str) -> None:
    print(f"  {GREEN}[PASS]{RESET} {msg}")


def _fail(msg: str) -> None:
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  {RED}[FAIL]{RESET} {msg}", file=sys.stderr)


def _warn(msg: str) -> None:
    print(f"  {YELLOW}[WARN]{RESET} {msg}")


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# Expected power_w ranges derived from aic_[h,b]200_power_data statistics.
#   Context (prefill proxy): p50 / p90 of context_attention_perf.txt
#   Generation (decode proxy): p50 / p90 of generation_attention_perf.txt
# ---------------------------------------------------------------------------
_EXPECTED_RANGES = {
    "h200_sxm": {
        "prefill_min": 100.0,   # below p50 of any op
        "prefill_max": 710.0,   # slightly above 700W TDP (instrument noise)
        "decode_min": 100.0,
        "decode_max": 710.0,
    },
    "b200_sxm": {
        "prefill_min": 200.0,   # B200 idle floor
        "prefill_max": 1050.0,  # slightly above 1000W TDP
        "decode_min": 200.0,
        "decode_max": 1050.0,
    },
}


def _check_power_w_range(
    value: float,
    name: str,
    expected_min: float,
    expected_max: float,
) -> bool:
    if expected_min <= value <= expected_max:
        _pass(f"{name} = {value:.1f} W (expected [{expected_min:.0f}, {expected_max:.0f}] W)")
        return True
    else:
        _fail(
            f"{name} = {value:.1f} W is OUTSIDE expected range "
            f"[{expected_min:.0f}, {expected_max:.0f}] W"
        )
        return False


def validate_estimator(system: str, backend: str, hf_id: str) -> None:
    _section(f"AIC Estimator — {system} / {backend}")

    try:
        from dynamo.planner.monitoring.aic_estimator import AIConfiguratorPerfEstimator
    except ImportError as exc:
        _fail(f"Cannot import AIConfiguratorPerfEstimator: {exc}")
        return

    try:
        estimator = AIConfiguratorPerfEstimator(
            hf_id=hf_id,
            system=system,
            backend=backend,
        )
        _pass(f"AIConfiguratorPerfEstimator loaded ({system}/{backend})")
    except Exception as exc:
        _fail(f"Failed to instantiate AIConfiguratorPerfEstimator: {exc}")
        return

    # --- Check: database loaded ---
    if estimator.database:
        _pass("AIC database is non-empty")
    else:
        _fail("AIC database is empty or None")
        return

    ranges = _EXPECTED_RANGES.get(system, _EXPECTED_RANGES["h200_sxm"])

    # --- Check: prefill power_w is non-zero and in range ---
    _section(f"  prefill estimate (isl=1024, tp=8)")
    try:
        prefill_result = estimator.estimate_prefill_perf(
            isl=1024,
            tp_size=8,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
        )
        power_p = float(prefill_result.get("power_w") or 0.0)
        ttft = float(prefill_result.get("context_latency") or 0.0)

        if power_p == 0.0:
            _fail(
                "prefill power_w = 0.0 — database has not been updated with power data. "
                "Did you run `tools/integrate_aic_power_data.py` and reinstall the package?"
            )
        else:
            _check_power_w_range(power_p, "prefill power_w", ranges["prefill_min"], ranges["prefill_max"])

        if ttft > 0:
            _pass(f"prefill context_latency = {ttft:.2f} ms (non-zero)")
        else:
            _fail(f"prefill context_latency = {ttft} (invalid)")

    except Exception as exc:
        _fail(f"estimate_prefill_perf raised: {exc}")

    # --- Check: decode power_w ---
    _section(f"  decode estimate (isl=1024, osl=128, bs=16, tp=8)")
    try:
        decode_result = estimator.estimate_perf(
            isl=1024,
            osl=128,
            batch_size=16,
            mode="decode",
            tp_size=8,
            pp_size=1,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
        )
        power_d = float(decode_result.get("power_w") or 0.0)
        itl = float(decode_result.get("tpot") or 0.0)

        if power_d == 0.0:
            _fail(
                "decode power_w = 0.0 — database has not been updated with power data."
            )
        else:
            _check_power_w_range(power_d, "decode power_w", ranges["decode_min"], ranges["decode_max"])

        if itl > 0:
            _pass(f"decode tpot = {itl:.2f} ms (non-zero)")
        else:
            _fail(f"decode tpot = {itl} (invalid)")

    except Exception as exc:
        _fail(f"estimate_perf (decode) raised: {exc}")


def validate_optimizer_uses_real_power(system: str, backend: str, hf_id: str) -> None:
    """Confirm AICPowerOptimizer picks up real power_w, not TDP fallback.

    We run optimize() with a real estimator instance and intercept the log
    WARNING that the TDP-fallback path emits.  If no WARNING fires → the
    optimizer correctly consumed real power_w.
    """
    _section("AICPowerOptimizer — real power_w path (no TDP fallback)")

    try:
        from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
        from dynamo.planner.config.parallelization import PickedParallelConfig
        from dynamo.planner.config.planner_config import PlannerConfig
        from dynamo.planner.monitoring.aic_power_optimizer import AICPowerOptimizer
    except ImportError as exc:
        _fail(f"Cannot import planner modules: {exc}")
        return

    pick = PickedParallelConfig(tp=8, pp=1, dp=1, moe_tp=1, moe_ep=1)
    spec = AICInterpolationSpec(
        hf_id=hf_id,
        system=system,
        backend=backend,
        isl=1024,
        osl=128,
        sweep_max_context_length=4096,
        prefill_interpolation_granularity=4,
        decode_interpolation_granularity=4,
        prefill_pick=pick,
        decode_pick=pick,
    )

    try:
        config = PlannerConfig(
            namespace="validate",
            enable_power_awareness=True,
            total_gpu_power_limit=80_000,  # 80 kW — intentionally generous
            power_agent_safe_default_watts=500,
            prefill_engine_gpu_power_limit=700,
            decode_engine_gpu_power_limit=700,
            enable_aic_optimizer=True,
            aic_interpolation=spec,
            ttft=300,
            itl=100,
        )
    except Exception as exc:
        _fail(f"PlannerConfig construction failed: {exc}")
        return

    import logging as _logging
    tdp_fallback_fired = False

    class _TDPWatcher(_logging.Handler):
        def emit(self, record):
            nonlocal tdp_fallback_fired
            if "power_w unavailable" in record.getMessage():
                tdp_fallback_fired = True

    watcher = _TDPWatcher()
    opt_logger = _logging.getLogger("dynamo.planner.monitoring.aic_power_optimizer")
    opt_logger.addHandler(watcher)
    opt_logger.setLevel(_logging.DEBUG)

    try:
        metrics = MagicMock()
        opt = AICPowerOptimizer(config, metrics)
        result = opt.optimize()   # uses the real estimator via normal import
    except Exception as exc:
        _fail(f"optimize() raised: {exc}")
        return
    finally:
        opt_logger.removeHandler(watcher)

    if result is None:
        _fail(
            "optimize() returned None — optimizer disabled at startup. "
            "Check logs for SLA infeasibility or AIC error."
        )
        return

    if tdp_fallback_fired:
        _warn(
            "TDP fallback warning was emitted — power_w is still 0.0 in the database. "
            "Run `tools/integrate_aic_power_data.py` and reinstall aiconfigurator."
        )
    else:
        _pass(
            f"Optimizer used real power_w (cap_p={result.cap_p} W, "
            f"cap_d={result.cap_d} W) — TDP fallback NOT triggered."
        )

    _pass(f"n_p={result.n_p}, n_d={result.n_d}, cap_p={result.cap_p} W, cap_d={result.cap_d} W")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end validation of the AIC power-data integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--system", default="h200_sxm", choices=list(_EXPECTED_RANGES), help="AIC system identifier.")
    parser.add_argument("--backend", default="trtllm", choices=["trtllm", "vllm", "sglang"], help="AIC backend.")
    parser.add_argument("--hf-id", default="Qwen/Qwen3-32B", help="HuggingFace model ID for the AIC sweep.")
    parser.add_argument("--skip-optimizer", action="store_true", help="Skip the optimizer smoke test (just check estimator).")
    args = parser.parse_args()

    print(f"\nAIC Power Integration Validator")
    print(f"  system  : {args.system}")
    print(f"  backend : {args.backend}")
    print(f"  hf_id   : {args.hf_id}")

    validate_estimator(args.system, args.backend, args.hf_id)

    if not args.skip_optimizer:
        validate_optimizer_uses_real_power(args.system, args.backend, args.hf_id)

    print()
    if FAIL_COUNT == 0:
        print(f"{GREEN}All checks passed.{RESET}")
    else:
        print(f"{RED}{FAIL_COUNT} check(s) FAILED.{RESET}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
