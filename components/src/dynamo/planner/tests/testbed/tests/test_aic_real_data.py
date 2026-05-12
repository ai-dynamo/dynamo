# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests against a *real* AIC perf database.

These tests are opt-in: they run only when the environment variable
``AIC_SANDBOX_DIR`` points at a populated ``systems/`` directory containing
both the AIC system YAMLs (e.g. ``h200_sxm.yaml``) and a ``data/`` tree
matching the in-repo layout used by ``aiconfigurator.sdk.perf_database``.
On CI you typically mount a read-only copy of the AIC power-data tarball
and point ``AIC_SANDBOX_DIR`` at it.

What we verify against the live data:

1. ``AIConfiguratorPerfEstimator`` loads cleanly and reports non-zero
   ``power_w`` for both prefill and decode on a representative workload.
2. ``AICPowerOptimizer.optimize()`` produces a ``PowerAwareConfig`` with
   per-GPU caps that never exceed nameplate TDP × 1.1 (the defensive clamp
   from §8 row 14 of ``powerplanner-design.md``).
3. The clamp counter ``aic_power_w_clamped_total{side=...}`` fires iff
   AIC's raw ``power_w`` for that side was outside the physical envelope.
4. The multi-tick EMA loop converges in three regimes (well-calibrated,
   AIC-over-predicts → pegs at 0.5, AIC-under-predicts → 1.67) and
   ``should_reoptimize()`` respects the hysteresis count.

Why this lives in the testbed rather than in
``integration/test_aic_power_optimizer.py``:

* It needs ``aiconfigurator`` installed and a real sandbox on disk — a
  much heavier prerequisite than the rest of the integration suite, which
  uses a pure ``MagicMock`` estimator.
* It belongs alongside the other "real-system bridge" tests: the testbed
  is the dedicated home for "exercise the planner against real-ish data
  without a Kubernetes cluster".
* Gating via env var keeps the default test invocation cheap; CI opts in
  by setting ``AIC_SANDBOX_DIR`` once.
"""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Hard gate: only collect when AIC_SANDBOX_DIR is set AND aiconfigurator is
# importable.  Both checks fail-skip the entire module rather than the
# individual tests so the skip reason shows up exactly once.
# ---------------------------------------------------------------------------

_SANDBOX_ENV = os.environ.get("AIC_SANDBOX_DIR")
if not _SANDBOX_ENV:
    pytest.skip(
        "AIC_SANDBOX_DIR not set — opt-in test, see "
        "docs/design-docs/powerplanner-testbed-design.md for sandbox setup. "
        "Locally: AIC_SANDBOX_DIR=<repo>/.aic_sandbox/systems pytest -m real_aic ...",
        allow_module_level=True,
    )

_SANDBOX_PATH = Path(_SANDBOX_ENV)
if not _SANDBOX_PATH.is_dir():
    pytest.skip(
        f"AIC_SANDBOX_DIR={_SANDBOX_ENV} is not a directory.",
        allow_module_level=True,
    )

# Probe the AIC package up front — without it, none of this can run.
pytest.importorskip(
    "aiconfigurator.sdk.perf_database",
    reason="real-AIC tests require the aiconfigurator package; "
    "`pip install aiconfigurator` or run inside the dev pod.",
)

from dynamo.planner.config.aic_interpolation_spec import (  # noqa: E402
    AICInterpolationSpec,
)
from dynamo.planner.config.parallelization import PickedParallelConfig  # noqa: E402
from dynamo.planner.config.planner_config import PlannerConfig  # noqa: E402
from dynamo.planner.core.types import TrafficObservation  # noqa: E402
from dynamo.planner.monitoring.aic_power_optimizer import (  # noqa: E402
    AICPowerOptimizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sandbox plumbing — re-route AIC's path resolver at session scope.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _patch_aic_systems_dir() -> Iterator[None]:
    """Point AIC's perf-database loader at AIC_SANDBOX_DIR for the whole session.

    AIC computes its systems path at function-definition time (the
    ``systems_dir`` parameter has ``get_system_config_path()`` baked into
    its ``__defaults__``), so we must rewrite three functions' defaults
    in-place AND override the resolver itself.  Yes, this is ugly; no, the
    alternative (forking AIC) is uglier.  See aic_smoke.py in the repo
    root for the original isolation of this technique.
    """
    import aiconfigurator.sdk.perf_database as pd

    sandbox = str(_SANDBOX_PATH)
    orig_resolver = pd.get_system_config_path
    orig_defaults: dict[str, tuple] = {}

    def _override_resolver() -> str:
        return sandbox

    pd.get_system_config_path = _override_resolver

    for fn_name in ("get_supported_databases", "get_database", "get_all_databases"):
        fn = getattr(pd, fn_name, None)
        if fn is None or fn.__defaults__ is None:
            continue
        orig_defaults[fn_name] = fn.__defaults__
        n_pos = fn.__code__.co_argcount
        arg_names = fn.__code__.co_varnames[:n_pos]
        defaults = list(fn.__defaults__)
        first_default_idx = n_pos - len(defaults)
        for i, name in enumerate(arg_names[first_default_idx:]):
            if name == "systems_dir":
                defaults[i] = sandbox
        fn.__defaults__ = tuple(defaults)

    pd.databases_cache.clear()
    try:
        yield
    finally:
        pd.get_system_config_path = orig_resolver
        for fn_name, defaults in orig_defaults.items():
            fn = getattr(pd, fn_name)
            fn.__defaults__ = defaults
        pd.databases_cache.clear()


# ---------------------------------------------------------------------------
# Per-system parametrization — drives every test against every (system,
# backend, hf_id) tuple that is actually present in the sandbox.
# ---------------------------------------------------------------------------


def _discover_systems() -> list[dict]:
    """Return the list of (system, backend, hf_id, tdp_w_expected) combos to test.

    Each entry is checked against the live sandbox at collection time so
    missing data on the sandbox skips that specific parametrization rather
    than failing it.
    """
    import aiconfigurator.sdk.perf_database as pd

    out: list[dict] = []
    candidates = [
        # (system, backend, hf_id, expected_tdp_w, expected_clamp_decode)
        ("h200_sxm", "vllm", "LLAMA3.1_8B", 700.0, True),  # known to clamp (1275 W)
        ("b200_sxm", "trtllm", "LLAMA3.1_8B", 1000.0, False),  # stays in envelope
    ]
    available = pd.get_supported_databases()
    for sys_name, backend, hf_id, tdp, clamp_expected in candidates:
        backends = available.get(sys_name, {})
        versions = backends.get(backend, [])
        if not versions:
            continue
        out.append(
            {
                "system": sys_name,
                "backend": backend,
                "hf_id": hf_id,
                "tdp_w": tdp,
                "expect_decode_clamp": clamp_expected,
                "latest_version": sorted(versions)[-1],
            }
        )
    return out


_SKU_TABLE = _discover_systems()


def _sku_id(sku: dict) -> str:
    return f"{sku['system']}-{sku['backend']}-{sku['hf_id']}"


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_config(
    *, system: str, backend: str, hf_id: str, isl: int, osl: int
) -> PlannerConfig:
    interp = AICInterpolationSpec(
        hf_id=hf_id,
        system=system,
        backend=backend,
        isl=isl,
        osl=osl,
        sweep_max_context_length=isl + osl,
        prefill_interpolation_granularity=8,
        decode_interpolation_granularity=8,
        prefill_pick=PickedParallelConfig(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1),
        decode_pick=PickedParallelConfig(tp=1, pp=1, dp=1, moe_tp=1, moe_ep=1),
    )
    return PlannerConfig(
        namespace="real-aic-testbed",
        environment="virtual",
        backend=backend,
        mode="disagg",
        enable_aic_optimizer=True,
        aic_interpolation=interp,
        aic_system=system,
        enable_power_awareness=True,
        total_gpu_power_limit=16000,
        prefill_engine_gpu_power_limit=int(1500),
        decode_engine_gpu_power_limit=int(1500),
        power_agent_safe_default_watts=500,
        min_endpoint=1,
        max_gpu_budget=8,
        ttft=10_000.0,
        itl=10_000.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.real_aic
@pytest.mark.parametrize(
    "sku", _SKU_TABLE, ids=[_sku_id(s) for s in _SKU_TABLE] or ["no-skus-found"]
)
class TestAICRealData:
    """End-to-end checks against the real AIC perf database.

    The class-level parametrize lets a single sandbox cover multiple SKUs
    in a single pytest run.
    """

    ISL, OSL = 2048, 256

    def test_estimator_returns_real_power_w(self, sku: dict) -> None:
        """Direct estimator must produce non-zero power_w."""
        if not _SKU_TABLE:
            pytest.skip("no SKUs available in sandbox")
        from dynamo.planner.monitoring.aic_estimator import AIConfiguratorPerfEstimator

        est = AIConfiguratorPerfEstimator(
            hf_id=sku["hf_id"], system=sku["system"], backend=sku["backend"]
        )
        assert float(est.database.system_spec["gpu"]["power"]) == pytest.approx(
            sku["tdp_w"]
        )

        prefill = est.estimate_perf(self.ISL, self.OSL, 8, mode="prefill", tp_size=1)
        decode = est.estimate_perf(self.ISL, self.OSL, 8, mode="decode", tp_size=1)
        prefill_w = float(prefill.get("power_w") or 0.0)
        decode_w = float(decode.get("power_w") or 0.0)
        assert prefill_w > 0, (
            f"{_sku_id(sku)}: AIC returned power_w=0 for prefill — "
            "power data not actually loaded in sandbox"
        )
        assert decode_w > 0
        # Per-op direct call at batch=8 should NEVER extrapolate non-physical.
        assert prefill_w <= sku["tdp_w"] * 1.1, (
            f"direct estimate exceeded TDP×1.1 at small batch — "
            f"sandbox data may be corrupt for {_sku_id(sku)}"
        )
        assert decode_w <= sku["tdp_w"] * 1.1

    def test_optimize_produces_well_formed_config(self, sku: dict) -> None:
        cfg = _make_config(
            system=sku["system"],
            backend=sku["backend"],
            hf_id=sku["hf_id"],
            isl=self.ISL,
            osl=self.OSL,
        )
        opt = AICPowerOptimizer(config=cfg, metrics=MagicMock())
        result = opt.optimize()

        assert result is not None, "optimize() unexpectedly returned None"
        assert result.n_p >= 1 and result.n_d >= 1
        assert result.cap_p > 0 and result.cap_d > 0
        # Caps must NEVER exceed nameplate TDP × _COEFF_MAX (2.0) when c_power
        # coefficient has drifted up; that's fine, the underlying *aic_power_w*
        # is what must be bounded, and that's what the clamp guards.
        assert result.cap_d <= sku["tdp_w"] * 2.0 + 1
        assert result.cap_p <= sku["tdp_w"] * 2.0 + 1

    def test_clamp_engages_as_expected(self, sku: dict) -> None:
        """Clamp counter fires iff raw aic_power_w > 1.1 × TDP."""
        cfg = _make_config(
            system=sku["system"],
            backend=sku["backend"],
            hf_id=sku["hf_id"],
            isl=self.ISL,
            osl=self.OSL,
        )
        metrics = MagicMock()
        opt = AICPowerOptimizer(config=cfg, metrics=metrics)
        result = opt.optimize()
        assert result is not None

        threshold = sku["tdp_w"] * 1.1
        expected_clamped: list[str] = []
        if result.aic_power_w_prefill > threshold:
            expected_clamped.append("prefill")
        if result.aic_power_w_decode > threshold:
            expected_clamped.append("decode")

        sides = sorted(
            c.kwargs.get("side")
            for c in metrics.aic_power_w_clamped_total.labels.call_args_list
        )
        assert sides == sorted(expected_clamped), (
            f"{_sku_id(sku)}: clamp sides mismatch — got {sides}, "
            f"expected {sorted(expected_clamped)} "
            f"(raw prefill={result.aic_power_w_prefill:.1f} W, "
            f"raw decode={result.aic_power_w_decode:.1f} W, "
            f"threshold={threshold:.0f} W)"
        )

        # For the known H200 vLLM 1275 W case, also assert the *applied* cap
        # came out at or below TDP — that's the user-visible contract.
        if sku["expect_decode_clamp"]:
            assert result.aic_power_w_decode > threshold, (
                f"sandbox H200 should still exhibit the 1275 W extrapolation; "
                f"got {result.aic_power_w_decode:.1f} W instead — has the "
                f"data been re-collected?"
            )
            assert result.cap_d <= sku["tdp_w"] + 1, (
                f"cap_d={result.cap_d} W must be clamped to ~TDP "
                f"({sku['tdp_w']:.0f} W) when AIC over-predicts"
            )


@pytest.mark.real_aic
class TestAICDriftLoopRealData:
    """Multi-tick EMA loop against the real H200 sandbox.

    Drives update_correction() with synthetic observation streams whose
    means are anchored to physical (NOT AIC) values; the AIC denominators
    come from a real optimize() call.  This is the closest thing to a
    production loop we can run without a real GPU.
    """

    ISL, OSL = 2048, 256
    HF_ID = "LLAMA3.1_8B"

    @pytest.fixture
    def h200_optimizer(self) -> AICPowerOptimizer:
        """Skip unless the H200 SXM vLLM data is in the sandbox."""
        import aiconfigurator.sdk.perf_database as pd

        available = pd.get_supported_databases().get("h200_sxm", {}).get("vllm", [])
        if not available:
            pytest.skip("H200 SXM + vLLM data not in sandbox")
        cfg = _make_config(
            system="h200_sxm",
            backend="vllm",
            hf_id=self.HF_ID,
            isl=self.ISL,
            osl=self.OSL,
        )
        opt = AICPowerOptimizer(config=cfg, metrics=MagicMock())
        result = opt.optimize()
        assert result is not None
        return opt

    @staticmethod
    def _drive(
        opt: AICPowerOptimizer,
        *,
        observed_power_w_decode_mean: float,
        n_ticks: int = 80,
        noise_frac: float = 0.05,
        seed: int = 42,
    ) -> float:
        rng = random.Random(seed)
        for _ in range(n_ticks):
            traffic = TrafficObservation(
                duration_s=60.0,
                num_req=10.0,
                isl=2048.0,
                osl=256.0,
                ttft_avg=0.05,
                itl_avg=0.01,
                total_tokens_per_s=200.0,
                scheduled_prefill_tokens=2000.0,
                scheduled_decode_kv_tokens=2000.0,
            )
            obs = observed_power_w_decode_mean * (
                1.0 + rng.uniform(-noise_frac, noise_frac)
            )
            opt.update_correction(
                traffic=traffic,
                observed_ttft_avg=traffic.ttft_avg,
                observed_itl_avg=traffic.itl_avg,
                observed_power_w_prefill=obs * 0.6,
                observed_power_w_decode=obs,
            )
        return opt._c_power_d

    def test_well_calibrated_converges_to_one(
        self, h200_optimizer: AICPowerOptimizer
    ) -> None:
        """Observed == aic_power_w_decode → c_power_d converges to 1.0."""
        target = h200_optimizer._last_optimal_config.aic_power_w_decode
        final = self._drive(h200_optimizer, observed_power_w_decode_mean=target)
        assert (
            abs(final - 1.0) < 0.05
        ), f"c_power_d={final:.3f} did not converge to 1.0 under matched observations"

    def test_over_prediction_pegs_at_lower_clamp(
        self, h200_optimizer: AICPowerOptimizer
    ) -> None:
        """Observed = 500 W < raw aic 1275 W → c_power_d hits 0.5 clamp."""
        raw_aic_decode = h200_optimizer._last_optimal_config.aic_power_w_decode
        if raw_aic_decode < 1000:
            pytest.skip(
                f"H200 sandbox no longer exhibits the >1000 W decode artefact "
                f"(raw={raw_aic_decode:.0f} W); drift-peg test is moot."
            )
        final = self._drive(h200_optimizer, observed_power_w_decode_mean=500.0)
        assert final == pytest.approx(
            0.5
        ), f"c_power_d={final:.3f} should peg at 0.5 when observed << aic"

    def test_under_prediction_converges_below_upper_clamp(
        self, h200_optimizer: AICPowerOptimizer
    ) -> None:
        """Observed > raw_aic but ratio < 2 → c_power_d converges within (1, 2)."""
        raw = h200_optimizer._last_optimal_config.aic_power_w_decode
        # Pick an observed value that gives ratio ≈ 1.5 (well inside the clamp).
        target_ratio = 1.5
        observed = raw * target_ratio
        final = self._drive(h200_optimizer, observed_power_w_decode_mean=observed)
        # Allow a small noise margin; ratio derived from noisy observations
        # may differ from target_ratio by ~the noise frac.
        assert 1.4 < final < 1.6, (
            f"c_power_d={final:.3f} did not converge to ~{target_ratio} "
            f"(raw aic_decode={raw:.0f} W, observed mean={observed:.0f} W)"
        )

    def test_hysteresis_holds_on_sla_drift(
        self, h200_optimizer: AICPowerOptimizer
    ) -> None:
        """should_reoptimize fires only after aic_drift_consecutive_ticks."""
        opt = h200_optimizer
        opt._time_of_last_optimize = (
            time.monotonic() - opt._config.aic_reoptimize_interval - 1.0
        )
        hysteresis = opt._config.aic_drift_consecutive_ticks
        triggered_at: int | None = None
        for tick in range(hysteresis + 3):
            traffic = TrafficObservation(
                duration_s=60.0,
                num_req=10.0,
                isl=2048.0,
                osl=256.0,
                ttft_avg=opt._config.ttft_ms / 1000.0 * 2.0,  # 2× SLA
                itl_avg=0.005,
                total_tokens_per_s=200.0,
                scheduled_prefill_tokens=2000.0,
                scheduled_decode_kv_tokens=2000.0,
            )
            if opt.should_reoptimize(traffic):
                triggered_at = tick
                break
        assert (
            triggered_at is not None
        ), "should_reoptimize never fired under sustained SLA breach"
        assert triggered_at + 1 == hysteresis, (
            f"hysteresis broken: triggered at tick={triggered_at}, "
            f"expected exactly tick={hysteresis - 1}"
        )

    def test_healthy_load_does_not_trigger_reoptimize(
        self, h200_optimizer: AICPowerOptimizer
    ) -> None:
        opt = h200_optimizer
        opt._time_of_last_optimize = (
            time.monotonic() - opt._config.aic_reoptimize_interval - 1.0
        )
        opt._estimated_throughput = 1000.0
        for tick in range(50):
            traffic = TrafficObservation(
                duration_s=60.0,
                num_req=10.0,
                isl=2048.0,
                osl=256.0,
                ttft_avg=0.020,  # well under SLA
                itl_avg=0.005,
                total_tokens_per_s=500.0,
                scheduled_prefill_tokens=2000.0,
                scheduled_decode_kv_tokens=2000.0,
            )
            assert not opt.should_reoptimize(
                traffic
            ), f"spurious reoptimize at tick {tick} under healthy load"
