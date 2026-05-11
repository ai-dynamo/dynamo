# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FakeAIC — testbed replacement for the AIConfigurator estimator.

Injected into ``AICPowerOptimizer`` via ``optimizer._aic_estimator_factory``.
The factory ignores ``hf_id / system / backend`` and returns a
``_FakeAICEstimator`` whose responses come from the per-SKU system spec.

Fault injection:
  - ``fault_mode="normal"``      → returns system-spec constants
  - ``fault_mode="raises"``      → estimate_prefill_perf raises RuntimeError
  - ``fault_mode="empty_pareto"``→ returns huge TTFT (> any SLA) to trigger
                                   infeasibility path in AICPowerOptimizer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.scenarios import SystemSpec


class _FakeDatabase:
    """Minimal stand-in for AIConfiguratorPerfEstimator.database."""

    def __init__(self, tdp_w: float) -> None:
        self.system_spec = {"gpu": {"power": tdp_w}}


class _FakeAICEstimator:
    """Minimal stand-in for AIConfiguratorPerfEstimator.

    Implements only the methods that AICPowerOptimizer.optimize() calls.
    """

    def __init__(self, system_spec: "SystemSpec", fault_mode: str) -> None:
        self._spec = system_spec
        self._fault_mode = fault_mode
        self.database = _FakeDatabase(system_spec.tdp_w)

    def estimate_prefill_perf(self, isl: int, **kwargs: Any) -> dict[str, Any]:
        if self._fault_mode == "raises":
            raise RuntimeError("synthetic AIC failure (estimate_prefill_perf)")
        if self._fault_mode == "empty_pareto":
            # Return huge TTFT to force SLA infeasibility in the optimizer.
            return {"context_latency": 999_999.0, "power_w": 0.0}
        return {
            "context_latency": self._spec.aic_ttft_ms,
            "power_w": self._spec.aic_power_w_prefill,
        }

    def get_max_kv_tokens(self, isl: int, osl: int, **kwargs: Any) -> int:
        if self._fault_mode == "raises":
            raise RuntimeError("synthetic AIC failure (get_max_kv_tokens)")
        return self._spec.max_kv_tokens

    def estimate_perf(
        self,
        isl: int,
        osl: int,
        batch_size: int,
        mode: str = "decode",
        **kwargs: Any,
    ) -> dict[str, Any]:
        if self._fault_mode == "raises":
            raise RuntimeError("synthetic AIC failure (estimate_perf)")
        if self._fault_mode == "empty_pareto":
            return {"tpot": 999_999.0, "power_w": 0.0}
        return {
            "tpot": self._spec.aic_itl_ms,
            "power_w": self._spec.aic_power_w_decode,
        }


class FakeAIC:
    """Testbed AIC replacement.

    Shared between α and γ-class scenarios.  The fault mode is controlled by
    the scenario's active event state; the runner calls ``set_fault_mode()``
    when processing ``aic_failure`` events.
    """

    def __init__(self, system_spec: "SystemSpec") -> None:
        self._system_spec = system_spec
        self._fault_mode: str = "normal"

    def set_fault_mode(self, mode: str) -> None:
        """Set fault mode: 'normal' | 'raises' | 'empty_pareto'."""
        assert mode in ("normal", "raises", "empty_pareto"), f"Unknown fault mode: {mode}"
        self._fault_mode = mode

    def reset_fault(self) -> None:
        self._fault_mode = "normal"

    def make_estimator_factory(self) -> Callable[..., _FakeAICEstimator]:
        """Return a callable compatible with optimizer._aic_estimator_factory.

        The factory is called as ``factory(hf_id=..., system=..., backend=...)``
        and should return an estimator-like object.  We capture ``self`` so the
        estimator always reads the current fault mode.
        """
        fake_aic = self

        def _factory(hf_id: str, system: str, backend: str) -> _FakeAICEstimator:
            return _FakeAICEstimator(fake_aic._system_spec, fake_aic._fault_mode)

        return _factory
