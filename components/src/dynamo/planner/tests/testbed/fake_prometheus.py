# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FakePrometheusClient — drop-in for PrometheusAPIClient.

Shared by both α-class (backed by SyntheticFleet) and γ-class (backed by
SyntheticPowerOverlay).  The discriminator is the ``source`` constructor arg.

The source object must expose ``observation_at(tick: int)`` returning an object
with attributes ``ttft_avg_s``, ``itl_avg_s``, ``power_w_prefill``,
``power_w_decode``, ``total_tokens_per_sec``.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class _ObservationSource(Protocol):
    def observation_at(self, tick: int) -> Any:
        ...

    def is_signal_in_outage(self, signal: str) -> bool:
        ...

    def prom_stale_lag(self) -> Optional[int]:
        ...


class FakePrometheusClient:
    """Drop-in for PrometheusAPIClient with same method signatures."""

    def __init__(self, source: Any) -> None:
        self._source = source
        self._current_tick: int = 0

    def set_tick(self, tick: int) -> None:
        """Called by runner at start of each tick."""
        self._current_tick = tick

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _effective_tick(self, signal: str) -> Optional[int]:
        """Return the tick to read from (applying stale lag) or None for outage."""
        if self._source.is_signal_in_outage(signal):
            return None
        lag = self._source.prom_stale_lag()
        t = self._current_tick - (lag or 0)
        return max(0, t)

    def _get_obs(self, signal: str):
        t = self._effective_tick(signal)
        if t is None:
            return None
        return self._source.observation_at(t)

    # ------------------------------------------------------------------
    # PrometheusAPIClient interface (same signatures as the real class)
    # ------------------------------------------------------------------

    def get_avg_time_to_first_token(self, *args: Any, **kwargs: Any) -> Optional[float]:
        obs = self._get_obs("ttft")
        return obs.ttft_avg_s if obs is not None else None

    def get_avg_inter_token_latency(self, *args: Any, **kwargs: Any) -> Optional[float]:
        obs = self._get_obs("itl")
        return obs.itl_avg_s if obs is not None else None

    def get_avg_per_gpu_power_by_component(
        self, *, component: str, **kwargs: Any
    ) -> Optional[float]:
        signal = f"power_{component[0]}"  # "power_p" or "power_d"
        obs = self._get_obs(signal)
        if obs is None:
            return None
        if component == "prefill":
            return obs.power_w_prefill
        elif component == "decode":
            return obs.power_w_decode
        return None

    def get_total_dgd_power(self, *args: Any, **kwargs: Any) -> Optional[float]:
        obs = self._get_obs("power_p")
        if obs is None:
            return None
        return obs.power_w_prefill + obs.power_w_decode

    def get_avg_request_count(self, *args: Any, **kwargs: Any) -> Optional[float]:
        obs = self._get_obs("capacity")
        if obs is None:
            return None
        return float(obs.traffic.num_req) if obs.traffic else 0.0

    def get_avg_input_sequence_tokens(self, *args: Any, **kwargs: Any) -> Optional[float]:
        obs = self._get_obs("capacity")
        if obs is None:
            return None
        return float(obs.traffic.isl) if obs.traffic else 0.0

    def get_avg_output_sequence_tokens(self, *args: Any, **kwargs: Any) -> Optional[float]:
        obs = self._get_obs("capacity")
        if obs is None:
            return None
        return float(obs.traffic.osl) if obs.traffic else 0.0

    def get_avg_kv_hit_rate(self, *args: Any, **kwargs: Any) -> Optional[float]:
        obs = self._get_obs("capacity")
        if obs is None:
            return None
        return obs.traffic.kv_hit_rate if obs.traffic else None

    def get_avg_request_duration(self, *args: Any, **kwargs: Any) -> Optional[float]:
        return None

    def warn_if_router_not_scraped(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_recent_and_averaged_metrics(self, *args: Any, **kwargs: Any):  # type: ignore[return]
        return None
