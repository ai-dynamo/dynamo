# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish worker load reports for the OpenAI backend worker.

The in-process backends read scheduler load directly from the engine and push
it over NATS via ``WorkerMetricsPublisher``. The OpenAI backend fronts the
engine as a separate HTTP server, so instead this module:

- fetches the engine's ``/get_server_info`` once at startup to size the KV
  cache (``ModelRuntimeConfig.total_kv_blocks``), and
- polls the engine's Prometheus ``/metrics`` to derive KV occupancy and queue
  depth, publishing them via ``WorkerMetricsPublisher`` so the frontend's
  worker monitor sees the same load signal as for in-process workers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx

from dynamo.llm import ModelRuntimeConfig, WorkerMetricsPublisher
from dynamo.runtime import Endpoint

LOGGER = logging.getLogger("dynamo.openai_backend.load_reporter")

DEFAULT_INTERVAL_SECONDS = 2.0
INTERVAL_ENV_VAR = "DYN_OPENAI_BACKEND_LOAD_REPORT_INTERVAL_SECS"
FETCH_TIMEOUT_SECONDS = 2.0

# (kv usage fraction, queued requests, running requests) gauge names, per
# engine family. Tried in order; the first family whose usage gauge is present
# wins.
_GAUGE_SETS = (
    ("sglang:token_usage", "sglang:num_queue_reqs", "sglang:num_running_reqs"),
    (
        "vllm:gpu_cache_usage_perc",
        "vllm:num_requests_waiting",
        "vllm:num_requests_running",
    ),
)


def load_report_interval_secs() -> float:
    """Poll interval from the environment; 0 disables load reporting."""
    raw = os.environ.get(INTERVAL_ENV_VAR)
    if raw is None:
        return DEFAULT_INTERVAL_SECONDS
    try:
        interval = float(raw)
    except ValueError:
        LOGGER.warning(
            "Invalid %s=%r; using default %.1fs",
            INTERVAL_ENV_VAR,
            raw,
            DEFAULT_INTERVAL_SECONDS,
        )
        return DEFAULT_INTERVAL_SECONDS
    return max(interval, 0.0)


def _engine_url(base_url: str, path: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"engine URL must be absolute http(s), got {base_url!r}")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _tokens_to_kv_blocks(tokens: int, page_size: int | None) -> int:
    # Same tokens->blocks convention as dynamo.sglang.capacity.
    if not page_size or page_size <= 1:
        return tokens

    return (tokens + page_size - 1) // page_size


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


@dataclass(frozen=True)
class EngineCapacity:
    total_kv_blocks: int | None
    max_num_batched_tokens: int | None
    data_parallel_size: int | None


async def fetch_engine_capacity(engine_base_url: str) -> Optional[EngineCapacity]:
    """Read KV capacity from the engine's ``/get_server_info`` (SGLang).

    Returns None when the endpoint is missing or unusable (e.g. vLLM), so the
    caller can register without a runtime config.
    """
    try:
        url = _engine_url(engine_base_url, "/get_server_info")
        async with httpx.AsyncClient(
            trust_env=False, timeout=FETCH_TIMEOUT_SECONDS
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            info = response.json()
    except Exception as exc:  # noqa: BLE001 - capacity is best-effort
        LOGGER.warning("Could not fetch engine capacity from /get_server_info: %s", exc)
        return None

    if not isinstance(info, dict):
        LOGGER.warning("Unexpected /get_server_info payload type: %s", type(info))
        return None

    max_total_tokens = _positive_int(info.get("max_total_num_tokens"))
    page_size = _positive_int(info.get("page_size"))
    total_kv_blocks = (
        _tokens_to_kv_blocks(max_total_tokens, page_size) if max_total_tokens else None
    )
    max_num_batched_tokens = (
        _positive_int(info.get("max_prefill_tokens")) or max_total_tokens
    )

    return EngineCapacity(
        total_kv_blocks=total_kv_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        data_parallel_size=_positive_int(info.get("dp_size")),
    )


def build_runtime_config(
    capacity: Optional[EngineCapacity],
) -> Optional[ModelRuntimeConfig]:
    if capacity is None or capacity.total_kv_blocks is None:
        return None

    runtime_config = ModelRuntimeConfig()
    runtime_config.total_kv_blocks = capacity.total_kv_blocks
    if capacity.max_num_batched_tokens is not None:
        runtime_config.max_num_batched_tokens = capacity.max_num_batched_tokens
    if capacity.data_parallel_size is not None:
        runtime_config.data_parallel_size = capacity.data_parallel_size
    return runtime_config


@dataclass
class _LoadSample:
    usage_frac: float | None = None
    waiting: float | None = None
    running: float | None = None


def parse_load_samples(metrics_text: str) -> Optional[dict[int, _LoadSample]]:
    """Extract per-dp-rank load gauges from engine Prometheus exposition text.

    Returns None when no known engine gauge family is present.
    """
    from prometheus_client.parser import text_string_to_metric_families

    values: dict[str, dict[int, float]] = {}
    wanted = {name for gauge_set in _GAUGE_SETS for name in gauge_set}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in wanted:
                continue
            try:
                dp_rank = int(sample.labels.get("dp_rank", 0))
            except (TypeError, ValueError):
                dp_rank = 0
            values.setdefault(sample.name, {})[dp_rank] = sample.value

    for usage_name, waiting_name, running_name in _GAUGE_SETS:
        if usage_name not in values:
            continue
        samples: dict[int, _LoadSample] = {}
        for dp_rank, usage in values[usage_name].items():
            samples[dp_rank] = _LoadSample(usage_frac=usage)
        for dp_rank, waiting in values.get(waiting_name, {}).items():
            samples.setdefault(dp_rank, _LoadSample()).waiting = waiting
        for dp_rank, running in values.get(running_name, {}).items():
            samples.setdefault(dp_rank, _LoadSample()).running = running
        return samples

    return None


class EngineLoadReporter:
    """Poll the engine's ``/metrics`` and publish worker load over NATS."""

    def __init__(
        self,
        endpoint: Endpoint,
        engine_base_url: str,
        *,
        total_kv_blocks: int | None,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._endpoint = endpoint
        self._metrics_url = _engine_url(engine_base_url, "/metrics")
        self._total_kv_blocks = total_kv_blocks
        self._interval = interval_seconds
        self._publisher = WorkerMetricsPublisher()
        self._client: httpx.AsyncClient | None = None
        self._task: asyncio.Task[None] | None = None
        self._scrape_warned = False

    async def start(self) -> None:
        await self._publisher.create_endpoint(self._endpoint)
        # Bootstrap the load signal so the frontend monitor sees this worker
        # before the first scrape (same dummy-init pattern as dynamo.sglang).
        self._publisher.publish(0, kv_used_blocks=0, num_waiting_reqs=0)
        self._client = httpx.AsyncClient(trust_env=False, timeout=FETCH_TIMEOUT_SECONDS)
        self._task = asyncio.create_task(self._run())
        LOGGER.info(
            "Engine load reporter started (url=%s, interval=%.1fs, total_kv_blocks=%s)",
            self._metrics_url,
            self._interval,
            self._total_kv_blocks,
        )

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never crash the worker
                self._log_scrape_failure(str(exc))

    async def _poll_once(self) -> None:
        assert self._client is not None
        response = await self._client.get(self._metrics_url)
        response.raise_for_status()

        samples = parse_load_samples(response.text)
        if samples is None:
            self._log_scrape_failure(
                f"no known engine load gauges in {self._metrics_url}"
            )
            return

        if self._scrape_warned:
            LOGGER.info("Engine load scrape recovered")
            self._scrape_warned = False

        for dp_rank, sample in sorted(samples.items()):
            kv_used_blocks = None
            if self._total_kv_blocks is not None and sample.usage_frac is not None:
                kv_used_blocks = int(sample.usage_frac * self._total_kv_blocks)
            num_waiting_reqs = (
                int(sample.waiting) if sample.waiting is not None else None
            )
            if kv_used_blocks is None and num_waiting_reqs is None:
                continue
            # The publisher dedupes unchanged values, so an idle engine
            # produces no NATS traffic.
            self._publisher.publish(
                dp_rank,
                kv_used_blocks=kv_used_blocks,
                num_waiting_reqs=num_waiting_reqs,
            )
            LOGGER.debug(
                "Published load report dp_rank=%s kv_used_blocks=%s "
                "num_waiting_reqs=%s num_running_reqs=%s",
                dp_rank,
                kv_used_blocks,
                num_waiting_reqs,
                sample.running,
            )

    def _log_scrape_failure(self, message: str) -> None:
        if not self._scrape_warned:
            self._scrape_warned = True
            LOGGER.warning(
                "Engine load scrape failed (%s); further failures logged at debug",
                message,
            )
        else:
            LOGGER.debug("Engine load scrape failed (%s)", message)
