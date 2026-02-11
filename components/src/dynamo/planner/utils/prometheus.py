# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import math
import typing
from dataclasses import dataclass, field
from typing import Optional

import aiohttp
from prometheus_api_client import PrometheusConnect
from prometheus_client.parser import text_string_to_metric_families
from pydantic import BaseModel, ValidationError

from dynamo import prometheus_names
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    ttft: Optional[float] = None
    itl: Optional[float] = None
    num_req: Optional[float] = None
    isl: Optional[float] = None
    osl: Optional[float] = None
    request_duration: Optional[float] = None
    p_load: Optional[float] = None
    d_load: Optional[float] = None

    def is_valid(self) -> bool:
        """Check if all required metrics are valid (not None and not NaN)."""
        required = [
            self.ttft,
            self.itl,
            self.isl,
            self.osl,
            self.num_req,
            self.request_duration,
        ]
        return all(v is not None and not math.isnan(v) for v in required)


@dataclass
class CachedLoadMetrics:
    """Container for load metrics used by load-based scaling.

    Attributes:
        recent:              Most recent per-worker metrics (from the latest sample).
                             Keyed by worker_id -> {metric_name: value}.
        per_worker_averaged: Per-worker metrics averaged over time (not across workers).
                             Keyed by worker_id -> {metric_name: value}.
        cluster_averaged:    Metrics averaged over time and all workers.
                             Flat dict {metric_name: value}.
    """

    recent: dict[str, dict[str, float]] = field(default_factory=dict)
    per_worker_averaged: dict[str, dict[str, float]] = field(default_factory=dict)
    cluster_averaged: dict[str, float] = field(default_factory=dict)


class FrontendMetric(BaseModel):
    container: typing.Optional[str] = None
    dynamo_namespace: typing.Optional[str] = None
    endpoint: typing.Optional[str] = None
    instance: typing.Optional[str] = None
    job: typing.Optional[str] = None
    model: typing.Optional[str] = None
    namespace: typing.Optional[str] = None
    pod: typing.Optional[str] = None


class FrontendMetricContainer(BaseModel):
    metric: FrontendMetric
    value: typing.Tuple[float, float]  # [timestamp, value]


class PrometheusAPIClient:
    def __init__(self, url: str, dynamo_namespace: str):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)
        self.dynamo_namespace = dynamo_namespace

    def _get_average_metric(
        self, full_metric_name: str, interval: str, operation_name: str, model_name: str
    ) -> float:
        """
        Helper method to get average metrics using the pattern:
        increase(metric_sum[interval])/increase(metric_count[interval])

        Args:
            full_metric_name: Full metric name (e.g., 'dynamo_frontend_inter_token_latency_seconds')
            interval: Time interval for the query (e.g., '60s')
            operation_name: Human-readable name for error logging

        Returns:
            Average metric value or 0 if no data/error
        """
        try:
            # Prepend the frontend metric prefix if not already present
            if not full_metric_name.startswith(prometheus_names.name_prefix.FRONTEND):
                full_metric_name = (
                    f"{prometheus_names.name_prefix.FRONTEND}_{full_metric_name}"
                )
            query = f"increase({full_metric_name}_sum[{interval}])/increase({full_metric_name}_count[{interval}])"
            result = self.prom.custom_query(query=query)
            if not result:
                # No data available yet (no requests made) - return 0 silently
                logger.warning(
                    f"No prometheus metric data available for {full_metric_name}, use 0 instead"
                )
                return 0
            metrics_containers = parse_frontend_metric_containers(result)

            values = []
            for container in metrics_containers:
                # Frontend lowercases model names for Prometheus labels so we need to do case-insensitive comparison
                if (
                    container.metric.model
                    and container.metric.model.lower() == model_name.lower()
                    and container.metric.dynamo_namespace == self.dynamo_namespace
                ):
                    values.append(container.value[1])

            if not values:
                logger.warning(
                    f"No prometheus metric data available for {full_metric_name} with model {model_name} and dynamo namespace {self.dynamo_namespace}, use 0 instead"
                )
                return 0
            return sum(values) / len(values)

        except Exception as e:
            logger.error(f"Error getting {operation_name}: {e}")
            return 0

    def get_avg_inter_token_latency(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend_service.INTER_TOKEN_LATENCY_SECONDS,
            interval,
            "avg inter token latency",
            model_name,
        )

    def get_avg_time_to_first_token(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend_service.TIME_TO_FIRST_TOKEN_SECONDS,
            interval,
            "avg time to first token",
            model_name,
        )

    def get_avg_request_duration(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend_service.REQUEST_DURATION_SECONDS,
            interval,
            "avg request duration",
            model_name,
        )

    def get_avg_request_count(self, interval: str, model_name: str):
        # This function follows a different query pattern than the other metrics
        try:
            requests_total_metric = prometheus_names.frontend_service.REQUESTS_TOTAL
            # Prepend the frontend metric prefix if not already present
            if not requests_total_metric.startswith(
                prometheus_names.name_prefix.FRONTEND
            ):
                requests_total_metric = (
                    f"{prometheus_names.name_prefix.FRONTEND}_{requests_total_metric}"
                )
            raw_res = self.prom.custom_query(
                query=f"increase({requests_total_metric}[{interval}])"
            )
            metrics_containers = parse_frontend_metric_containers(raw_res)
            total_count = 0.0
            for container in metrics_containers:
                # Frontend lowercases model names for Prometheus labels so we need to do case-insensitive comparison
                if (
                    container.metric.model
                    and container.metric.model.lower() == model_name.lower()
                    and container.metric.dynamo_namespace == self.dynamo_namespace
                ):
                    total_count += container.value[1]
            return total_count
        except Exception as e:
            logger.error(f"Error getting avg request count: {e}")
            return 0

    def get_avg_input_sequence_tokens(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend_service.INPUT_SEQUENCE_TOKENS,
            interval,
            "avg input sequence tokens",
            model_name,
        )

    def get_avg_output_sequence_tokens(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend_service.OUTPUT_SEQUENCE_TOKENS,
            interval,
            "avg output sequence tokens",
            model_name,
        )


def parse_frontend_metric_containers(
    result: list[dict],
) -> list[FrontendMetricContainer]:
    metrics_containers: list[FrontendMetricContainer] = []
    for res in result:
        try:
            metrics_containers.append(FrontendMetricContainer.model_validate(res))
        except ValidationError as e:
            logger.error(f"Error parsing frontend metric container: {e}")
            continue
    return metrics_containers


# Metric names for per-worker load metrics (gauge-type, queried directly from router)
_WORKER_METRIC_NAMES = {
    "active_prefill_tokens": f"{prometheus_names.name_prefix.FRONTEND}_{prometheus_names.frontend_service.WORKER_ACTIVE_PREFILL_TOKENS}",
    "active_decode_blocks": f"{prometheus_names.name_prefix.FRONTEND}_{prometheus_names.frontend_service.WORKER_ACTIVE_DECODE_BLOCKS}",
    "last_ttft": f"{prometheus_names.name_prefix.FRONTEND}_{prometheus_names.frontend_service.WORKER_LAST_TIME_TO_FIRST_TOKEN_SECONDS}",
    "last_isl": f"{prometheus_names.name_prefix.FRONTEND}_{prometheus_names.frontend_service.WORKER_LAST_INPUT_SEQUENCE_TOKENS}",
    "last_itl": f"{prometheus_names.name_prefix.FRONTEND}_{prometheus_names.frontend_service.WORKER_LAST_INTER_TOKEN_LATENCY_SECONDS}",
}


class DirectRouterMetricsClient:
    """Query router's /metrics endpoint directly for real-time per-worker metrics.

    Runs a continuous background sampling loop that collects metrics at
    evenly-spaced intervals (interval / num_samples). At decision time,
    the load-based loop reads the buffer via get_recent_and_averaged_metrics().
    """

    def __init__(self, router_metrics_url: str, dynamo_namespace: str):
        self.router_metrics_url = router_metrics_url
        self.dynamo_namespace = dynamo_namespace
        self._sample_buffer: list[dict[str, dict[str, dict[str, float]]]] = []
        self._num_samples: int = 10

    def _parse_prometheus_text(
        self, text: str
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Parse Prometheus text exposition format and extract per-worker metrics.

        Uses prometheus_client.parser to parse the text exposition format.
        Groups results by worker_type label (prefill/decode) so callers
        can access only the workers they care about.

        Args:
            text: Raw Prometheus text from /metrics endpoint

        Returns:
            {"prefill": {worker_id: {metric: float, ...}},
             "decode":  {worker_id: {metric: float, ...}}}
        """
        target_metrics = set(_WORKER_METRIC_NAMES.values())
        reverse_map = {v: k for k, v in _WORKER_METRIC_NAMES.items()}
        result: dict[str, dict[str, dict[str, float]]] = {}

        for family in text_string_to_metric_families(text):
            if family.name not in target_metrics:
                continue

            field_name = reverse_map[family.name]

            for sample in family.samples:
                labels = sample.labels
                worker_type = labels.get("worker_type", "unknown")
                worker_id = labels.get("worker_id", "unknown")
                value = sample.value

                if worker_type not in result:
                    result[worker_type] = {}
                if worker_id not in result[worker_type]:
                    result[worker_type][worker_id] = {}
                result[worker_type][worker_id][field_name] = value

        return result

    async def _fetch_and_parse(self) -> dict[str, dict[str, dict[str, float]]]:
        """Fetch /metrics from router and parse into per-worker metrics."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.router_metrics_url, timeout=aiohttp.ClientTimeout(total=2)
                ) as response:
                    text = await response.text()
            return self._parse_prometheus_text(text)
        except Exception as e:
            logger.warning(f"Failed to fetch router metrics: {e}")
            return {}

    async def run_sampling_loop(self, num_samples: int, interval: float) -> None:
        """Background coroutine: continuously sample at evenly-spaced intervals.

        Runs alongside the load-based loop via asyncio.gather().
        sample_interval = interval / num_samples (e.g., 5s / 10 = 0.5s)
        Keeps only the last num_samples in the buffer (rolling window).
        """
        self._num_samples = num_samples
        sample_interval = interval / num_samples
        while True:
            metrics = await self._fetch_and_parse()
            if metrics:
                self._sample_buffer.append(metrics)
                if len(self._sample_buffer) > num_samples:
                    self._sample_buffer.pop(0)
            await asyncio.sleep(sample_interval)

    def get_recent_and_averaged_metrics(
        self, worker_type: str
    ) -> typing.Optional[
        tuple[
            dict[str, dict[str, float]],
            dict[str, dict[str, float]],
            dict[str, float],
        ]
    ]:
        """Return recent, per-worker time-averaged, and cluster-averaged metrics.

        Called by the load-based loop at decision time. Non-blocking.

        Args:
            worker_type: "prefill" or "decode" — only workers matching
                         the worker_type label are included.

        Returns:
            A tuple of (recent, per_worker_averaged, cluster_averaged):
            - recent:              {worker_id: {metric: float}} from the latest sample
            - per_worker_averaged: {worker_id: {metric: float}} averaged over time per worker
            - cluster_averaged:    {metric: float} averaged over all samples and all workers
            Returns None if the sample buffer is empty.
        """
        if not self._sample_buffer:
            return None

        # --- Recent: last sample only ---
        latest_sample = self._sample_buffer[-1]
        recent: dict[str, dict[str, float]] = {}
        for worker_id, metrics in latest_sample.get(worker_type, {}).items():
            recent[worker_id] = dict(metrics)

        # --- Per-worker averaged: across time, grouped by worker_id ---
        pw_sums: dict[str, dict[str, float]] = {}
        pw_counts: dict[str, dict[str, int]] = {}

        for sample in self._sample_buffer:
            typed_workers = sample.get(worker_type, {})
            for worker_id, metrics in typed_workers.items():
                if worker_id not in pw_sums:
                    pw_sums[worker_id] = {}
                    pw_counts[worker_id] = {}
                for metric_name, value in metrics.items():
                    pw_sums[worker_id][metric_name] = (
                        pw_sums[worker_id].get(metric_name, 0.0) + value
                    )
                    pw_counts[worker_id][metric_name] = (
                        pw_counts[worker_id].get(metric_name, 0) + 1
                    )

        if not pw_sums and not recent:
            return None

        per_worker_averaged: dict[str, dict[str, float]] = {}
        for worker_id in pw_sums:
            per_worker_averaged[worker_id] = {}
            for metric_name in pw_sums[worker_id]:
                per_worker_averaged[worker_id][metric_name] = (
                    pw_sums[worker_id][metric_name] / pw_counts[worker_id][metric_name]
                )

        # --- Cluster averaged: across time AND worker_id ---
        cluster_sums: dict[str, float] = {}
        cluster_counts: dict[str, int] = {}
        for worker_id in pw_sums:
            for metric_name in pw_sums[worker_id]:
                cluster_sums[metric_name] = (
                    cluster_sums.get(metric_name, 0.0) + pw_sums[worker_id][metric_name]
                )
                cluster_counts[metric_name] = (
                    cluster_counts.get(metric_name, 0)
                    + pw_counts[worker_id][metric_name]
                )

        cluster_averaged: dict[str, float] = {}
        for metric_name in cluster_sums:
            cluster_averaged[metric_name] = (
                cluster_sums[metric_name] / cluster_counts[metric_name]
            )

        return recent, per_worker_averaged, cluster_averaged
