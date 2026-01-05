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

import logging
import typing
from enum import Enum

from prometheus_api_client import PrometheusConnect
from pydantic import BaseModel, ValidationError

from dynamo import prometheus_names
from dynamo.prometheus_names import (
    frontend_service as metric_names,  # Note that we are mapping from frontend metric names to VLLM
)
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class FrontendMetric(BaseModel):
    container: typing.Optional[str] = None
    dynamo_namespace: typing.Optional[str] = None
    endpoint: typing.Optional[str] = None
    instance: typing.Optional[str] = None
    job: typing.Optional[str] = None
    model: typing.Optional[str] = None  # Frontend uses this label
    model_name: typing.Optional[str] = None  # Backend (vLLM) uses this label
    namespace: typing.Optional[str] = None  # Kubernetes namespace
    pod: typing.Optional[str] = None  # Pod name (used for backend filtering)
    engine: typing.Optional[str] = None  # vLLM engine index


class FrontendMetricContainer(BaseModel):
    metric: FrontendMetric
    value: typing.Tuple[float, float]  # [timestamp, value]


class MetricSource(Enum):
    FRONTEND = "frontend"
    VLLM = "vllm"
    SGLANG = "sglang"  # not supported yet
    TRTLLM = "trtllm"  # not supported yet


METRIC_SOURCE_MAP = {  # sourced from prometheus_names.py
    MetricSource.VLLM: {
        metric_names.TIME_TO_FIRST_TOKEN_SECONDS: "vllm:time_to_first_token_seconds",  # histogram
        metric_names.INTER_TOKEN_LATENCY_SECONDS: "vllm:inter_token_latency_seconds",  # histogram
        metric_names.REQUEST_DURATION_SECONDS: "vllm:e2e_request_latency_seconds",  # histogram - vLLM's e2e latency
        metric_names.INPUT_SEQUENCE_TOKENS: "vllm:prompt_tokens_total",  # counter - total prompt tokens
        metric_names.OUTPUT_SEQUENCE_TOKENS: "vllm:generation_tokens_total",  # counter - total generation tokens
        metric_names.REQUESTS_TOTAL: "vllm:request_success_total",  # counter
    },
    MetricSource.FRONTEND: {
        metric_names.TIME_TO_FIRST_TOKEN_SECONDS: f"{prometheus_names.name_prefix.FRONTEND}_{metric_names.TIME_TO_FIRST_TOKEN_SECONDS}",
        metric_names.INTER_TOKEN_LATENCY_SECONDS: f"{prometheus_names.name_prefix.FRONTEND}_{metric_names.INTER_TOKEN_LATENCY_SECONDS}",
        metric_names.REQUEST_DURATION_SECONDS: f"{prometheus_names.name_prefix.FRONTEND}_{metric_names.REQUEST_DURATION_SECONDS}",
        metric_names.INPUT_SEQUENCE_TOKENS: f"{prometheus_names.name_prefix.FRONTEND}_{metric_names.INPUT_SEQUENCE_TOKENS}",
        metric_names.OUTPUT_SEQUENCE_TOKENS: f"{prometheus_names.name_prefix.FRONTEND}_{metric_names.OUTPUT_SEQUENCE_TOKENS}",
        metric_names.REQUESTS_TOTAL: f"{prometheus_names.name_prefix.FRONTEND}_{metric_names.REQUESTS_TOTAL}",
    },
}

METRIC_SOURCE_MODEL_ATTR = {
    MetricSource.VLLM: "model_name",
    MetricSource.FRONTEND: "model",
}


class PrometheusAPIClient:
    """
    Client for querying Dynamo metrics from Prometheus.

    Supports querying both frontend and backend metrics:
    - Frontend metrics: {prometheus_names.name_prefix.FRONTEND}_* (from Dynamo HTTP frontend)
    - Backend metrics: vllm:* (from vLLM engine workers)

    Usage:
        # Query frontend metrics (default)
        frontend_client = PrometheusAPIClient(url="http://prometheus:9090",
                                             dynamo_namespace="my-deployment")
        ttft = frontend_client.get_avg_time_to_first_token("60s", "llama-3-8b")

        # Query backend worker metrics
        backend_client = PrometheusAPIClient(url="http://prometheus:9090",
                                            dynamo_namespace="my-deployment",
                                            metric_source=MetricSource.VLLM)
        ttft = backend_client.get_avg_time_to_first_token("60s", "llama-3-8b")
    """

    def __init__(
        self,
        url: str,
        dynamo_namespace: str,
        metric_source: MetricSource = MetricSource.FRONTEND,
    ):
        """
        Initialize Prometheus API client.

        Args:
            url: Prometheus server URL
            dynamo_namespace: Dynamo namespace to filter metrics
            metric_source: Either MetricSource.FRONTEND or MetricSource.VLLM.
        """

        self.prom = PrometheusConnect(url=url, disable_ssl=True)
        self.dynamo_namespace = dynamo_namespace
        self.metric_source = metric_source
        self.model_attr = METRIC_SOURCE_MODEL_ATTR[self.metric_source]

    def _get_average_metric(
        self, full_metric_name: str, interval: str, operation_name: str, model_name: str
    ) -> float:
        """
        Helper method to get average metrics using the pattern:
        increase(metric_sum[interval])/increase(metric_count[interval])

        Args:
            full_metric_name: Full metric name (e.g., metric_names.INTER_TOKEN_LATENCY_SECONDS or metric_names.TIME_TO_FIRST_TOKEN_SECONDS)
            interval: Time interval for the query (e.g., '60s')
            operation_name: Human-readable name for error logging
            model_name: Model name to filter by

        Returns:
            Average metric value or 0 if no data/error
        """
        try:
            full_metric_name = METRIC_SOURCE_MAP[self.metric_source][full_metric_name]

            # Query sum and count separately
            sum_query = f"increase({full_metric_name}_sum[{interval}])"
            count_query = f"increase({full_metric_name}_count[{interval}])"

            sum_result = self.prom.custom_query(query=sum_query)
            count_result = self.prom.custom_query(query=count_query)

            if not sum_result or not count_result:
                # No data available yet (no requests made) - return 0 silently
                logger.warning(
                    f"No prometheus metric data available for {full_metric_name}, use 0 instead"
                )
                return 0

            sum_containers = parse_frontend_metric_containers(sum_result)
            count_containers = parse_frontend_metric_containers(count_result)

            # Sum up values for matching containers
            total_sum = 0.0
            total_count = 0.0

            for container in sum_containers:
                model_value = getattr(container.metric, self.model_attr, None)
                model_match = model_value and model_value.lower() == model_name.lower()
                namespace_match = (
                    container.metric.dynamo_namespace == self.dynamo_namespace
                )

                # Filter by model and namespace
                if model_match and namespace_match:
                    total_sum += container.value[1]

            for container in count_containers:
                model_value = getattr(container.metric, self.model_attr, None)
                model_match = model_value and model_value.lower() == model_name.lower()
                namespace_match = (
                    container.metric.dynamo_namespace == self.dynamo_namespace
                )

                # Filter by model and namespace
                if model_match and namespace_match:
                    total_count += container.value[1]

            if total_count == 0:
                logger.warning(
                    f"No prometheus metric data available for {full_metric_name} with model {model_name} and dynamo namespace {self.dynamo_namespace}, use 0 instead"
                )
                return 0

            return total_sum / total_count

        except Exception as e:
            logger.error(f"Error getting {operation_name}: {e}")
            return 0

    def _get_counter_average(
        self, counter_metric: str, interval: str, model_name: str, operation_name: str
    ) -> float:
        """
        Get average value from a counter metric by dividing total increase by request count increase.
        Used for vLLM token counters (prompt_tokens_total, generation_tokens_total).

        Formula: increase(counter_total[interval]) / increase(request_success_total[interval])
        """
        try:
            full_metric_name = METRIC_SOURCE_MAP[self.metric_source][counter_metric]
            requests_metric = METRIC_SOURCE_MAP[self.metric_source][
                metric_names.REQUESTS_TOTAL
            ]

            # Query both the counter and request count
            counter_query = f"increase({full_metric_name}[{interval}])"
            requests_query = f"increase({requests_metric}[{interval}])"

            counter_result = self.prom.custom_query(query=counter_query)
            requests_result = self.prom.custom_query(query=requests_query)

            if not counter_result or not requests_result:
                logger.warning(
                    f"No prometheus metric data available for {full_metric_name}, use 0 instead"
                )
                return 0

            counter_containers = parse_frontend_metric_containers(counter_result)
            requests_containers = parse_frontend_metric_containers(requests_result)

            # Sum up values for matching pods
            total_counter = 0.0
            total_requests = 0.0

            for container in counter_containers:
                model_value = getattr(container.metric, self.model_attr, None)
                if model_value and model_value.lower() == model_name.lower():
                    if container.metric.dynamo_namespace == self.dynamo_namespace:
                        total_counter += container.value[1]

            for container in requests_containers:
                model_value = getattr(container.metric, self.model_attr, None)
                if model_value and model_value.lower() == model_name.lower():
                    if container.metric.dynamo_namespace == self.dynamo_namespace:
                        total_requests += container.value[1]

            if total_requests == 0:
                logger.warning(
                    f"No requests for {operation_name} calculation, use 0 instead"
                )
                return 0

            average = total_counter / total_requests
            return average

        except Exception as e:
            logger.error(f"Error getting {operation_name}: {e}")
            return 0

    def get_avg_inter_token_latency(self, interval: str, model_name: str):
        return self._get_average_metric(
            metric_names.INTER_TOKEN_LATENCY_SECONDS,
            interval,
            "avg inter token latency",
            model_name,
        )

    def get_avg_time_to_first_token(self, interval: str, model_name: str):
        return self._get_average_metric(
            metric_names.TIME_TO_FIRST_TOKEN_SECONDS,
            interval,
            "avg time to first token",
            model_name,
        )

    def get_avg_request_duration(self, interval: str, model_name: str):
        return self._get_average_metric(
            metric_names.REQUEST_DURATION_SECONDS,
            interval,
            "avg request duration",
            model_name,
        )

    def get_avg_request_count(self, interval: str, model_name: str):
        """
        Get request count over the specified interval.

        For frontend: queries dynamo_frontend_requests_total
        For backend: queries vllm:request_success_total
        """
        try:
            requests_total_metric = METRIC_SOURCE_MAP[self.metric_source][
                metric_names.REQUESTS_TOTAL
            ]

            raw_res = self.prom.custom_query(
                query=f"increase({requests_total_metric}[{interval}])"
            )
            metrics_containers = parse_frontend_metric_containers(raw_res)
            total_count = 0.0
            for container in metrics_containers:
                model_value = getattr(container.metric, self.model_attr, None)
                model_match = model_value and model_value.lower() == model_name.lower()
                namespace_match = (
                    container.metric.dynamo_namespace == self.dynamo_namespace
                )

                # Filter by model and namespace
                if model_match and namespace_match:
                    total_count += container.value[1]
            return total_count
        except Exception as e:
            logger.error(f"Error getting avg request count: {e}")
            return 0

    def get_avg_input_sequence_tokens(self, interval: str, model_name: str):
        if self.metric_source == MetricSource.VLLM:
            # Backend uses prompt_tokens counter (not histogram)
            return self._get_counter_average(
                metric_names.INPUT_SEQUENCE_TOKENS,
                interval,
                model_name,
                "input_sequence_tokens",
            )
        return self._get_average_metric(
            metric_names.INPUT_SEQUENCE_TOKENS,
            interval,
            "avg input sequence tokens",
            model_name,
        )

    def get_avg_output_sequence_tokens(self, interval: str, model_name: str):
        if self.metric_source == MetricSource.VLLM:
            # Backend uses generation_tokens counter (not histogram)
            return self._get_counter_average(
                metric_names.OUTPUT_SEQUENCE_TOKENS,
                interval,
                model_name,
                "output_sequence_tokens",
            )
        return self._get_average_metric(
            metric_names.OUTPUT_SEQUENCE_TOKENS,
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
