# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from prometheus_api_client import PrometheusConnect
from pydantic import BaseModel, ValidationError

from dynamo._core import prometheus_names
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


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
                if (
                    container.metric.model == model_name
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
            prometheus_names.frontend.inter_token_latency_seconds,
            interval,
            "avg inter token latency",
            model_name,
        )

    def get_avg_time_to_first_token(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend.time_to_first_token_seconds,
            interval,
            "avg time to first token",
            model_name,
        )

    def get_avg_request_duration(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend.request_duration_seconds,
            interval,
            "avg request duration",
            model_name,
        )

    def get_avg_request_count(self, interval: str, model_name: str):
        # This function follows a different query pattern than the other metrics
        try:
            requests_total_metric = prometheus_names.frontend.requests_total
            raw_res = self.prom.custom_query(
                query=f"increase({requests_total_metric}[{interval}])"
            )
            metrics_containers = parse_frontend_metric_containers(raw_res)
            total_count = 0.0
            for container in metrics_containers:
                if (
                    container.metric.model == model_name
                    and container.metric.dynamo_namespace == self.dynamo_namespace
                ):
                    total_count += container.value[1]
            return total_count
        except Exception as e:
            logger.error(f"Error getting avg request count: {e}")
            return 0

    def get_avg_input_sequence_tokens(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend.input_sequence_tokens,
            interval,
            "avg input sequence tokens",
            model_name,
        )

    def get_avg_output_sequence_tokens(self, interval: str, model_name: str):
        return self._get_average_metric(
            prometheus_names.frontend.output_sequence_tokens,
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
