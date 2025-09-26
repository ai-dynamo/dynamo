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

from prometheus_api_client import PrometheusConnect

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class PrometheusAPIClient:
    def __init__(self, url: str):
        self.prom = PrometheusConnect(url=url, disable_ssl=True)

    def _get_average_metric(
        self, metric_name: str, interval: str, operation_name: str
    ) -> float:
        """
        Helper method to get average metrics using the pattern:
        increase(metric_sum[interval])/increase(metric_count[interval])

        Args:
            metric_name: Base metric name (e.g., 'inter_token_latency_seconds')
            interval: Time interval for the query (e.g., '60s')
            operation_name: Human-readable name for error logging

        Returns:
            Average metric value or 0 if no data/error
        """
        try:
            # TODO: use prometheus_names.rs
            full_metric_name = f"dynamo_frontend_{metric_name}"
            query = f"increase({full_metric_name}_sum[{interval}])/increase({full_metric_name}_count[{interval}])"
            result = self.prom.custom_query(query=query)
            if not result:
                # No data available yet (no requests made) - return 0 silently
                return 0
            return float(result[0]["value"][1])
        except Exception as e:
            logger.error(f"Error getting {operation_name}: {e}")
            return 0

    def get_avg_inter_token_latency(self, interval: str):
        return self._get_average_metric(
            "inter_token_latency_seconds",
            interval,
            "avg inter token latency",
        )

    def get_avg_time_to_first_token(self, interval: str):
        return self._get_average_metric(
            "time_to_first_token_seconds",
            interval,
            "avg time to first token",
        )

    def get_avg_request_duration(self, interval: str):
        return self._get_average_metric(
            "request_duration_seconds",
            interval,
            "avg request duration",
        )

    def get_avg_request_count(self, interval: str):
        # This function follows a different query pattern than the other metrics
        try:
            raw_res = self.prom.custom_query(
                query=f"increase(dynamo_frontend_requests_total[{interval}])"
            )
            total_count = 0.0
            for res in raw_res:
                # count all success/failed and stream/non-stream requests
                total_count += float(res["value"][1])
            return total_count
        except Exception as e:
            logger.error(f"Error getting avg request count: {e}")
            return 0

    def get_avg_input_sequence_tokens(self, interval: str):
        return self._get_average_metric(
            "input_sequence_tokens",
            interval,
            "avg input sequence tokens",
        )

    def get_avg_output_sequence_tokens(self, interval: str):
        return self._get_average_metric(
            "output_sequence_tokens",
            interval,
            "avg output sequence tokens",
        )
