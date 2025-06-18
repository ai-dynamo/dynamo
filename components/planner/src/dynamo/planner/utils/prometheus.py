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

    def get_avg_inter_token_latency(self, interval: str):
        try:
            return float(
                self.prom.custom_query(
                    query=f"increase(nv_llm_http_service_inter_token_latency_seconds_sum[{interval}])/increase(nv_llm_http_service_inter_token_latency_seconds_count[{interval}])",
                )[0]["value"][1]
            )
        except Exception as e:
            logger.error(f"Error getting avg inter token latency: {e}")
            return 0

    def get_avg_time_to_first_token(self, interval: str):
        try:
            return float(
                self.prom.custom_query(
                    query=f"increase(nv_llm_http_service_time_to_first_token_seconds_sum[{interval}])/increase(nv_llm_http_service_time_to_first_token_seconds_count[{interval}])",
                )[0]["value"][1]
            )
        except Exception as e:
            logger.error(f"Error getting avg time to first token: {e}")
            return 0

    def get_avg_request_duration(self, interval: str):
        try:
            return float(
                self.prom.custom_query(
                    query=f"increase(nv_llm_http_service_request_duration_seconds_sum[{interval}])/increase(nv_llm_http_service_request_duration_seconds_count[{interval}])",
                )[0]["value"][1]
            )
        except Exception as e:
            logger.error(f"Error getting avg request duration: {e}")
            return 0

    def get_avg_request_count(self, interval: str):
        try:
            raw_res = self.prom.custom_query(
                query=f"increase(nv_llm_http_service_requests_total[{interval}])"
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
        try:
            return float(
                self.prom.custom_query(
                    query=f"increase(nv_llm_http_service_input_sequence_tokens_sum[{interval}])/increase(nv_llm_http_service_input_sequence_tokens_count[{interval}])",
                )[0]["value"][1]
            )
        except Exception as e:
            logger.error(f"Error getting avg input sequence tokens: {e}")
            return 0

    def get_avg_output_sequence_tokens(self, interval: str):
        try:
            return float(
                self.prom.custom_query(
                    query=f"increase(nv_llm_http_service_output_sequence_tokens_sum[{interval}])/increase(nv_llm_http_service_output_sequence_tokens_count[{interval}])",
                )[0]["value"][1]
            )
        except Exception as e:
            logger.error(f"Error getting avg output sequence tokens: {e}")
            return 0
