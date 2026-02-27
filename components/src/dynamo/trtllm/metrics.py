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

"""
Additional Prometheus metrics for dynamo-trtllm beyond what the engine provides.

The TRT-LLM engine MetricsCollector already provides 5 core metrics:
  request_success_total, e2e_request_latency_seconds,
  time_to_first_token_seconds, inter_token_latency_seconds,
  request_queue_time_seconds

The Rust frontend (metrics.rs) provides token counters:
  input_tokens_total, output_tokens_total, cached_tokens

This module adds metrics that have no engine/runtime/frontend equivalent:
  - Request types (image, structured output)
  - KV transfer metrics (success/failure counters)
  - Abort tracking
"""

import logging

from prometheus_client import Counter

logger = logging.getLogger(__name__)


class AdditionalMetricsCollector:
    """
    Additional Prometheus metrics for dynamo-trtllm.

    Only creates metrics that have no engine/runtime/frontend equivalent.
    Metrics are registered in the default prometheus_client.REGISTRY.

    Args:
        labels: Dict with keys like model_name, disaggregation_mode, engine_type.
    """

    def __init__(self, labels: dict):
        self._labelnames = list(labels.keys())
        self._labelvalues = list(labels.values())

        # --- Abort tracking ---
        self.num_aborted_requests = Counter(
            "num_aborted_requests_total",
            "Total number of aborted/cancelled requests",
            labelnames=self._labelnames,
        )

        # --- Request type counters ---
        self.request_type_image = Counter(
            "request_type_image_total",
            "Total number of requests containing image content",
            labelnames=self._labelnames,
        )
        self.request_type_structured_output = Counter(
            "request_type_structured_output_total",
            "Total number of requests using guided/structured decoding",
            labelnames=self._labelnames,
        )

        # --- KV cache transfer metrics ---
        self.kv_transfer_success = Counter(
            "kv_transfer_success_total",
            "Total number of successful KV cache transfers",
            labelnames=self._labelnames,
        )
        self.kv_transfer_failure = Counter(
            "kv_transfer_failure_total",
            "Total number of failed KV cache transfers",
            labelnames=self._labelnames,
        )

        logger.info("AdditionalMetricsCollector initialized")

    # --- Request helpers ---

    def record_request_abort(self):
        """Increment aborted requests counter."""
        self.num_aborted_requests.labels(*self._labelvalues).inc()

    # --- Request type tracking ---

    def record_request_type_image(self):
        """Increment the image request type counter."""
        self.request_type_image.labels(*self._labelvalues).inc()

    def record_request_type_structured_output(self):
        """Increment the structured output request type counter."""
        self.request_type_structured_output.labels(*self._labelvalues).inc()

    # --- KV transfer ---

    def record_kv_transfer_success(self):
        """Increment the KV transfer success counter."""
        self.kv_transfer_success.labels(*self._labelvalues).inc()

    def record_kv_transfer_failure(self):
        """Increment the KV transfer failure counter."""
        self.kv_transfer_failure.labels(*self._labelvalues).inc()


# Backwards compatibility alias
UnifiedMetricsCollector = AdditionalMetricsCollector
