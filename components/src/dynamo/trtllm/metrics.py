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
  - KV transfer metrics (speed, latency, bytes, success/failure)
  - Config info (model, parallel, detailed, cache, engine startup)
  - Abort tracking
"""

import logging

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

KV_TRANSFER_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 5.0,
)


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
        self.kv_transfer_speed = Gauge(
            "kv_transfer_speed_gb_s",
            "KV cache transfer speed in GB/s",
            labelnames=self._labelnames,
        )
        self.kv_transfer_latency = Histogram(
            "kv_transfer_latency_seconds",
            "KV cache transfer duration in seconds",
            labelnames=self._labelnames,
            buckets=KV_TRANSFER_BUCKETS,
        )
        self.kv_transfer_bytes = Counter(
            "kv_transfer_bytes_total",
            "Total bytes transferred for KV cache",
            labelnames=self._labelnames,
        )
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

        # --- Config info metrics (set once at startup) ---
        self.model_config_info = Gauge(
            "model_config_info",
            "Model configuration info (set to 1.0; labels carry config details)",
            labelnames=["model", "served_model_name", "dtype", "gpu_type"],
        )
        self.parallel_config_info = Gauge(
            "parallel_config_info",
            "Parallelism configuration info (set to 1.0; labels carry config details)",
            labelnames=["tensor_parallel_size", "pipeline_parallel_size", "gpu_count"],
        )
        self.engine_startup_time = Gauge(
            "engine_startup_time",
            "Engine initialization time in seconds",
            labelnames=self._labelnames,
        )
        self.detailed_config_info = Gauge(
            "detailed_config_info",
            "Detailed engine configuration (set to 1.0; labels carry config details)",
            labelnames=[
                "max_batch_size", "max_num_tokens", "max_seq_len",
                "kv_block_size", "free_gpu_memory_fraction",
                "disaggregation_mode", "modality",
            ],
        )
        self.cache_config_info = Gauge(
            "cache_config_info",
            "KV cache configuration (set to 1.0; labels carry config details)",
            labelnames=["free_gpu_memory_fraction", "kv_block_size"],
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

    # --- Config info ---

    def set_model_config(
        self,
        model: str,
        served_model_name: str,
        dtype: str = "auto",
        gpu_type: str = "unknown",
    ):
        """Set one-time model configuration info gauge."""
        self.model_config_info.labels(
            model=model,
            served_model_name=served_model_name,
            dtype=dtype,
            gpu_type=gpu_type,
        ).set(1.0)

    def set_parallel_config(
        self,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        gpu_count: int,
    ):
        """Set one-time parallelism configuration info gauge."""
        self.parallel_config_info.labels(
            tensor_parallel_size=str(tensor_parallel_size),
            pipeline_parallel_size=str(pipeline_parallel_size),
            gpu_count=str(gpu_count),
        ).set(1.0)

    def set_engine_startup_time(self, seconds: float):
        """Record engine initialization time.

        Note: Dynamo also tracks engine load time internally (engine.py).
        This metric is part of the unified metrics spec for cross-backend
        parity with vLLM/SGLang, measured from before get_llm_engine to
        after the engine context is entered.
        """
        self.engine_startup_time.labels(*self._labelvalues).set(seconds)

    def set_detailed_config(
        self,
        max_batch_size: int,
        max_num_tokens: int,
        max_seq_len: int,
        kv_block_size: int,
        free_gpu_memory_fraction: float,
        disaggregation_mode: str,
        modality: str,
    ):
        """Set one-time detailed engine configuration info gauge."""
        self.detailed_config_info.labels(
            max_batch_size=str(max_batch_size),
            max_num_tokens=str(max_num_tokens),
            max_seq_len=str(max_seq_len),
            kv_block_size=str(kv_block_size),
            free_gpu_memory_fraction=str(free_gpu_memory_fraction),
            disaggregation_mode=str(disaggregation_mode),
            modality=str(modality),
        ).set(1.0)

    def set_cache_config(
        self,
        free_gpu_memory_fraction: float,
        kv_block_size: int,
    ):
        """Set one-time KV cache configuration info gauge."""
        self.cache_config_info.labels(
            free_gpu_memory_fraction=str(free_gpu_memory_fraction),
            kv_block_size=str(kv_block_size),
        ).set(1.0)


# Backwards compatibility alias
UnifiedMetricsCollector = AdditionalMetricsCollector
