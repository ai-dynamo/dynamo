# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Python type stubs for Prometheus metric name constants

⚠️  **CRITICAL: SYNC WITH RUST SOURCE** ⚠️
This file must stay in sync with:
- Source: `lib/runtime/src/metrics/prometheus_names.rs`
- Bindings: `lib/bindings/python/rust/prometheus_names.rs`

When the Rust source is modified, update all three files immediately.
"""

class PrometheusNames:
    """
    Main container for all Prometheus metric name constants
    """

    @property
    def frontend(self) -> FrontendService:
        """
        Frontend service metrics
        """
        ...

    @property
    def work_handler(self) -> WorkHandler:
        """
        Work handler metrics
        """
        ...

    @property
    def kvstats(self) -> KvStatsMetrics:
        """
        KV stats metrics
        """
        ...

class FrontendService:
    """
    Frontend service metrics (LLM HTTP service)
    These methods return the full metric names with the "dynamo_frontend_" prefix
    """

    @property
    def requests_total(self) -> str:
        """
        Total number of LLM requests processed
        """
        ...

    @property
    def queued_requests(self) -> str:
        """
        Number of requests waiting in HTTP queue before receiving the first response
        """
        ...

    @property
    def inflight_requests(self) -> str:
        """
        Number of inflight requests going to the engine (vLLM, SGLang, ...)
        """
        ...

    @property
    def request_duration_seconds(self) -> str:
        """
        Duration of LLM requests
        """
        ...

    @property
    def input_sequence_tokens(self) -> str:
        """
        Input sequence length in tokens
        """
        ...

    @property
    def output_sequence_tokens(self) -> str:
        """
        Output sequence length in tokens
        """
        ...

    @property
    def time_to_first_token_seconds(self) -> str:
        """
        Time to first token in seconds
        """
        ...

    @property
    def inter_token_latency_seconds(self) -> str:
        """
        Inter-token latency in seconds
        """
        ...

    @property
    def disconnected_clients(self) -> str:
        """
        Number of disconnected clients
        """
        ...

    @property
    def model_total_kv_blocks(self) -> str:
        """
        Model total KV blocks
        """
        ...

    @property
    def model_max_num_seqs(self) -> str:
        """
        Model max number of sequences
        """
        ...

    @property
    def model_max_num_batched_tokens(self) -> str:
        """
        Model max number of batched tokens
        """
        ...

    @property
    def model_context_length(self) -> str:
        """
        Model context length
        """
        ...

    @property
    def model_kv_cache_block_size(self) -> str:
        """
        Model KV cache block size
        """
        ...

    @property
    def model_migration_limit(self) -> str:
        """
        Model migration limit
        """
        ...

class WorkHandler:
    """
    Work handler metrics (component request processing)
    These methods return the full metric names with the "dynamo_component_" prefix
    """

    @property
    def requests_total(self) -> str:
        """
        Total number of requests processed by work handler
        """
        ...

    @property
    def request_bytes_total(self) -> str:
        """
        Total number of bytes received in requests by work handler
        """
        ...

    @property
    def response_bytes_total(self) -> str:
        """
        Total number of bytes sent in responses by work handler
        """
        ...

    @property
    def inflight_requests(self) -> str:
        """
        Number of requests currently being processed by work handler
        """
        ...

    @property
    def request_duration_seconds(self) -> str:
        """
        Time spent processing requests by work handler (histogram)
        """
        ...

    @property
    def errors_total(self) -> str:
        """
        Total number of errors in work handler processing
        """
        ...

class KvStatsMetrics:
    """
    KV stats metrics (KV cache statistics)
    These methods return the metric names with the "kvstats_" prefix
    """

    @property
    def active_blocks(self) -> str:
        """
        Number of active KV cache blocks currently in use
        """
        ...

    @property
    def total_blocks(self) -> str:
        """
        Total number of KV cache blocks available
        """
        ...

    @property
    def gpu_cache_usage_percent(self) -> str:
        """
        GPU cache usage as a percentage (0.0-1.0)
        """
        ...

    @property
    def gpu_prefix_cache_hit_rate(self) -> str:
        """
        GPU prefix cache hit rate as a percentage (0.0-1.0)
        """
        ...

# Module-level singleton instance for convenient access
prometheus_names: PrometheusNames


