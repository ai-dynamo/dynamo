# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Optional

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import TrafficObservation
from dynamo.planner.environment.interface import (
    DeploymentStateSource,
    RuntimeNamespaceSource,
)
from dynamo.planner.environment.metrics_provider.interface import TrafficMetricsProvider
from dynamo.planner.monitoring.traffic_metrics import Metrics, PrometheusAPIClient

logger = logging.getLogger(__name__)


class PrometheusTrafficProvider(TrafficMetricsProvider):
    def __init__(
        self,
        *,
        config: PlannerConfig,
        state_source: DeploymentStateSource,
        metrics_state: Metrics,
        namespace_source: Optional[RuntimeNamespaceSource] = None,
    ) -> None:
        self.config = config
        self.state_source = state_source
        self.metrics_state = metrics_state
        self.namespace_source = namespace_source
        self.prometheus_traffic_client = PrometheusAPIClient(
            config.metric_pulling_prometheus_endpoint,
            config.namespace,
            metrics_source=config.throughput_metrics_source,
            bearer_token=config.metric_pulling_prometheus_token,
            bearer_token_file=config.metric_pulling_prometheus_token_file,
            ssl_verify=config.metric_pulling_prometheus_ssl_verify,
            extra_query_params=config.metric_pulling_prometheus_extra_query_params,
            ca_bundle=config.metric_pulling_prometheus_ca_bundle,
        )
        if config.throughput_metrics_source == "router":
            self.prometheus_traffic_client.warn_if_router_not_scraped()

    async def collect_traffic(self) -> Optional[TrafficObservation]:
        model_name = self._model_name()
        if model_name is None:
            logger.info("Model name is not available, skipping traffic collection")
            return None

        interval_str = f"{self.config.throughput_adjustment_interval_seconds}s"
        m = self.metrics_state
        ttft = self.prometheus_traffic_client.get_avg_time_to_first_token(
            interval_str, model_name
        )
        m.ttft = ttft * 1000 if ttft is not None else None
        itl = self.prometheus_traffic_client.get_avg_inter_token_latency(
            interval_str, model_name
        )
        m.itl = itl * 1000 if itl is not None else None
        m.num_req = self.prometheus_traffic_client.get_avg_request_count(
            interval_str, model_name
        )
        m.request_duration = self.prometheus_traffic_client.get_avg_request_duration(
            interval_str, model_name
        )
        m.isl = self.prometheus_traffic_client.get_avg_input_sequence_tokens(
            interval_str, model_name
        )
        m.osl = self.prometheus_traffic_client.get_avg_output_sequence_tokens(
            interval_str, model_name
        )
        m.kv_hit_rate = self.prometheus_traffic_client.get_avg_kv_hit_rate(
            interval_str, model_name
        )
        m.accept_length = self.collect_accept_length(interval_str)

        normalized_idle_metrics = m.normalize_idle_nans()
        if normalized_idle_metrics:
            logger.info(
                "Zero traffic observed; treating undefined averages as 0: %s",
                ", ".join(normalized_idle_metrics),
            )

        if not m.is_valid():
            logger.info("Metrics contain None or NaN values, skipping")
            return None

        hit_rate_str = f"{m.kv_hit_rate:.3f}" if m.kv_hit_rate is not None else "n/a"
        accept_length_str = (
            f"{m.accept_length:.3f}" if m.accept_length is not None else "n/a"
        )
        logger.info(
            "Observed num_req: %.2f isl: %.2f osl: %.2f kv_hit_rate: %s "
            "accept_length: %s",
            m.num_req,
            m.isl,
            m.osl,
            hit_rate_str,
            accept_length_str,
        )

        return TrafficObservation(
            duration_s=self.config.throughput_adjustment_interval_seconds,
            num_req=m.num_req,
            isl=m.isl,
            osl=m.osl,
            kv_hit_rate=m.kv_hit_rate,
            accept_length=m.accept_length,
        )

    def collect_accept_length(self, interval_str: str) -> Optional[float]:
        if self.config.mode not in ("disagg", "decode", "agg"):
            return None
        state = self.state_source.deployment_state()
        model_name = self._model_name()
        decode_info = state.decode.info
        if model_name is None or decode_info is None:
            return None
        if not decode_info.component_name or not decode_info.endpoint:
            return None
        return self.prometheus_traffic_client.get_avg_spec_decode_accept_length(
            interval_str,
            self.config.backend,
            decode_info.component_name,
            model_name,
            namespace=self._runtime_namespace(),
            endpoint_name=decode_info.endpoint,
        )

    async def collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        model_name = self._model_name()
        if model_name is None or duration_s <= 0:
            return None

        interval_str = f"{int(duration_s)}s"
        hit_rate = self.prometheus_traffic_client.get_avg_kv_hit_rate(
            interval_str, model_name
        )
        accept_length = self.collect_accept_length(interval_str)
        self.metrics_state.kv_hit_rate = hit_rate
        self.metrics_state.accept_length = accept_length
        hit_rate_str = f"{hit_rate:.3f}" if hit_rate is not None else "n/a"
        accept_length_str = (
            f"{accept_length:.3f}" if accept_length is not None else "n/a"
        )
        logger.info(
            "Observed kv_hit_rate over %s: %s; accept_length: %s",
            interval_str,
            hit_rate_str,
            accept_length_str,
        )
        return TrafficObservation(
            duration_s=duration_s,
            num_req=0.0,
            isl=0.0,
            osl=0.0,
            kv_hit_rate=hit_rate,
            accept_length=accept_length,
        )

    def _model_name(self) -> Optional[str]:
        state = self.state_source.deployment_state()
        if state.model_name:
            return state.model_name
        if state.decode.info is not None and state.decode.info.model_name:
            return state.decode.info.model_name
        if state.prefill.info is not None and state.prefill.info.model_name:
            return state.prefill.info.model_name
        return None

    def _runtime_namespace(self) -> str:
        if self.namespace_source is None:
            return self.config.namespace
        return self.namespace_source.runtime_namespace() or self.config.namespace
