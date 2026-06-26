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
import math
import typing
from dataclasses import dataclass
from typing import Dict, Optional

from prometheus_api_client import PrometheusConnect
from pydantic import BaseModel, ValidationError

from dynamo import prometheus_names
from dynamo.runtime.logging import configure_dynamo_logging


class _BearerTokenFileAuth:
    """Auth callable that re-reads a bearer token from disk on every request.

    Assigned to requests.Session.auth. Any callable that takes a PreparedRequest
    and returns it qualifies — no AuthBase subclass required.
    Useful for rotating tokens (Kubernetes projected ServiceAccount tokens).
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def __call__(self, request):  # type: ignore[override]
        with open(self._path) as f:
            token = f.read().strip()
        request.headers["Authorization"] = f"Bearer {token}"
        return request


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
    kv_hit_rate: Optional[float] = None
    accept_length: Optional[float] = None

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
    def __init__(
        self,
        url: str,
        dynamo_namespace: str,
        metrics_source: str = "frontend",
        bearer_token: Optional[str] = None,
        bearer_token_file: Optional[str] = None,
        ssl_verify: bool = False,
        extra_query_params: Optional[Dict[str, str]] = None,
        ca_bundle: Optional[str] = None,
    ):
        self.prom = PrometheusConnect(url=url, disable_ssl=not ssl_verify)
        if bearer_token:
            self.prom._session.headers["Authorization"] = f"Bearer {bearer_token}"
        if bearer_token_file:
            self.prom._session.auth = _BearerTokenFileAuth(bearer_token_file)
        if extra_query_params:
            self.prom._session.params = dict(extra_query_params)
        if ca_bundle:
            self.prom._session.verify = ca_bundle
        self.dynamo_namespace = dynamo_namespace
        self.metrics_source = metrics_source  # "frontend" | "router"

    def _frontend_metric_name(self, metric_name: str) -> str:
        if metric_name.startswith(prometheus_names.name_prefix.FRONTEND):
            return metric_name
        return f"{prometheus_names.name_prefix.FRONTEND}_{metric_name}"

    def _sum_frontend_metric(self, result, model_name: str) -> Optional[float]:
        if not result:
            return None

        metrics_containers = parse_frontend_metric_containers(result)
        total = 0.0
        matched = False
        for container in metrics_containers:
            # Frontend lowercases model names in Prometheus labels.
            if (
                container.metric.model
                and container.metric.model.lower() == model_name.lower()
                and container.metric.dynamo_namespace == self.dynamo_namespace
                and not math.isnan(container.value[1])
            ):
                matched = True
                total += container.value[1]
        return total if matched else None

    def _get_average_metric(
        self,
        full_metric_name: str,
        interval: str,
        operation_name: str,
        model_name: Optional[str] = None,
    ) -> float:
        """Query average histogram metric.

        When model_name is None (router source): queries aggregate metrics via
        sum(increase(metric_sum[interval])) / sum(increase(metric_count[interval])),
        filtered by dynamo_namespace. DYN_NAMESPACE uses dashes but Prometheus labels
        use underscores, so dashes are normalized before building the PromQL filter.

        When model_name is provided (frontend source): queries per-model metrics
        via increase(metric_sum)/increase(metric_count), filtered by model and
        dynamo_namespace labels. The dynamo_frontend_ prefix is prepended
        automatically if absent.

        Returns:
            Average metric value, or 0 if no data/error.
        """
        try:
            if model_name is None:
                # Router aggregate path: filter by dynamo_namespace so each pool
                # planner only reads its own LocalRouter's metrics.
                # dynamo_component_router_* metrics are registered via MetricsHierarchy
                # which auto-injects dynamo_namespace with underscores (e.g.
                # "darfeen_dynamo_cloud_gp_prefill_1"). DYN_NAMESPACE uses dashes, so
                # normalize before building the PromQL filter.
                ns = self.dynamo_namespace.replace("-", "_")
                ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
                query = (
                    f"sum(increase({full_metric_name}_sum{{{ns_filter}}}[{interval}])) / "
                    f"sum(increase({full_metric_name}_count{{{ns_filter}}}[{interval}]))"
                )
                result = self.prom.custom_query(query=query)
                if not result:
                    logger.warning(
                        f"No prometheus metric data available for {full_metric_name}, use 0 instead"
                    )
                    return 0
                value = float(result[0]["value"][1])
                return 0 if math.isnan(value) else value
            else:
                # Frontend per-model path: filter by model and dynamo_namespace labels
                if not full_metric_name.startswith(
                    prometheus_names.name_prefix.FRONTEND
                ):
                    full_metric_name = (
                        f"{prometheus_names.name_prefix.FRONTEND}_{full_metric_name}"
                    )
                query = f"increase({full_metric_name}_sum[{interval}])/increase({full_metric_name}_count[{interval}])"
                result = self.prom.custom_query(query=query)
                if not result:
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
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.INTER_TOKEN_LATENCY_SECONDS}",
                interval,
                "avg inter token latency",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.INTER_TOKEN_LATENCY_SECONDS,
            interval,
            "avg inter token latency",
            model_name,
        )

    def get_avg_time_to_first_token(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.TIME_TO_FIRST_TOKEN_SECONDS}",
                interval,
                "avg time to first token",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.TIME_TO_FIRST_TOKEN_SECONDS,
            interval,
            "avg time to first token",
            model_name,
        )

    def get_avg_request_duration(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            # TODO: Replace work_handler.REQUEST_DURATION_SECONDS with
            #       prometheus_names.router.REQUEST_DURATION_SECONDS once
            #       RouterRequestMetrics in lib/llm/src/kv_router/metrics.rs
            #       registers dynamo_component_router_request_duration_seconds.
            #       Until then this queries a non-existent metric and returns 0,
            #       which causes throughput planning to see
            #       concurrency=0 (under-estimated), inflating replica recommendations.
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.work_handler.REQUEST_DURATION_SECONDS}",
                interval,
                "avg request duration",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.REQUEST_DURATION_SECONDS,
            interval,
            "avg request duration",
            model_name,
        )

    def get_avg_request_count(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            try:
                router_req_total = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.REQUESTS_TOTAL}"
                ns = self.dynamo_namespace.replace("-", "_")
                ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
                query = f"sum(increase({router_req_total}{{{ns_filter}}}[{interval}]))"
                result = self.prom.custom_query(query=query)
                if not result:
                    logger.warning(
                        f"No prometheus metric data available for "
                        f"{router_req_total}, use 0 instead"
                    )
                    return 0
                value = float(result[0]["value"][1])
                return 0 if math.isnan(value) else value
            except Exception as e:
                logger.error(f"Error getting avg request count: {e}")
                return 0
        # This function follows a different query pattern than the other metrics:
        # use frontend-started requests so throughput planning sees offered load,
        # not only completed responses.
        try:
            requests_started_metric = self._frontend_metric_name(
                prometheus_names.frontend_service.REQUESTS_STARTED_TOTAL
            )
            started_res = self.prom.custom_query(
                query=f"increase({requests_started_metric}[{interval}])"
            )
            started_count = self._sum_frontend_metric(started_res, model_name)
            if started_count is not None:
                return started_count

            logger.warning(
                f"No prometheus metric data available for {requests_started_metric} "
                f"with model {model_name} and dynamo namespace "
                f"{self.dynamo_namespace}; falling back to completed request count"
            )

            requests_total_metric = self._frontend_metric_name(
                prometheus_names.frontend_service.REQUESTS_TOTAL
            )
            completed_res = self.prom.custom_query(
                query=f"increase({requests_total_metric}[{interval}])"
            )
            completed_count = self._sum_frontend_metric(completed_res, model_name)
            return completed_count or 0
        except Exception as e:
            logger.error(f"Error getting avg request count: {e}")
            return 0

    def get_avg_input_sequence_tokens(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.INPUT_SEQUENCE_TOKENS}",
                interval,
                "avg input sequence tokens",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.INPUT_SEQUENCE_TOKENS,
            interval,
            "avg input sequence tokens",
            model_name,
        )

    def get_avg_output_sequence_tokens(self, interval: str, model_name: str):
        if self.metrics_source == "router":
            return self._get_average_metric(
                f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.OUTPUT_SEQUENCE_TOKENS}",
                interval,
                "avg output sequence tokens",
            )
        return self._get_average_metric(
            prometheus_names.frontend_service.OUTPUT_SEQUENCE_TOKENS,
            interval,
            "avg output sequence tokens",
            model_name,
        )

    def get_avg_kv_hit_rate(self, interval: str, model_name: str) -> Optional[float]:
        """Average predicted KV cache hit rate (0.0-1.0) from the router.

        Only available when metrics_source == "router" (the histogram lives on
        the LocalRouter component). In disagg deployments the scrape is
        namespace-filtered, so if the planner's ``dynamo_namespace`` matches
        the prefill pool, the returned value pools only prefill-router
        observations.

        Returns ``None`` (not ``0.0``) on missing data — Prometheus scrape
        gaps must not be confused with a real "no reuse" signal: the state
        machine treats a real ``0.0`` as a valid observation and would
        otherwise drag the predictor / sticky value down toward zero on
        every scrape failure. The caller's ``_clamp_kv_hit_rate(None)``
        falls back to no-discount behavior, which is the safe choice.
        """
        if self.metrics_source != "router":
            return None
        full_metric_name = (
            f"{prometheus_names.name_prefix.COMPONENT}_"
            f"{prometheus_names.router.KV_HIT_RATE}"
        )
        try:
            ns = self.dynamo_namespace.replace("-", "_")
            ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
            query = (
                f"sum(increase({full_metric_name}_sum{{{ns_filter}}}[{interval}])) / "
                f"sum(increase({full_metric_name}_count{{{ns_filter}}}[{interval}]))"
            )
            result = self.prom.custom_query(query=query)
            if not result:
                logger.info(
                    f"No prometheus data for {full_metric_name}, returning None"
                )
                return None
            value = float(result[0]["value"][1])
            return None if math.isnan(value) else value
        except Exception as e:
            logger.warning(f"Error getting avg kv hit rate: {e}")
            return None

    @staticmethod
    def _quote_label_value(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def _engine_metric_filter(
        self,
        component_name: Optional[str],
        model_name: Optional[str],
        namespace: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> str:
        metric_namespace = namespace or self.dynamo_namespace
        metric_endpoint = endpoint_name or "generate"
        filters = [
            f'{prometheus_names.labels.NAMESPACE}="{self._quote_label_value(metric_namespace)}"',
            f'{prometheus_names.labels.ENDPOINT}="{self._quote_label_value(metric_endpoint)}"',
        ]
        if component_name:
            filters.append(
                f'{prometheus_names.labels.COMPONENT}="{self._quote_label_value(component_name)}"'
            )
        if model_name:
            filters.append(
                f'{prometheus_names.labels.MODEL}="{self._quote_label_value(model_name)}"'
            )
        return ",".join(filters)

    def _query_single_value(self, query: str, operation_name: str) -> Optional[float]:
        try:
            result = self.prom.custom_query(query=query)
            if not result:
                logger.info(f"No prometheus data for {operation_name}")
                return None
            value = float(result[0]["value"][1])
            return value if math.isfinite(value) else None
        except Exception as e:
            logger.warning(f"Error getting {operation_name}: {e}")
            return None

    def get_avg_spec_decode_accept_length(
        self,
        interval: str,
        backend: str,
        component_name: Optional[str],
        model_name: Optional[str],
        namespace: Optional[str] = None,
        endpoint_name: Optional[str] = None,
    ) -> Optional[float]:
        """Average spec-decode accept length from worker engine metrics.

        Returns tokens produced per decode forward, including the base token.
        Missing data returns ``None`` so callers can fall back to no discount.
        """
        selector = self._engine_metric_filter(
            component_name, model_name, namespace, endpoint_name
        )
        if backend == "vllm":
            accepted = (
                f"sum(rate(vllm:spec_decode_num_accepted_tokens_total"
                f"{{{selector}}}[{interval}]))"
            )
            drafts = (
                f"sum(rate(vllm:spec_decode_num_drafts_total"
                f"{{{selector}}}[{interval}]))"
            )
            return self._query_single_value(
                f"1 + ({accepted}) / ({drafts})",
                "vLLM spec decode accept length",
            )
        if backend == "sglang":
            return self._query_single_value(
                f"avg(avg_over_time(sglang:spec_accept_length"
                f"{{{selector}}}[{interval}]))",
                "SGLang spec decode accept length",
            )
        if backend == "trtllm":
            return self._query_single_value(
                f"avg(avg_over_time(trtllm_spec_decode_acceptance_length"
                f"{{{selector}}}[{interval}]))",
                "TRT-LLM spec decode accept length",
            )
        return None

    def warn_if_router_not_scraped(self) -> None:
        """Warn if Prometheus is not scraping any dynamo_component_router_* series.

        Called once at planner startup when throughput_metrics_source="router".
        Detects a missing or misconfigured PodMonitor early so the operator
        sees a clear warning rather than silent zero metrics.

        Uses absent() to check whether any dynamo_component_router_requests_total
        series exist for this namespace. MetricsHierarchy injects dynamo_namespace
        with underscores, so DYN_NAMESPACE dashes are normalized before the query.
        """
        try:
            metric = f"{prometheus_names.name_prefix.COMPONENT}_{prometheus_names.router.REQUESTS_TOTAL}"
            ns = self.dynamo_namespace.replace("-", "_")
            ns_filter = f'{prometheus_names.labels.NAMESPACE}="{ns}"'
            result = self.prom.custom_query(query=f"absent({metric}{{{ns_filter}}})")
            if result:
                logger.warning(
                    f"[throughput_metrics_source=router] No '{metric}' series found "
                    f"for namespace '{ns}' in Prometheus. "
                    "Router metrics will read as zero until scraping is working. "
                    "Check: (1) PodMonitor 'dynamo-router' is installed in the operator namespace, "
                    "(2) LocalRouter pods have DYN_SYSTEM_PORT=9090, "
                    "(3) pods have label nvidia.com/metrics-enabled=true."
                )
        except Exception as e:
            logger.warning(f"Could not check router scraping status: {e}")


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
