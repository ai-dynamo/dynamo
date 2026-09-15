# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External observations and actuation for batch-aware planning.

This module defines narrow source and actuator primitives. Native lifecycle
ownership lives in ``batch_runtime`` so the policy core remains independent of
HTTP, OpenMetrics, Prometheus, and Redis clients.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Protocol, Union
from urllib.parse import quote

import aiohttp
from dynamo.planner.core.types import (
    BatchDispatcherFeedback,
    BatchJobDemand,
    BatchSchedulingObservation,
    PoolTrafficDemand,
)
from prometheus_client.parser import text_string_to_metric_families

__all__ = [
    "BatchGatewayJobSource",
    "BatchJobSource",
    "BatchResolver",
    "BatchSchedulingCollector",
    "DispatcherFeedbackSource",
    "LlmdAsyncPrometheusSource",
    "LlmdAsyncOpenMetricsSource",
    "OpenMetricsOnlineTrafficSource",
    "OnlineTrafficSource",
    "PrometheusQueryClient",
]

logger = logging.getLogger(__name__)

_KNOWN_BATCH_STATUSES = frozenset(
    {
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
    }
)
_TERMINAL_BATCH_STATUSES = frozenset({"completed", "failed", "expired", "cancelled"})
_PLANNER_REQUEST_COUNT_METADATA_KEY = "planner_request_count"
_MAX_SIGNED_INT64 = (1 << 63) - 1
_NANOSECONDS_PER_SECOND = 1_000_000_000
_GO_DURATION_COMPONENT = re.compile(
    r"(?P<value>(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))" r"(?P<unit>ns|us|µs|μs|ms|s|m|h)"
)
_GO_DURATION_UNIT_NANOSECONDS = {
    "ns": 1,
    "us": 1_000,
    "µs": 1_000,
    "μs": 1_000,
    "ms": 1_000_000,
    "s": _NANOSECONDS_PER_SECOND,
    "m": 60 * _NANOSECONDS_PER_SECOND,
    "h": 60 * 60 * _NANOSECONDS_PER_SECOND,
}


class BatchJobSource(Protocol):
    """Collect batch-job demand at a caller-supplied sample time."""

    async def collect_batch_jobs(
        self, *, observed_at_s: float
    ) -> list[BatchJobDemand]: ...


class OnlineTrafficSource(Protocol):
    """Collect non-batch offered load at a caller-supplied sample time."""

    async def collect_online_traffic(
        self, *, observed_at_s: float
    ) -> list[PoolTrafficDemand]: ...


class DispatcherFeedbackSource(Protocol):
    """Collect llm-d Async dispatcher feedback for configured pools."""

    async def collect_dispatcher_feedback(
        self, *, observed_at_s: float
    ) -> list[BatchDispatcherFeedback]: ...


class PrometheusQueryClient(Protocol):
    """Subset of a Prometheus client used by the dispatcher source."""

    def custom_query(self, query: str) -> object: ...


BatchResolver = Callable[[Mapping[str, Any]], str]
PrometheusQuery = Callable[[str], object]


class OpenMetricsOnlineTrafficSource:
    """Derive a strict online-request rate from a directly scraped counter.

    The first successful scrape only establishes a counter baseline and raises
    ``RuntimeError``.  This intentionally makes the first native Planner tick
    publish a zero-rate safety lease instead of treating an unknown rate as
    zero. Counter resets are handled the same way: the new value becomes the
    next baseline, but the current observation fails closed.

    ``match_labels`` is explicit because frontend metrics do not currently
    carry a general online-vs-batch origin label.  A deployment can select a
    dedicated request type (the POC uses streaming requests for online load)
    without baking that convention into Planner policy.
    """

    def __init__(
        self,
        *,
        pool_id: str,
        metrics_url: str,
        session: aiohttp.ClientSession,
        match_labels: Mapping[str, str],
        metric_name: str = "dynamo_frontend_requests_started_total",
    ) -> None:
        if not isinstance(pool_id, str) or not pool_id:
            raise ValueError("pool_id must be non-empty")
        if not isinstance(metrics_url, str) or not metrics_url:
            raise ValueError("metrics_url must be non-empty")
        if not isinstance(metric_name, str) or not metric_name:
            raise ValueError("metric_name must be non-empty")
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in match_labels.items()
        ):
            raise ValueError("OpenMetrics match labels must be non-empty strings")

        self._pool_id = pool_id
        self._metrics_url = metrics_url
        self._session = session
        self._match_labels = dict(match_labels)
        self._metric_name = metric_name
        self._previous_counter: Optional[float] = None
        self._previous_observed_at_s: Optional[float] = None

    async def collect_online_traffic(
        self, *, observed_at_s: float
    ) -> list[PoolTrafficDemand]:
        observed_at_s = _require_finite_non_negative("observed_at_s", observed_at_s)
        payload = await _scrape_openmetrics(self._session, self._metrics_url)
        counter = _sum_matching_metric(
            payload,
            metric_name=self._metric_name,
            match_labels=self._match_labels,
            allow_missing=True,
        )

        previous_counter = self._previous_counter
        previous_observed_at_s = self._previous_observed_at_s
        self._previous_counter = counter
        self._previous_observed_at_s = observed_at_s

        if previous_counter is None or previous_observed_at_s is None:
            raise RuntimeError("online request-rate counter is warming up")
        elapsed_s = observed_at_s - previous_observed_at_s
        if elapsed_s <= 0:
            raise RuntimeError("online request-rate sample time did not advance")
        if counter < previous_counter:
            raise RuntimeError("online request-rate counter reset")

        return [
            PoolTrafficDemand(
                observed_at_s=observed_at_s,
                pool_id=self._pool_id,
                online_offered_rps=(counter - previous_counter) / elapsed_s,
            )
        ]


class LlmdAsyncOpenMetricsSource:
    """Read dispatcher state directly from llm-d Async's metrics endpoint.

    This is the in-cluster counterpart to ``LlmdAsyncPrometheusSource`` for
    environments where the central Prometheus API is not reachable from
    workload namespaces. Availability is still strict: every configured pool
    must expose a successful broker-backlog anchor with value exactly one.
    """

    def __init__(
        self,
        *,
        pools: Sequence[str],
        metrics_url: str,
        session: aiohttp.ClientSession,
    ) -> None:
        if isinstance(pools, (str, bytes, bytearray)):
            raise ValueError("pools must be a sequence of pool IDs, not a string")
        pool_ids = tuple(pools)
        if not pool_ids or any(
            not isinstance(pool_id, str) or not pool_id for pool_id in pool_ids
        ):
            raise ValueError("pools must contain non-empty pool IDs")
        if len(set(pool_ids)) != len(pool_ids):
            raise ValueError("pool IDs must be unique")
        if not isinstance(metrics_url, str) or not metrics_url:
            raise ValueError("metrics_url must be non-empty")

        self._pools = pool_ids
        self._metrics_url = metrics_url
        self._session = session
        self._previous_dispatch_counters: dict[str, tuple[float, float]] = {}

    async def collect_dispatcher_feedback(
        self, *, observed_at_s: float
    ) -> list[BatchDispatcherFeedback]:
        observed_at_s = _require_finite_non_negative("observed_at_s", observed_at_s)
        payload = await _scrape_openmetrics(self._session, self._metrics_url)
        result = [
            self._collect_pool(payload, pool_id, observed_at_s)
            for pool_id in self._pools
        ]
        _validate_dispatcher_feedback(result)
        return result

    def _collect_pool(
        self,
        payload: str,
        pool_id: str,
        observed_at_s: float,
    ) -> BatchDispatcherFeedback:
        labels = {"pool_name": pool_id}
        backlog = _sum_matching_metric(
            payload,
            metric_name="llm_d_async_async_broker_backlog",
            match_labels=labels,
            allow_missing=False,
        )
        availability_values = _matching_metric_values(
            payload,
            metric_name="llm_d_async_async_broker_backlog_source_available",
            match_labels=labels,
        )
        if not availability_values or any(
            value != 1.0 for value in availability_values
        ):
            raise ValueError(
                f"broker backlog source availability for pool {pool_id!r} "
                "must be exactly 1"
            )
        queue_depth = _sum_matching_metric(
            payload,
            metric_name="llm_d_async_async_queue_depth",
            match_labels=labels,
            allow_missing=True,
        )
        inflight = _sum_matching_metric(
            payload,
            metric_name="llm_d_async_async_inflight_requests",
            match_labels=labels,
            allow_missing=True,
        )
        dispatched = _sum_matching_metric(
            payload,
            metric_name="llm_d_async_async_dispatched_requests_total",
            match_labels=labels,
            allow_missing=True,
        )

        previous = self._previous_dispatch_counters.get(pool_id)
        dispatch_rps = 0.0
        if previous is not None:
            previous_counter, previous_observed_at_s = previous
            elapsed_s = observed_at_s - previous_observed_at_s
            if elapsed_s > 0 and dispatched >= previous_counter:
                dispatch_rps = (dispatched - previous_counter) / elapsed_s
        self._previous_dispatch_counters[pool_id] = (dispatched, observed_at_s)

        lease_valid_values = _matching_metric_values(
            payload,
            metric_name="llm_d_async_async_drain_limit_lease_valid",
            match_labels=labels,
        )
        lease_valid = bool(lease_valid_values) and all(
            value == 1.0 for value in lease_valid_values
        )
        cap_values = _matching_metric_values(
            payload,
            metric_name="llm_d_async_async_drain_limit_rps",
            match_labels=labels,
        )
        expiry_values = _matching_metric_values(
            payload,
            metric_name="llm_d_async_async_drain_limit_valid_until_seconds",
            match_labels=labels,
        )
        applied_cap: Optional[float] = None
        if lease_valid and cap_values and expiry_values:
            expiry_s = min(expiry_values)
            if expiry_s > observed_at_s:
                applied_cap = min(cap_values)

        queued_requests = _require_integral_metric(
            "queued_requests", backlog + queue_depth, pool_id
        )
        inflight_requests = _require_integral_metric(
            "inflight_requests", inflight, pool_id
        )
        return BatchDispatcherFeedback(
            observed_at_s=observed_at_s,
            pool_id=pool_id,
            observation_window_s=(
                observed_at_s - previous[1]
                if previous is not None and observed_at_s > previous[1]
                else 1.0
            ),
            queued_requests=queued_requests,
            inflight_requests=inflight_requests,
            actual_dispatch_rps=dispatch_rps,
            applied_max_admission_rps=applied_cap,
        )


class BatchSchedulingCollector:
    """Collect independently implemented batch inputs into one observation."""

    def __init__(
        self,
        *,
        batch_jobs: BatchJobSource,
        online_traffic: OnlineTrafficSource,
        dispatcher_feedback: DispatcherFeedbackSource,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._batch_jobs = batch_jobs
        self._online_traffic = online_traffic
        self._dispatcher_feedback = dispatcher_feedback
        self._clock = clock

    async def collect(self) -> BatchSchedulingObservation:
        """Sample all sources against one wall-clock timestamp.

        Gateway job state is the durable demand authority, so its failure is
        propagated. Traffic and dispatcher samples are independently optional:
        omitting a failed sample preserves successfully observed jobs without
        representing the missing value as zero. The policy treats a missing
        traffic sample as a fail-closed admission pause.
        """

        observed_at_s = _require_finite_non_negative("observed_at_s", self._clock())
        results = await asyncio.gather(
            self._batch_jobs.collect_batch_jobs(observed_at_s=observed_at_s),
            self._online_traffic.collect_online_traffic(observed_at_s=observed_at_s),
            self._dispatcher_feedback.collect_dispatcher_feedback(
                observed_at_s=observed_at_s
            ),
            return_exceptions=True,
        )
        # All sibling operations have settled before any propagation. This
        # prevents an orphaned request from racing provider/session shutdown.
        # Cancellation and process-control BaseExceptions are never degraded
        # into a missing optional observation.
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result

        jobs_result, traffic_result, feedback_result = results
        if isinstance(jobs_result, BaseException):
            raise jobs_result
        jobs = jobs_result
        _validate_job_demands(jobs)

        traffic: Sequence[PoolTrafficDemand] = []
        if isinstance(traffic_result, BaseException):
            self._log_optional_source_failure("online traffic", traffic_result)
        else:
            try:
                _validate_pool_traffic(traffic_result)
            except Exception:
                logger.exception(
                    "Batch scheduling online traffic observation was invalid; "
                    "continuing with the source missing"
                )
            else:
                traffic = traffic_result

        feedback: Sequence[BatchDispatcherFeedback] = []
        if isinstance(feedback_result, BaseException):
            self._log_optional_source_failure("dispatcher feedback", feedback_result)
        else:
            try:
                _validate_dispatcher_feedback(feedback_result)
            except Exception:
                logger.exception(
                    "Batch scheduling dispatcher feedback observation was invalid; "
                    "continuing with the source missing"
                )
            else:
                feedback = feedback_result

        return BatchSchedulingObservation(
            job_demands=list(jobs),
            pool_traffic=list(traffic),
            dispatcher_feedback=list(feedback),
        )

    @staticmethod
    def _log_optional_source_failure(name: str, error: BaseException) -> None:
        logger.warning(
            "Batch scheduling %s observation failed; continuing with the "
            "source missing",
            name,
            exc_info=(type(error), error, error.__traceback__),
        )


class BatchGatewayJobSource:
    """Read batch jobs from the OpenAI-compatible Batch Gateway API.

    The list endpoint contains database snapshots. For every job whose listed
    status is nonterminal, this source retrieves ``/v1/batches/{id}`` so the
    Gateway can merge its live Redis progress counters before Planner sees the
    job.
    """

    def __init__(
        self,
        *,
        base_url: str,
        session: aiohttp.ClientSession,
        pool_resolver: BatchResolver,
        work_class_resolver: BatchResolver,
        headers: Optional[Mapping[str, str]] = None,
        page_size: int = 100,
        max_pages: int = 10_000,
    ) -> None:
        if not isinstance(base_url, str):
            raise ValueError("base_url must be a string")
        normalized_url = base_url.rstrip("/")
        if not normalized_url:
            raise ValueError("base_url must be non-empty")
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= 100
        ):
            raise ValueError("page_size must be between 1 and 100")
        if (
            isinstance(max_pages, bool)
            or not isinstance(max_pages, int)
            or max_pages <= 0
        ):
            raise ValueError("max_pages must be positive")

        self._base_url = normalized_url
        self._session = session
        self._pool_resolver = pool_resolver
        self._work_class_resolver = work_class_resolver
        self._headers = dict(headers or {})
        self._page_size = page_size
        self._max_pages = max_pages

    async def collect_batch_jobs(self, *, observed_at_s: float) -> list[BatchJobDemand]:
        observed_at_s = _require_finite_non_negative("observed_at_s", observed_at_s)
        listed_jobs = await self._list_all_batches()
        jobs: list[BatchJobDemand] = []
        for listed_job in listed_jobs:
            status = _batch_status(listed_job)
            job = listed_job
            if status not in _TERMINAL_BATCH_STATUSES:
                job_id = _required_string(listed_job, "id")
                job = await self._get_json(f"/v1/batches/{quote(job_id, safe='')}")
                if not isinstance(job, Mapping):
                    raise ValueError(
                        f"batch {job_id!r} detail response must be an object"
                    )
                detailed_id = _required_string(job, "id")
                if detailed_id != job_id:
                    raise ValueError(
                        f"batch detail ID {detailed_id!r} does not match {job_id!r}"
                    )
            jobs.append(self._to_job_demand(job, observed_at_s))

        _validate_job_demands(jobs)
        return jobs

    async def _list_all_batches(self) -> list[Mapping[str, Any]]:
        after = 0
        listed_jobs: list[Mapping[str, Any]] = []
        seen_ids: set[str] = set()

        for _page_number in range(self._max_pages):
            payload = await self._get_json(
                "/v1/batches",
                params={"limit": str(self._page_size), "after": str(after)},
            )
            if not isinstance(payload, Mapping):
                raise ValueError("batch list response must be an object")
            raw_jobs = payload.get("data")
            has_more = payload.get("has_more")
            if not isinstance(raw_jobs, list):
                raise ValueError("batch list data must be a list")
            if not isinstance(has_more, bool):
                raise ValueError("batch list has_more must be a boolean")

            for raw_job in raw_jobs:
                if not isinstance(raw_job, Mapping):
                    raise ValueError("each listed batch must be an object")
                job_id = _required_string(raw_job, "id")
                _batch_status(raw_job)
                if job_id in seen_ids:
                    raise ValueError(f"duplicate batch ID {job_id!r} across pages")
                seen_ids.add(job_id)
                listed_jobs.append(raw_job)

            if not has_more:
                return listed_jobs
            if not raw_jobs:
                raise ValueError("batch list returned has_more=true with an empty page")
            after += len(raw_jobs)

        raise ValueError(f"batch list exceeded max_pages={self._max_pages}")

    async def _get_json(
        self,
        path: str,
        *,
        params: Optional[Mapping[str, str]] = None,
    ) -> object:
        async with self._session.get(
            f"{self._base_url}{path}",
            params=params,
            headers=self._headers,
        ) as response:
            response.raise_for_status()
            return await response.json()

    def _to_job_demand(
        self, job: Mapping[str, Any], observed_at_s: float
    ) -> BatchJobDemand:
        job_id = _required_string(job, "id")
        status = _batch_status(job)
        request_counts = job.get("request_counts")
        if not isinstance(request_counts, Mapping):
            raise ValueError(f"batch {job_id!r} request_counts must be an object")

        deadline_at_s = _batch_deadline_at_s(job, context=job_id)
        total_requests = _batch_total_requests(
            job,
            request_counts=request_counts,
            status=status,
            context=job_id,
        )

        pool_id = self._pool_resolver(job)
        work_class = self._work_class_resolver(job)
        if not isinstance(pool_id, str) or not pool_id:
            raise ValueError(f"batch {job_id!r} resolved to an empty pool_id")
        if not isinstance(work_class, str) or not work_class:
            raise ValueError(f"batch {job_id!r} resolved to an empty work_class")

        return BatchJobDemand(
            observed_at_s=observed_at_s,
            pool_id=pool_id,
            job_id=job_id,
            status=status,
            total_requests=total_requests,
            completed_requests=_required_non_negative_int(
                request_counts, "completed", context=job_id
            ),
            failed_requests=_required_non_negative_int(
                request_counts, "failed", context=job_id
            ),
            deadline_at_s=deadline_at_s,
            work_class=work_class,
        )


class LlmdAsyncPrometheusSource:
    """Collect pool-level llm-d Async feedback through Prometheus queries."""

    def __init__(
        self,
        *,
        pools: Sequence[str],
        query: Union[PrometheusQuery, PrometheusQueryClient],
        observation_window_s: int = 30,
        max_sample_age_s: float = 60.0,
    ) -> None:
        if isinstance(pools, (str, bytes, bytearray)):
            raise ValueError("pools must be a sequence of pool IDs, not a string")
        pool_ids = tuple(pools)
        if not pool_ids:
            raise ValueError("at least one pool must be configured")
        if any(not isinstance(pool_id, str) or not pool_id for pool_id in pool_ids):
            raise ValueError("pool IDs must be non-empty strings")
        if len(set(pool_ids)) != len(pool_ids):
            raise ValueError("pool IDs must be unique")
        if (
            isinstance(observation_window_s, bool)
            or not isinstance(observation_window_s, int)
            or observation_window_s <= 0
        ):
            raise ValueError("observation_window_s must be a positive integer")
        max_sample_age_s = _require_finite_non_negative(
            "max_sample_age_s", max_sample_age_s
        )

        self._pools = pool_ids
        self._observation_window_s = observation_window_s
        self._max_sample_age_s = max_sample_age_s
        if callable(query):
            self._query = query
        else:
            self._query = query.custom_query

    async def collect_dispatcher_feedback(
        self, *, observed_at_s: float
    ) -> list[BatchDispatcherFeedback]:
        observed_at_s = _require_finite_non_negative("observed_at_s", observed_at_s)
        feedback = await asyncio.gather(
            *(self._collect_pool(pool_id, observed_at_s) for pool_id in self._pools)
        )
        result = list(feedback)
        _validate_dispatcher_feedback(result)
        return result

    async def _collect_pool(
        self, pool_id: str, observed_at_s: float
    ) -> BatchDispatcherFeedback:
        label = _escape_promql_label_value(pool_id)
        selector = f'pool_name="{label}"'
        window = f"{self._observation_window_s}s"

        # Broker backlog is the pool-presence anchor and intentionally has no
        # zero fallback. Availability is also mandatory; min() makes one failed
        # queue/replica fail the entire pool sample. The other three activity
        # series are absent before the first dequeue/dispatch, where zero is the
        # only valid interpretation.
        queued_query = (
            "sum(max by (queue_id, queue_name, pool_name) "
            f"(llm_d_async_async_broker_backlog{{{selector}}})) + "
            f"(sum(llm_d_async_async_queue_depth{{{selector}}}) or vector(0))"
        )
        backlog_available_query = (
            f"min(llm_d_async_async_broker_backlog_source_available{{{selector}}})"
        )
        # An instant-query result's sample timestamp is normally the PromQL
        # evaluation time, even when the underlying series came from the
        # lookback window. Put the oldest mandatory anchor's scrape age in the
        # query value itself so a dead exporter cannot be replayed as fresh.
        # Measuring age on the Prometheus server also avoids local/server clock
        # skew; subtracting it from the collection timestamp is conservative.
        sample_age_query = (
            "time() - min(timestamp("
            "llm_d_async_async_broker_backlog_source_available"
            f"{{{selector}}}))"
        )
        inflight_query = (
            f"(sum(llm_d_async_async_inflight_requests{{{selector}}}) or vector(0))"
        )
        rate_query = (
            "(sum(rate(llm_d_async_async_dispatched_requests_total"
            f"{{{selector}}}[{window}])) or vector(0))"
        )
        cap_query = f"min(llm_d_async_async_drain_limit_rps{{{selector}}})"
        lease_query = f"min(llm_d_async_async_drain_limit_lease_valid{{{selector}}})"
        # lease_valid is updated when the gate runs and can remain stale while
        # an idle lease expires. Re-evaluate the published absolute expiry at
        # Prometheus query time and require both signals below.
        lease_unexpired_query = (
            "min(llm_d_async_async_drain_limit_valid_until_seconds"
            f"{{{selector}}}) > bool time()"
        )

        (
            queued,
            backlog_available,
            sample_age,
            inflight,
            dispatch_rate,
            cap,
            lease_valid,
            lease_unexpired,
        ) = await asyncio.gather(
            self._query_scalar(queued_query),
            self._query_scalar(backlog_available_query),
            self._query_scalar(sample_age_query),
            self._query_scalar(inflight_query),
            self._query_scalar(rate_query),
            self._query_scalar(cap_query),
            self._query_scalar(lease_query),
            self._query_scalar(lease_unexpired_query),
        )

        queued_requests = _require_integral_metric("queued_requests", queued, pool_id)
        if backlog_available != 1.0:
            raise ValueError(
                f"broker backlog source availability for pool {pool_id!r} "
                "must be exactly 1"
            )
        sample_age = _require_finite_non_negative(
            f"Prometheus sample age for pool {pool_id!r}", sample_age
        )
        if sample_age > self._max_sample_age_s:
            raise ValueError(
                f"Prometheus sample age for pool {pool_id!r} is {sample_age}s, "
                f"exceeding the {self._max_sample_age_s}s maximum"
            )
        feedback_observed_at_s = max(0.0, observed_at_s - sample_age)
        inflight_requests = _require_integral_metric(
            "inflight_requests", inflight, pool_id
        )
        dispatch_rate = _require_finite_non_negative(
            f"actual_dispatch_rps for pool {pool_id!r}", dispatch_rate
        )
        cap = _require_finite_non_negative(f"applied cap for pool {pool_id!r}", cap)
        if lease_valid not in (0.0, 1.0):
            raise ValueError(
                f"lease validity for pool {pool_id!r} must be exactly 0 or 1"
            )
        if lease_unexpired not in (0.0, 1.0):
            raise ValueError(
                f"lease expiry state for pool {pool_id!r} must be exactly 0 or 1"
            )
        effective_lease_valid = lease_valid == 1.0 and lease_unexpired == 1.0

        return BatchDispatcherFeedback(
            observed_at_s=feedback_observed_at_s,
            pool_id=pool_id,
            observation_window_s=float(self._observation_window_s),
            queued_requests=queued_requests,
            inflight_requests=inflight_requests,
            actual_dispatch_rps=dispatch_rate,
            applied_max_admission_rps=cap if effective_lease_valid else None,
        )

    async def _query_scalar(self, query: str) -> float:
        if inspect.iscoroutinefunction(self._query):
            payload = await self._query(query)
        else:
            payload = await asyncio.to_thread(self._query, query)
            if inspect.isawaitable(payload):
                payload = await payload
        return _parse_prometheus_scalar(payload, query=query)


OpenMetricsSamples = dict[str, list[tuple[Mapping[str, str], float]]]


async def _scrape_openmetrics(
    session: aiohttp.ClientSession, metrics_url: str
) -> OpenMetricsSamples:
    """Fetch and parse one Prometheus/OpenMetrics exposition atomically."""

    async with session.get(metrics_url) as response:
        response.raise_for_status()
        payload = await response.text()

    samples: OpenMetricsSamples = {}
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            value = float(sample.value)
            if not math.isfinite(value):
                raise ValueError(f"OpenMetrics sample {sample.name!r} is not finite")
            samples.setdefault(sample.name, []).append((dict(sample.labels), value))
    return samples


def _matching_metric_values(
    samples: OpenMetricsSamples,
    *,
    metric_name: str,
    match_labels: Mapping[str, str],
) -> list[float]:
    values = [
        value
        for labels, value in samples.get(metric_name, [])
        if all(labels.get(key) == expected for key, expected in match_labels.items())
    ]
    if any(value < 0 for value in values):
        raise ValueError(f"OpenMetrics sample {metric_name!r} must be non-negative")
    return values


def _sum_matching_metric(
    samples: OpenMetricsSamples,
    *,
    metric_name: str,
    match_labels: Mapping[str, str],
    allow_missing: bool,
) -> float:
    values = _matching_metric_values(
        samples,
        metric_name=metric_name,
        match_labels=match_labels,
    )
    if not values and not allow_missing:
        raise ValueError(
            f"required OpenMetrics sample {metric_name!r} was not present "
            f"for labels {dict(match_labels)!r}"
        )
    return sum(values)


def _batch_status(job: Mapping[str, Any]) -> str:
    status = _required_string(job, "status")
    if status not in _KNOWN_BATCH_STATUSES:
        raise ValueError(f"batch {_job_context(job)!r} has unknown status {status!r}")
    return status


def _job_context(job: Mapping[str, Any]) -> str:
    job_id = job.get("id")
    return job_id if isinstance(job_id, str) and job_id else "<unknown>"


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"batch {_job_context(payload)!r} field {field_name!r} "
            "must be a non-empty string"
        )
    return value


def _required_non_negative_int(
    payload: Mapping[str, Any], field_name: str, *, context: str
) -> int:
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"batch {context!r} field {field_name!r} must be a non-negative integer"
        )
    return value


def _batch_deadline_at_s(job: Mapping[str, Any], *, context: str) -> float:
    """Resolve the best public deadline representation exposed by the Gateway."""

    if job.get("expires_at") is not None:
        expires_at_s = _required_non_negative_int64(job, "expires_at", context=context)
        return _conservative_seconds_from_nanoseconds(
            expires_at_s * _NANOSECONDS_PER_SECOND,
            context=context,
        )

    created_at_s = _required_non_negative_int64(job, "created_at", context=context)
    completion_window = job.get("completion_window")
    if not isinstance(completion_window, str) or not completion_window:
        raise ValueError(
            f"batch {context!r} field 'completion_window' must be a non-empty string"
        )

    duration_ns = _parse_positive_go_duration_nanoseconds(
        completion_window, context=context
    )
    deadline_ns = created_at_s * _NANOSECONDS_PER_SECOND + duration_ns
    max_timestamp_ns = (
        _MAX_SIGNED_INT64 * _NANOSECONDS_PER_SECOND + _NANOSECONDS_PER_SECOND - 1
    )
    if deadline_ns > max_timestamp_ns:
        raise ValueError(
            f"batch {context!r} derived deadline overflows int64 Unix time"
        )
    return _conservative_seconds_from_nanoseconds(deadline_ns, context=context)


def _batch_total_requests(
    job: Mapping[str, Any],
    *,
    request_counts: Mapping[str, Any],
    status: str,
    context: str,
) -> int:
    live_total = _required_non_negative_int(request_counts, "total", context=context)
    if live_total > 0:
        declared_total = _planner_request_count(job, context=context)
        if declared_total is not None and declared_total != live_total:
            raise ValueError(
                f"batch {context!r} metadata."
                f"{_PLANNER_REQUEST_COUNT_METADATA_KEY}={declared_total} does not "
                f"match request_counts.total={live_total}"
            )
        return live_total

    if status in _TERMINAL_BATCH_STATUSES:
        return live_total

    declared_total = _planner_request_count(job, context=context)
    if declared_total is None:
        raise ValueError(
            f"active batch {context!r} has request_counts.total=0 and requires "
            f"metadata.{_PLANNER_REQUEST_COUNT_METADATA_KEY}"
        )
    return declared_total


def _planner_request_count(job: Mapping[str, Any], *, context: str) -> Optional[int]:
    metadata = job.get("metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"batch {context!r} metadata must be an object or null")
    if _PLANNER_REQUEST_COUNT_METADATA_KEY not in metadata:
        return None

    raw_value = metadata[_PLANNER_REQUEST_COUNT_METADATA_KEY]
    if (
        not isinstance(raw_value, str)
        or re.fullmatch(r"[1-9][0-9]*", raw_value) is None
    ):
        raise ValueError(
            f"batch {context!r} metadata.{_PLANNER_REQUEST_COUNT_METADATA_KEY} "
            "must be a canonical positive decimal string"
        )
    declared_total = int(raw_value)
    if declared_total > _MAX_SIGNED_INT64:
        raise ValueError(
            f"batch {context!r} metadata.{_PLANNER_REQUEST_COUNT_METADATA_KEY} "
            "exceeds signed int64"
        )
    return declared_total


def _required_non_negative_int64(
    payload: Mapping[str, Any], field_name: str, *, context: str
) -> int:
    value = _required_non_negative_int(payload, field_name, context=context)
    if value > _MAX_SIGNED_INT64:
        raise ValueError(f"batch {context!r} field {field_name!r} exceeds signed int64")
    return value


def _parse_positive_go_duration_nanoseconds(value: str, *, context: str) -> int:
    """Parse the positive subset of Go's ``time.ParseDuration`` grammar."""

    original_value = value
    value = value.removeprefix("+")
    if not value or value.startswith("-"):
        raise ValueError(
            f"batch {context!r} completion_window {original_value!r} "
            "must be a positive Go duration"
        )

    total_ns = 0
    offset = 0
    while offset < len(value):
        match = _GO_DURATION_COMPONENT.match(value, offset)
        if match is None:
            raise ValueError(
                f"batch {context!r} completion_window {original_value!r} "
                "must be a positive Go duration"
            )

        number = match.group("value")
        unit_ns = _GO_DURATION_UNIT_NANOSECONDS[match.group("unit")]
        whole_text, separator, fraction_text = number.partition(".")
        whole = int(whole_text or "0")
        component_ns = whole * unit_ns
        if separator and fraction_text:
            component_ns += int(fraction_text) * unit_ns // (10 ** len(fraction_text))

        if component_ns > _MAX_SIGNED_INT64 - total_ns:
            raise ValueError(
                f"batch {context!r} completion_window {original_value!r} "
                "overflows Go duration"
            )
        total_ns += component_ns
        offset = match.end()

    if total_ns <= 0:
        raise ValueError(
            f"batch {context!r} completion_window {original_value!r} "
            "must resolve to a positive Go duration"
        )
    return total_ns


def _conservative_seconds_from_nanoseconds(value_ns: int, *, context: str) -> float:
    seconds = value_ns / _NANOSECONDS_PER_SECOND
    if not math.isfinite(seconds):
        raise ValueError(f"batch {context!r} deadline must be finite")

    numerator, denominator = seconds.as_integer_ratio()
    if numerator * _NANOSECONDS_PER_SECOND > value_ns * denominator:
        seconds = math.nextafter(seconds, -math.inf)
    if seconds < 0 or not math.isfinite(seconds):
        raise ValueError(f"batch {context!r} deadline must be finite and non-negative")
    return seconds


def _escape_promql_label_value(value: str) -> str:
    """Escape a value for a double-quoted PromQL label matcher."""

    escaped: list[str] = []
    for character in value:
        if character == "\\":
            escaped.append("\\\\")
        elif character == '"':
            escaped.append('\\"')
        elif character == "\n":
            escaped.append("\\n")
        elif character == "\r":
            escaped.append("\\r")
        elif character == "\t":
            escaped.append("\\t")
        elif ord(character) < 0x20 or ord(character) == 0x7F:
            escaped.append(f"\\x{ord(character):02x}")
        else:
            escaped.append(character)
    return "".join(escaped)


def _parse_prometheus_scalar(payload: object, *, query: str) -> float:
    if isinstance(payload, Mapping) and "status" in payload:
        if payload.get("status") != "success":
            raise ValueError(f"Prometheus query failed: {query}")
        data = payload.get("data")
        if not isinstance(data, Mapping):
            raise ValueError(f"Prometheus response data must be an object: {query}")
        payload = data.get("result")

    if isinstance(payload, bool):
        raise ValueError(f"Prometheus returned a boolean instead of a scalar: {query}")
    if isinstance(payload, (int, float)):
        value = float(payload)
    else:
        if not isinstance(payload, Sequence) or isinstance(
            payload, (str, bytes, bytearray)
        ):
            raise ValueError(f"Prometheus query result must be a vector: {query}")
        if len(payload) != 1:
            raise ValueError(
                f"Prometheus query must return exactly one sample, got {len(payload)}: "
                f"{query}"
            )
        sample = payload[0]
        if not isinstance(sample, Mapping):
            raise ValueError(f"Prometheus vector sample must be an object: {query}")
        sample_value = sample.get("value")
        if (
            not isinstance(sample_value, Sequence)
            or isinstance(sample_value, (str, bytes, bytearray))
            or len(sample_value) != 2
        ):
            raise ValueError(f"Prometheus sample value is malformed: {query}")
        raw_value = sample_value[1]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (str, int, float)):
            raise ValueError(f"Prometheus sample scalar is malformed: {query}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Prometheus sample is not numeric: {query}") from exc

    if not math.isfinite(value):
        raise ValueError(f"Prometheus sample must be finite: {query}")
    return value


def _require_finite_non_negative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_integral_metric(name: str, value: float, pool_id: str) -> int:
    value = _require_finite_non_negative(f"{name} for pool {pool_id!r}", value)
    if not value.is_integer():
        raise ValueError(f"{name} for pool {pool_id!r} must be an integer")
    return int(value)


def _require_non_negative_int_value(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_job_demands(jobs: Sequence[BatchJobDemand]) -> None:
    seen_ids: set[str] = set()
    for job in jobs:
        if not isinstance(job, BatchJobDemand):
            raise TypeError("batch job source returned a non-BatchJobDemand value")
        _require_finite_non_negative("job observed_at_s", job.observed_at_s)
        if any(
            not isinstance(value, str) or not value
            for value in (job.pool_id, job.job_id, job.work_class)
        ):
            raise ValueError("batch job identifiers and work_class must be non-empty")
        if not isinstance(job.status, str) or job.status not in _KNOWN_BATCH_STATUSES:
            raise ValueError(f"unknown batch job status {job.status!r}")
        total_requests = _require_non_negative_int_value(
            "batch total_requests", job.total_requests
        )
        completed_requests = _require_non_negative_int_value(
            "batch completed_requests", job.completed_requests
        )
        failed_requests = _require_non_negative_int_value(
            "batch failed_requests", job.failed_requests
        )
        if completed_requests + failed_requests > total_requests:
            raise ValueError(
                "batch completed_requests + failed_requests must not exceed "
                "total_requests"
            )
        if job.deadline_at_s is not None:
            _require_finite_non_negative("job deadline_at_s", job.deadline_at_s)
        if job.job_id in seen_ids:
            raise ValueError(f"duplicate batch job ID {job.job_id!r}")
        seen_ids.add(job.job_id)


def _validate_pool_traffic(traffic: Sequence[PoolTrafficDemand]) -> None:
    seen_pools: set[str] = set()
    for demand in traffic:
        if not isinstance(demand, PoolTrafficDemand):
            raise TypeError(
                "online traffic source returned a non-PoolTrafficDemand value"
            )
        _require_finite_non_negative("traffic observed_at_s", demand.observed_at_s)
        _require_finite_non_negative("online_offered_rps", demand.online_offered_rps)
        if not isinstance(demand.pool_id, str) or not demand.pool_id:
            raise ValueError("online traffic pool_id must be non-empty")
        if demand.pool_id in seen_pools:
            raise ValueError(f"duplicate online traffic pool {demand.pool_id!r}")
        seen_pools.add(demand.pool_id)


def _validate_dispatcher_feedback(
    feedback: Sequence[BatchDispatcherFeedback],
) -> None:
    seen_pools: set[str] = set()
    for item in feedback:
        if not isinstance(item, BatchDispatcherFeedback):
            raise TypeError(
                "dispatcher source returned a non-BatchDispatcherFeedback value"
            )
        _require_finite_non_negative("dispatcher observed_at_s", item.observed_at_s)
        window_s = _require_finite_non_negative(
            "dispatcher observation_window_s", item.observation_window_s
        )
        if window_s <= 0:
            raise ValueError("dispatcher observation_window_s must be positive")
        _require_non_negative_int_value(
            "dispatcher queued_requests", item.queued_requests
        )
        _require_non_negative_int_value(
            "dispatcher inflight_requests", item.inflight_requests
        )
        _require_finite_non_negative(
            "dispatcher actual_dispatch_rps", item.actual_dispatch_rps
        )
        if item.applied_max_admission_rps is not None:
            _require_finite_non_negative(
                "dispatcher applied_max_admission_rps",
                item.applied_max_admission_rps,
            )
        if not isinstance(item.pool_id, str) or not item.pool_id:
            raise ValueError("dispatcher pool_id must be non-empty")
        if item.pool_id in seen_pools:
            raise ValueError(f"duplicate dispatcher pool {item.pool_id!r}")
        seen_pools.add(item.pool_id)
