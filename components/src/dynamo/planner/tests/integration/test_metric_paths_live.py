# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster integration tests for the planner's metric-collection paths.

Run inside the dev pod (`planner-dev`) to exercise real Prometheus, real
router /metrics scrapes, real DCGM samples, and live MDC CRs.

Skipped unless `DYN_PARENT_DGD_K8S_NAME` is set (i.e. running in the dev pod).

Run from the dev pod:
    export PYTHONPATH=/workspace/repo/components/src:/workspace/repo
    cd /workspace/repo
    python3 -m pytest \
        components/src/dynamo/planner/tests/integration/test_metric_paths_live.py \
        -v --tb=short -m 'integration'
"""

from __future__ import annotations

import os
import urllib.parse
import urllib.request
from typing import Optional

import pytest

# ---------------------------------------------------------------------------
# Module-level gating
# ---------------------------------------------------------------------------

try:
    from kubernetes import config as k8s_config

    try:
        k8s_config.load_incluster_config()
        _IN_CLUSTER = True
    except k8s_config.ConfigException:
        _IN_CLUSTER = False
except ImportError:
    _IN_CLUSTER = False

_DGD_NAME = os.environ.get("DYN_PARENT_DGD_K8S_NAME")
_K8S_NAMESPACE = os.environ.get("DYN_PARENT_DGD_K8S_NAMESPACE") or os.environ.get(
    "POD_NAMESPACE"
)
_DYN_NAMESPACE = os.environ.get(
    "DYN_NAMESPACE", f"{_K8S_NAMESPACE}-{_DGD_NAME}" if _DGD_NAME else "dynamo"
)

if not (_IN_CLUSTER and _DGD_NAME):
    pytest.skip(
        "Live metric-path tests require running inside the dev pod with "
        "DYN_PARENT_DGD_K8S_NAME set.",
        allow_module_level=True,
    )

# Cluster service URLs (these match the ones documented in
# docs/components/planner/dpp-dev-env.md).
_PROM_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"

# Deferred imports — these pull in heavy planner deps that need the runtime ready.
from dynamo.planner.connectors.kubernetes import KubernetesConnector  # noqa: E402
from dynamo.planner.monitoring.traffic_metrics import (  # noqa: E402
    _WORKER_METRIC_NAMES,
    DirectRouterMetricsClient,
    PrometheusAPIClient,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def connector() -> KubernetesConnector:
    return KubernetesConnector(
        dynamo_namespace=_DYN_NAMESPACE,
        k8s_namespace=_K8S_NAMESPACE,
        parent_dgd_name=_DGD_NAME,
    )


@pytest.fixture(scope="module")
def model_name(connector: KubernetesConnector) -> str:
    """Resolve model name from the live DGD; fall back to env var if absent."""
    try:
        return connector.get_model_name(require_prefill=False, require_decode=False)
    except Exception:
        return os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def prom_frontend() -> PrometheusAPIClient:
    return PrometheusAPIClient(
        url=_PROM_URL, dynamo_namespace=_DYN_NAMESPACE, metrics_source="frontend"
    )


@pytest.fixture(scope="module")
def prom_router() -> PrometheusAPIClient:
    return PrometheusAPIClient(
        url=_PROM_URL, dynamo_namespace=_DYN_NAMESPACE, metrics_source="router"
    )


def _raw_prom_query(query: str) -> list:
    url = f"{_PROM_URL}/api/v1/query?query={urllib.parse.quote(query)}"
    import json

    return json.loads(urllib.request.urlopen(url, timeout=10).read())["data"]["result"]


# ---------------------------------------------------------------------------
# 1. Prometheus connectivity (smoke)
# ---------------------------------------------------------------------------


class TestPrometheusReachable:
    def test_prometheus_responds_to_up_query(self):
        result = _raw_prom_query("up")
        assert result, "Prometheus did not return any 'up' samples"


# ---------------------------------------------------------------------------
# 2. Frontend-source metrics path
# ---------------------------------------------------------------------------


class TestPrometheusFrontendMetrics:
    """PrometheusAPIClient with metrics_source='frontend' against live data.

    These methods drive the throughput-based scaling loop. They must:
      - Connect to the live Prometheus,
      - Parse the response containers without ValidationError,
      - Filter by model + dynamo_namespace correctly.
    """

    def test_frontend_metric_series_exists(self):
        """Sanity: the frontend is producing metrics for our DGD."""
        result = _raw_prom_query(
            f'dynamo_frontend_requests_total{{dynamo_namespace="{_DYN_NAMESPACE}"}}'
        )
        if not result:
            pytest.skip(
                f"No frontend metric series for dynamo_namespace={_DYN_NAMESPACE!r}; "
                "send at least one chat/completions request to populate."
            )

    def test_get_avg_request_count_returns_numeric(
        self, prom_frontend: PrometheusAPIClient, model_name: str
    ):
        v = prom_frontend.get_avg_request_count("5m", model_name)
        assert isinstance(v, (int, float)), f"Expected numeric, got {type(v)}"
        assert v >= 0, f"Negative request count: {v}"

    def test_get_avg_time_to_first_token_returns_numeric(
        self, prom_frontend: PrometheusAPIClient, model_name: str
    ):
        v = prom_frontend.get_avg_time_to_first_token("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_get_avg_inter_token_latency_returns_numeric(
        self, prom_frontend: PrometheusAPIClient, model_name: str
    ):
        v = prom_frontend.get_avg_inter_token_latency("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_get_avg_request_duration_returns_numeric(
        self, prom_frontend: PrometheusAPIClient, model_name: str
    ):
        v = prom_frontend.get_avg_request_duration("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_get_avg_input_sequence_tokens_returns_numeric(
        self, prom_frontend: PrometheusAPIClient, model_name: str
    ):
        v = prom_frontend.get_avg_input_sequence_tokens("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_get_avg_output_sequence_tokens_returns_numeric(
        self, prom_frontend: PrometheusAPIClient, model_name: str
    ):
        v = prom_frontend.get_avg_output_sequence_tokens("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_model_name_filtering_excludes_other_models(
        self, prom_frontend: PrometheusAPIClient
    ):
        """A bogus model name should produce 0, proving the filter is active."""
        v = prom_frontend.get_avg_request_count("5m", "this/model-does-not-exist")
        assert v == 0


# ---------------------------------------------------------------------------
# 3. Router-source metrics path
# ---------------------------------------------------------------------------


class TestPrometheusRouterMetrics:
    """PrometheusAPIClient with metrics_source='router'.

    Router-source queries normalise dashes to underscores when building the
    dynamo_namespace label filter. The live router series confirms the
    underscore form (e.g. 'myns_dynamo_system_qwen3_quickstart').
    """

    def test_router_metric_series_exists(self):
        ns = _DYN_NAMESPACE.replace("-", "_")
        result = _raw_prom_query(
            f'dynamo_component_router_requests_total{{dynamo_namespace="{ns}"}}'
        )
        if not result:
            pytest.skip(
                f"No router metric series for namespace={ns!r}; "
                "either no router is running or PodMonitor is not scraping it."
            )

    def test_router_avg_request_count_returns_numeric(
        self, prom_router: PrometheusAPIClient, model_name: str
    ):
        v = prom_router.get_avg_request_count("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_router_avg_ttft_returns_numeric(
        self, prom_router: PrometheusAPIClient, model_name: str
    ):
        v = prom_router.get_avg_time_to_first_token("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_router_avg_itl_returns_numeric(
        self, prom_router: PrometheusAPIClient, model_name: str
    ):
        v = prom_router.get_avg_inter_token_latency("5m", model_name)
        assert isinstance(v, (int, float))
        assert v >= 0

    def test_warn_if_router_not_scraped_succeeds(
        self, prom_router: PrometheusAPIClient
    ):
        """Smoke: function must not raise even if scraping is fine."""
        prom_router.warn_if_router_not_scraped()


# ---------------------------------------------------------------------------
# 4. KV-hit-rate query (router-source only)
# ---------------------------------------------------------------------------


class TestKvHitRateLive:
    def test_router_source_kv_hit_rate_numeric_or_none(
        self, prom_router: PrometheusAPIClient, model_name: str
    ):
        v = prom_router.get_avg_kv_hit_rate("5m", model_name)
        # On a quiet cluster Prometheus returns no data → planner returns None.
        # If there is data, it must be a finite float in [0,1].
        assert v is None or (isinstance(v, float) and 0.0 <= v <= 1.0), v

    def test_frontend_source_kv_hit_rate_returns_none(
        self, prom_frontend: PrometheusAPIClient, model_name: str
    ):
        """The KV-hit-rate histogram only exists on the router; frontend
        source must short-circuit to None instead of querying."""
        assert prom_frontend.get_avg_kv_hit_rate("5m", model_name) is None


# ---------------------------------------------------------------------------
# 5. DCGM per-pod GPU power
# ---------------------------------------------------------------------------


class TestDcgmPerPodPower:
    """DCGM query path. On clusters with working DCGM workload attribution,
    these queries return watt values; otherwise the test skips with a clear
    message identifying the cluster-side gap (DCGM exporter or kubelet
    pod-info mapping not configured).

    The planner-side query construction was previously broken in three
    ways and has been fixed:
      1. Used bare ``pod`` (DCGM exporter pod) instead of ``exported_pod``
         (workload pod).
      2. Used ``exported_namespace=<dynamo-namespace>`` instead of the
         actual K8s namespace value.
      3. Used a regex ``<dgd>-<component>-.*`` that did not accommodate
         the operator's ``<dgd>-<replica-idx>-<service-key-lc>-<hash>``
         format.

    These tests pin the new behavior: planner queries must reach the
    DCGM samples whenever attribution exists, not silently return None.
    """

    def test_dcgm_samples_exist_for_dgd(self):
        """Smoke: verify DCGM is scraping pod-attribution for *something*."""
        result = _raw_prom_query("DCGM_FI_DEV_POWER_USAGE")
        assert result, "Prometheus reports no DCGM_FI_DEV_POWER_USAGE samples at all"

    def test_dcgm_pod_attribution_for_this_dgd(self):
        """Try to find DCGM samples attributed to a workload pod of this DGD.

        Uses the canonical ``exported_pod`` label (the planner now uses
        the same label internally).  Skip — not fail — when this cluster
        lacks attribution; downstream tests in this class then skip too.
        """
        result = _raw_prom_query(
            f'DCGM_FI_DEV_POWER_USAGE{{exported_pod=~"^{_DGD_NAME}-[0-9]+-.*"}}'
        )
        if not result:
            pytest.skip(
                f"DCGM has no exported_pod attribution for {_DGD_NAME!r} on this "
                "cluster (cluster-side DCGM/kubelet config issue, not a planner "
                "bug). The planner query construction has been fixed; this skip "
                "indicates the *cluster* needs DCGM exporter pod-info wiring."
            )

    def test_planner_get_total_dgd_power_runs_without_error(
        self, prom_frontend: PrometheusAPIClient
    ):
        """The planner method must not raise. With attribution it should
        also return a positive numeric value (real watt reading)."""
        v = prom_frontend.get_total_dgd_power(
            k8s_namespace=_K8S_NAMESPACE, dgd_name=_DGD_NAME
        )
        assert v is None or isinstance(v, float)
        # If the cluster has DCGM attribution wired (smoke test above
        # passed without skipping) and the planner queries are correct,
        # the value must not be None — that would indicate the fix
        # regressed.  Only assert this when raw attribution is present.
        attributed = _raw_prom_query(
            f"DCGM_FI_DEV_POWER_USAGE{{"
            f'exported_namespace="{_K8S_NAMESPACE}",'
            f'exported_pod=~"^{_DGD_NAME}-[0-9]+-.*"}}'
        )
        if attributed:
            assert v is not None, (
                f"Raw DCGM query found {len(attributed)} samples for "
                f"{_DGD_NAME} in {_K8S_NAMESPACE}, but planner method "
                "returned None — selector-construction regression."
            )
            assert v > 0, f"Total power should be positive, got {v}"

    def test_planner_per_component_query_runs_without_error(
        self, prom_frontend: PrometheusAPIClient
    ):
        """Per-component query: with attribution it should reach the
        right pod via the new ``<dgd>-<idx>-<service-key-lc>-`` regex."""
        # qwen3-quickstart deploys as agg with VllmWorker as the decode-side
        # service key.  Operator embeds it lowercase in pod names like
        # qwen3-quickstart-0-vllmworker-86nvj.
        v = prom_frontend.get_avg_per_gpu_power_by_component(
            interval="5m",
            k8s_namespace=_K8S_NAMESPACE,
            dgd_name=_DGD_NAME,
            component="agg",
            service_key="VllmWorker",
        )
        assert v is None or isinstance(v, float)
        attributed = _raw_prom_query(
            f"DCGM_FI_DEV_POWER_USAGE{{"
            f'exported_namespace="{_K8S_NAMESPACE}",'
            f'exported_pod=~"^{_DGD_NAME}-[0-9]+-vllmworker-.*"}}'
        )
        if attributed:
            assert v is not None, (
                f"Raw DCGM query found {len(attributed)} samples for "
                f"vllmworker pods in {_DGD_NAME}, but planner method "
                "returned None — selector-construction regression."
            )
            assert v > 0, f"Per-component power should be positive, got {v}"

    def test_planner_per_component_empty_service_key_short_circuits(
        self, prom_frontend: PrometheusAPIClient
    ):
        """Empty service_key (e.g. agg-mode DGD with no decode worker info)
        must short-circuit to None without issuing a Prometheus query that
        would degenerate to a never-matching regex."""
        v = prom_frontend.get_avg_per_gpu_power_by_component(
            interval="5m",
            k8s_namespace=_K8S_NAMESPACE,
            dgd_name=_DGD_NAME,
            component="prefill",
            service_key="",
        )
        assert v is None


# ---------------------------------------------------------------------------
# 6. Direct router /metrics scrape
# ---------------------------------------------------------------------------


def _discover_kv_router_metrics_url() -> Optional[str]:
    """Discover a KV-router /metrics endpoint reachable from the dev pod.

    The planner's ``DirectRouterMetricsClient`` only emits per-worker
    ``dynamo_frontend_worker_*`` gauges when traffic flows through the
    frontend HTTP service's *in-process* KV router, because
    ``register_worker_load_metrics`` / ``register_worker_timing_metrics``
    are only called from ``service_v2.rs``.  This means a single supported
    deployment topology:

    - **KV-routed aggregated/disagg DGDs**: the frontend runs the KV router
      in-process when started with ``--router-mode kv``; metrics live on
      ``<dgd>-frontend:8000/metrics``.  Detected here only if at least one
      of the target gauges actually appears in the exposition (presence
      proves the routing mode is KV).

    Specifically NOT supported:

    - **Global Planner standalone LocalRouter** (``python3 -m dynamo.router``):
      the LocalRouter's ``system_status_server`` registry is *separate* from
      the frontend HTTP service registry, so the per-worker gauges are
      never registered there even though the global ``WORKER_LOAD_METRICS``
      static is updated by the in-process KvScheduler.  Pool planners must
      use ``PrometheusAPIClient`` (router-source histograms, filtered by
      ``dynamo_namespace``) instead.  See open-question #14 in
      ``powerplanner-design.md``.
    - **Round-robin / random / etc. frontends** never populate the gauges,
      so Prometheus omits the families entirely.

    Discovery checks for *any* of the target gauges, not just one specific
    family.  The ``active_*`` gauges require the worker to emit
    ``ActiveLoad`` events through ``KvWorkerMonitor`` (KV-events plumbing
    on the worker side), while the ``last_*`` gauges populate from
    request-level observations on the router side.  A KV-mode frontend
    talking to a worker that doesn't publish KV events still exposes the
    ``last_*`` family as soon as the first request streams a token, which
    is enough to qualify the endpoint as "KV-router-bearing".
    """
    target_metrics = list(_WORKER_METRIC_NAMES.values())

    def _fetch(url: str) -> Optional[str]:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                return resp.read().decode("utf-8")
        except Exception:
            return None

    def _any_gauge_present(text: str) -> bool:
        return any(name in text for name in target_metrics)

    frontend_url = f"http://{_DGD_NAME}-frontend:8000/metrics"
    text = _fetch(frontend_url)
    if text and _any_gauge_present(text):
        return frontend_url

    return None


class TestDirectRouterMetricsClientLive:
    """DirectRouterMetricsClient against a live KV-router /metrics endpoint.

    The class targets the gauges populated by ``KvWorkerMonitor`` and
    ``observe_*_gauges`` in ``lib/llm`` — both KV-router code paths.  Only
    deployments that run an in-process KV router on the frontend HTTP
    service expose these gauges (Global-Planner LocalRouter pods do *not*,
    despite running KV routing — see ``_discover_kv_router_metrics_url``
    docstring).  We discover the endpoint at runtime and skip cleanly when
    none exists, rather than asserting against a hardcoded URL that
    doesn't reflect the live topology.
    """

    @pytest.fixture(scope="class")
    def metrics_url(self) -> str:
        url = _discover_kv_router_metrics_url()
        if url is None:
            pytest.skip(
                f"DGD {_DGD_NAME!r} has no reachable KV-mode frontend "
                "/metrics endpoint exposing dynamo_frontend_worker_* gauges. "
                "DirectRouterMetricsClient applies only to frontends started "
                "with --router-mode kv (the gauges are registered exclusively "
                "by lib/llm/src/http/service/service_v2.rs). Standalone "
                "LocalRouter pods do not expose these gauges, and "
                "round-robin/random/etc. frontends never populate them."
            )
        return url

    def test_endpoint_responds_with_prometheus_text(self, metrics_url: str):
        # _discover_kv_router_metrics_url already verified reachability +
        # presence of at least one target gauge; re-fetch here as a smoke
        # check that the endpoint is still up (catches mid-run flakiness).
        try:
            resp = urllib.request.urlopen(metrics_url, timeout=5)
            text = resp.read().decode("utf-8")
        except Exception as exc:
            pytest.skip(f"Discovered KV-router /metrics endpoint flapped: {exc}")
        assert (
            "# HELP" in text or "# TYPE" in text
        ), "Response is not Prometheus text exposition format"

    @pytest.mark.asyncio
    async def test_fetch_and_parse_returns_dict(self, metrics_url: str):
        client = DirectRouterMetricsClient(
            router_metrics_url=metrics_url, dynamo_namespace=_DYN_NAMESPACE
        )
        parsed = await client._fetch_and_parse()
        assert isinstance(parsed, dict)
        # The endpoint was discovered by presence of a target gauge family,
        # so an empty dict here means the parser regressed (e.g. label name
        # changed) — promote to a hard failure instead of a skip so we
        # actually catch that.
        assert parsed, (
            "DirectRouterMetricsClient returned empty dict from an endpoint "
            f"({metrics_url}) that the discovery probe confirmed exposes "
            f"{sorted(_WORKER_METRIC_NAMES.values())}.  Likely a parser "
            "regression (worker_id / worker_type label names changed?)."
        )
        for worker_type, by_id in parsed.items():
            assert isinstance(by_id, dict), worker_type
            for worker_id, metrics in by_id.items():
                assert isinstance(metrics, dict), (worker_type, worker_id)
                for name, value in metrics.items():
                    assert isinstance(value, (int, float)), (name, value)


# ---------------------------------------------------------------------------
# 7. MDC (DynamoWorkerMetadata CR) read path
# ---------------------------------------------------------------------------


class TestMdcReadLive:
    """Verify the planner can read DynamoWorkerMetadata CRs from the live cluster."""

    def test_list_worker_metadata_crs_returns_items(
        self, connector: KubernetesConnector
    ):
        crs = connector._list_worker_metadata_crs()
        assert isinstance(crs, list)
        assert crs, (
            "No DynamoWorkerMetadata CRs found in namespace "
            f"{_K8S_NAMESPACE!r}; the worker may not have registered yet."
        )

    def test_extract_mdc_entries_for_this_dgd(self, connector: KubernetesConnector):
        entries = connector._extract_mdc_entries()
        assert isinstance(entries, list)
        if not entries:
            pytest.skip(
                f"No MDC entries found prefixed with {_DGD_NAME!r}-; check that "
                "the worker pod started cleanly and registered its model card."
            )
        for entry in entries:
            assert entry.card_json is not None
            assert isinstance(entry.card_json, dict)
