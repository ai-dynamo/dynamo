# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check that frontend metrics name the namespace that served a request.

One frontend, two aggregated deployments serving the *same* model name in two
different namespaces. The frontend spans both via ``--namespace-prefix`` and
picks a namespace per request by weighted random choice, so ``dynamo_namespace``
on its response metrics is the only thing that says which deployment did the
work.

This closes the seam the Rust unit tests leave open: selection attribution
(``lib/llm/src/discovery/model.rs``) and metric-label independence
(``lib/llm/src/http/service/metrics.rs``) are each covered in isolation, but
nothing else drives real requests through a real multi-namespace frontend and
reads the scraped exposition.

Uses ``dynamo.mocker`` rather than a GPU backend. Everything under test is
frontend-side -- worker-set selection and label plumbing -- and the workers only
need to be genuine registered deployments in distinct namespaces, which the
mocker is. That keeps this on the gpu_0 lane and lets every assertion below be
strict.
"""

import json
import logging
import time

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.prometheus import metric_label_sets, sum_metric_samples

logger = logging.getLogger(__name__)

MODEL = QWEN
# Shared prefix so one frontend discovers both deployments; the prefix itself is
# never a namespace.
NAMESPACE_PREFIX = "nsmetrics"
NAMESPACE_A = f"{NAMESPACE_PREFIX}-a"
NAMESPACE_B = f"{NAMESPACE_PREFIX}-b"
NAMESPACES = {NAMESPACE_A, NAMESPACE_B}

NUM_REQUESTS = 40
MAX_TOKENS = 8

# Frontend response families that must name the namespace that served the
# request. These cover three distinct write paths: handles resolved in
# ResponseMetricCollector::new, the lazily-resolved inter-token-latency handle,
# and the output-sequence-length write in the collector's Drop.
NAMESPACED_RESPONSE_METRICS = [
    "dynamo_frontend_output_tokens_total",
    "dynamo_frontend_time_to_first_token_seconds_count",
    "dynamo_frontend_input_sequence_tokens_count",
    "dynamo_frontend_inter_token_latency_seconds_count",
    "dynamo_frontend_output_sequence_tokens_count",
]

# Deliberately namespace-free: recorded by InflightGuard, which is built before a
# worker set is selected so failures during model resolution are still counted.
# Pinned so closing that gap has to be a deliberate change.
NAMESPACE_FREE_METRICS = [
    "dynamo_frontend_requests_total",
    "dynamo_frontend_inflight_requests",
]

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.model(MODEL),
    # ~12s locally across four runs. Bounded so a hung frontend, a worker that
    # never reaches ready, or a stalled stream fails the job instead of holding a
    # CI slot until the suite-level timeout.
    pytest.mark.timeout(180),
]


def _wait_for_both_namespaces_ready(frontend_port: int, timeout_s: int = 180) -> None:
    """Block until the frontend reports both deployments servable.

    Uses ``GET /v1/models/{model}/ready``, which reports readiness per namespace
    from the same evaluation the serving gate uses. Gating on ``/v1/models``
    instead would let the test start while only one worker set was registered,
    so every request would land on one namespace and the test would pass without
    ever exercising cross-namespace selection.
    """
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            response = requests.get(
                f"http://localhost:{frontend_port}/v1/models/{MODEL}/ready", timeout=30
            )
            if response.status_code == 200:
                last = response.json()
                ready = {
                    name
                    for name, detail in (last.get("namespaces") or {}).items()
                    if detail.get("ready")
                }
                if NAMESPACES.issubset(ready):
                    return
        except (requests.RequestException, ValueError) as exc:
            last = repr(exc)
        time.sleep(2)

    raise AssertionError(
        f"both namespaces should be ready within {timeout_s}s; "
        f"last /ready payload: {json.dumps(last, default=str)}"
    )


def _stream_completion(frontend_port: int, index: int) -> None:
    """Send one streaming completion and drain it.

    Streaming means the response arrives as several chunks, which is what makes
    inter-token latency observable.
    """
    with requests.post(
        f"http://localhost:{frontend_port}/v1/completions",
        json={
            "model": MODEL,
            "prompt": f"namespace attribution probe {index}",
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
        },
        stream=True,
        timeout=120,
    ) as response:
        assert response.status_code == 200, response.text
        for _ in response.iter_lines():
            pass


def _namespaces_for(body: str, metric: str) -> set[str]:
    return {
        labels["dynamo_namespace"]
        for labels in metric_label_sets(body, metric)
        if labels.get("model") == MODEL and "dynamo_namespace" in labels
    }


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_frontend_metrics_attribute_requests_to_serving_namespace(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """Frontend response metrics name the namespace that served each request,
    per-namespace series stay independent, and they still aggregate to the total.
    """
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_ports = dynamo_dynamic_ports.system_ports

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--namespace-prefix", NAMESPACE_PREFIX],
    ):
        with MockerWorkerProcess(
            request,
            model=MODEL,
            frontend_port=frontend_port,
            system_port=system_ports[0],
            worker_id=f"mocker-{NAMESPACE_A}",
            extra_env={"DYN_NAMESPACE": NAMESPACE_A},
        ), MockerWorkerProcess(
            request,
            model=MODEL,
            frontend_port=frontend_port,
            system_port=system_ports[1],
            worker_id=f"mocker-{NAMESPACE_B}",
            extra_env={"DYN_NAMESPACE": NAMESPACE_B},
        ):
            _wait_for_both_namespaces_ready(frontend_port)

            # Enough requests that weighted-random selection reaches both
            # namespaces with overwhelming probability (~2^-40 to miss one).
            for i in range(NUM_REQUESTS):
                _stream_completion(frontend_port, i)

            body = requests.get(
                f"http://localhost:{frontend_port}/metrics", timeout=30
            ).text

            # 1. Every response family names both deployments. A regression that
            #    pins one namespace, or drops the label, fails here.
            for metric in NAMESPACED_RESPONSE_METRICS:
                assert _namespaces_for(body, metric) == NAMESPACES, (
                    f"{metric} should be split across both namespaces, got "
                    f"{sorted(_namespaces_for(body, metric))}"
                )

            # 2. Aggregation is lossless: per-namespace counts add back up to the
            #    request total, and neither namespace is starved (which would let
            #    assertion 1 pass on a technicality).
            per_namespace = {
                ns: sum_metric_samples(
                    body,
                    "dynamo_frontend_time_to_first_token_seconds_count",
                    {"model": MODEL, "dynamo_namespace": ns},
                )
                for ns in NAMESPACES
            }
            assert all(
                count > 0 for count in per_namespace.values()
            ), f"both namespaces should have served traffic, got {per_namespace}"
            assert sum(per_namespace.values()) == NUM_REQUESTS, (
                f"per-namespace TTFT counts should sum to {NUM_REQUESTS}, "
                f"got {per_namespace}"
            )

            successes = sum_metric_samples(
                body,
                "dynamo_frontend_requests_total",
                {"model": MODEL, "endpoint": "completions", "status": "success"},
            )
            assert successes == NUM_REQUESTS, (
                f"frontend should have counted {NUM_REQUESTS} successful "
                f"completions, got {successes}"
            )

            # 3. Per-deployment config gauges keep one series per namespace
            #    instead of the last card overwriting the first.
            assert (
                _namespaces_for(body, "dynamo_frontend_model_context_length")
                == NAMESPACES
            ), "model config gauges should keep one series per namespace"

            # 4. The documented gap stays a gap. If this fails the namespace was
            #    added to InflightGuard -- update the aggregation guidance in
            #    docs/fern/pages/reference/observability/metric-labels.mdx too.
            for metric in NAMESPACE_FREE_METRICS:
                for labels in metric_label_sets(body, metric):
                    assert "dynamo_namespace" not in labels, (
                        f"{metric} gained a dynamo_namespace label; that gap is "
                        "deliberate (the guard predates worker-set selection)"
                    )
