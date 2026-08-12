# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check that frontend metrics name the namespace that served a request.

One frontend, two aggregated vLLM deployments serving the *same* model name in
two different namespaces, one GPU each. The frontend spans both via
``--namespace-prefix`` and picks a namespace per request by weighted random
choice, so ``dynamo_namespace`` on its response metrics is the only thing that
says which deployment did the work.

This closes the seam the unit tests leave open: selection attribution
(``lib/llm/src/discovery/model.rs``) and label independence
(``lib/llm/src/http/service/metrics.rs``) are each covered in isolation, but
nothing else drives real requests through a real multi-namespace frontend and
reads the scraped exposition.
"""

import json
import logging
import os
import shutil
import time

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
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

# Written unconditionally by ResponseMetricCollector::observe_response on the
# first token-bearing chunk, so every successful request contributes.
ALWAYS_OBSERVED_METRICS = [
    "dynamo_frontend_output_tokens_total",
    "dynamo_frontend_time_to_first_token_seconds_count",
    "dynamo_frontend_input_sequence_tokens_count",
]

# Written from different code paths that need more than a first chunk: ITL is
# resolved lazily on the second observation, OSL in the collector's Drop and only
# when an LLMMetricAnnotation carried an output-token count. Streaming with
# several tokens makes both overwhelmingly likely, but the test does not fail if
# a backend skips them -- it fails if they are present and *not* split.
CONDITIONALLY_OBSERVED_METRICS = [
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

# TODO: add @pytest.mark.profiled_vram_gib(N) and
# @pytest.mark.requested_vllm_kv_cache_bytes(N) once this has been profiled on a
# GPU host:
#   python tests/utils/profile_pytest.py \
#     tests/frontend/test_namespace_metrics_attribution.py -xvs
# Until then `--max-vram-gib` filtering deselects it as unsized. See
# .ai/test-model-size-guardrails.md.
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_2,
    pytest.mark.post_merge,
    pytest.mark.model(MODEL),
]


class AggWorkerProcess(ManagedProcess):
    """Aggregated vLLM worker pinned to one GPU inside one namespace."""

    def __init__(self, request, *, namespace: str, system_port: int, gpu_id: int):
        self.namespace = namespace
        self.system_port = int(system_port)

        # Honour the GPU-parallel scheduler's KV-cache budget when it sets one,
        # so this test can be profiled and scheduled alongside others. Falls back
        # to a fixed fraction, which leaves headroom on a shared node.
        mem_args = build_gpu_mem_args("build_vllm_gpu_mem_args") or [
            "--gpu-memory-utilization",
            "0.35",
        ]

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            MODEL,
            "--enforce-eager",
            "--max-model-len",
            "4096",
            *mem_args,
        ]

        env = os.environ.copy()
        env["DYN_NAMESPACE"] = namespace
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        # Canary probes issue their own generate calls; they never reach the
        # frontend's HTTP metrics, but disabling them keeps request counts exact.
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        # One GPU each, so the pair fits the gpu_2 lane.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        log_dir = f"{request.node.name}_worker_{namespace}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{self.system_port}/health", self._is_ready)
            ],
            timeout=900,
            display_output=False,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

    @staticmethod
    def _is_ready(response) -> bool:
        try:
            return (response.json() or {}).get("status") == "ready"
        except ValueError:
            return False


def _wait_for_both_namespaces_ready(frontend_port: int, timeout_s: int = 300) -> None:
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

    Streaming guarantees the response arrives as several chunks, which is what
    makes inter-token latency observable.
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
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """Frontend response metrics name the namespace that served each request,
    per-namespace series stay independent, and they still aggregate to the total.
    """
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_ports = dynamo_dynamic_ports.system_ports

    frontend = DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_args=["--namespace-prefix", NAMESPACE_PREFIX],
    )
    with frontend:
        worker_a = AggWorkerProcess(
            request, namespace=NAMESPACE_A, system_port=system_ports[0], gpu_id=0
        )
        worker_b = AggWorkerProcess(
            request, namespace=NAMESPACE_B, system_port=system_ports[1], gpu_id=1
        )
        with worker_a, worker_b:
            _wait_for_both_namespaces_ready(frontend_port)

            # Enough requests that weighted-random selection reaches both
            # namespaces with overwhelming probability (~2^-40 to miss one).
            for i in range(NUM_REQUESTS):
                _stream_completion(frontend_port, i)

            body = requests.get(
                f"http://localhost:{frontend_port}/metrics", timeout=30
            ).text

            # 1. Every always-observed response family names both deployments.
            #    A regression that pins one namespace, or drops the label, fails
            #    here.
            for metric in ALWAYS_OBSERVED_METRICS:
                assert _namespaces_for(body, metric) == NAMESPACES, (
                    f"{metric} should be split across both namespaces, got "
                    f"{sorted(_namespaces_for(body, metric))}"
                )

            # 2. Families written from the lazy and Drop paths must be split too
            #    when they were observed at all.
            for metric in CONDITIONALLY_OBSERVED_METRICS:
                seen = _namespaces_for(body, metric)
                if not seen:
                    logger.warning(
                        "%s was never observed; skipping split check", metric
                    )
                    continue
                assert seen == NAMESPACES, (
                    f"{metric} was observed but is not split across both "
                    f"namespaces, got {sorted(seen)}"
                )

            # 3. Aggregation is lossless: per-namespace counts add back up to the
            #    request total, and neither namespace is starved (which would
            #    make assertion 1 pass on a technicality).
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

            # 4. Per-deployment config gauges keep one series per namespace
            #    instead of the last card overwriting the first.
            assert (
                _namespaces_for(body, "dynamo_frontend_model_context_length")
                == NAMESPACES
            ), "model config gauges should keep one series per namespace"

            # 5. The documented gap stays a gap. If this fails the namespace was
            #    added to InflightGuard -- update the aggregation guidance in
            #    docs/fern/pages/reference/observability/metric-labels.mdx too.
            for metric in NAMESPACE_FREE_METRICS:
                for labels in metric_label_sets(body, metric):
                    assert "dynamo_namespace" not in labels, (
                        f"{metric} gained a dynamo_namespace label; that gap is "
                        "deliberate (the guard predates worker-set selection)"
                    )
