# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image-aware KV routing between two TensorRT-LLM sidecar workers over HTTP.

`lib/sidecar/trtllm/launch/agg_kv_router.sh` starts a `--router-mode kv`
frontend and two aggregated OpenEngine engines, each behind its own sidecar.
The test caches one image on each worker with pinned requests, then sends
unpinned requests and checks the worker the router picks:

- same-image reuse: an image goes back to the worker that cached it, and that
  worker reports cached prompt tokens;
- different-image isolation: the other worker's image does not match the
  first worker's cache, and an image no worker has seen gets no image-block
  overlap.

The image prefixes the prompt, so every complete block's hash depends on the
image hash: text-only routing would score both workers the same.
"""

from __future__ import annotations

import os
import time

import pytest
import requests

from lib.sidecar.trtllm.ci.run_image_test import routing_fixture
from tests.router.helper import generate_random_suffix
from tests.serve.common import managed_serve_deployment
from tests.serve.test_sidecar import (
    KV_ROUTING_ENV,
    TRTLLM_MULTI_ENGINE_ENV,
    TRTLLM_OPENENGINE_SKIP_REASON,
    _trtllm_serves_openengine,
    trtllm_sidecar_dir,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.gpu_args import map_cuda_visible_devices
from tests.utils.router_nvext import require_router_worker_id

TRTLLM_MM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

pytestmark = [
    pytest.mark.trtllm,
    pytest.mark.sidecar,
    pytest.mark.multimodal,
    pytest.mark.e2e,
    # Both engines share one GPU, as in `trtllm_kv_router` (tests/serve/test_sidecar.py).
    pytest.mark.gpu_1,
    pytest.mark.post_merge,
    pytest.mark.model(TRTLLM_MM_MODEL),
    # Measured nvidia-smi peak with both engines loaded.
    pytest.mark.profiled_vram_gib(21.0),
    pytest.mark.requested_trtllm_kv_tokens(2048),
    # Two engines load serially before both sidecars register.
    pytest.mark.timeout(1200),
    pytest.mark.engine_ports,
    pytest.mark.skipif(
        not _trtllm_serves_openengine(), reason=TRTLLM_OPENENGINE_SKIP_REASON
    ),
]

# The native routing test's text: complete blocks after the image.
_PROMPT = (
    "Inspect the image carefully and answer only with the two colors in order. " * 8
    + "Name the colors from left to right."
)
# A warm image prefix covers most of the prompt; a cold one covers none of it.
_WARM_HIT_RATE = 0.5


def _generate_instances(frontend_port: int, namespace: str) -> list[int]:
    """The two `generate` instance IDs the frontend has discovered."""
    deadline = time.monotonic() + 60
    instances: list[int] = []
    while time.monotonic() < deadline:
        response = requests.get(
            f"http://localhost:{frontend_port}/health?instances=true", timeout=5
        )
        if response.status_code == 200:
            instances = sorted(
                {
                    instance["instance_id"]
                    for instance in response.json()["instances"]
                    if instance["namespace"] == namespace
                    and instance["endpoint"] == "generate"
                }
            )
            if len(instances) == 2:
                return instances
        time.sleep(0.25)
    raise AssertionError(f"expected two generate instances, got {instances}")


def _send(
    frontend_port: int, image_uri: str, label: str, worker_id: int | None = None
) -> tuple[int, float, int]:
    """Send one image request; return the chosen worker, its router hit rate and
    the engine's cached prompt tokens."""
    headers = (
        {"x-dynamo-worker-instance-id": str(worker_id)}
        if worker_id is not None
        else None
    )
    response = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json={
            "model": TRTLLM_MM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
            "max_tokens": 8,
            "temperature": 0,
            "nvext": {"extra_fields": ["worker_id", "timing"]},
        },
        headers=headers,
        timeout=120,
    )
    assert response.status_code == 200, f"{label}: {response.text}"
    data = response.json()
    selected = require_router_worker_id(data, context=label)["decode_worker_id"]
    hit_rate = data["nvext"].get("timing", {}).get("kv_hit_rate")
    assert isinstance(hit_rate, (int, float)), f"{label}: {data['nvext']}"
    usage = data.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    print(
        f"[TRTLLM_MM_ROUTER] {label}: worker_id={selected} kv_hit_rate={hit_rate:.3f} "
        f"cached_tokens={cached} prompt_tokens={usage.get('prompt_tokens')}"
    )
    return selected, hit_rate, cached


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_trtllm_sidecar_image_cache_selection_across_two_workers(
    request,
    monkeypatch,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    namespace = f"trtllm-mm-router-{generate_random_suffix()}"
    monkeypatch.setenv("MODEL", TRTLLM_MM_MODEL)
    monkeypatch.setenv("DYN_NAMESPACE", namespace)
    gpu = map_cuda_visible_devices((0,), os.environ.get("CUDA_VISIBLE_DEVICES"))
    monkeypatch.setenv("TRTLLM_WORKER1_GPU", gpu)
    monkeypatch.setenv("TRTLLM_WORKER2_GPU", gpu)

    config = EngineConfig(
        name="trtllm_mm_kv_router",
        directory=trtllm_sidecar_dir,
        script_name="agg_kv_router.sh",
        marks=[],
        request_payloads=[],
        model=TRTLLM_MM_MODEL,
        timeout=1000,
        health_check_workers=True,
        health_check_worker_count=2,
        env={**KV_ROUTING_ENV, **TRTLLM_MULTI_ENGINE_ENV},
    )
    image_x, image_y, image_alias = routing_fixture()
    frontend_port = dynamo_dynamic_ports.frontend_port

    with managed_serve_deployment(config, request, ports=dynamo_dynamic_ports):
        worker_a, worker_b = _generate_instances(frontend_port, namespace)
        owners = {image_x: worker_a, image_y: worker_b}

        for image, owner in owners.items():
            selected, _, _ = _send(frontend_port, image, "warm", owner)
            assert selected == owner, (selected, owner)

        # Wait for each worker's Stored events to reach the router: pinned
        # repeats see their image prefix once the router has indexed it.
        deadline = time.monotonic() + 60
        hit_rates: list[float] = []
        while True:
            hit_rates = [
                _send(frontend_port, image, "converge", owner)[1]
                for image, owner in owners.items()
            ]
            if all(rate >= _WARM_HIT_RATE for rate in hit_rates):
                break
            assert (
                time.monotonic() < deadline
            ), f"router did not index the cached images: kv_hit_rate={hit_rates}"
            time.sleep(0.5)

        x_hit_rate, x_cached = 0.0, 0
        for image in (image_x, image_x, image_y, image_x, image_y, image_y):
            label = "same-image X" if image == image_x else "different-image Y"
            selected, hit_rate, cached = _send(frontend_port, image, label)
            assert selected == owners[image], (
                f"{label}: routed to {selected}, cached on {owners[image]}; "
                f"workers A={worker_a} B={worker_b}"
            )
            assert hit_rate >= _WARM_HIT_RATE, (label, hit_rate)
            assert cached > 0, f"{label}: the engine reported no cached tokens"
            if image == image_x:
                x_hit_rate, x_cached = hit_rate, cached

        # Same pixels, new URI: no worker holds its image blocks.
        _, alias_hit_rate, alias_cached = _send(frontend_port, image_alias, "alias")
        assert alias_hit_rate < x_hit_rate, (alias_hit_rate, x_hit_rate)
        assert alias_cached < x_cached, (alias_cached, x_cached)
