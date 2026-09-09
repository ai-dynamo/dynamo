# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for lib/sidecar/{vllm,sglang,trtllm}/launch/agg.sh (native-gRPC sidecar + engine).

TRT-LLM's agg.sh pip-installs smg-grpc-proto at runtime if the test image does
not already carry it (see lib/sidecar/trtllm/launch/agg.sh). The 1.3.0rc25
runtime image does not bundle it, so lib/sidecar/ci/sidecar-test-image.Dockerfile
pre-bakes it into the sidecar test image to keep this suite hermetic and
avoid a live network fetch on every run.
"""

import dataclasses
import os

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload_default

vllm_sidecar_dir = os.environ.get("VLLM_SIDECAR_DIR") or os.path.join(
    WORKSPACE_DIR, "lib/sidecar/vllm"
)
sglang_sidecar_dir = os.environ.get("SGLANG_SIDECAR_DIR") or os.path.join(
    WORKSPACE_DIR, "lib/sidecar/sglang"
)
trtllm_sidecar_dir = os.environ.get("TRTLLM_SIDECAR_DIR") or os.path.join(
    WORKSPACE_DIR, "lib/sidecar/trtllm"
)


# Sequential stage only: no profiled_vram_gib mark yet, since actual peak VRAM
# has not been profiled for the sidecar launch path. Add one once measured, to
# admit these into the parallel stage alongside the equivalent dynamo.{backend}
# scenarios in tests/serve/test_{vllm,sglang,trtllm}.py.
sidecar_configs = {
    "vllm_aggregated": EngineConfig(
        name="vllm_aggregated",
        directory=vllm_sidecar_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.vllm,
            pytest.mark.gpu_1,
            # First-run wiring smoke: adjust once CI has measured actual duration.
            pytest.mark.timeout(610),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
        ],
    ),
    "sglang_aggregated": EngineConfig(
        name="sglang_aggregated",
        directory=sglang_sidecar_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.sglang,
            pytest.mark.gpu_1,
            # First-run wiring smoke: adjust once CI has measured actual duration.
            pytest.mark.timeout(360),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[
            chat_payload_default(),
        ],
    ),
    "trtllm_aggregated": EngineConfig(
        name="trtllm_aggregated",
        directory=trtllm_sidecar_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.trtllm,
            pytest.mark.gpu_1,
            # First-run wiring smoke: adjust once CI has measured actual duration.
            pytest.mark.timeout(650),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        # TRT-LLM blocks greedy n>1 by default; matches the guard already
        # enabled for the equivalent dynamo.trtllm scenario in test_trtllm.py.
        env={"TLLM_ALLOW_N_GREEDY_DECODING": "1"},
        request_payloads=[
            chat_payload_default(),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(sidecar_configs))
def sidecar_config_test(request):
    """Fixture that provides different sidecar test configurations"""
    return sidecar_configs[request.param]


@pytest.mark.sidecar
@pytest.mark.e2e
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_serve_deployment(
    sidecar_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """
    Launch a lib/sidecar/<backend>/launch/agg.sh script end-to-end (Dynamo
    frontend + native-gRPC engine + dynamo-<backend>-sidecar) and confirm it
    serves a real chat completion.
    """
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    config = dataclasses.replace(
        sidecar_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
