# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for lib/sidecar/{vllm,sglang,trtllm}/launch/agg.sh (native-gRPC sidecar + engine)."""

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
            # EngineConfig.timeout defaults to 600s for the internal
            # health-check loop (tests/utils/engine_process.py); that loop's
            # own timeout path logs a clean, detailed failure (attempt count,
            # last failure reason, log tail). A pytest.mark.timeout with too
            # little margin over 600s + setup overhead (etcd/nats/process
            # launch, observed ~70s here) lets pytest-timeout's blunt global
            # signal fire first mid-loop, discarding that diagnostic path —
            # exactly what happened at 610s in run 34552442779/103122249777.
            pytest.mark.timeout(780),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        # Piped (non-tty) stdout is block-buffered by default, so without this
        # a hung/slow launch shows literally nothing in CI logs until the
        # process is killed.
        env={"PYTHONUNBUFFERED": "1"},
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
            # See vllm_aggregated above: needs margin over EngineConfig's
            # 600s internal health-check timeout, not just over historically
            # observed run time, so a real timeout logs its own diagnostics
            # instead of being cut off by pytest-timeout first.
            pytest.mark.timeout(780),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={"PYTHONUNBUFFERED": "1"},
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
            # See vllm_aggregated above re: margin over EngineConfig's 600s
            # internal timeout. Observed ~173s in CI in practice.
            pytest.mark.timeout(780),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            # TRT-LLM blocks greedy n>1 by default; matches the guard already
            # enabled for the equivalent dynamo.trtllm scenario in test_trtllm.py.
            "TLLM_ALLOW_N_GREEDY_DECODING": "1",
            "PYTHONUNBUFFERED": "1",
        },
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
