# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for lib/sidecar/{vllm,sglang,trtllm}/launch/*.sh (native-gRPC sidecar + engine).

Aggregated for all three sidecars; disaggregated for TensorRT-LLM, which is the
only one whose prefill/decode launcher this suite drives so far.
"""

import dataclasses
import importlib.util
import os
import pathlib

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DynamoPortRange
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload_default
from tests.utils.port_utils import reserved_ports

vllm_sidecar_dir = os.environ.get("VLLM_SIDECAR_DIR") or os.path.join(
    WORKSPACE_DIR, "lib/sidecar/vllm"
)
sglang_sidecar_dir = os.environ.get("SGLANG_SIDECAR_DIR") or os.path.join(
    WORKSPACE_DIR, "lib/sidecar/sglang"
)
trtllm_sidecar_dir = os.environ.get("TRTLLM_SIDECAR_DIR") or os.path.join(
    WORKSPACE_DIR, "lib/sidecar/trtllm"
)


def _trtllm_serves_openengine() -> bool:
    """Is the installed TensorRT-LLM new enough to serve `openengine.v1`?

    `lib/sidecar/trtllm` speaks OpenEngine, which the servicer provides from
    release 1.3.0rc27 onward. `container/context.yaml` pins that release, so
    this passes in the stock container and the case runs; it guards the
    off-container and downgraded-image paths, where
    `trtllm-serve --grpc-protocol openengine` does not exist and the launcher
    cannot come up at all.

    Probing for the servicer module beats comparing version strings: it is the
    thing the sidecar actually needs, and it tracks the pinned image by itself.
    `find_spec` on the parent does not import TensorRT-LLM, so this stays cheap
    and works on a host without a GPU.
    """
    spec = importlib.util.find_spec("tensorrt_llm")
    locations = getattr(spec, "submodule_search_locations", None) if spec else None
    if not locations:
        return False
    return any(
        (pathlib.Path(root) / "grpc" / "openengine" / "servicer.py").is_file()
        for root in locations
    )


TRTLLM_OPENENGINE_SKIP_REASON = (
    "the installed TensorRT-LLM has no OpenEngine servicer; the sidecar needs "
    "release 1.3.0rc27 or newer, which container/context.yaml pins. Seeing this "
    "skip in CI means the pin moved backwards."
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
            # Let the 600s health check report failure before pytest times out.
            pytest.mark.timeout(780),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        # Flush Python output promptly into CI logs.
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
            pytest.mark.timeout(780),
            pytest.mark.pre_merge,
            pytest.mark.skipif(
                not _trtllm_serves_openengine(),
                reason=TRTLLM_OPENENGINE_SKIP_REASON,
            ),
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            # Kept for parity with the dynamo.trtllm scenario in test_trtllm.py,
            # though this sidecar rejects `n > 1` in `validate_request` before
            # the engine ever sees it.
            "TLLM_ALLOW_N_GREEDY_DECODING": "1",
            "PYTHONUNBUFFERED": "1",
        },
        request_payloads=[
            chat_payload_default(),
        ],
    ),
    # The disaggregated launcher runs two engines and two sidecars, so it is
    # the only config here that uses both DYN_SYSTEM_PORT1 and _PORT2 -- which
    # is what `num_system_ports=2` on the test is sized for. A green run proves
    # the prefill worker produced a `KvSessionRef` handoff and the decode worker
    # replayed it: with the handoff broken, the decode leg has no KV to resume
    # and the request fails rather than returning a short answer.
    "trtllm_disaggregated": EngineConfig(
        name="trtllm_disaggregated",
        directory=trtllm_sidecar_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.trtllm,
            # Prefill on GPU 0, decode on GPU 1 -- the launcher's defaults.
            pytest.mark.gpu_2,
            # Two engines load serially before either sidecar can register, so
            # this needs longer than the single-engine configs above.
            pytest.mark.timeout(1200),
            pytest.mark.pre_merge,
            pytest.mark.skipif(
                not _trtllm_serves_openengine(),
                reason=TRTLLM_OPENENGINE_SKIP_REASON,
            ),
        ],
        model="Qwen/Qwen3-0.6B",
        env={
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


@pytest.mark.core
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
    if config.name == "vllm_aggregated":
        with reserved_ports(2, start_port=DynamoPortRange.SERVE.value) as engine_ports:
            run_serve_deployment(
                config,
                request,
                ports=dynamo_dynamic_ports,
                extra_env={
                    "VLLM_RS_HTTP_PORT": str(engine_ports[0]),
                    "VLLM_GRPC_PORT": str(engine_ports[1]),
                },
            )
    else:
        run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
