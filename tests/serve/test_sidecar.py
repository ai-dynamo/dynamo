# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for native-gRPC sidecar launch scripts."""

import dataclasses
import os

import pytest

from tests.router.common import _test_frontend_kv_routing
from tests.router.helper import generate_random_suffix
from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DynamoPortRange
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import LONG_PROMPT_FOR_CACHING, chat_payload_default
from tests.utils.payloads import ChatPayload, DisaggregatedChatPayload
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


def _sidecar_worker_gpu_env(
    backend: str, roles: tuple[str, str] = ("WORKER1", "WORKER2")
) -> dict[str, str]:
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    env = {}
    for index, role in enumerate(roles):
        key = f"{backend.upper()}_{role}_GPU"
        device = os.environ.get(key)
        if device is None:
            assert len(devices) > index and devices[index].strip() not in (
                "",
                "-1",
            ), f"Two visible GPUs are required; CUDA_VISIBLE_DEVICES={devices}"
            device = devices[index].strip()
        env[key] = device
    return env


def _disaggregated_chat_payload() -> DisaggregatedChatPayload:
    return DisaggregatedChatPayload(
        body={
            "messages": [{"role": "user", "content": LONG_PROMPT_FOR_CACHING}],
            "max_tokens": 64,
            "n": 1,
            "temperature": 0,
            "stream": False,
            "nvext": {"extra_fields": ["worker_id"]},
        },
        repeat_count=1,
        expected_response=[],
        expected_log=[],
        expected_num_choices=1,
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
    # Prefill/decode handoff is a critical native-sidecar path.
    "vllm_disaggregated": EngineConfig(
        name="vllm_disaggregated",
        directory=vllm_sidecar_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.vllm,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
            pytest.mark.requested_vllm_kv_cache_bytes(1119388000),
        ],
        model="Qwen/Qwen3-0.6B",
        health_check_workers=True,
        health_check_worker_count=2,
        env={"PYTHONUNBUFFERED": "1", "MAX_MODEL_LEN": "2048"},
        request_payloads=[_disaggregated_chat_payload()],
    ),
    "sglang_disaggregated": EngineConfig(
        name="sglang_disaggregated",
        directory=sglang_sidecar_dir,
        script_name="disagg.sh",
        script_args=["--disable-cuda-graph"],
        marks=[
            pytest.mark.sglang,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
            pytest.mark.requested_sglang_kv_tokens(2048),
        ],
        model="Qwen/Qwen3-0.6B",
        health_check_workers=True,
        health_check_worker_count=2,
        env={"PYTHONUNBUFFERED": "1", "MAX_MODEL_LEN": "2048"},
        request_payloads=[_disaggregated_chat_payload()],
    ),
    "trtllm_disaggregated": EngineConfig(
        name="trtllm_disaggregated",
        directory=trtllm_sidecar_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.trtllm,
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
            pytest.mark.requested_trtllm_kv_tokens(2048),
        ],
        model="Qwen/Qwen3-0.6B",
        health_check_workers=True,
        health_check_worker_count=2,
        env={"PYTHONUNBUFFERED": "1", "TRTLLM_CONTEXT_LENGTH": "2048"},
        request_payloads=[_disaggregated_chat_payload()],
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
    monkeypatch,
):
    """Launch a native engine and sidecar deployment and validate chat completion."""
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    config = dataclasses.replace(
        sidecar_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    if config.name.endswith("_disaggregated"):
        monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)
        monkeypatch.setenv("DYN_REQUEST_PLANE", "tcp")
        backend = config.name.removesuffix("_disaggregated")
        roles = ("DECODE", "PREFILL") if backend == "vllm" else ("PREFILL", "DECODE")
        engine_env = {
            **_sidecar_worker_gpu_env(backend, roles),
            "DYN_NAMESPACE": f"sidecar-disagg-{generate_random_suffix()}",
            "MODEL": config.model,
        }
        num_engine_ports = {"vllm": 4, "sglang": 5, "trtllm": 2}[backend]
        with reserved_ports(
            num_engine_ports, start_port=DynamoPortRange.SERVE.value
        ) as engine_ports:
            for index, role in enumerate(roles):
                prefix = f"{backend.upper()}_{role}"
                if backend == "trtllm":
                    engine_env[f"{prefix}_GRPC_PORT"] = str(engine_ports[index])
                else:
                    engine_env[f"{prefix}_HTTP_PORT"] = str(engine_ports[index * 2])
                    engine_env[f"{prefix}_GRPC_PORT"] = str(engine_ports[index * 2 + 1])
                if backend == "vllm":
                    engine_env[f"{prefix}_NIXL_SIDE_CHANNEL_PORT"] = str(
                        dynamo_dynamic_ports.nixl_side_channel_ports[index]
                    )
            if backend == "vllm":
                engine_env["VLLM_PREFILL_KV_EVENT_PORT"] = str(
                    dynamo_dynamic_ports.kv_event_ports[1]
                )
            elif backend == "sglang":
                engine_env["SGLANG_DISAGGREGATION_BOOTSTRAP_PORT"] = str(
                    engine_ports[4]
                )
            run_serve_deployment(
                config, request, ports=dynamo_dynamic_ports, extra_env=engine_env
            )
    elif config.name == "vllm_aggregated":
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


@pytest.mark.router
@pytest.mark.sidecar
@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.pre_merge  # Guard native KV-event discovery on every sidecar change.
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.timeout(780)
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "vllm",
            marks=[
                pytest.mark.vllm,
                pytest.mark.requested_vllm_kv_cache_bytes(1119388000),
            ],
        ),
        pytest.param(
            "sglang",
            marks=[pytest.mark.sglang, pytest.mark.requested_sglang_kv_tokens(2048)],
        ),
    ],
)
def test_sidecar_kv_routing(
    backend,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    monkeypatch,
):
    monkeypatch.delenv("DYN_ROUTER_PREDICTED_TTL_SECS", raising=False)
    monkeypatch.delenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", raising=False)
    monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)
    namespace = f"sidecar-kv-{generate_random_suffix()}"
    block_size = 64
    config = EngineConfig(
        name=f"{backend}_kv_routing",
        directory=vllm_sidecar_dir if backend == "vllm" else sglang_sidecar_dir,
        script_name="agg_kv_router.sh",
        script_args=["--disable-cuda-graph"] if backend == "sglang" else [],
        marks=[],
        model="Qwen/Qwen3-0.6B",
        health_check_workers=True,
        health_check_worker_count=2,
        request_payloads=[
            ChatPayload(
                body={
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 1,
                    "temperature": 0,
                },
                expected_response=[],
                expected_log=[],
            )
        ],
        env={
            "PYTHONUNBUFFERED": "1",
            "DYN_NAMESPACE": namespace,
            "DYN_COMPONENT": "backend",
            "DYN_ENDPOINT": "generate",
            "DYN_ROUTER_USE_KV_EVENTS": "true",
            "DYN_ROUTER_TEMPERATURE": "0",
            "DYN_ROUTER_MIN_INITIAL_WORKERS": "2",
            "DYN_REQUEST_PLANE": "tcp",
            "MAX_MODEL_LEN": "2048",
            "VLLM_BLOCK_SIZE": str(block_size),
            "SGLANG_PAGE_SIZE": str(block_size),
        },
    )
    with reserved_ports(4, start_port=DynamoPortRange.SERVE.value) as engine_ports:
        engine_env = _sidecar_worker_gpu_env(backend)
        for worker_index in range(2):
            prefix = f"{backend.upper()}_WORKER{worker_index + 1}"
            engine_env[f"{prefix}_HTTP_PORT"] = str(engine_ports[worker_index * 2])
            engine_env[f"{prefix}_GRPC_PORT"] = str(engine_ports[worker_index * 2 + 1])
            engine_env[f"{prefix}_KV_EVENT_PORT"] = str(
                dynamo_dynamic_ports.kv_event_ports[worker_index]
            )
        run_serve_deployment(
            config,
            request,
            ports=dynamo_dynamic_ports,
            extra_env=engine_env,
            post_validation=lambda: _test_frontend_kv_routing(
                frontend_port=dynamo_dynamic_ports.frontend_port,
                system_ports=dynamo_dynamic_ports.system_ports,
                namespace=namespace,
                model_name=config.model,
                block_size=block_size,
            ),
        )
