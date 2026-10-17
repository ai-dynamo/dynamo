# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E coverage for native-gRPC sidecar launch scripts."""

import importlib.util
import json
import os
import pathlib

import pytest

from tests.router.common import _test_frontend_kv_routing
from tests.router.helper import generate_random_suffix
from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DynamoPortRange
from tests.utils.engine_metrics import (
    EngineMetrics,
    SGLangMetricsChecker,
    VllmMetricsChecker,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.gpu_args import map_cuda_visible_devices
from tests.utils.payload_builder import (
    chat_logprobs_and_structured_output_payloads,
    disaggregated_token_count_payload,
    http_cancellation_payloads,
)
from tests.utils.payloads import (
    ChatPayload,
    KvTransferPayload,
    OmittedMaxTokensChatPayload,
)
from tests.utils.port_utils import allocate_contiguous_ports, deallocate_ports

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


CANCELLATION_MAX_TOKENS = 2048
CANCELLATION_CONTEXT_LEN = 4096


def _aggregated_payloads(metrics: EngineMetrics):
    return [
        *chat_logprobs_and_structured_output_payloads(),
        *http_cancellation_payloads(metrics, max_tokens=CANCELLATION_MAX_TOKENS),
    ]


def _disaggregated_payloads(
    *,
    prefill_metrics: EngineMetrics,
    decode_metrics: EngineMetrics,
    transfer_metrics: EngineMetrics,
):
    transfer = disaggregated_token_count_payload()
    return [
        disaggregated_token_count_payload(),
        KvTransferPayload(
            body=transfer.body,
            expected_response=[],
            expected_log=[],
            expected_finish_reason=transfer.expected_finish_reason,
            expected_completion_tokens=transfer.expected_completion_tokens,
            prefill_metrics=prefill_metrics,
            decode_metrics=decode_metrics,
            transfer_metrics=transfer_metrics,
        ),
    ]


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
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            "PYTHONUNBUFFERED": "1",
            "DYN_REQUEST_PLANE": "tcp",
            "MAX_MODEL_LEN": str(CANCELLATION_CONTEXT_LEN),
            "VLLM_DATA_PARALLEL_SIZE": "1",
        },
        request_payloads=_aggregated_payloads(
            VllmMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT1"),
        ),
    ),
    "sglang_aggregated": EngineConfig(
        name="sglang_aggregated",
        directory=sglang_sidecar_dir,
        script_name="agg.sh",
        script_args=["--enable-metrics", "--decode-log-interval", "1"],
        marks=[
            pytest.mark.sglang,
            pytest.mark.gpu_1,
            pytest.mark.timeout(780),
            pytest.mark.post_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            "PYTHONUNBUFFERED": "1",
            "DYN_REQUEST_PLANE": "tcp",
            "MAX_MODEL_LEN": str(CANCELLATION_CONTEXT_LEN),
        },
        request_payloads=[
            *_aggregated_payloads(
                SGLangMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT1"),
            ),
            OmittedMaxTokensChatPayload(
                body={
                    "messages": [{"role": "user", "content": "Tell me about rivers."}],
                    "ignore_eos": True,
                },
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "trtllm_aggregated": EngineConfig(
        name="trtllm_aggregated",
        directory=trtllm_sidecar_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.trtllm,
            pytest.mark.gpu_1,
            pytest.mark.requested_trtllm_kv_tokens(4096),
            pytest.mark.timeout(780),
            pytest.mark.post_merge,
            pytest.mark.skipif(
                not _trtllm_serves_openengine(),
                reason=TRTLLM_OPENENGINE_SKIP_REASON,
            ),
        ],
        model="Qwen/Qwen3-0.6B",
        env={
            # TRT-LLM blocks greedy n>1 by default; matches the guard already
            # enabled for the equivalent dynamo.trtllm scenario in test_trtllm.py.
            "TLLM_ALLOW_N_GREEDY_DECODING": "1",
            "PYTHONUNBUFFERED": "1",
        },
        # OpenEngine exposes output logprobs, but not prompt logprobs.
        request_payloads=[
            *chat_logprobs_and_structured_output_payloads(prompt_logprobs=None),
            # Public disconnect/recovery only: no native scheduler observations.
            *http_cancellation_payloads(None, max_tokens=CANCELLATION_MAX_TOKENS),
        ],
    ),
    "trtllm_disaggregated": EngineConfig(
        name="trtllm_disaggregated",
        directory=trtllm_sidecar_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.trtllm,
            # Both engines share one GPU: the handoff is what this covers, and
            # a co-resident pair keeps it on the existing 1-GPU sidecar runner.
            # `requested_trtllm_kv_tokens` caps each engine's KV pool so the
            # second one has memory left to load. The KV cache still moves over
            # the launcher's NIXL default; note that NIXL's shared-memory
            # transport needs more than a 64 MiB `/dev/shm` to bring up a
            # second agent, as `deploy/disagg.yaml` also calls out.
            pytest.mark.gpu_1,
            pytest.mark.requested_trtllm_kv_tokens(2048),
            # Two engines load serially before either sidecar can register, so
            # this needs longer than the single-engine configs above. Raise the
            # readiness budget with it: `pytest.mark.timeout` is only the outer
            # kill timer, while EngineConfig.timeout is what the harness gives
            # the health checks (engine_process.py passes it as their deadline).
            # Leaving that at its 600s default would let the health check fail
            # at the single-engine budget and then idle until the kill timer.
            pytest.mark.timeout(1200),
            pytest.mark.post_merge,
            pytest.mark.skipif(
                not _trtllm_serves_openengine(),
                reason=TRTLLM_OPENENGINE_SKIP_REASON,
            ),
        ],
        model="Qwen/Qwen3-0.6B",
        timeout=1000,
        health_check_workers=True,
        health_check_worker_count=2,
        env={
            "TLLM_ALLOW_N_GREEDY_DECODING": "1",
            "PYTHONUNBUFFERED": "1",
            "DYN_REQUEST_PLANE": "tcp",
            # The TensorRT-LLM release image runs as uid 0 and PRTE refuses
            # to run as root unless told twice; without these the engines'
            # worker spawn fails with `MPI_ERR_UNKNOWN` and neither binds its
            # gRPC port. Same pair, same reason, as `deploy/disagg.yaml`.
            "PRTE_ALLOW_RUN_AS_ROOT": "1",
            "PRTE_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        },
        request_payloads=[disaggregated_token_count_payload(logprobs=True)],
    ),
    "vllm_disaggregated": EngineConfig(
        name="vllm_disaggregated",
        directory=vllm_sidecar_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.vllm,
            pytest.mark.gpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            pytest.mark.requested_vllm_kv_cache_bytes(1119388000),
        ],
        model="Qwen/Qwen3-0.6B",
        health_check_workers=True,
        health_check_worker_count=2,
        env={
            "PYTHONUNBUFFERED": "1",
            "DYN_REQUEST_PLANE": "tcp",
            "MAX_MODEL_LEN": "2048",
        },
        # disagg.sh launches decode as worker 1 and prefill as worker 2.
        request_payloads=_disaggregated_payloads(
            prefill_metrics=VllmMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT2"),
            decode_metrics=VllmMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT1"),
            transfer_metrics=VllmMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT1"),
        ),
    ),
    "sglang_disaggregated": EngineConfig(
        name="sglang_disaggregated",
        directory=sglang_sidecar_dir,
        script_name="disagg.sh",
        script_args=[
            "--disable-cuda-graph",
            "--enable-metrics",
            "--decode-log-interval",
            "1",
        ],
        marks=[
            pytest.mark.sglang,
            pytest.mark.gpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            pytest.mark.requested_sglang_kv_tokens(2048),
        ],
        model="Qwen/Qwen3-0.6B",
        health_check_workers=True,
        health_check_worker_count=2,
        env={
            "PYTHONUNBUFFERED": "1",
            "DYN_REQUEST_PLANE": "tcp",
            "MAX_MODEL_LEN": "2048",
        },
        request_payloads=_disaggregated_payloads(
            prefill_metrics=SGLangMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT1"),
            decode_metrics=SGLangMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT2"),
            transfer_metrics=SGLangMetricsChecker(port_env="DYN_ENGINE_HTTP_PORT1"),
        ),
    ),
}


@pytest.fixture(params=params_with_model_mark(sidecar_configs))
def sidecar_config_test(request, monkeypatch, discovery_backend):
    config = sidecar_configs[request.param]
    monkeypatch.setenv("MODEL", config.model)
    monkeypatch.setenv("DYN_DISCOVERY_BACKEND", discovery_backend)
    monkeypatch.setenv("DYN_NAMESPACE", f"sidecar-{generate_random_suffix()}")
    monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)
    backend, layout = config.name.split("_", 1)
    if layout == "disaggregated":
        device = map_cuda_visible_devices([0], os.environ.get("CUDA_VISIBLE_DEVICES"))
        assert device != "-1", "One visible GPU is required"
        for role in ("PREFILL", "DECODE"):
            monkeypatch.setenv(f"{backend.upper()}_{role}_GPU", device)
    return config


@pytest.mark.core
@pytest.mark.sidecar
@pytest.mark.e2e
@pytest.mark.engine_ports
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_serve_deployment(
    sidecar_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """Launch a native engine and sidecar deployment and validate chat completion."""
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    run_serve_deployment(
        sidecar_config_test,
        request,
        ports=dynamo_dynamic_ports,
    )


@pytest.mark.router
@pytest.mark.sidecar
@pytest.mark.e2e
@pytest.mark.engine_ports
@pytest.mark.post_merge
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.parametrize(
    "dep,num_system_ports,model_name",
    [
        pytest.param(
            False,
            2,
            "Qwen/Qwen3-0.6B",
            id="replicas",
            marks=[pytest.mark.gpu_1, pytest.mark.model("Qwen/Qwen3-0.6B")],
        ),
        pytest.param(
            True,
            1,
            "silence09/DeepSeek-R1-Small-2layers",
            id="dep",
            marks=[
                pytest.mark.gpu_2,
                pytest.mark.model("silence09/DeepSeek-R1-Small-2layers"),
            ],
        ),
    ],
    indirect=["num_system_ports"],
)
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
    dep,
    model_name,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    monkeypatch,
):
    """Verify native sidecar KV events route requests to the cached worker or rank."""
    monkeypatch.delenv("DYN_ROUTER_PREDICTED_TTL_SECS", raising=False)
    monkeypatch.delenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", raising=False)
    monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)
    namespace = f"sidecar-kv-{generate_random_suffix()}"
    block_size = 64
    worker_count = 1 if dep else 2
    config = EngineConfig(
        name=f"{backend}_{'dep' if dep else 'kv'}_routing",
        directory=vllm_sidecar_dir if backend == "vllm" else sglang_sidecar_dir,
        script_name="agg.sh" if dep else "agg_kv_router.sh",
        script_args=(
            [
                "--disable-cuda-graph",
                "--cuda-graph-backend-prefill",
                "disabled",
            ]
            if backend == "sglang"
            else []
        ),
        marks=[],
        model=model_name,
        health_check_workers=True,
        health_check_worker_count=worker_count,
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
            "PYTHONHASHSEED": "0",
            "MODEL": model_name,
            "DYN_NAMESPACE": namespace,
            "DYN_COMPONENT": "backend",
            "DYN_ENDPOINT": "generate",
            "DYN_ROUTER_USE_KV_EVENTS": "true",
            "DYN_ROUTER_MODE": "kv",
            "DYN_ROUTER_TEMPERATURE": "0",
            "DYN_ROUTER_MIN_INITIAL_WORKERS": str(worker_count),
            "DYN_REQUEST_PLANE": "tcp",
            "MAX_MODEL_LEN": "2048",
            "VLLM_BLOCK_SIZE": str(block_size),
            "SGLANG_PAGE_SIZE": str(block_size),
        },
    )
    if dep:
        devices = map_cuda_visible_devices(
            [0, 1], os.environ.get("CUDA_VISIBLE_DEVICES")
        ).split(",")
        assert (
            len(set(devices)) == 2 and "-1" not in devices
        ), "DEP requires two distinct GPUs"
        engine_env = {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
        # KV publishers offset by rank; SGLang also derives communication ports.
        dep_ports = allocate_contiguous_ports(
            1, 12 if backend == "sglang" else 2, DynamoPortRange.SERVE.value
        )
        request.addfinalizer(lambda: deallocate_ports(dep_ports))
        config.script_args += [
            "--kv-events-config",
            json.dumps(
                {
                    "publisher": "zmq",
                    "topic": "kv-events",
                    "endpoint": f"tcp://*:{dep_ports[0]}",
                    **({"enable_kv_cache_events": True} if backend == "vllm" else {}),
                }
            ),
        ]
        if backend == "vllm":
            engine_env.update(VLLM_DATA_PARALLEL_SIZE="2")
            config.script_args += [
                "--tensor-parallel-size",
                "1",
                "--enable-expert-parallel",
                "--block-size",
                str(block_size),
                "--attention-backend",
                "TRITON_MLA",
            ]
        else:
            engine_env.update(
                # Remove after pinning https://github.com/sgl-project/sglang/pull/35347 and DEP passes on SM86 without it.
                SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK="1",
            )
            config.script_args += [
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--ep-size",
                "2",
                "--enable-dp-attention",
                "--page-size",
                str(block_size),
                "--attention-backend",
                "triton",
                "--moe-runner-backend",
                "triton",
                "--moe-a2a-backend",
                "none",
                "--dist-init-addr",
                f"127.0.0.1:{dep_ports[2]}",
                "--nccl-port",
                str(dep_ports[-1]),
            ]
    else:
        device = map_cuda_visible_devices([0], os.environ.get("CUDA_VISIBLE_DEVICES"))
        assert device != "-1", "One visible GPU is required"
        engine_env = {
            f"{backend.upper()}_WORKER{worker}_GPU": device for worker in (1, 2)
        }
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
            dp_ranks=(0, 1) if dep else (0,),
        ),
    )
