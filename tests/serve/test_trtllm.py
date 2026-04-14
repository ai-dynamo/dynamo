# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.serve.common import (
    SERVE_TEST_DIR,
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import (
    TEXT_PROMPT,
    chat_payload,
    chat_payload_default,
    completion_payload,
    completion_payload_default,
    metric_payload_default,
    multimodal_payload_default,
)
from tests.utils.payloads import BasePayload

logger = logging.getLogger(__name__)


@dataclass
class VideoGenerationPayload(BasePayload):
    """Payload for /v1/videos endpoint (TRT-LLM video diffusion)."""

    endpoint: str = "/v1/videos"
    timeout: int = 300

    def response_handler(self, response: Any) -> str:
        response.raise_for_status()
        result = response.json()
        assert result.get("status") == "completed", (
            f"Video generation not completed. Status: {result.get('status')}, "
            f"Error: {result.get('error', 'none')}"
        )
        assert (
            "data" in result
        ), f"Missing 'data' in response. Keys: {list(result.keys())}"
        assert len(result["data"]) > 0, "Empty data in video response"
        entry = result["data"][0]
        if "url" in entry:
            assert entry["url"], "Video response url is empty"
            return entry["url"]
        assert entry.get("b64_json"), "Video response b64_json is empty"
        return "b64_video_returned"

    def validate(self, response: Any, content: str) -> None:
        assert content, "Video response content is empty"


@dataclass
class TRTLLMConfig(EngineConfig):
    """Configuration for trtllm test scenarios"""

    stragglers: list[str] = field(default_factory=lambda: ["TRTLLM:EngineCore"])


trtllm_dir = os.environ.get("TRTLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/trtllm"
)

# TensorRT-LLM test configurations
# NOTE: pytest.mark.gpu_1 tests take ~442s (7m 22s) total to run sequentially (with models pre-cached)
# TODO: Parallelize these tests to reduce total execution time
trtllm_configs = {
    "aggregated": TRTLLMConfig(
        name="aggregated",
        directory=trtllm_dir,
        script_name="agg_metrics.sh",
        marks=[
            pytest.mark.gpu_1,  # 1 GPU(s) used, peak 3.9 GiB
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.profiled_vram_gib(3.9),  # actual nvidia-smi peak 3.9 GiB
            pytest.mark.requested_trtllm_kv_tokens(
                2592
            ),  # KV cache cap (2x safety over min=1296)
            pytest.mark.timeout(
                300
            ),  # 3x measured time (44.66s) + download time (150s)
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        delayed_start=5,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(min_num_requests=6, backend="trtllm"),
        ],
    ),
    "disaggregated": TRTLLMConfig(
        name="disaggregated",
        directory=trtllm_dir,
        script_name="disagg.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm, pytest.mark.pre_merge],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_same_gpu": TRTLLMConfig(
        name="disaggregated_same_gpu",
        directory=trtllm_dir,
        script_name="disagg_same_gpu.sh",
        marks=[
            pytest.mark.gpu_1,  # 1 GPU(s) used, peak 6.6 GiB
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.profiled_vram_gib(6.6),  # actual nvidia-smi peak 6.6 GiB
            pytest.mark.requested_trtllm_kv_tokens(
                512
            ),  # KV cache cap (2x safety over min=256)
            pytest.mark.timeout(432),  # ~6x profiled wall time 72s
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        delayed_start=10,
        health_check_workers=True,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
            metric_payload_default(
                port=DefaultPort.SYSTEM1.value, min_num_requests=6, backend="trtllm"
            ),
            metric_payload_default(
                port=DefaultPort.SYSTEM2.value, min_num_requests=6, backend="trtllm"
            ),
        ],
    ),
    "aggregated_logprobs": TRTLLMConfig(
        name="aggregated_logprobs",
        directory=trtllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.gpu_1,  # 1 GPU(s) used, peak 3.8 GiB
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.profiled_vram_gib(3.8),  # actual nvidia-smi peak 3.8 GiB
            pytest.mark.requested_trtllm_kv_tokens(
                2592
            ),  # KV cache cap (2x safety over min=1296)
            pytest.mark.timeout(300),  # 3x measured time (~44s) + download time (150s)
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        delayed_start=5,
        request_payloads=[
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=False, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=None),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=0),
        ],
    ),
    "disaggregated_logprobs": TRTLLMConfig(
        name="disaggregated_logprobs",
        directory=trtllm_dir,
        script_name="disagg.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=False, top_logprobs=5),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=None),
            chat_payload(content=TEXT_PROMPT, logprobs=True, top_logprobs=0),
        ],
    ),
    "aggregated_router": TRTLLMConfig(
        name="aggregated_router",
        directory=trtllm_dir,
        script_name="agg_router.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.pre_merge,
            pytest.mark.trtllm,
            pytest.mark.timeout(
                300
            ),  # 3x measured time (37.91s) + download time (180s)
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(
                expected_log=[
                    r"Event processor for worker_id \d+ processing event: Stored\(",
                    r"Selected worker: worker_type=\w+, worker_id=\d+ dp_rank=.*?, logit: ",
                ]
            )
        ],
        env={
            "DYN_LOG": "dynamo_llm::kv_router::publisher=trace,dynamo_kv_router::scheduling::selector=info",
        },
    ),
    "disaggregated_router": TRTLLMConfig(
        name="disaggregated_router",
        directory=trtllm_dir,
        script_name="disagg_router.sh",
        marks=[pytest.mark.gpu_2, pytest.mark.trtllm, pytest.mark.nightly],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
    "disaggregated_multimodal": TRTLLMConfig(
        name="disaggregated_multimodal",
        directory=trtllm_dir,
        script_name="disagg_multimodal.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.trtllm,
            pytest.mark.multimodal,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen2-VL-7B-Instruct",
        frontend_port=DefaultPort.FRONTEND.value,
        timeout=900,
        delayed_start=60,
        request_payloads=[multimodal_payload_default()],
    ),
    "aggregated_multimodal_router": TRTLLMConfig(
        name="aggregated_multimodal_router",
        directory=trtllm_dir,
        script_name="agg_multimodal_router.sh",
        marks=[
            pytest.mark.skip(
                reason="Nightly CI failure: https://linear.app/nvidia/issue/DYN-2608"
            ),
            pytest.mark.gpu_1,
            pytest.mark.trtllm,
            pytest.mark.multimodal,
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        frontend_port=DefaultPort.FRONTEND.value,
        timeout=900,
        delayed_start=60,
        request_payloads=[
            multimodal_payload_default(
                text="Describe what you see in this image.",
                expected_response=["mountain", "rock", "trees", "road"],
            )
        ],
    ),
    # TensorRT-LLM EPD (Encode-Prefill-Decode) multimodal test for pre-merge CI
    # Uses Qwen3-VL-2B-Instruct model with 1 GPU (all workers share same GPU)
    #
    # TODO: Add Llama-4-Scout multimodal tests (agg_multimodal_llama, disagg_multimodal_llama)
    #       once CI supports gpu_8 runners and launch scripts are available
    "epd_multimodal": TRTLLMConfig(
        name="epd_multimodal",
        directory=trtllm_dir,
        script_name="epd_multimodal_image_and_embeddings.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.trtllm,
            pytest.mark.multimodal,
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        frontend_port=DefaultPort.FRONTEND.value,
        timeout=900,
        delayed_start=120,
        request_payloads=[
            multimodal_payload_default(
                text="Describe what you see in this image.",
                expected_response=["mountain", "rock", "trees", "road"],
            )
        ],
        env={
            "PREFILL_CUDA_VISIBLE_DEVICES": "0",
            "DECODE_CUDA_VISIBLE_DEVICES": "0",
            "ENCODE_CUDA_VISIBLE_DEVICES": "0",
        },
    ),
    # Test Encoder with Aggregated PD worker on same GPU
    # Make this pre-merge after TRTLLM #5938603 is fixed
    "e_pd_multimodal": TRTLLMConfig(
        name="e_pd_multimodal",
        directory=trtllm_dir,
        script_name="disagg_e_pd.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.trtllm,
            pytest.mark.multimodal,
            pytest.mark.nightly,
        ],
        model="Qwen/Qwen3-VL-2B-Instruct",
        frontend_port=DefaultPort.FRONTEND.value,
        timeout=900,
        delayed_start=120,
        request_payloads=[
            multimodal_payload_default(
                text="Describe what you see in this image.",
                expected_response=["mountain", "rock", "trees", "road"],
            )
        ],
        env={
            "ENCODE_CUDA_VISIBLE_DEVICES": "0",
        },
    ),
    # LLaVA raw-embeddings E/PD test
    # Validates the raw-embeddings code path where pre-computed vision embeddings
    # (.pt tensor file) are sent via file:// URL instead of a raw image URL.
    #
    # Flow:
    #   1. Launch script generates embeddings using standalone HF vision encoder
    #   2. Encode + Aggregated PD workers start for LLaVA
    #   3. Test sends chat/completions request with file:///tmp/llava_embeddings.pt
    #
    # Uses gpu_2: encode worker on GPU 0, PD worker on GPU 1.
    # The 7B LLaVA model requires two GPUs because both encode and PD workers
    # load the full model (~14GB each in bfloat16), exceeding a single L4's 22GB.
    # Runs in the multi-GPU pre-merge CI (marker: pre_merge and trtllm and gpu_2).
    "raw_embeddings_epd": TRTLLMConfig(
        name="raw_embeddings_epd",
        directory=SERVE_TEST_DIR,
        script_name="agg_raw_embeddings_llava.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.trtllm,
            pytest.mark.multimodal,
            pytest.mark.pre_merge,
            pytest.mark.timeout(
                900
            ),  # Embeddings generation (~60s) + model load (~120s) + inference
        ],
        model="llava-hf/llava-v1.6-mistral-7b-hf",
        frontend_port=DefaultPort.FRONTEND.value,
        timeout=600,
        # Embeddings generation + worker startup takes longer than normal
        delayed_start=180,
        request_payloads=[
            multimodal_payload_default(
                image_url="file:///tmp/llava_embeddings.pt",
                text="Describe what this image shows.",
                expected_response=["bench", "person", "image", "picture"],
            )
        ],
        env={
            "ENCODE_CUDA_VISIBLE_DEVICES": "0",
            "PD_CUDA_VISIBLE_DEVICES": "1",
        },
    ),
    # TensorRT-LLM video diffusion test using Wan2.1-T2V-1.3B model.
    # Validates the end-to-end video generation pipeline (frontend → worker → /v1/videos).
    # Uses --skip-warmup (warmup at default resolution OOMs on 22 GB L4 GPU),
    # --disable-torch-compile, and small default resolution (480x272, 17 frames)
    # to fit within CI GPU memory constraints.
    "video_diffusion": TRTLLMConfig(
        name="video_diffusion",
        directory=trtllm_dir,
        script_name="agg_video_diffusion.sh",
        script_args=[
            "--skip-warmup",
            "--disable-torch-compile",
            "--default-height",
            "272",
            "--default-width",
            "480",
            "--default-num-frames",
            "17",
        ],
        marks=[
            pytest.mark.gpu_1,  # 1 GPU(s) used, peak 17.1 GiB
            pytest.mark.trtllm,
            pytest.mark.pre_merge,
            # Diffusion models don't use KV cache, so requested_trtllm_kv_tokens
            # doesn't apply.  requested_trtllm_vram_gib maps to
            # KvCacheConfig.max_gpu_total_bytes which has no effect on the
            # diffusion engine itself, but the parallel scheduler requires one
            # of the KV/VRAM markers to accept the test.  We set it to the
            # profiled peak so the scheduler's VRAM budget is accurate.
            pytest.mark.profiled_vram_gib(17.1),  # actual nvidia-smi peak 17.1 GiB
            pytest.mark.requested_trtllm_vram_gib(17.1),
            pytest.mark.timeout(
                600
            ),  # Video generation is slow even at small resolution
        ],
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        frontend_port=DefaultPort.FRONTEND.value,
        timeout=300,
        delayed_start=5,
        request_payloads=[
            VideoGenerationPayload(
                body={
                    "prompt": "A golden retriever running on a beach",
                    "size": "480x272",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 10,
                        "num_frames": 17,
                        "guidance_scale": 5.0,
                        "seed": 42,
                    },
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "completions_only": TRTLLMConfig(
        name="completions_only",
        directory=trtllm_dir,
        script_name="agg.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.trtllm,
            pytest.mark.post_merge,
            pytest.mark.skip(reason="DIS-1566"),
            pytest.mark.timeout(
                480
            ),  # 3x measured time (83.85s) + download time (210s) for 7B model
        ],
        model="deepseek-ai/deepseek-llm-7b-base",
        script_args=["--dyn-endpoint-types", "completions"],
        env={
            "MODEL_PATH": "deepseek-ai/deepseek-llm-7b-base",
            "SERVED_MODEL_NAME": "deepseek-ai/deepseek-llm-7b-base",
        },
        request_payloads=[
            completion_payload_default(),
            completion_payload(prompt=TEXT_PROMPT, logprobs=3),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(trtllm_configs))
def trtllm_config_test(request):
    """Fixture that provides different trtllm test configurations"""
    return trtllm_configs[request.param]


@pytest.mark.trtllm
@pytest.mark.e2e
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_deployment(
    trtllm_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    num_system_ports,
    predownload_models,
):
    """
    Test dynamo deployments with different configurations.
    """
    assert (
        num_system_ports >= 2
    ), "serve tests require at least SYSTEM_PORT1 + SYSTEM_PORT2"
    # Use per-test ports so tests can run safely under pytest-xdist.
    config = dataclasses.replace(
        trtllm_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    # Non-port env stays here; ports are wired by run_serve_deployment(ports=...).
    config.env.update(
        {
            "MODEL_PATH": config.model,
            "SERVED_MODEL_NAME": config.model,
        }
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)


# TODO make this a normal guy
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.trtllm
@pytest.mark.pre_merge
@pytest.mark.timeout(660)  # 3x measured time (159.68s) + download time (180s)
def test_chat_only_aggregated_with_test_logits_processor(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
    monkeypatch,
):
    """
    Run a single aggregated chat-completions test using Qwen 0.6B with the
    test logits processor enabled, and expect "Hello world" in the response.
    """

    # Enable HelloWorld logits processor only for this test
    monkeypatch.setenv("DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR", "1")

    base = trtllm_configs["aggregated"]
    config = TRTLLMConfig(
        name="aggregated_qwen_chatonly",
        directory=base.directory,
        script_name=base.script_name,  # agg.sh
        marks=[],  # not used by this direct test
        request_payloads=[
            chat_payload_default(expected_response=["Hello world!"]),
        ],
        model="Qwen/Qwen3-0.6B",
        delayed_start=base.delayed_start,
        timeout=base.timeout,
    )

    config = dataclasses.replace(
        config, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    config.env.update(
        {
            "MODEL_PATH": config.model,
            "SERVED_MODEL_NAME": config.model,
        }
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
