# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import pytest

try:
    from dynamo.vllm.omni.args import OmniConfig  # noqa: F401
except Exception:
    # vllm_omni's import chain can raise NotImplementedError (and other
    # non-ImportError types) on platforms it doesn't support — e.g. a
    # CPU-only runner where vllm._C can't load libcuda.so.1.
    pytest.skip("vLLM omni dependencies not available", allow_module_level=True)

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.engine_process import EngineConfig
from tests.utils.payloads import (
    AudioSpeechPayload,
    ChatPayload,
    I2VPayload,
    ImageGenerationPayload,
    VideoGenerationPayload,
)

logger = logging.getLogger(__name__)

vllm_dir = os.environ.get("VLLM_DIR") or os.path.join(
    WORKSPACE_DIR, "examples/backends/vllm"
)


@dataclass
class VLLMOmniConfig(EngineConfig):
    """Configuration for vLLM-Omni test scenarios."""

    stragglers: list[str] = field(default_factory=lambda: ["VLLM:EngineCore"])


vllm_omni_configs = {
    "omni_disagg_t2i": VLLMOmniConfig(
        name="omni_disagg_t2i",
        directory=vllm_dir,
        script_name="disagg_omni_glm_image.sh",
        marks=[
            pytest.mark.gpu_2,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
            pytest.mark.skip(
                reason="zai-org/GLM-Image requires ~23GB per GPU across 2 GPUs, exceeds CI capacity"
            ),
        ],
        model="zai-org/GLM-Image",
        request_payloads=[
            ImageGenerationPayload(
                body={
                    "prompt": "A red apple on a white table",
                    "size": "1024x1024",
                    "response_format": "url",
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "omni_text": VLLMOmniConfig(
        name="omni_text",
        directory=vllm_dir,
        script_name="agg_omni.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.xpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            pytest.mark.skip(
                reason="Qwen2.5-Omni-7B requires ~80GB GPU memory, exceeds CI capacity (22GB)"
            ),
        ],
        model="Qwen/Qwen2.5-Omni-7B",
        request_payloads=[
            ChatPayload(
                body={
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 32,
                    "temperature": 0.0,
                },
                repeat_count=1,
                expected_response=["hello", "Hello"],
                expected_log=[],
            ),
        ],
    ),
    "omni_image": VLLMOmniConfig(
        name="omni_image",
        directory=vllm_dir,
        script_name="agg_omni_image.sh",
        script_args=[
            "--vae-use-slicing",
            "--vae-use-tiling",
            "--enforce-eager",
        ],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.xpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            pytest.mark.skip(
                reason="Qwen/Qwen-Image requires ~40GB GPU memory, exceeds CI capacity (22GB)"
            ),
        ],
        model="Qwen/Qwen-Image",
        request_payloads=[
            ImageGenerationPayload(
                body={
                    "prompt": "A red apple on a table",
                    "size": "512x512",
                    "num_inference_steps": 20,
                    "response_format": "url",
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "omni_i2v": VLLMOmniConfig(
        name="omni_i2v",
        directory=vllm_dir,
        script_name="agg_omni_i2v.sh",
        script_args=[
            "--vae-use-slicing",
            "--vae-use-tiling",
            "--enforce-eager",
            "--enable-cpu-offload",
        ],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
        ],
        model="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        request_payloads=[
            I2VPayload(
                body={
                    "prompt": "Make it dance",
                    "size": "320x192",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 5,
                        "num_frames": 9,
                        "guidance_scale": 1.0,
                        "boundary_ratio": 0.875,
                        "guidance_scale_2": 1.0,
                        "seed": 42,
                    },
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "omni_audio": VLLMOmniConfig(
        name="omni_audio",
        directory=vllm_dir,
        script_name="agg_omni_audio.sh",
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.xpu_1,
            pytest.mark.pre_merge,
            pytest.mark.timeout(1200),
            pytest.mark.skip(
                reason="vLLM-Omni audio release/v0.19.0rc1 uses the pre-vLLM 0.20 "
                "GPUModelRunner._bookkeeping_sync signature"
            ),
        ],
        model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        request_payloads=[
            AudioSpeechPayload(
                body={
                    "input": "Hello, this is a test of Dynamo audio generation.",
                    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                    "voice": "vivian",
                    "language": "English",
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    "omni_audex": VLLMOmniConfig(
        name="omni_audex",
        directory=vllm_dir,
        # Audex shares the audio launch script; only the model differs.
        script_name="agg_omni_audio.sh",
        script_args=["--model", "nvidia/Nemotron-Labs-Audex-2B"],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.xpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            # Profiled with tests/utils/profile_pytest.py on 1x H200: peak
            # 117.6 GiB, unchanged at every probed KV cap (9-75 GiB). vLLM-Omni's
            # audex_tts.yaml sizes each stage as a fraction of *total* device
            # memory (0.4 for the thinker, 0.25 for code2wav) and the aggregated
            # worker forwards no engine memory args to AsyncOmni, so neither
            # --kv-cache-memory-bytes nor --gpu-memory-utilization can cap the
            # footprint from the CLI. Both VRAM markers are therefore omitted:
            # the H200 number is card-relative, not a portable requirement.
            # Re-enable once per-stage memory overrides are plumbed through
            # omni's _build_omni_kwargs, then re-profile on a 24 GiB card.
            pytest.mark.skip(
                reason="Audex peaked at 117.6 GiB on 1x H200; the shipped stage "
                "config sizes stages as a fraction of total device memory and "
                "the worker cannot cap it, so it exceeds CI capacity (24GB)"
            ),
        ],
        model="nvidia/Nemotron-Labs-Audex-2B",
        request_payloads=[
            AudioSpeechPayload(
                body={
                    "model": "nvidia/Nemotron-Labs-Audex-2B",
                    "input": "Hey, this is generated using Dynamo!",
                    # Audex is the only audio model that takes cfg_scale, and it
                    # travels as a top-level protocol field: unknown fields are
                    # dropped silently, so a plumbing regression would leave
                    # guidance unapplied without failing the request.
                    "cfg_scale": 1.5,
                },
                repeat_count=1,
                # Stage 1 emits cumulative waveform snapshots, the first of
                # which is empty. Mis-assembling that stream yields a short or
                # silent WAV that still parses, so check the waveform itself:
                # this prompt decodes to ~2.3s at rms ~0.055.
                min_duration_s=1.0,
                min_rms=0.01,
                expected_sample_rate=16000,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
    # Known flake (post-merge): URL check fails after 600s with "StageDiffusionProc
    # died during handshake (exit code 143)" — the diffusion child process is
    # SIGTERM'd before the handshake completes. Bumping the timeout will not fix this;
    # needs investigation of why StageDiffusionProc is dying. On XPU, this
    # currently manifests as `RuntimeError: level_zero backend failed with error:
    # 20 (UR_RESULT_ERROR_DEVICE_LOST)`, so skip the XPU variant until the
    # backend path is stabilized.
    "omni_t2v": VLLMOmniConfig(
        name="omni_t2v",
        directory=vllm_dir,
        script_name="agg_omni_video.sh",
        script_args=[
            "--vae-use-slicing",
            "--vae-use-tiling",
            "--enforce-eager",
        ],
        marks=[
            pytest.mark.gpu_1,
            pytest.mark.post_merge,
            pytest.mark.timeout(1200),
            pytest.mark.profiled_vram_gib(16.8),  # actual profiled peak with kv-bytes
            pytest.mark.requested_vllm_kv_cache_bytes(
                6_473_647_000
            ),  # KV cache cap (2x safety over min=3_236_823_040)
        ],
        model="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        request_payloads=[
            VideoGenerationPayload(
                body={
                    "prompt": "Dog running on a beach",
                    "size": "480x272",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 10,
                        "num_frames": 17,
                    },
                },
                repeat_count=1,
                expected_response=[],
                expected_log=[],
            ),
            # Streaming video generation
            VideoGenerationPayload(
                body={
                    "prompt": "Dog running on a beach",
                    "size": "480x272",
                    "response_format": "url",
                    "nvext": {
                        "num_inference_steps": 10,
                        "num_frames": 17,
                    },
                },
                repeat_count=1,
                http_stream=True,
                expected_response=[],
                expected_log=[],
            ),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(vllm_omni_configs))
def vllm_omni_config_test(request):
    """Fixture that provides different vLLM-Omni test configurations."""
    return vllm_omni_configs[request.param]


@pytest.mark.vllm
@pytest.mark.multimodal
@pytest.mark.e2e
def test_omni_serve_deployment(
    vllm_omni_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
    """Test dynamo serve deployments with vLLM-Omni configurations."""
    config = dataclasses.replace(
        vllm_omni_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
