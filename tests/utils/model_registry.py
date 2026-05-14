# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central Hugging Face model registry for Dynamo CI tests.

The size policy uses Hugging Face snapshot download size: the sum of file sizes
returned for the current repo snapshot by the Hugging Face model API. This is
the closest static proxy for CI download and cache cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB = 20.0


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a model or adapter used by CI tests."""

    repo_id: str
    snapshot_size_gib: float
    kind: str
    architecture_tags: tuple[str, ...]
    gated: bool = False
    download_required: bool = True
    over_cap_exception: bool = False
    exception_reason: str = ""

    @property
    def hf_url(self) -> str:
        return f"https://huggingface.co/{self.repo_id}"

    @property
    def exceeds_default_cap(self) -> bool:
        return self.snapshot_size_gib > DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB


QWEN_QWEN_IMAGE = "Qwen/Qwen-Image"
QWEN_QWEN2_AUDIO_7B_INSTRUCT = "Qwen/Qwen2-Audio-7B-Instruct"
QWEN_QWEN2_VL_2B_INSTRUCT = "Qwen/Qwen2-VL-2B-Instruct"
QWEN_QWEN2_VL_7B_INSTRUCT = "Qwen/Qwen2-VL-7B-Instruct"
QWEN_QWEN2_5_OMNI_7B = "Qwen/Qwen2.5-Omni-7B"
QWEN_QWEN2_5_VL_3B_INSTRUCT = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_QWEN2_5_VL_7B_INSTRUCT = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_QWEN3_0_6B = "Qwen/Qwen3-0.6B"
QWEN_QWEN3_32B = "Qwen/Qwen3-32B"
QWEN_QWEN3_5_0_8B = "Qwen/Qwen3.5-0.8B"
QWEN_QWEN3_EMBEDDING_4B = "Qwen/Qwen3-Embedding-4B"
QWEN_QWEN3_TTS_12HZ_1_7B_CUSTOMVOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_QWEN3_VL_2B_INSTRUCT = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_QWEN3_VL_8B_INSTRUCT = "Qwen/Qwen3-VL-8B-Instruct"

BLACK_FOREST_LABS_FLUX_2_KLEIN_4B = "black-forest-labs/FLUX.2-klein-4B"

TINYLLAMA_TINYLLAMA_1_1B_CHAT_V1_0 = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

TONGYI_MAI_Z_IMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"

WAN_AI_WAN2_1_T2V_1_3B_DIFFUSERS = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_AI_WAN2_2_TI2V_5B_DIFFUSERS = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

CODELION_QWEN3_0_6B_ACCURACY_RECOVERY_LORA = (
    "codelion/Qwen3-0.6B-accuracy-recovery-lora"
)

DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B = (
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
DEEPSEEK_AI_DEEPSEEK_V2_LITE = "deepseek-ai/DeepSeek-V2-Lite"
DEEPSEEK_AI_DEEPSEEK_LLM_7B_BASE = "deepseek-ai/deepseek-llm-7b-base"

GOOGLE_GEMMA_4_E2B_IT = "google/gemma-4-E2B-it"
GOOGLE_GEMMA_3_4B_IT = "google/gemma-3-4b-it"
LLAVA_HF_LLAVA_1_5_7B_HF = "llava-hf/llava-1.5-7b-hf"
LLAVA_HF_LLAVA_V1_6_MISTRAL_7B_HF = "llava-hf/llava-v1.6-mistral-7b-hf"
MICROSOFT_PHI_3_VISION_128K_INSTRUCT = "microsoft/Phi-3-vision-128k-instruct"
MISTRALAI_MINISTRAL_3_3B_REASONING_2512 = (
    "mistralai/Ministral-3-3B-Reasoning-2512"
)
OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"
SILENCE09_DEEPSEEK_R1_SMALL_2LAYERS = "silence09/DeepSeek-R1-Small-2layers"
ZAI_ORG_GLM_IMAGE = "zai-org/GLM-Image"


def constant_name_for_repo_id(repo_id: str) -> str:
    """Return the exported constant name for a Hugging Face repo id."""

    return (
        repo_id.upper()
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace("__", "_")
    )


MODEL_SPECS: tuple[ModelSpec, ...] = (
        ModelSpec(
            repo_id=QWEN_QWEN_IMAGE,
            snapshot_size_gib=53.7,
            kind="image_generation",
            architecture_tags=("diffusion", "image"),
            over_cap_exception=True,
            exception_reason=(
                "Legacy skipped vLLM-Omni image-generation coverage; exceeds "
                "CI capacity and should be replaced."
            ),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN2_AUDIO_7B_INSTRUCT,
            snapshot_size_gib=15.7,
            kind="audio_text_to_text",
            architecture_tags=("audio", "dense", "multimodal"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN2_VL_2B_INSTRUCT,
            snapshot_size_gib=4.1,
            kind="vlm",
            architecture_tags=("dense", "gqa", "vision"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN2_VL_7B_INSTRUCT,
            snapshot_size_gib=15.5,
            kind="vlm",
            architecture_tags=("dense", "gqa", "vision"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN2_5_OMNI_7B,
            snapshot_size_gib=20.8,
            kind="omni",
            architecture_tags=("audio", "dense", "multimodal", "vision"),
            over_cap_exception=True,
            exception_reason=(
                "Legacy skipped vLLM-Omni text coverage; exceeds the default "
                "CI snapshot cap."
            ),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN2_5_VL_3B_INSTRUCT,
            snapshot_size_gib=7.0,
            kind="vlm",
            architecture_tags=("dense", "gqa", "vision"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN2_5_VL_7B_INSTRUCT,
            snapshot_size_gib=15.5,
            kind="vlm",
            architecture_tags=("dense", "gqa", "vision"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN3_0_6B,
            snapshot_size_gib=1.4,
            kind="llm",
            architecture_tags=("dense", "gqa"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN3_32B,
            snapshot_size_gib=61.0,
            kind="llm",
            architecture_tags=("aic_metadata", "dense", "gqa"),
            download_required=False,
            over_cap_exception=True,
            exception_reason=(
                "Used as AIC metadata in router tests; not predownloaded by CI."
            ),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN3_5_0_8B,
            snapshot_size_gib=1.6,
            kind="vlm",
            architecture_tags=("dense", "hybrid_attention", "mamba", "vision"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN3_EMBEDDING_4B,
            snapshot_size_gib=7.5,
            kind="embedding",
            architecture_tags=("dense", "gqa"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN3_TTS_12HZ_1_7B_CUSTOMVOICE,
            snapshot_size_gib=4.2,
            kind="tts",
            architecture_tags=("audio", "dense"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN3_VL_2B_INSTRUCT,
            snapshot_size_gib=4.0,
            kind="vlm",
            architecture_tags=("dense", "gqa", "vision"),
        ),
        ModelSpec(
            repo_id=QWEN_QWEN3_VL_8B_INSTRUCT,
            snapshot_size_gib=16.3,
            kind="vlm",
            architecture_tags=("dense", "gqa", "vision"),
        ),
        ModelSpec(
            repo_id=BLACK_FOREST_LABS_FLUX_2_KLEIN_4B,
            snapshot_size_gib=22.1,
            kind="image_generation",
            architecture_tags=("diffusion", "image"),
            over_cap_exception=True,
            exception_reason=(
                "Current TRT-LLM diffusion pre-merge smoke coverage; exceeds "
                "the default CI snapshot cap."
            ),
        ),
        ModelSpec(
            repo_id=TINYLLAMA_TINYLLAMA_1_1B_CHAT_V1_0,
            snapshot_size_gib=2.1,
            kind="llm",
            architecture_tags=("dense", "gqa"),
        ),
        ModelSpec(
            repo_id=TONGYI_MAI_Z_IMAGE_TURBO,
            snapshot_size_gib=30.6,
            kind="image_generation",
            architecture_tags=("diffusion", "image"),
            over_cap_exception=True,
            exception_reason=(
                "Current SGLang diffusion nightly coverage; exceeds the "
                "default CI snapshot cap."
            ),
        ),
        ModelSpec(
            repo_id=WAN_AI_WAN2_1_T2V_1_3B_DIFFUSERS,
            snapshot_size_gib=26.9,
            kind="video_generation",
            architecture_tags=("diffusion", "video"),
            over_cap_exception=True,
            exception_reason=(
                "Current TRT-LLM/vLLM-Omni video-generation CI coverage; "
                "replace with a smaller fixture when available."
            ),
        ),
        ModelSpec(
            repo_id=WAN_AI_WAN2_2_TI2V_5B_DIFFUSERS,
            snapshot_size_gib=31.9,
            kind="video_generation",
            architecture_tags=("diffusion", "image_to_video", "video"),
            over_cap_exception=True,
            exception_reason=(
                "Current vLLM-Omni image-to-video CI coverage; replace with a "
                "smaller fixture when available."
            ),
        ),
        ModelSpec(
            repo_id=CODELION_QWEN3_0_6B_ACCURACY_RECOVERY_LORA,
            snapshot_size_gib=0.2,
            kind="lora_adapter",
            architecture_tags=("adapter", "lora"),
        ),
        ModelSpec(
            repo_id=DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B,
            snapshot_size_gib=15.0,
            kind="llm",
            architecture_tags=("dense", "gqa"),
        ),
        ModelSpec(
            repo_id=DEEPSEEK_AI_DEEPSEEK_V2_LITE,
            snapshot_size_gib=29.3,
            kind="llm",
            architecture_tags=("mla", "moe"),
            over_cap_exception=True,
            exception_reason=(
                "Legacy MoE/MLA coverage for vLLM, KVBM, and fault-tolerance "
                "tests; intentionally noisy until reduced."
            ),
        ),
        ModelSpec(
            repo_id=DEEPSEEK_AI_DEEPSEEK_LLM_7B_BASE,
            snapshot_size_gib=12.9,
            kind="llm",
            architecture_tags=("dense", "mha"),
        ),
        ModelSpec(
            repo_id=GOOGLE_GEMMA_4_E2B_IT,
            snapshot_size_gib=9.6,
            kind="vlm",
            architecture_tags=("dense", "vision"),
        ),
        ModelSpec(
            repo_id=GOOGLE_GEMMA_3_4B_IT,
            snapshot_size_gib=8.0,
            kind="vlm",
            architecture_tags=("dense", "gated", "vision"),
            gated=True,
        ),
        ModelSpec(
            repo_id=LLAVA_HF_LLAVA_1_5_7B_HF,
            snapshot_size_gib=13.2,
            kind="vlm",
            architecture_tags=("dense", "vision"),
        ),
        ModelSpec(
            repo_id=LLAVA_HF_LLAVA_V1_6_MISTRAL_7B_HF,
            snapshot_size_gib=14.1,
            kind="vlm",
            architecture_tags=("dense", "vision"),
        ),
        ModelSpec(
            repo_id=MICROSOFT_PHI_3_VISION_128K_INSTRUCT,
            snapshot_size_gib=7.7,
            kind="vlm",
            architecture_tags=("dense", "vision"),
        ),
        ModelSpec(
            repo_id=MISTRALAI_MINISTRAL_3_3B_REASONING_2512,
            snapshot_size_gib=14.4,
            kind="llm",
            architecture_tags=("dense", "gqa"),
        ),
        ModelSpec(
            repo_id=OPENAI_GPT_OSS_20B,
            snapshot_size_gib=38.5,
            kind="llm",
            architecture_tags=("gqa", "moe"),
            over_cap_exception=True,
            exception_reason=(
                "Required by current GPT-OSS reasoning/tool-calling frontend "
                "coverage; exceeds CI size cap."
            ),
        ),
        ModelSpec(
            repo_id=SILENCE09_DEEPSEEK_R1_SMALL_2LAYERS,
            snapshot_size_gib=4.5,
            kind="llm",
            architecture_tags=("mla", "moe"),
        ),
        ModelSpec(
            repo_id=ZAI_ORG_GLM_IMAGE,
            snapshot_size_gib=33.3,
            kind="image_generation",
            architecture_tags=("diffusion", "image"),
            over_cap_exception=True,
            exception_reason=(
                "Legacy skipped vLLM-Omni image-generation coverage; exceeds "
                "CI capacity and should be replaced."
            ),
        ),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.repo_id: spec for spec in MODEL_SPECS}


DEFAULT_TEST_MODELS = (
    QWEN_QWEN3_0_6B,
    DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B,
    OPENAI_GPT_OSS_20B,
    QWEN_QWEN3_EMBEDDING_4B,
)


def get_model_spec(repo_id: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[repo_id]
    except KeyError as exc:
        raise KeyError(
            f"{repo_id!r} is not registered in tests.utils.model_registry. "
            "Add a ModelSpec with HF snapshot size and architecture metadata."
        ) from exc


def validate_ci_model_ids(repo_ids: Iterable[str]) -> tuple[str, ...]:
    """Validate and normalize CI model ids collected from pytest marks."""

    unique_repo_ids = tuple(dict.fromkeys(repo_ids))
    missing = [repo_id for repo_id in unique_repo_ids if repo_id not in MODEL_REGISTRY]
    if missing:
        raise ValueError(
            "Unregistered CI model id(s): "
            + ", ".join(sorted(missing))
            + ". Add them to tests.utils.model_registry before using them in CI."
        )
    return unique_repo_ids


def downloadable_model_ids(repo_ids: Iterable[str]) -> tuple[str, ...]:
    """Return registered ids that should be downloaded by CI predownload fixtures."""

    return tuple(
        repo_id
        for repo_id in validate_ci_model_ids(repo_ids)
        if MODEL_REGISTRY[repo_id].download_required
    )
