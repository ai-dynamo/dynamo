# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central Hugging Face model registry for Dynamo CI tests.

The size policy uses Hugging Face snapshot download size: the sum of file sizes
returned for the current repo snapshot by the Hugging Face model API. This is
the closest static proxy for CI download and cache cost.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

DEFAULT_CI_MODEL_SNAPSHOT_CAP_GIB = 20.0
GLOBAL_CI_MODEL_OVERRIDE_ENV_VAR = "DYN_CI_MODEL"


@dataclass(frozen=True)
class ModelSpec:
    """Metadata for a model or adapter used by CI tests."""

    repo_id: str
    snapshot_size_gib: float
    kind: str
    characteristics: tuple[str, ...]
    architecture: str = ""
    backends: tuple[str, ...] = ()
    parameter_count_millions: int | None = None
    context_length: int | None = None
    release_year: int | None = None
    license: str = ""
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


@dataclass(frozen=True)
class ModelQuery:
    """Compatibility constraints used to select a CI model."""

    kind: str | None = None
    required_characteristics: frozenset[str] = frozenset()
    excluded_characteristics: frozenset[str] = frozenset()
    required_backends: frozenset[str] = frozenset()
    max_parameter_count_millions: int | None = None
    max_snapshot_size_gib: float | None = None
    allow_gated: bool = False
    include_metadata_only: bool = False


@dataclass(frozen=True)
class ModelProfile:
    """A stable CI role whose concrete model can be safely overridden."""

    default_repo_id: str
    query: ModelQuery
    override_env_var: str
    allow_global_override: bool = True


QWEN_QWEN_IMAGE = "Qwen/Qwen-Image"
QWEN_QWEN2_AUDIO_7B_INSTRUCT = "Qwen/Qwen2-Audio-7B-Instruct"
QWEN_QWEN2_VL_2B_INSTRUCT = "Qwen/Qwen2-VL-2B-Instruct"
QWEN_QWEN2_VL_7B_INSTRUCT = "Qwen/Qwen2-VL-7B-Instruct"
QWEN_QWEN2_5_OMNI_7B = "Qwen/Qwen2.5-Omni-7B"
QWEN_QWEN2_5_VL_3B_INSTRUCT = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_QWEN2_5_VL_7B_INSTRUCT = "Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_QWEN2_5_1_5B_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_QWEN3_0_6B = "Qwen/Qwen3-0.6B"
QWEN_QWEN3_32B = "Qwen/Qwen3-32B"
QWEN_QWEN3_235B_A22B_FP8 = "Qwen/Qwen3-235B-A22B-FP8"
QWEN_QWEN3_5_0_8B = "Qwen/Qwen3.5-0.8B"
QWEN_QWEN3_EMBEDDING_0_6B = "Qwen/Qwen3-Embedding-0.6B"
QWEN_QWEN3_EMBEDDING_4B = "Qwen/Qwen3-Embedding-4B"
QWEN_QWEN3_TTS_12HZ_1_7B_CUSTOMVOICE = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
QWEN_QWEN3_VL_2B_INSTRUCT = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_QWEN3_VL_2B_INSTRUCT_FP8 = "Qwen/Qwen3-VL-2B-Instruct-FP8"
QWEN_QWEN3_VL_8B_INSTRUCT = "Qwen/Qwen3-VL-8B-Instruct"

BLACK_FOREST_LABS_FLUX_2_KLEIN_4B = "black-forest-labs/FLUX.2-klein-4B"

TINYLLAMA_TINYLLAMA_1_1B_CHAT_V1_0 = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TINYLLAMA_TINYLLAMA_1_1B_INTERMEDIATE_STEP_1431K_3T = (
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

TONGYI_MAI_Z_IMAGE_TURBO = "Tongyi-MAI/Z-Image-Turbo"

WAN_AI_WAN2_1_T2V_1_3B_DIFFUSERS = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_AI_WAN2_2_TI2V_5B_DIFFUSERS = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

CODELION_QWEN3_0_6B_ACCURACY_RECOVERY_LORA = (
    "codelion/Qwen3-0.6B-accuracy-recovery-lora"
)

DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEEPSEEK_AI_DEEPSEEK_V2_LITE = "deepseek-ai/DeepSeek-V2-Lite"
DEEPSEEK_AI_DEEPSEEK_LLM_7B_BASE = "deepseek-ai/deepseek-llm-7b-base"

BAAI_BGE_SMALL_EN_V1_5 = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_NLI_MINILM2_L6_H768 = "cross-encoder/nli-MiniLM2-L6-H768"
GOOGLE_GEMMA_3_270M_IT = "google/gemma-3-270m-it"
GOOGLE_GEMMA_4_E2B_IT = "google/gemma-4-E2B-it"
GOOGLE_GEMMA_3_4B_IT = "google/gemma-3-4b-it"
IBM_GRANITE_GRANITE_4_0_H_350M = "ibm-granite/granite-4.0-h-350m"
INCLUSIONAI_LLADA2_0_MINI_PREVIEW = "inclusionAI/LLaDA2.0-mini-preview"
LIQUIDAI_LFM2_5_350M = "LiquidAI/LFM2.5-350M"
LLAVA_HF_LLAVA_1_5_7B_HF = "llava-hf/llava-1.5-7b-hf"
LLAVA_HF_LLAVA_V1_6_MISTRAL_7B_HF = "llava-hf/llava-v1.6-mistral-7b-hf"
MICROSOFT_PHI_3_VISION_128K_INSTRUCT = "microsoft/Phi-3-vision-128k-instruct"
MISTRALAI_MINISTRAL_3_3B_REASONING_2512 = "mistralai/Ministral-3-3B-Reasoning-2512"
OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"
META_LLAMA_META_LLAMA_3_1_70B = "meta-llama/Meta-Llama-3.1-70B"
META_LLAMA_META_LLAMA_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SILENCE09_DEEPSEEK_R1_SMALL_2LAYERS = "silence09/DeepSeek-R1-Small-2layers"
YUHUILI_EAGLE3_LLAMA3_1_INSTRUCT_8B = "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
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
        repo_id=BAAI_BGE_SMALL_EN_V1_5,
        snapshot_size_gib=0.4,
        kind="embedding",
        characteristics=("bert", "dense", "encoder"),
    ),
    ModelSpec(
        repo_id=CROSS_ENCODER_NLI_MINILM2_L6_H768,
        snapshot_size_gib=2.7,
        kind="reranker",
        characteristics=("dense", "encoder", "roberta"),
    ),
    ModelSpec(
        repo_id=GOOGLE_GEMMA_3_270M_IT,
        snapshot_size_gib=0.54,
        kind="llm",
        characteristics=(
            "chat",
            "dense",
            "instruction_tuned",
            "sliding_window",
            "small_ci_candidate",
            "text_generation",
        ),
        architecture="Gemma3ForCausalLM",
        backends=("sglang", "trtllm", "vllm"),
        parameter_count_millions=270,
        context_length=32768,
        release_year=2025,
        license="gemma",
        gated=True,
    ),
    ModelSpec(
        repo_id=IBM_GRANITE_GRANITE_4_0_H_350M,
        snapshot_size_gib=0.64,
        kind="llm",
        characteristics=(
            "chat",
            "hybrid_state_space",
            "instruction_tuned",
            "mamba2",
            "small_ci_candidate",
            "text_generation",
            "tool_calling",
        ),
        architecture="GraniteMoeHybridForCausalLM",
        backends=("sglang", "vllm"),
        parameter_count_millions=340,
        context_length=32768,
        release_year=2025,
        license="apache-2.0",
    ),
    ModelSpec(
        repo_id=LIQUIDAI_LFM2_5_350M,
        snapshot_size_gib=0.66,
        kind="llm",
        characteristics=(
            "chat",
            "hybrid_state_space",
            "instruction_tuned",
            "short_convolution",
            "small_ci_candidate",
            "text_generation",
            "tool_calling",
        ),
        architecture="Lfm2ForCausalLM",
        backends=("sglang", "vllm"),
        parameter_count_millions=350,
        context_length=32768,
        release_year=2026,
        license="lfm1.0",
    ),
    ModelSpec(
        repo_id=QWEN_QWEN_IMAGE,
        snapshot_size_gib=53.7,
        kind="image_generation",
        characteristics=("diffusion", "image"),
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
        characteristics=("audio", "dense", "multimodal"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN2_VL_2B_INSTRUCT,
        snapshot_size_gib=4.1,
        kind="vlm",
        characteristics=("dense", "gqa", "vision"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN2_VL_7B_INSTRUCT,
        snapshot_size_gib=15.5,
        kind="vlm",
        characteristics=("dense", "gqa", "vision"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN2_5_OMNI_7B,
        snapshot_size_gib=20.8,
        kind="omni",
        characteristics=("audio", "dense", "multimodal", "vision"),
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
        characteristics=("dense", "gqa", "vision"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN2_5_1_5B_INSTRUCT,
        snapshot_size_gib=2.9,
        kind="llm",
        characteristics=("dense", "gqa"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN2_5_VL_7B_INSTRUCT,
        snapshot_size_gib=15.5,
        kind="vlm",
        characteristics=("dense", "gqa", "vision"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_0_6B,
        snapshot_size_gib=1.4,
        kind="llm",
        characteristics=(
            "chat",
            "dense",
            "full_attention",
            "gqa",
            "instruction_tuned",
            "text_generation",
            "tool_calling",
        ),
        architecture="Qwen3ForCausalLM",
        backends=("sglang", "trtllm", "vllm"),
        parameter_count_millions=600,
        context_length=40960,
        release_year=2025,
        license="apache-2.0",
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_32B,
        snapshot_size_gib=61.0,
        kind="llm",
        characteristics=("aic_metadata", "dense", "gqa"),
        download_required=False,
        over_cap_exception=True,
        exception_reason=(
            "Used as AIC metadata in router tests; not predownloaded by CI."
        ),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_235B_A22B_FP8,
        snapshot_size_gib=222.6,
        kind="llm",
        characteristics=("fp8", "gqa", "moe"),
        over_cap_exception=True,
        exception_reason="H100 DGDR deployment coverage for a large FP8 MoE model.",
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_5_0_8B,
        snapshot_size_gib=1.6,
        kind="vlm",
        characteristics=("dense", "hybrid_attention", "mamba", "vision"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_EMBEDDING_0_6B,
        snapshot_size_gib=1.2,
        kind="embedding",
        characteristics=("dense", "gqa"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_EMBEDDING_4B,
        snapshot_size_gib=7.5,
        kind="embedding",
        characteristics=("dense", "gqa"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_TTS_12HZ_1_7B_CUSTOMVOICE,
        snapshot_size_gib=4.2,
        kind="tts",
        characteristics=("audio", "dense"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_VL_2B_INSTRUCT,
        snapshot_size_gib=4.0,
        kind="vlm",
        characteristics=("dense", "gqa", "vision"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_VL_2B_INSTRUCT_FP8,
        snapshot_size_gib=3.3,
        kind="vlm",
        characteristics=("fp8", "gqa", "vision"),
    ),
    ModelSpec(
        repo_id=QWEN_QWEN3_VL_8B_INSTRUCT,
        snapshot_size_gib=16.3,
        kind="vlm",
        characteristics=("dense", "gqa", "vision"),
    ),
    ModelSpec(
        repo_id=BLACK_FOREST_LABS_FLUX_2_KLEIN_4B,
        snapshot_size_gib=22.1,
        kind="image_generation",
        characteristics=("diffusion", "image"),
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
        characteristics=("dense", "gqa"),
    ),
    ModelSpec(
        repo_id=TINYLLAMA_TINYLLAMA_1_1B_INTERMEDIATE_STEP_1431K_3T,
        snapshot_size_gib=8.2,
        kind="llm",
        characteristics=("dense", "gqa"),
    ),
    ModelSpec(
        repo_id=TONGYI_MAI_Z_IMAGE_TURBO,
        snapshot_size_gib=30.6,
        kind="image_generation",
        characteristics=("diffusion", "image"),
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
        characteristics=("diffusion", "video"),
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
        characteristics=("diffusion", "image_to_video", "video"),
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
        characteristics=("adapter", "lora"),
    ),
    ModelSpec(
        repo_id=DEEPSEEK_AI_DEEPSEEK_R1_DISTILL_LLAMA_8B,
        snapshot_size_gib=15.0,
        kind="llm",
        characteristics=("dense", "gqa"),
    ),
    ModelSpec(
        repo_id=DEEPSEEK_AI_DEEPSEEK_V2_LITE,
        snapshot_size_gib=29.3,
        kind="llm",
        characteristics=("mla", "moe"),
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
        characteristics=("dense", "mha"),
    ),
    ModelSpec(
        repo_id=GOOGLE_GEMMA_4_E2B_IT,
        snapshot_size_gib=9.6,
        kind="vlm",
        characteristics=("dense", "vision"),
    ),
    ModelSpec(
        repo_id=GOOGLE_GEMMA_3_4B_IT,
        snapshot_size_gib=8.0,
        kind="vlm",
        characteristics=("dense", "gated", "vision"),
        gated=True,
    ),
    ModelSpec(
        repo_id=INCLUSIONAI_LLADA2_0_MINI_PREVIEW,
        snapshot_size_gib=30.3,
        kind="llm",
        characteristics=("diffusion", "moe"),
        over_cap_exception=True,
        exception_reason="SGLang diffusion-language-model coverage.",
    ),
    ModelSpec(
        repo_id=LLAVA_HF_LLAVA_1_5_7B_HF,
        snapshot_size_gib=13.2,
        kind="vlm",
        characteristics=("dense", "vision"),
    ),
    ModelSpec(
        repo_id=LLAVA_HF_LLAVA_V1_6_MISTRAL_7B_HF,
        snapshot_size_gib=14.1,
        kind="vlm",
        characteristics=("dense", "vision"),
    ),
    ModelSpec(
        repo_id=MICROSOFT_PHI_3_VISION_128K_INSTRUCT,
        snapshot_size_gib=7.7,
        kind="vlm",
        characteristics=("dense", "vision"),
    ),
    ModelSpec(
        repo_id=MISTRALAI_MINISTRAL_3_3B_REASONING_2512,
        snapshot_size_gib=14.4,
        kind="llm",
        characteristics=("dense", "gqa"),
    ),
    ModelSpec(
        repo_id=OPENAI_GPT_OSS_20B,
        snapshot_size_gib=38.5,
        kind="llm",
        characteristics=("gqa", "moe"),
        over_cap_exception=True,
        exception_reason=(
            "Required by current GPT-OSS reasoning/tool-calling frontend "
            "coverage; exceeds CI size cap."
        ),
    ),
    ModelSpec(
        repo_id=META_LLAMA_META_LLAMA_3_1_70B,
        snapshot_size_gib=131.0,
        kind="llm",
        characteristics=("dense", "gated", "gqa"),
        gated=True,
        over_cap_exception=True,
        exception_reason="H100 DGDR deployment coverage for Llama 3.1 70B.",
    ),
    ModelSpec(
        repo_id=META_LLAMA_META_LLAMA_3_1_8B_INSTRUCT,
        snapshot_size_gib=15.0,
        kind="llm",
        characteristics=("dense", "gated", "gqa"),
        gated=True,
    ),
    ModelSpec(
        repo_id=SILENCE09_DEEPSEEK_R1_SMALL_2LAYERS,
        snapshot_size_gib=4.5,
        kind="llm",
        characteristics=("mla", "moe"),
    ),
    ModelSpec(
        repo_id=YUHUILI_EAGLE3_LLAMA3_1_INSTRUCT_8B,
        snapshot_size_gib=0.9,
        kind="speculative_draft",
        characteristics=("dense", "eagle3", "gqa"),
    ),
    ModelSpec(
        repo_id=ZAI_ORG_GLM_IMAGE,
        snapshot_size_gib=33.3,
        kind="image_generation",
        characteristics=("diffusion", "image"),
        over_cap_exception=True,
        exception_reason=(
            "Legacy skipped vLLM-Omni image-generation coverage; exceeds "
            "CI capacity and should be replaced."
        ),
    ),
)

MODEL_REGISTRY: dict[str, ModelSpec] = {spec.repo_id: spec for spec in MODEL_SPECS}


_SMOKE_CHARACTERISTICS = frozenset({"chat", "instruction_tuned", "text_generation"})

MODEL_PROFILES: dict[str, ModelProfile] = {
    "cross_backend_smoke": ModelProfile(
        default_repo_id=QWEN_QWEN3_0_6B,
        query=ModelQuery(
            kind="llm",
            required_characteristics=_SMOKE_CHARACTERISTICS,
            required_backends=frozenset({"sglang", "trtllm", "vllm"}),
            max_parameter_count_millions=700,
            allow_gated=True,
        ),
        override_env_var="DYN_CI_CROSS_BACKEND_SMOKE_MODEL",
    ),
    "kv_transfer": ModelProfile(
        default_repo_id=QWEN_QWEN3_0_6B,
        query=ModelQuery(
            kind="llm",
            required_characteristics=_SMOKE_CHARACTERISTICS | {"full_attention"},
            required_backends=frozenset({"vllm"}),
            max_parameter_count_millions=700,
        ),
        override_env_var="DYN_CI_KV_TRANSFER_MODEL",
        allow_global_override=False,
    ),
    "sglang_smoke": ModelProfile(
        default_repo_id=QWEN_QWEN3_0_6B,
        query=ModelQuery(
            kind="llm",
            required_characteristics=_SMOKE_CHARACTERISTICS,
            required_backends=frozenset({"sglang"}),
            max_parameter_count_millions=700,
            allow_gated=True,
        ),
        override_env_var="DYN_CI_SGLANG_SMOKE_MODEL",
    ),
    "trtllm_smoke": ModelProfile(
        default_repo_id=QWEN_QWEN3_0_6B,
        query=ModelQuery(
            kind="llm",
            required_characteristics=_SMOKE_CHARACTERISTICS,
            required_backends=frozenset({"trtllm"}),
            max_parameter_count_millions=700,
            allow_gated=True,
        ),
        override_env_var="DYN_CI_TRTLLM_SMOKE_MODEL",
    ),
    "vllm_smoke": ModelProfile(
        default_repo_id=QWEN_QWEN3_0_6B,
        query=ModelQuery(
            kind="llm",
            required_characteristics=_SMOKE_CHARACTERISTICS,
            required_backends=frozenset({"vllm"}),
            max_parameter_count_millions=700,
            allow_gated=True,
        ),
        override_env_var="DYN_CI_VLLM_SMOKE_MODEL",
    ),
}


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


def model_matches_query(spec: ModelSpec, query: ModelQuery) -> bool:
    """Return whether a model satisfies every compatibility constraint."""

    characteristics = frozenset(spec.characteristics)
    backends = frozenset(spec.backends)
    return all(
        (
            query.kind is None or spec.kind == query.kind,
            query.required_characteristics <= characteristics,
            not (query.excluded_characteristics & characteristics),
            query.required_backends <= backends,
            query.max_parameter_count_millions is None
            or (
                spec.parameter_count_millions is not None
                and spec.parameter_count_millions <= query.max_parameter_count_millions
            ),
            query.max_snapshot_size_gib is None
            or spec.snapshot_size_gib <= query.max_snapshot_size_gib,
            query.allow_gated or not spec.gated,
            query.include_metadata_only or spec.download_required,
        )
    )


def select_models(query: ModelQuery) -> tuple[ModelSpec, ...]:
    """Return compatible models, smallest parameter count first."""

    matches = [
        spec for spec in MODEL_REGISTRY.values() if model_matches_query(spec, query)
    ]
    return tuple(
        sorted(
            matches,
            key=lambda spec: (
                spec.parameter_count_millions
                if spec.parameter_count_millions is not None
                else float("inf"),
                spec.snapshot_size_gib,
                spec.repo_id,
            ),
        )
    )


def select_model(query: ModelQuery) -> str:
    """Return the smallest compatible model id, or fail with the query."""

    matches = select_models(query)
    if not matches:
        raise ValueError(f"No registered CI model satisfies {query!r}")
    return matches[0].repo_id


def resolve_model_profile(profile_name: str, *, override: str | None = None) -> str:
    """Resolve a stable CI role and validate any environment override."""

    try:
        profile = MODEL_PROFILES[profile_name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PROFILES))
        raise KeyError(
            f"Unknown CI model profile {profile_name!r}; choose {choices}"
        ) from exc

    global_override = (
        os.environ.get(GLOBAL_CI_MODEL_OVERRIDE_ENV_VAR)
        if profile.allow_global_override
        else None
    )
    repo_id = (
        override
        or os.environ.get(profile.override_env_var)
        or global_override
        or profile.default_repo_id
    )
    spec = get_model_spec(repo_id)
    if not model_matches_query(spec, profile.query):
        raise ValueError(
            f"{repo_id!r} does not satisfy CI model profile {profile_name!r}: "
            f"{profile.query!r}"
        )
    return repo_id


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


def _main() -> None:
    parser = argparse.ArgumentParser(description="Select a compatible Dynamo CI model")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    parser.add_argument("--kind", default="llm")
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument("--backend", action="append", default=[])
    parser.add_argument("--max-parameters-millions", type=int)
    parser.add_argument("--max-snapshot-gib", type=float)
    parser.add_argument("--allow-gated", action="store_true")
    args = parser.parse_args()

    if args.profile:
        print(resolve_model_profile(args.profile))
        return
    print(
        select_model(
            ModelQuery(
                kind=args.kind,
                required_characteristics=frozenset(args.require),
                required_backends=frozenset(args.backend),
                max_parameter_count_millions=args.max_parameters_millions,
                max_snapshot_size_gib=args.max_snapshot_gib,
                allow_gated=args.allow_gated,
            )
        )
    )


if __name__ == "__main__":
    _main()
