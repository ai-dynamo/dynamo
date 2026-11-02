# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the pinned optional vLLM source manifest before loading native APIs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .decision_cache_provenance import (
    CAUSAL_CACHE_SOURCE_BLOBS,
    verify_installed_sources,
)
from .decision_native_errors import NativeRuntimeError
from .decision_runtime_policy import is_exact_vllm_release

PINNED_VLLM_TAG = "v0.33.0"
PINNED_VLLM_COMMIT = "4ba787ec9959e566d3b587a796dff07673324da5"

PINNED_SOURCE_BLOBS = (
    ("vllm/exceptions.py", "85863c2fbf1db06e3d3c3119ac0f2ad7526faf15"),
    ("vllm/v1/engine/exceptions.py", "e1d8e5f20b10ea882017644e01b65d4c532b5783"),
    ("vllm/inputs/engine.py", "aa8c9054335e75242dc2d89b80fc364e590f0de5"),
    ("vllm/config/diffusion.py", "6f59c40a8366910b41630ad62bfe824def738364"),
    (
        "vllm/entrypoints/generate/label_reads.py",
        "065d9767d68e06a5456a01a240b55886317d0b36",
    ),
    (
        "vllm/entrypoints/openai/decisions/adapters.py",
        "449ededb549743e0b767183aea9b6046580c55d8",
    ),
    (
        "vllm/entrypoints/openai/decisions/question_types.py",
        "7ff319edc0c274ef355b2180ff26dbf17f7fe8af",
    ),
    (
        # Liquid label budgeting relies on _label_groups -> _single_tokens -> _tokenizer.
        # This blob also pins private NextTokenStrategy._question_read reduction
        # and the copied-strategy boundary.
        "vllm/entrypoints/openai/decisions/strategies.py",
        "2159657dcbff622079875452129060f3887de124",
    ),
    (
        "vllm/model_executor/models/diffusion_gemma.py",
        "f2f49cf032976ac4903daa29f99403d93160e845",
    ),
    (
        "vllm/model_executor/models/diffusion_gemma_sampler.py",
        "9b1523379197e2d460e935812af734bb372dd2a6",
    ),
    (
        "vllm/model_executor/models/lfm2_vl.py",
        "dd58707b855ff7b28327c072b82b4623c323aa11",
    ),
    ("vllm/renderers/inputs/preprocess.py", "9daf38fdbd3bd54a9cb32175329af9ca5bc07099"),
    ("vllm/renderers/online_renderer.py", "941d173bd1d5cb2ad2c6e3f07f8d052c54770739"),
    ("vllm/sampling_params.py", "466c19e6bb74c2d6127e0c2ab986631f3beea69f"),
    ("vllm/utils/async_utils.py", "b2c0d57dc921bd71eda85b5c7472dfc03c6cdd3e"),
    ("vllm/utils/diffusion.py", "f3018c18e7bdeaee0850c96773f0b6638950d3d2"),
    (
        "vllm/v1/core/sched/diffusion_scheduler.py",
        "9803307037670ea0be87d286111e732ed4bedf98",
    ),
    ("vllm/v1/engine/async_llm.py", "9fdf687d1345c89f386d5ae2ee121074150364ec"),
    ("vllm/v1/worker/gpu_model_runner.py", "ac3ff3b658655215da135797215ca549e9f7bc6d"),
    ("vllm/inputs/__init__.py", "7f89a598b77946e9e44b265472be1aebfc408d1b"),
    ("vllm/pooling_params.py", "7d9a022c5ae05126561c4131d7c564c1cb8d4806"),
    ("vllm/model_executor/models/d3.py", "da9019d66fb8de53c267c898f5e9ffa878a607a4"),
    (
        "vllm/model_executor/models/clef_joint_schema.py",
        "678c354387ce6ca9d0c399057e9a4f6ad744806f",
    ),
    *CAUSAL_CACHE_SOURCE_BLOBS["0.33.0"],
)


@dataclass(frozen=True)
class NativeApi:
    strategies: Any
    questions: Any
    pooling_params: Any
    tokens_input: Any
    online_renderer: Any
    validation_error: type[Exception]
    template_error: type[Exception]
    overload_errors: tuple[type[Exception], ...]
    overload_error: type[Exception]
    engine_generate_error: type[Exception]
    engine_dead_error: type[Exception]
    retryable_request_error: type[Exception]


def load_native() -> NativeApi:
    """Verify the complete shared tag manifest before importing native adapters."""
    try:
        required_release = PINNED_VLLM_TAG.removeprefix("v")
        if not is_exact_vllm_release(version("vllm"), required_release):
            raise NativeRuntimeError(
                f"native decisions require vLLM {required_release}"
            )
        if not verify_installed_sources(PINNED_SOURCE_BLOBS):
            raise NativeRuntimeError(
                "installed native source is unavailable or differs from pinned runtime"
            )
        exceptions = import_module("vllm.exceptions")
        engine_errors = import_module("vllm.v1.engine.exceptions")
        return NativeApi(
            strategies=import_module("vllm.entrypoints.openai.decisions.strategies"),
            questions=import_module("vllm.entrypoints.openai.decisions.question_types"),
            pooling_params=import_module("vllm.pooling_params").PoolingParams,
            tokens_input=import_module("vllm.inputs").tokens_input,
            online_renderer=import_module(
                "vllm.renderers.online_renderer"
            ).OnlineRenderer,
            validation_error=exceptions.VLLMValidationError,
            template_error=import_module("jinja2").TemplateError,
            overload_errors=(
                exceptions.QueueOverflowError,
                exceptions.MaxQueuedTokensError,
            ),
            overload_error=import_module("dynamo.llm.exceptions").WorkerOverloaded,
            engine_generate_error=engine_errors.EngineGenerateError,
            engine_dead_error=engine_errors.EngineDeadError,
            retryable_request_error=exceptions.RetryableRequestError,
        )
    except (
        ImportError,
        OSError,
        AttributeError,
        TypeError,
        PackageNotFoundError,
    ) as cause:
        raise NativeRuntimeError(
            "pinned optional native runtime is unavailable"
        ) from cause
