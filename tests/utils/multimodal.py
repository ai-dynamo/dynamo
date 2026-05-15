# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

import pytest

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.serve.conftest import MULTIMODAL_IMG_URL, get_multimodal_test_image_bytes
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload
from tests.utils.payloads import BasePayload, CachedTokensChatPayload, ChatPayload

LOCAL_VIDEO_TEST_PATH = Path(
    WORKSPACE_DIR, "lib/llm/tests/data/media/240p_10.mp4"
).resolve()
LOCAL_VIDEO_TEST_URI = LOCAL_VIDEO_TEST_PATH.as_uri()

AUDIO_TEST_URL = (
    "https://raw.githubusercontent.com/yuekaizhang/Triton-ASR-Client"
    "/main/datasets/mini_en/wav/1221-135766-0002.wav"
)


# ---------------------------------------------------------------------------
# Payload factories
# ---------------------------------------------------------------------------


_MULTIMODAL_COLOR_PROMPT = (
    "What colors are in the following image? Respond only with the colors."
)


def make_image_payload(
    expected_response: list[str], *, max_attempts: int = 1
) -> ChatPayload:
    """Standard image color-identification payload using MULTIMODAL_IMG_URL.

    ``max_attempts`` is the cap on validation attempts inside
    ``run_serve_deployment`` — set >1 only for known-flaky multimodal
    smoke checks (see tests/README.md "Flaky Tests"). The server stays
    up across attempts; only the request/response is re-issued.
    """
    return chat_payload(
        [
            {"type": "text", "text": _MULTIMODAL_COLOR_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": MULTIMODAL_IMG_URL},
            },
        ],
        repeat_count=1,
        expected_response=expected_response,
        temperature=0.0,
        max_tokens=100,
        max_attempts=max_attempts,
    )


def make_image_payload_cached_tokens(
    expected_response: list[str],
    *,
    repeat_count: int = 2,
    min_cached_tokens: int = 1,
) -> CachedTokensChatPayload:
    """Image payload that also asserts MM-aware KV cache reuse on repeats.

    Same body shape as :func:`make_image_payload`, but wrapped in a
    :class:`CachedTokensChatPayload` so the 2nd+ request validates that
    ``usage.prompt_tokens_details.cached_tokens >= min_cached_tokens``.
    Two identical MM requests through an MM-routing-aware frontend must
    land on the same warm worker and reuse the prefix cache; if routing
    silently regresses to text-only the second request will report 0
    cached tokens and this payload fails.

    Used to harden the ``agg_router`` pre_merge smoke against silent
    regressions in the Rust+lightseek routing path.
    """
    return CachedTokensChatPayload(
        body={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _MULTIMODAL_COLOR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": MULTIMODAL_IMG_URL},
                        },
                    ],
                }
            ],
            "max_tokens": 100,
            "temperature": 0.0,
            "stream": False,
        },
        repeat_count=repeat_count,
        expected_response=expected_response,
        min_cached_tokens=min_cached_tokens,
    )


class Base64LazyChatPayload(ChatPayload):
    """ChatPayload variant that defers reading the multimodal test PNG until
    the first `.body` access.

    The LFS-tracked test image may be a pointer file when the workspace is
    fresh; eager reads at module import (i.e. pytest collection) would fail
    before pytest can apply the test's hardware preconditions. Materializing
    on first `.body` read defers that I/O to test execution.
    """

    def __init__(
        self,
        *,
        prompt: str,
        expected_response: list[str],
        max_tokens: int = 100,
        temperature: float = 0.0,
        repeat_count: int = 1,
        timeout: int = 60,
    ):
        """Stash payload params for lazy body construction, then run parent init."""
        # Initialize lazy state and the init flag BEFORE super().__init__ —
        # the parent dataclass __init__ assigns self.body = ..., which routes
        # through our @body.setter.
        object.__setattr__(self, "_b64_initializing", True)
        object.__setattr__(self, "_b64_prompt", prompt)
        object.__setattr__(self, "_b64_max_tokens", max_tokens)
        object.__setattr__(self, "_b64_temperature", temperature)
        object.__setattr__(self, "_b64_storage", None)
        object.__setattr__(self, "_b64_materialized", False)
        super().__init__(
            body=None,  # placeholder; replaced when .body is first read
            expected_response=expected_response,
            expected_log=[],
            repeat_count=repeat_count,
            timeout=timeout,
        )
        object.__setattr__(self, "_b64_initializing", False)

    def _materialize_body(self) -> dict:
        """Read the LFS image, base64-encode it, and build the chat body dict."""
        b64 = base64.b64encode(get_multimodal_test_image_bytes()).decode()
        ref = chat_payload(
            [
                {"type": "text", "text": self._b64_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
            repeat_count=1,
            expected_response=list(self.expected_response),
            max_tokens=self._b64_max_tokens,
            temperature=self._b64_temperature,
        )
        return ref.body

    @property
    def body(self) -> dict:  # type: ignore[override]
        """Return the chat body, materializing the base64 image on first access."""
        if not self._b64_materialized:
            object.__setattr__(self, "_b64_storage", self._materialize_body())
            object.__setattr__(self, "_b64_materialized", True)
        return self._b64_storage

    @body.setter
    def body(self, value) -> None:
        """Store body; flip materialized so post-init writes survive next read."""
        object.__setattr__(self, "_b64_storage", value)
        if not getattr(self, "_b64_initializing", True):
            object.__setattr__(self, "_b64_materialized", True)


def make_image_payload_b64(expected_response: list[str]) -> ChatPayload:
    """Inline-base64 PNG variant of :func:`make_image_payload`.

    The image bytes are read lazily on first `.body` access so pytest
    collection does not fail when the LFS image pointer has not been pulled.
    """
    return Base64LazyChatPayload(
        prompt=_MULTIMODAL_COLOR_PROMPT,
        expected_response=expected_response,
        max_tokens=100,
        temperature=0.0,
        repeat_count=1,
    )


def make_video_payload(expected_response: list[str]) -> ChatPayload:
    """Standard video description payload using the local test video."""
    return chat_payload(
        [
            {"type": "text", "text": "Describe the video in detail"},
            {
                "type": "video_url",
                "video_url": {"url": LOCAL_VIDEO_TEST_URI},
            },
        ],
        repeat_count=1,
        expected_response=expected_response,
        temperature=0.0,
        max_tokens=100,
    )


def make_audio_payload(expected_response: list[str]) -> ChatPayload:
    """Standard audio transcription payload using the remote test WAV."""
    return chat_payload(
        [
            {"type": "text", "text": "What is recited in the audio?"},
            {
                "type": "audio_url",
                "audio_url": {"url": AUDIO_TEST_URL},
            },
        ],
        repeat_count=1,
        expected_response=expected_response,
        temperature=0.0,
        max_tokens=100,
    )


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MmCase:
    """One test case emitted by a topology.

    Each :class:`MmCase` produces exactly one ``EngineConfig`` keyed
    ``mm_{topology}_{short_name}[_{suffix}]``. The case carries its own
    payload (HTTP-URL, base64-inline, video, audio) and any per-case
    overrides on top of the parent :class:`TopologyConfig`.

    ``payload`` is the only required field; everything else either inherits
    from the parent ``TopologyConfig`` (marks, timeout) or extends it
    (extra script args, env overlay).
    """

    payload: BasePayload  # required — fully formed test payload
    suffix: str = ""  # appended to config key; "" yields the bare topology key
    extra_script_args: list[str] = field(default_factory=list)
    marks: list[Any] = field(default_factory=list)  # if empty, inherit topology marks
    timeout_s: Optional[int] = None  # if None, inherit topology timeout
    env: dict[str, str] = field(
        default_factory=dict
    )  # extra env on top of topology env


@dataclass
class TopologyConfig:
    """Per-topology overrides for marks, timeout, and VRAM profiling.

    A topology must declare its tests explicitly via ``tests``. There is no
    implicit HTTP-URL test — each test case (HTTP, b64, video, audio) is a
    first-class :class:`MmCase` entry.
    """

    timeout_s: int = 300  # default for cases that don't override
    marks: list[Any] = field(default_factory=list)  # default for cases
    profiled_vram_gib: Optional[float] = None
    requested_vllm_kv_cache_bytes: Optional[int] = None
    delayed_start: int = 0
    directory: Optional[str] = None  # override profile-level directory
    gpu_marker: Optional[str] = None  # override profile-level gpu_marker
    single_gpu: bool = False  # append --single-gpu to script_args
    two_gpu: bool = False  # append --two-gpu to script_args (epd only)
    env: dict[str, str] = field(default_factory=dict)  # extra env vars for subprocess
    tests: list[MmCase] = field(default_factory=list)


@dataclass
class MultimodalModelProfile:
    """Describes a multimodal model's test-relevant properties.

    Each profile generates one config per :class:`MmCase` per topology
    via :func:`make_multimodal_configs`.
    """

    name: str  # HuggingFace model ID
    short_name: str  # kebab-case slug for config key
    topologies: dict[str, TopologyConfig]
    gpu_marker: str = "gpu_1"
    extra_vllm_args: list[str] = field(default_factory=list)
    marks: list[Any] = field(default_factory=list)  # shared across all topologies
    gated: bool = False  # if True, skip unless DYN_HF_GATED_MODELS_ENABLED=1


# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------


def make_multimodal_configs(
    profile: MultimodalModelProfile,
    config_cls: Type[EngineConfig],
    directory: str,
    topology_scripts: dict[str, str],
) -> dict[str, EngineConfig]:
    """Generate one ``EngineConfig`` per :class:`MmCase` per topology.

    Each case in ``topology.tests`` produces a config keyed
    ``mm_{topology}_{short_name}[_{case.suffix}]``. There is no implicit
    HTTP-URL config — a topology with empty ``tests`` emits nothing.

    Parameters
    ----------
    config_cls:
        The concrete config class to instantiate (e.g. ``VLLMConfig``).
    directory:
        Default directory; overridden by ``TopologyConfig.directory`` if set.
    topology_scripts:
        Mapping from topology key to shell script filename.
    """
    configs: dict[str, EngineConfig] = {}
    for topology, topo_cfg in profile.topologies.items():
        script_name = topology_scripts[topology]
        base_script_args = ["--model", profile.name] + profile.extra_vllm_args
        if topo_cfg.single_gpu:
            base_script_args.append("--single-gpu")
        if topo_cfg.two_gpu:
            base_script_args.append("--two-gpu")

        gpu = topo_cfg.gpu_marker or profile.gpu_marker
        topo_env = {
            "DYN_MM_ALLOW_INTERNAL": "1",
            "DYN_MM_LOCAL_PATH": str(WORKSPACE_DIR),
            **topo_cfg.env,
        }

        for case in topo_cfg.tests:
            key_suffix = f"_{case.suffix}" if case.suffix else ""
            key = f"mm_{topology}_{profile.short_name}{key_suffix}"

            timeout = case.timeout_s or topo_cfg.timeout_s
            marks: list[Any] = [
                getattr(pytest.mark, gpu),
                pytest.mark.timeout(timeout),
                pytest.mark.multimodal,
            ]
            marks.extend(case.marks if case.marks else topo_cfg.marks)
            if topo_cfg.profiled_vram_gib is not None:
                marks.append(pytest.mark.profiled_vram_gib(topo_cfg.profiled_vram_gib))
            if topo_cfg.requested_vllm_kv_cache_bytes is not None:
                marks.append(
                    pytest.mark.requested_vllm_kv_cache_bytes(
                        topo_cfg.requested_vllm_kv_cache_bytes
                    )
                )
            if profile.gated:
                marks.append(
                    pytest.mark.skipif(
                        not os.environ.get("DYN_HF_GATED_MODELS_ENABLED"),
                        reason=(
                            f"{profile.name} is gated; set "
                            "DYN_HF_GATED_MODELS_ENABLED=1 with an HF_TOKEN "
                            "that has accepted the license"
                        ),
                    )
                )
            marks.extend(profile.marks)

            configs[key] = config_cls(
                name=key,
                directory=topo_cfg.directory or directory,
                script_name=script_name,
                model=profile.name,
                script_args=list(base_script_args) + list(case.extra_script_args),
                marks=marks,
                delayed_start=topo_cfg.delayed_start,
                request_payloads=[case.payload],
                env={**topo_env, **case.env},
            )

    return configs
