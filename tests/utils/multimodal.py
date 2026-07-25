# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import os
import re
from copy import deepcopy
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
IMAGE_COLOR_PROMPT = _MULTIMODAL_COLOR_PROMPT

# Topic-aligned filler sentence used by callers that need to pad the prompt
# past a coarse routing-block threshold without confusing the model's
# color-identification objective. ~30 tokens per copy.
_COLOR_PROMPT_FILLER = (
    " Analyze the image in detail; identify dominant tones, secondary hues, "
    "and surface textures across the foreground and background regions."
)


def make_image_payload(
    expected_response: list[str],
    *,
    max_attempts: int = 1,
    repeat_count: int = 1,
    expected_log: Optional[list[str]] = None,
) -> ChatPayload:
    """Standard image color-identification payload using MULTIMODAL_IMG_URL.

    ``max_attempts`` is the cap on validation attempts inside
    ``run_serve_deployment`` — set >1 only for known-flaky multimodal
    smoke checks (see tests/README.md "Flaky Tests"). The server stays
    up across attempts; only the request/response is re-issued.
    ``repeat_count`` sends the same image request repeatedly, which is useful
    for cache smoke coverage.
    ``expected_log`` contains regex patterns that must appear in the deployment
    log, allowing callers to verify that optional multimodal paths were enabled.
    """
    return chat_payload(
        [
            {"type": "text", "text": _MULTIMODAL_COLOR_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": MULTIMODAL_IMG_URL},
            },
        ],
        repeat_count=repeat_count,
        expected_response=expected_response,
        expected_log=expected_log,
        temperature=0.0,
        max_tokens=100,
        max_attempts=max_attempts,
    )


def make_image_payload_cached_tokens(
    expected_response: list[str],
    *,
    repeat_count: int = 3,
    min_cached_tokens: int = 1,
    require_rust_processor_init: bool = False,
    require_vllm_mm_processor_init: bool = False,
    min_routing_total_blocks: int = 0,
    min_avg_kv_hit_rate: float = 0.0,
    prompt_filler_repeats: int = 0,
) -> CachedTokensChatPayload:
    """Image payload that asserts MM-aware KV cache reuse on repeats.

    ``require_rust_processor_init`` / ``require_vllm_mm_processor_init`` assert
    the MM-routing init log fired. ``min_routing_total_blocks`` asserts the
    [ROUTING] block count is well above text-prefix fallback (~1-3 blocks).
    ``min_avg_kv_hit_rate`` asserts the post-R1 mean of router_kv_hit_rate
    >= threshold (fails closed when router-side hashes diverge from the worker).
    ``prompt_filler_repeats`` prepends N copies of a topic-aligned filler so
    the routing-token sequence spans multiple blocks on specs that use a
    coarse routing-block size (e.g. qwen3_5 family at block_size=544).
    """
    prompt = _MULTIMODAL_COLOR_PROMPT
    if prompt_filler_repeats > 0:
        prompt = (_COLOR_PROMPT_FILLER * prompt_filler_repeats).strip() + " " + prompt
    return CachedTokensChatPayload(
        body={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": MULTIMODAL_IMG_URL},
                        },
                    ],
                }
            ],
            "max_tokens": 50,
            "temperature": 0.0,
            "stream": False,
        },
        repeat_count=repeat_count,
        expected_response=expected_response,
        min_cached_tokens=min_cached_tokens,
        require_rust_processor_init=require_rust_processor_init,
        require_vllm_mm_processor_init=require_vllm_mm_processor_init,
        min_routing_total_blocks=min_routing_total_blocks,
        min_avg_kv_hit_rate=min_avg_kv_hit_rate,
    )


class UuidPassthroughChatPayload(ChatPayload):
    """Send a URL+UUID cache fill followed by a UUID-only cache hit.

    When ``exercise_embedding_cache`` is true, insert a second URL+UUID fill
    with a different UUID. Tests can pair this sequence with a one-image GPU
    encoder cache so the final request must load the first embedding from
    Dynamo's CPU embedding cache.
    """

    def __init__(
        self,
        *,
        expected_response: list[str],
        image_uuid: str = "dynamo-mm-cache-image-1",
        max_tokens: int = 100,
        temperature: float = 0.0,
        timeout: int = 60,
        exercise_embedding_cache: bool = False,
    ):
        self._uuid_image_uuid = image_uuid
        self._uuid_eviction_image_uuid = (
            f"{image_uuid}-eviction" if exercise_embedding_cache else None
        )
        self._uuid_final_expected_log = (
            [
                "Dynamo multimodal embedding cache hit: "
                f"identifier={re.escape(repr(image_uuid))}"
            ]
            if exercise_embedding_cache
            else []
        )
        repeat_count = 3 if exercise_embedding_cache else 2
        self._uuid_bodies = [
            self._build_uuid_body(index, max_tokens, temperature)
            for index in range(repeat_count)
        ]
        self._uuid_iterations_requested: set[int] = set()
        super().__init__(
            body=self._uuid_bodies[0],
            expected_response=expected_response,
            expected_log=[],
            repeat_count=repeat_count,
            timeout=timeout,
        )

    def with_model(self, model):
        payload = deepcopy(self)
        for body in payload._uuid_bodies:
            body["model"] = model
        payload.body = payload._uuid_bodies[0]
        payload._uuid_iterations_requested = set()
        payload.expected_log = []
        return payload

    def _build_uuid_body(
        self,
        request_index: int,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        eviction_image_uuid = self._uuid_eviction_image_uuid
        if request_index == 0:
            image_url = {"url": MULTIMODAL_IMG_URL}
            image_uuid = self._uuid_image_uuid
            text = _MULTIMODAL_COLOR_PROMPT
        elif request_index == 1 and eviction_image_uuid is not None:
            image_url = {"url": MULTIMODAL_IMG_URL}
            image_uuid = eviction_image_uuid
            text = _MULTIMODAL_COLOR_PROMPT
        else:
            image_url = None
            image_uuid = self._uuid_image_uuid
            text = (
                "What colors are prominent in the same image? "
                "Respond only with the colors."
            )

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": image_url,
                            "uuid": image_uuid,
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

    def body_for_iteration(self, iteration: int) -> dict:
        if not 0 <= iteration < self.repeat_count:
            raise IndexError(f"UUID payload iteration {iteration} is out of range")
        self._uuid_iterations_requested.add(iteration)
        self.expected_log = (
            list(self._uuid_final_expected_log)
            if iteration == self.repeat_count - 1
            else []
        )
        return self._uuid_bodies[iteration]

    def final_validation(self) -> None:
        assert self._uuid_iterations_requested == set(
            range(self.repeat_count)
        ), "UUID passthrough payload did not send the expected request sequence"


def make_image_payload_uuid_passthrough(
    expected_response: list[str],
    *,
    exercise_embedding_cache: bool = False,
) -> ChatPayload:
    """Build the maintained vLLM cached multimodal UUID smoke payload."""
    return UuidPassthroughChatPayload(
        expected_response=expected_response,
        exercise_embedding_cache=exercise_embedding_cache,
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


def make_custom_encoder_payload() -> ChatPayload:
    """Semantic check for the aggregated CustomEncoder path.

    The example HitchhikersVisionEncoder splices the embeddings of "the
    Ultimate Question of Life, the Universe, and Everything" at the image
    placeholder, so the assembled prompt must answer "42". The served image
    content is irrelevant (the encoder ignores the URL). ``expected_log``
    asserts the encoder loaded in-process at worker startup.
    """
    return chat_payload(
        [
            {
                "type": "text",
                "text": "Based on The Hitchhiker's Guide to the Galaxy, The Answer to",
            },
            {"type": "image_url", "image_url": {"url": MULTIMODAL_IMG_URL}},
            {"type": "text", "text": " is?"},
        ],
        repeat_count=1,
        expected_response=["42"],
        expected_log=["Loaded CustomEncoder", "_LinearEmbedsAdapter"],
        max_tokens=32,
        temperature=0.0,
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
    requested_sglang_kv_tokens: Optional[int] = None
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
            if topo_cfg.requested_sglang_kv_tokens is not None:
                marks.append(
                    pytest.mark.requested_sglang_kv_tokens(
                        topo_cfg.requested_sglang_kv_tokens
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
