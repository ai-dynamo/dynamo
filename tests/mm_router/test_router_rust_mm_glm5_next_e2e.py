# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GLM-5.3-Flash image routing against real vLLM and SGLang workers.

Architecture:
  Frontend (Rust preprocessor + KV router)
       └─ glm5_next planner: reads image geometry from the checkpoint's
          processor_config.json, validates the rendered prompt layout,
          expands <|image|> for routing only, forwards mm_hashes
       → vLLM or SGLang worker: receives the unexpanded prompt, expands the
         placeholder with its own processor, publishes KV events keyed by
         the forwarded mm_hash

The Rust tests pin the planner against transformers goldens and the Python
binding tests pin the resolver, but neither runs an engine. They cannot catch
the planner and the worker disagreeing on how many tokens an image expands
to, or a hash mismatch between the frontend's routing blocks and the worker's
KV events. These tests do, with ``tiny-random/glm-5.3-flash``: a randomly
initialized 227 MB checkpoint that keeps the tokenizer, chat template,
processor config and token ids of ``zai-org/GLM-5.3-Flash``.

Checks, per engine:
  1. Worker expansion: the router's ``total_blocks`` (planner-expanded prompt)
     must equal ``ceil(usage.prompt_tokens / block)``, where the engine reports
     ``prompt_tokens`` after its own placeholder expansion. Per-image token
     increments must follow GLM geometry.
  2. KV-event hash parity: a repeated request matches nearly every block the
     worker published, and the engine reports a prefix-cache hit.
  3. Different-image isolation: a new image on the same prompt matches neither
     at the router nor in the engine's prefix cache.
  4. Multiple images across block boundaries: non-block-aligned image runs
     agree block for block, and swapping the second image invalidates exactly
     the blocks from the boundary on.

Hardware: GLM5-Next's DSA layers need sparse MLA attention, which vLLM 0.32
and SGLang 0.5.22 implement only for Hopper and Blackwell (FlashMLA sparse,
FlashAttention-3, NSA). The checkpoint is tiny but cannot run on the 24 GiB
Ada runners behind the post-merge lanes, so this file is ``nightly`` +
``h100`` and runs in the ``vllm-h100-test`` / ``sglang-h100-test`` lanes.
"""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
import time
from typing import Any, Generator

import pytest
import requests

from tests.mm_router.utils import COMMON_PROCESS_KWARGS, make_png_bytes
from tests.utils.gpu_args import build_gpu_mem_args, build_vllm_gpu_mem_args
from tests.utils.managed_process import ManagedProcess, check_health_ready
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import reserved_ports
from tests.utils.router_logs import (
    extract_router_kv_overlap_records,
    wait_for_router_kv_overlap,
)

GLM_MM_MODEL = os.getenv("DYN_TEST_GLM_MM_MODEL", "tiny-random/glm-5.3-flash")
# FlashMLA sparse (vLLM) and the NSA backend (SGLang) page KV at 64 tokens. A
# 16-token frontend block would desync the router radix from the worker's KV
# events, so the frontend, the worker and the assertions all use 64.
BLOCK_SIZE = 64
MAX_MODEL_LEN = 4096
# Enough for every prompt this module caches at once; the tiny model's KV is
# a few KiB per token.
SGLANG_MAX_TOTAL_TOKENS = 32768

# GLM geometry (patch 14, merge 2): a square image whose side is 28*k expands
# to k*k tokens. 616 -> 484 and 364 -> 169, neither a multiple of 64, so every
# image run straddles block boundaries. The chat template wraps each image in
# <|begin_of_image|> ... <|end_of_image|>, two more tokens per image.
LARGE_IMAGE_SIDE = 616
LARGE_IMAGE_TOKENS = 484
SMALL_IMAGE_SIDE = 364
SMALL_IMAGE_TOKENS = 169
IMAGE_DELIMITER_TOKENS = 2

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.h100,
    pytest.mark.model(GLM_MM_MODEL),
]


def _make_process_env(namespace: str, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env["DYN_LOG"] = (
        "info,mm_routing=debug,"
        "dynamo_kv_router::scheduling=debug,"
        "dynamo_llm::kv_router=debug"
    )
    env["DYN_NAMESPACE"] = namespace
    env["DYN_REQUEST_PLANE"] = "tcp"
    env.update(extra)
    return env


def _prepare_log_dir(request, suffix: str) -> str:
    return tempfile.mkdtemp(prefix=f"{request.node.name}_{suffix}_")


def _kv_events_config(kv_event_port: int, **extra: Any) -> str:
    return json.dumps(
        {
            "publisher": "zmq",
            "topic": "kv-events",
            "endpoint": f"tcp://*:{kv_event_port}",
            **extra,
        }
    )


class VllmGlmWorkerProcess(ManagedProcess):
    """vLLM worker serving the tiny GLM checkpoint with KV events enabled."""

    def __init__(
        self, request, *, namespace: str, system_port: int, kv_event_port: int
    ):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                GLM_MM_MODEL,
                "--enable-multimodal",
                "--block-size",
                str(BLOCK_SIZE),
                "--enforce-eager",
                "--max-model-len",
                str(MAX_MODEL_LEN),
                # The checkpoint keeps the source's block-FP8 weights with 32x32
                # scale blocks. DeepGEMM and CUTLASS block GEMMs assume 128x128,
                # so pin the Triton backends the checkpoint card validates.
                "--kernel-config",
                json.dumps({"moe_backend": "triton", "linear_backend": "triton"}),
                *build_vllm_gpu_mem_args("0.30"),
                "--kv-events-config",
                _kv_events_config(kv_event_port, enable_kv_cache_events=True),
            ],
            env=_make_process_env(namespace, DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", check_health_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=_prepare_log_dir(request, "vllm-glm-worker"),
            **COMMON_PROCESS_KWARGS,
        )


class SglangGlmWorkerProcess(ManagedProcess):
    """SGLang worker serving the tiny GLM checkpoint with KV events enabled."""

    def __init__(
        self,
        request,
        *,
        namespace: str,
        system_port: int,
        kv_event_port: int,
        nccl_port: int,
    ):
        # Token-cap form, see tests/README.md "SGLang KV tokens". The scheduler
        # override wins when a profiling run sets it.
        mem_args = build_gpu_mem_args("build_sglang_gpu_mem_args") or [
            "--max-total-tokens",
            str(SGLANG_MAX_TOTAL_TOKENS),
            "--mem-fraction-static",
            "0.9",
        ]
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.sglang",
                "--model-path",
                GLM_MM_MODEL,
                "--served-model-name",
                GLM_MM_MODEL,
                # The Rust frontend tokenizes; the worker receives token ids plus
                # image URLs and lets SGLang's processor expand the placeholder.
                "--skip-tokenizer-init",
                "--trust-remote-code",
                "--tp",
                "1",
                "--page-size",
                str(BLOCK_SIZE),
                "--context-length",
                str(MAX_MODEL_LEN),
                "--disable-cuda-graph",
                "--disable-piecewise-cuda-graph",
                *mem_args,
                "--kv-events-config",
                _kv_events_config(kv_event_port),
                # Pin the torch.distributed rendezvous port so a parallel test
                # cannot take it between SGLang's probe and its real bind.
                "--nccl-port",
                str(nccl_port),
            ],
            env=_make_process_env(
                namespace,
                DYN_SYSTEM_PORT=str(system_port),
                # Same 32x32 FP8 scale blocks as above: DeepGEMM expects 128-wide
                # blocks, so keep SGLang on its Triton block-FP8 kernels.
                SGLANG_ENABLE_JIT_DEEPGEMM="0",
            ),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", check_health_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.sglang"],
            log_dir=_prepare_log_dir(request, "sglang-glm-worker"),
            **COMMON_PROCESS_KWARGS,
        )


class FrontendProcess(ManagedProcess):
    """Rust frontend with the KV router and the default Dynamo chat processor."""

    def __init__(self, request, *, namespace: str, frontend_port: int):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(frontend_port),
                "--router-mode",
                "kv",
                "--kv-cache-block-size",
                str(BLOCK_SIZE),
                "--model-name",
                GLM_MM_MODEL,
            ],
            env=_make_process_env(namespace),
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=240,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=_prepare_log_dir(request, "router-rust-glm-frontend"),
            **COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("vllm", marks=pytest.mark.vllm),
        pytest.param("sglang", marks=pytest.mark.sglang),
    ],
)
def start_glm_mm_services(
    request, mm_runtime_services
) -> Generator[tuple[int, ManagedProcess], None, None]:
    """One worker plus one KV-routing frontend per engine.

    Each engine gets its own namespace so the second frontend never sees the
    first worker's registration while its etcd lease expires.
    """
    backend = request.param
    namespace = f"router-rust-mm-glm-{backend}"
    with reserved_ports(count=4, start_port=12000) as ports:
        frontend_port, system_port, kv_event_port, nccl_port = ports
        worker: ManagedProcess
        if backend == "vllm":
            worker = VllmGlmWorkerProcess(
                request,
                namespace=namespace,
                system_port=system_port,
                kv_event_port=kv_event_port,
            )
        else:
            worker = SglangGlmWorkerProcess(
                request,
                namespace=namespace,
                system_port=system_port,
                kv_event_port=kv_event_port,
                nccl_port=nccl_port,
            )
        with worker:
            time.sleep(2)  # allow the ZMQ KV-event publisher to bind
            with FrontendProcess(
                request, namespace=namespace, frontend_port=frontend_port
            ) as frontend_proc:
                yield frontend_port, frontend_proc


def _make_data_uri(color: tuple[int, int, int], side: int) -> str:
    b64 = base64.b64encode(make_png_bytes(color, side)).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _build_payload(image_uris: list[str], prompt: str) -> dict[str, Any]:
    # Always a content list, so a text-only prompt renders through the same
    # template branch as the image prompts it is compared against.
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for uri in image_uris:
        content.append({"type": "image_url", "image_url": {"url": uri}})
    return {
        "model": GLM_MM_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4,
    }


def _send(
    frontend_port: int,
    router_proc: ManagedProcess,
    payload: dict[str, Any],
    label: str,
) -> tuple[int, int, dict[str, Any], str]:
    """Send one request; return router overlap/total, engine usage, and logs."""
    pre_request_logs = router_proc.read_logs()
    start_offset = len(pre_request_logs)
    pre_request_record_count = len(extract_router_kv_overlap_records(pre_request_logs))
    resp = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=payload,
        timeout=240,
    )
    assert resp.status_code == 200, f"{label}: HTTP {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "choices" in data, f"{label}: missing choices in response: {data}"
    usage = data.get("usage")
    assert isinstance(usage, dict), f"{label}: missing usage in response: {data}"
    overlap, total, recent_logs = wait_for_router_kv_overlap(
        router_proc.read_logs,
        start_offset=start_offset,
        pre_request_record_count=pre_request_record_count,
        context=label,
        log_label="frontend",
    )
    print(f"[ROUTER_RUST_MM_GLM] {label}: overlap={overlap}/{total} usage={usage}")
    # Let the worker's KV events for this request reach the router before the
    # next request is scored against them.
    time.sleep(2)
    return overlap, total, usage, recent_logs


def _prompt_tokens(label: str, usage: dict[str, Any]) -> int:
    prompt_tokens = usage.get("prompt_tokens")
    assert (
        isinstance(prompt_tokens, int) and prompt_tokens > 0
    ), f"{label}: engine did not report prompt_tokens, usage={usage}"
    return prompt_tokens


def _cached_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens")


def _assert_worker_expansion_matches_router(
    label: str, total_blocks: int, usage: dict[str, Any], recent_logs: str
) -> int:
    """The planner-expanded routing prompt and the engine's expanded prompt
    must cover the same number of blocks."""
    prompt_tokens = _prompt_tokens(label, usage)
    expected_blocks = math.ceil(prompt_tokens / BLOCK_SIZE)
    assert total_blocks == expected_blocks, (
        f"{label}: the router hashed {total_blocks} blocks from the planner-expanded "
        f"prompt, but the worker reports prompt_tokens={prompt_tokens}, which is "
        f"{expected_blocks} blocks of {BLOCK_SIZE}. The glm5_next planner and the "
        f"engine processor disagree on image token expansion.\n"
        f"Recent frontend logs:\n{recent_logs[-4000:]}"
    )
    return prompt_tokens


def _assert_nearly_full_repeat_overlap(
    label: str, overlap: int, total: int, recent_logs: str
) -> None:
    """Every full block the worker published must match; only the trailing
    partial block, which the engine never caches, may miss."""
    min_expected = max(1, total - 1)
    assert overlap >= min_expected, (
        f"{label}: expected the repeat to match nearly every cached block, got "
        f"{overlap}/{total}, expected >= {min_expected}/{total}. The frontend's "
        f"block hashes do not reproduce the worker's KV events.\n"
        f"Recent frontend logs:\n{recent_logs[-4000:]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_glm_repeated_image_cache_overlap(start_glm_mm_services, predownload_models):
    """Same image twice: the worker expands the placeholder to the planner's
    count, and the repeat hits the worker's KV blocks at the router and in the
    engine's prefix cache."""
    frontend_port, router_proc = start_glm_mm_services
    image = _make_data_uri((201, 17, 99), LARGE_IMAGE_SIDE)
    payload = _build_payload([image], "GLM routing e2e: repeated single image.")

    overlap_1, total_1, usage_1, logs_1 = _send(
        frontend_port, router_proc, payload, "single_req1"
    )
    overlap_2, total_2, usage_2, logs_2 = _send(
        frontend_port, router_proc, payload, "single_req2"
    )

    prompt_tokens_1 = _assert_worker_expansion_matches_router(
        "single_req1", total_1, usage_1, logs_1
    )
    _assert_worker_expansion_matches_router("single_req2", total_2, usage_2, logs_2)
    assert prompt_tokens_1 > LARGE_IMAGE_TOKENS, (
        f"single_req1: prompt_tokens={prompt_tokens_1} is not larger than the "
        f"{LARGE_IMAGE_TOKENS} image tokens a {LARGE_IMAGE_SIDE}px image expands to; "
        f"the worker did not expand the image placeholder."
    )
    assert (
        total_1 == total_2
    ), f"total blocks changed between repeats: {total_1} -> {total_2}"

    assert overlap_1 <= 1, (
        f"single_req1: a never-seen image should overlap at most the text prefix "
        f"block, got {overlap_1}/{total_1}.\nRecent frontend logs:\n{logs_1[-4000:]}"
    )
    _assert_nearly_full_repeat_overlap("single_req2", overlap_2, total_2, logs_2)
    assert overlap_2 > overlap_1 + 1, (
        f"single_req2: the repeat must match the image blocks, not just the text "
        f"prefix, got req1={overlap_1}/{total_1}, req2={overlap_2}/{total_2}"
    )

    cached_2 = _cached_tokens(usage_2)
    prompt_tokens_2 = _prompt_tokens("single_req2", usage_2)
    assert cached_2 is not None and cached_2 > prompt_tokens_2 // 2, (
        f"single_req2: expected an engine-side prefix-cache hit on the repeat, got "
        f"cached_tokens={cached_2} of prompt_tokens={prompt_tokens_2}"
    )

    log_text = router_proc.read_logs()
    assert (
        "MM-aware KV routing enabled" in log_text
    ), "frontend should log MM-routing initialization for the GLM checkpoint"
    assert (
        "resolved image-placeholder token id" in log_text
    ), "frontend should log which tier resolved the GLM image placeholder"


@pytest.mark.timeout(900)
def test_glm_different_image_isolation(start_glm_mm_services, predownload_models):
    """Same prompt, different image: the warm image keeps its overlap while the
    new image matches neither the router's blocks nor the engine's cache."""
    frontend_port, router_proc = start_glm_mm_services
    prompt = "GLM routing e2e: different image isolation."
    payload_a = _build_payload([_make_data_uri((11, 22, 33), LARGE_IMAGE_SIDE)], prompt)
    payload_b = _build_payload(
        [_make_data_uri((222, 111, 200), LARGE_IMAGE_SIDE)], prompt
    )

    _send(frontend_port, router_proc, payload_a, "warm_A")
    overlap_a_repeat, total_a, usage_a, logs_a = _send(
        frontend_port, router_proc, payload_a, "A_repeat"
    )
    overlap_b_cold, total_b, usage_b, logs_b = _send(
        frontend_port, router_proc, payload_b, "B_cold"
    )

    _assert_worker_expansion_matches_router("A_repeat", total_a, usage_a, logs_a)
    prompt_tokens_b = _assert_worker_expansion_matches_router(
        "B_cold", total_b, usage_b, logs_b
    )
    assert (
        total_b == total_a
    ), f"same geometry must expand identically, got A={total_a}, B={total_b} blocks"

    _assert_nearly_full_repeat_overlap("A_repeat", overlap_a_repeat, total_a, logs_a)
    assert overlap_b_cold <= 1, (
        f"B_cold: a first-time image must overlap at most the text prefix block; "
        f"got {overlap_b_cold}/{total_b}. The image content is not part of the "
        f"routing key.\nRecent frontend logs:\n{logs_b[-4000:]}"
    )
    assert overlap_a_repeat > overlap_b_cold, (
        f"warm-A overlap ({overlap_a_repeat}/{total_a}) must exceed cold-B overlap "
        f"({overlap_b_cold}/{total_b}); equal means image hashes collide"
    )

    cached_b = _cached_tokens(usage_b)
    assert cached_b is not None and cached_b <= BLOCK_SIZE, (
        f"B_cold: the engine reused {cached_b} of {prompt_tokens_b} prompt tokens "
        f"for an image it had never seen; the forwarded mm_hash does not isolate "
        f"image B from image A in the engine's prefix cache"
    )


@pytest.mark.timeout(900)
def test_glm_multiple_images_across_block_boundaries(
    start_glm_mm_services, predownload_models
):
    """Two images whose token runs are not block aligned: per-image expansion
    follows GLM geometry on the worker, repeats agree block for block, and
    swapping the second image invalidates exactly the blocks after the first
    image's last full block."""
    frontend_port, router_proc = start_glm_mm_services
    prompt = "GLM routing e2e: multiple images across block boundaries."
    image_a = _make_data_uri((17, 99, 201), LARGE_IMAGE_SIDE)
    image_b = _make_data_uri((89, 210, 34), SMALL_IMAGE_SIDE)
    image_c = _make_data_uri((144, 61, 230), SMALL_IMAGE_SIDE)
    payload_text = _build_payload([], prompt)
    payload_a = _build_payload([image_a], prompt)
    payload_ab = _build_payload([image_a, image_b], prompt)
    payload_ac = _build_payload([image_a, image_c], prompt)
    payload_aa = _build_payload([image_a, image_a], prompt)

    _, total_text, usage_text, logs_text = _send(
        frontend_port, router_proc, payload_text, "text_only"
    )
    _, total_a1, usage_a1, logs_a1 = _send(
        frontend_port, router_proc, payload_a, "A_cold"
    )
    overlap_a2, total_a2, usage_a2, logs_a2 = _send(
        frontend_port, router_proc, payload_a, "A_repeat"
    )
    _, total_ab1, usage_ab1, logs_ab1 = _send(
        frontend_port, router_proc, payload_ab, "AB_cold"
    )
    overlap_ab2, total_ab2, usage_ab2, logs_ab2 = _send(
        frontend_port, router_proc, payload_ab, "AB_repeat"
    )
    overlap_ac, total_ac, usage_ac, logs_ac = _send(
        frontend_port, router_proc, payload_ac, "AC_cold"
    )
    _, total_aa, usage_aa, logs_aa = _send(
        frontend_port, router_proc, payload_aa, "AA_cold"
    )

    # Worker expansion: every request's engine token count covers exactly the
    # blocks the router hashed from the planner's expansion.
    tokens_text = _assert_worker_expansion_matches_router(
        "text_only", total_text, usage_text, logs_text
    )
    tokens_a = _assert_worker_expansion_matches_router(
        "A_cold", total_a1, usage_a1, logs_a1
    )
    _assert_worker_expansion_matches_router("A_repeat", total_a2, usage_a2, logs_a2)
    tokens_ab = _assert_worker_expansion_matches_router(
        "AB_cold", total_ab1, usage_ab1, logs_ab1
    )
    _assert_worker_expansion_matches_router("AB_repeat", total_ab2, usage_ab2, logs_ab2)
    tokens_ac = _assert_worker_expansion_matches_router(
        "AC_cold", total_ac, usage_ac, logs_ac
    )
    tokens_aa = _assert_worker_expansion_matches_router(
        "AA_cold", total_aa, usage_aa, logs_aa
    )

    # Each image adds its geometry's token count plus the template delimiters,
    # measured entirely on the engine side.
    large_increment = LARGE_IMAGE_TOKENS + IMAGE_DELIMITER_TOKENS
    small_increment = SMALL_IMAGE_TOKENS + IMAGE_DELIMITER_TOKENS
    assert tokens_a - tokens_text == large_increment, (
        f"one {LARGE_IMAGE_SIDE}px image added {tokens_a - tokens_text} prompt tokens "
        f"on the worker, expected {large_increment}"
    )
    assert tokens_ab - tokens_a == small_increment, (
        f"appending a {SMALL_IMAGE_SIDE}px image added {tokens_ab - tokens_a} prompt "
        f"tokens on the worker, expected {small_increment}"
    )
    assert tokens_aa - tokens_a == large_increment, (
        f"appending a second {LARGE_IMAGE_SIDE}px image added {tokens_aa - tokens_a} "
        f"prompt tokens on the worker, expected {large_increment}"
    )
    assert tokens_ac == tokens_ab, (
        f"same geometry, different content must expand identically, got "
        f"AB={tokens_ab}, AC={tokens_ac}"
    )

    # KV-event hash parity across both image runs and the boundary they share.
    _assert_nearly_full_repeat_overlap("A_repeat", overlap_a2, total_a2, logs_a2)
    _assert_nearly_full_repeat_overlap("AB_repeat", overlap_ab2, total_ab2, logs_ab2)

    # A then C shares header, text, image A and C's opening delimiter with the
    # cached A-then-B prompt. Blocks wholly inside that prefix must match; the
    # block where C's tokens begin must not, and nothing after it may either.
    # Image A's cached blocks bound the match: at most all of them, at least
    # all but the one that in the single-image prompt held the suffix instead.
    assert overlap_a2 - 1 <= overlap_ac <= overlap_a2, (
        f"AC_cold: expected the overlap to stop at image A's last full block, "
        f"within [{overlap_a2 - 1}, {overlap_a2}], got {overlap_ac}/{total_ac} "
        f"(A repeat matched {overlap_a2}/{total_a2}).\n"
        f"Recent frontend logs:\n{logs_ac[-4000:]}"
    )
    assert overlap_ac < overlap_ab2, (
        f"AC_cold: swapping the second image must lose overlap versus the AB repeat, "
        f"got AC={overlap_ac}/{total_ac}, AB repeat={overlap_ab2}/{total_ab2}; the "
        f"second image's content is not part of its blocks' routing keys"
    )
