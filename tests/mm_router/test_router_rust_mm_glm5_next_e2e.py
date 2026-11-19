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
KV events. These tests do, with a pinned revision of
``inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small``: a 5 GB BF16 cut of
``zai-org/GLM-5.3-Flash`` down to three text layers (two KDA linear-attention
layers and one DSA layer) that keeps the base model's tokenizer, chat
template, processor config and token ids.

Block size: the engines size KV blocks for the hybrid layers themselves. vLLM
raises the requested 64 tokens until an attention page is at least as large as
a linear-attention state, and SGLang can raise its page size as well (on GB200
in November 2026, vLLM advertised 4352 tokens and SGLang 256). The router
hashes at the size the worker advertises, so the module reads that size from
the frontend's metrics and derives the image sizes and every block assertion
from it.

Checks, per engine:
  1. Worker expansion, on every request: the router's ``total_blocks``
     (planner-expanded prompt) must equal ``ceil(usage.prompt_tokens / block)``,
     where the engine reports ``prompt_tokens`` after its own placeholder
     expansion. Per-image token increments must follow GLM geometry.
  2. KV-event hash parity: a repeated request matches nearly every block the
     worker published, and the engine reports a prefix-cache hit.
  3. Different-image isolation: a new image on the same prompt matches neither
     at the router nor in the engine's prefix cache.
  4. Multiple images across block boundaries: image runs that straddle block
     boundaries agree block for block, and swapping the second image keeps
     the first image's blocks and invalidates exactly the blocks from the
     boundary on.

Hardware: GLM5-Next's DSA layers need sparse MLA attention, which vLLM and
SGLang implement only for Hopper and Blackwell as of November 2026 (FlashMLA
sparse, FlashAttention-3, NSA). The checkpoint is small but cannot run on the
24 GiB Ada runners behind the post-merge lanes, so this file is ``nightly`` +
``h100`` and runs in the ``vllm-h100-test`` / ``sglang-h100-test`` lanes.

SGLang builds that fill image tokens with pad values of their own (see
``sglang_has_wide_mm_pad_values``) publish KV events the router cannot match
to image content: it keeps every block from the first such value on out of
its index, and every image prompt here holds one in its first block. The
SGLang tests skip on those builds; the skip reason names the change that
restores exact routing.
"""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Generator

import pytest
import requests
from huggingface_hub import snapshot_download

from dynamo.prometheus_names import frontend_service, name_prefix
from tests.mm_router.utils import COMMON_PROCESS_KWARGS, make_png_bytes
from tests.utils.gpu_args import build_gpu_mem_args, build_vllm_gpu_mem_args
from tests.utils.managed_process import ManagedProcess, check_health_ready
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import reserved_ports
from tests.utils.prometheus import find_metric_samples
from tests.utils.router_logs import (
    extract_router_kv_overlap_records,
    wait_for_router_kv_overlap,
)

DEFAULT_GLM_MM_MODEL = "inference-optimization/GLM-5.3-Flash-2.6B-A2.3B-small"
# Pinned so a later push to the checkpoint repository cannot change what the
# nightly lanes run. An overridden model follows its default branch unless
# DYN_TEST_GLM_MM_REVISION names a revision.
DEFAULT_GLM_MM_REVISION = "a2d7ca22e78d7961d47ce820af0c37e67a62a657"
GLM_MM_MODEL = os.getenv("DYN_TEST_GLM_MM_MODEL", DEFAULT_GLM_MM_MODEL)
GLM_MM_REVISION = os.getenv("DYN_TEST_GLM_MM_REVISION") or (
    DEFAULT_GLM_MM_REVISION if GLM_MM_MODEL == DEFAULT_GLM_MM_MODEL else None
)
# Requested KV block size. The engines raise it for the hybrid layers, so the
# assertions use the size the worker advertises instead.
REQUESTED_BLOCK_SIZE = 64
# GLM caps an image at 8000 tokens, so the longest prompt here, two capped
# images, stays under half of this.
MAX_MODEL_LEN = 32768
# Room for every prompt this module caches at once, even when a large block
# size pushes the images toward GLM's 8000-token cap.
SGLANG_MAX_TOTAL_TOKENS = 131072

# GLM geometry (patch 14, merge 2): a square image whose side is 28*k expands
# to k*k tokens, for k from 4 to 89 (GLM's 16- and 8000-token limits). The
# chat template wraps each image in <|begin_of_image|> ... <|end_of_image|>,
# two more tokens per image.
GLM_IMAGE_SIDE_UNIT = 28
GLM_MAX_IMAGE_TOKENS = 89 * 89
IMAGE_DELIMITER_TOKENS = 2

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.e2e,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.h100,
    pytest.mark.model(GLM_MM_MODEL, revision=GLM_MM_REVISION),
]


@dataclass(frozen=True)
class SquareImage:
    """A solid square image that GLM expands without resizing."""

    side: int
    tokens: int


@dataclass(frozen=True)
class GlmServices:
    """One engine's frontend, with image sizes derived from its KV blocks."""

    frontend_port: int
    frontend: ManagedProcess
    block_size: int
    large_image: SquareImage
    small_image: SquareImage


@dataclass(frozen=True)
class RequestObservation:
    """What the router and the engine reported for one request."""

    label: str
    overlap_blocks: int
    total_blocks: int
    prompt_tokens: int
    cached_tokens: int | None
    logs: str


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
    """vLLM worker serving the GLM checkpoint with KV events enabled."""

    def __init__(
        self,
        request,
        *,
        model_path: str,
        namespace: str,
        system_port: int,
        kv_event_port: int,
    ):
        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                model_path,
                "--served-model-name",
                GLM_MM_MODEL,
                "--enable-multimodal",
                "--block-size",
                str(REQUESTED_BLOCK_SIZE),
                "--enforce-eager",
                "--max-model-len",
                str(MAX_MODEL_LEN),
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
    """SGLang worker serving the GLM checkpoint with KV events enabled."""

    def __init__(
        self,
        request,
        *,
        model_path: str,
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
                model_path,
                "--served-model-name",
                GLM_MM_MODEL,
                # The Rust frontend tokenizes; the worker receives token ids plus
                # image URLs and lets SGLang's processor expand the placeholder.
                "--skip-tokenizer-init",
                "--trust-remote-code",
                "--tp",
                "1",
                "--page-size",
                str(REQUESTED_BLOCK_SIZE),
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
            env=_make_process_env(namespace, DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", check_health_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.sglang"],
            log_dir=_prepare_log_dir(request, "sglang-glm-worker"),
            **COMMON_PROCESS_KWARGS,
        )


class FrontendProcess(ManagedProcess):
    """Rust frontend with the KV router and the default Dynamo chat processor.

    The router takes its KV block size from the worker's registration.
    """

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


def _square_image(min_tokens: int) -> SquareImage:
    """Return the smallest square image that expands to at least ``min_tokens``."""
    k = math.isqrt(min_tokens - 1) + 1
    if k * k > GLM_MAX_IMAGE_TOKENS:
        pytest.fail(
            f"a {min_tokens}-token image exceeds GLM's {GLM_MAX_IMAGE_TOKENS}-token "
            f"square-image limit: the worker's KV block is too large for these "
            f"prompts to span several blocks"
        )
    return SquareImage(side=GLM_IMAGE_SIDE_UNIT * k, tokens=k * k)


def _images_for_block(block_size: int) -> tuple[SquareImage, SquareImage]:
    """Size the test images from the worker's KV block size.

    The large image covers about one and a half blocks, so a single-image
    prompt fills at least one block that holds image tokens. The small image
    is longer than a block, so a second image always carries a prompt into at
    least one more full block than its first image reached.
    """
    large = _square_image(min(block_size + block_size // 2, GLM_MAX_IMAGE_TOKENS))
    small = _square_image(block_size + 1)
    return large, small


def _advertised_block_size(frontend_port: int, timeout_s: float = 60.0) -> int:
    """Read the KV block size the worker registered, as the frontend exports it.

    The router hashes prompts at this size, whatever size the worker was asked
    for.
    """
    metric = f"{name_prefix.FRONTEND}_{frontend_service.MODEL_KV_CACHE_BLOCK_SIZE}"
    deadline = time.monotonic() + timeout_s
    while True:
        response = requests.get(f"http://localhost:{frontend_port}/metrics", timeout=10)
        response.raise_for_status()
        samples = find_metric_samples(response.text, metric, {"model": GLM_MM_MODEL})
        if samples and samples[0] > 0:
            return int(samples[0])
        if time.monotonic() >= deadline:
            pytest.fail(
                f"frontend did not export {metric} for {GLM_MM_MODEL} within "
                f"{timeout_s:.0f}s"
            )
        time.sleep(1)


@pytest.fixture(scope="module")
def glm_model_path(predownload_models) -> str:
    """Local snapshot of the pinned checkpoint revision.

    The ``model`` mark carries the revision, so predownload_models caches this
    snapshot before it turns on HF_HUB_OFFLINE; here it is only looked up. The
    workers load and register this directory, so both engines and the
    frontend's planner read the same tokenizer, chat template and processor
    config, whatever the repository's default branch holds later.
    """
    return snapshot_download(
        repo_id=GLM_MM_MODEL, revision=GLM_MM_REVISION, local_files_only=True
    )


@pytest.fixture(
    scope="module",
    params=[
        pytest.param("vllm", marks=pytest.mark.vllm),
        pytest.param("sglang", marks=pytest.mark.sglang),
    ],
)
def start_glm_mm_services(
    request, mm_runtime_services, glm_model_path
) -> Generator[GlmServices, None, None]:
    """One worker plus one KV-routing frontend per engine.

    Each engine gets its own namespace so the second frontend never sees the
    first worker's registration while its etcd lease expires.
    """
    backend = request.param
    if backend == "sglang":
        # Deferred so that collection never imports SGLang.
        from dynamo.sglang._compat import sglang_has_wide_mm_pad_values

        if sglang_has_wide_mm_pad_values():
            # The router drops image blocks from wide-pad sources (#17126).
            pytest.skip("exact SGLang image routing needs #17082")

    namespace = f"router-rust-mm-glm-{backend}"
    with reserved_ports(count=4, start_port=12000) as ports:
        frontend_port, system_port, kv_event_port, nccl_port = ports
        worker: ManagedProcess
        if backend == "vllm":
            worker = VllmGlmWorkerProcess(
                request,
                model_path=glm_model_path,
                namespace=namespace,
                system_port=system_port,
                kv_event_port=kv_event_port,
            )
        else:
            worker = SglangGlmWorkerProcess(
                request,
                model_path=glm_model_path,
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
                block_size = _advertised_block_size(frontend_port)
                large_image, small_image = _images_for_block(block_size)
                print(
                    f"[ROUTER_RUST_MM_GLM] {backend}: worker block_size={block_size}, "
                    f"large image {large_image}, small image {small_image}"
                )
                yield GlmServices(
                    frontend_port=frontend_port,
                    frontend=frontend_proc,
                    block_size=block_size,
                    large_image=large_image,
                    small_image=small_image,
                )


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
    services: GlmServices, payload: dict[str, Any], label: str
) -> RequestObservation:
    """Send one request and return what the router and the engine reported,
    after checking that both expanded the prompt to the same block count."""
    router_proc = services.frontend
    pre_request_logs = router_proc.read_logs()
    start_offset = len(pre_request_logs)
    pre_request_record_count = len(extract_router_kv_overlap_records(pre_request_logs))
    resp = requests.post(
        f"http://localhost:{services.frontend_port}/v1/chat/completions",
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
    observation = RequestObservation(
        label=label,
        overlap_blocks=overlap,
        total_blocks=total,
        prompt_tokens=_prompt_tokens(label, usage),
        cached_tokens=_cached_tokens(usage),
        logs=recent_logs,
    )
    _assert_worker_expansion_matches_router(observation, services.block_size)
    # Let the worker's KV events for this request reach the router before the
    # next request is scored against them.
    time.sleep(2)
    return observation


def _prompt_tokens(label: str, usage: dict[str, Any]) -> int:
    prompt_tokens = usage.get("prompt_tokens")
    assert (
        isinstance(prompt_tokens, int) and prompt_tokens > 0
    ), f"{label}: engine did not report prompt_tokens, usage={usage}"
    return prompt_tokens


def _cached_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens")


def _text_only_blocks(services: GlmServices, prompt: str, label: str) -> int:
    """Count the full blocks a prompt's text fills before its first image.

    Image prompts with the same text share at most the text-only prompt's
    header and text, so only these blocks may match across different images.
    """
    text_only = _send(services, _build_payload([], prompt), label)
    return text_only.prompt_tokens // services.block_size


def _assert_worker_expansion_matches_router(
    observation: RequestObservation, block_size: int
) -> None:
    """The planner-expanded routing prompt and the engine's expanded prompt
    must cover the same number of blocks."""
    expected_blocks = math.ceil(observation.prompt_tokens / block_size)
    assert observation.total_blocks == expected_blocks, (
        f"{observation.label}: the router hashed {observation.total_blocks} blocks "
        f"from the planner-expanded prompt, but the worker reports "
        f"prompt_tokens={observation.prompt_tokens}, which is {expected_blocks} "
        f"blocks of {block_size}. The glm5_next planner and the engine processor "
        f"disagree on image token expansion.\n"
        f"Recent frontend logs:\n{observation.logs[-4000:]}"
    )


def _assert_nearly_full_repeat_overlap(repeat: RequestObservation) -> None:
    """Every full block the worker published must match; only the trailing
    partial block, which the engine never caches, may miss."""
    min_expected = max(1, repeat.total_blocks - 1)
    assert repeat.overlap_blocks >= min_expected, (
        f"{repeat.label}: expected the repeat to match nearly every cached block, "
        f"got {repeat.overlap_blocks}/{repeat.total_blocks}, expected >= "
        f"{min_expected}/{repeat.total_blocks}. The frontend's block hashes do not "
        f"reproduce the worker's KV events.\n"
        f"Recent frontend logs:\n{repeat.logs[-4000:]}"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_glm_repeated_image_cache_overlap(start_glm_mm_services, predownload_models):
    """Same image twice: the worker expands the placeholder to the planner's
    count, and the repeat hits the worker's KV blocks at the router and in the
    engine's prefix cache."""
    services = start_glm_mm_services
    image = services.large_image
    prompt = "GLM routing e2e: repeated single image."
    text_blocks = _text_only_blocks(services, prompt, "single_text_only")
    payload = _build_payload([_make_data_uri((201, 17, 99), image.side)], prompt)

    req1 = _send(services, payload, "single_req1")
    req2 = _send(services, payload, "single_req2")

    assert req1.prompt_tokens > image.tokens, (
        f"single_req1: prompt_tokens={req1.prompt_tokens} is not larger than the "
        f"{image.tokens} image tokens a {image.side}px image expands to; "
        f"the worker did not expand the image placeholder."
    )
    assert req1.total_blocks == req2.total_blocks, (
        f"total blocks changed between repeats: "
        f"{req1.total_blocks} -> {req2.total_blocks}"
    )

    assert req1.overlap_blocks <= text_blocks, (
        f"single_req1: a never-seen image should overlap at most the "
        f"{text_blocks} text-only block(s), got "
        f"{req1.overlap_blocks}/{req1.total_blocks}.\n"
        f"Recent frontend logs:\n{req1.logs[-4000:]}"
    )
    _assert_nearly_full_repeat_overlap(req2)
    assert req2.overlap_blocks > text_blocks, (
        f"single_req2: the repeat must match blocks holding image tokens, not just "
        f"the {text_blocks} text-only block(s), got "
        f"req1={req1.overlap_blocks}/{req1.total_blocks}, "
        f"req2={req2.overlap_blocks}/{req2.total_blocks}"
    )

    min_cached = (text_blocks + 1) * services.block_size
    assert req2.cached_tokens is not None and req2.cached_tokens >= min_cached, (
        f"single_req2: expected an engine-side prefix-cache hit covering at least "
        f"one block of image tokens ({min_cached} tokens), got "
        f"cached_tokens={req2.cached_tokens} of prompt_tokens={req2.prompt_tokens}"
    )

    log_text = services.frontend.read_logs()
    assert (
        "MM-aware KV routing enabled" in log_text
    ), "frontend should log MM-routing initialization for the GLM checkpoint"


@pytest.mark.timeout(900)
def test_glm_different_image_isolation(start_glm_mm_services, predownload_models):
    """Same prompt, different image: the warm image keeps its overlap while the
    new image matches neither the router's blocks nor the engine's cache."""
    services = start_glm_mm_services
    side = services.large_image.side
    prompt = "GLM routing e2e: different image isolation."
    text_blocks = _text_only_blocks(services, prompt, "isolation_text_only")
    payload_a = _build_payload([_make_data_uri((11, 22, 33), side)], prompt)
    payload_b = _build_payload([_make_data_uri((222, 111, 200), side)], prompt)

    _send(services, payload_a, "warm_A")
    a_repeat = _send(services, payload_a, "A_repeat")
    b_cold = _send(services, payload_b, "B_cold")

    assert b_cold.total_blocks == a_repeat.total_blocks, (
        f"same geometry must expand identically, got A={a_repeat.total_blocks}, "
        f"B={b_cold.total_blocks} blocks"
    )

    _assert_nearly_full_repeat_overlap(a_repeat)
    assert b_cold.overlap_blocks <= text_blocks, (
        f"B_cold: a first-time image must overlap at most the {text_blocks} "
        f"text-only block(s); got {b_cold.overlap_blocks}/{b_cold.total_blocks}. "
        f"The image content is not part of the routing key.\n"
        f"Recent frontend logs:\n{b_cold.logs[-4000:]}"
    )
    assert a_repeat.overlap_blocks > b_cold.overlap_blocks, (
        f"warm-A overlap ({a_repeat.overlap_blocks}/{a_repeat.total_blocks}) must "
        f"exceed cold-B overlap ({b_cold.overlap_blocks}/{b_cold.total_blocks}); "
        f"equal means image hashes collide"
    )

    max_cached = text_blocks * services.block_size
    assert b_cold.cached_tokens is not None and b_cold.cached_tokens <= max_cached, (
        f"B_cold: the engine reused {b_cold.cached_tokens} of "
        f"{b_cold.prompt_tokens} prompt tokens for an image it had never seen, "
        f"more than the {max_cached} text-only tokens; the forwarded mm_hash does "
        f"not isolate image B from image A in the engine's prefix cache"
    )


@pytest.mark.timeout(900)
def test_glm_multiple_images_across_block_boundaries(
    start_glm_mm_services, predownload_models
):
    """Two images whose token runs straddle block boundaries: per-image
    expansion follows GLM geometry on the worker, repeats agree block for
    block, and swapping the second image keeps the first image's blocks and
    invalidates exactly the blocks after the first image's last full block."""
    services = start_glm_mm_services
    large, small = services.large_image, services.small_image
    prompt = "GLM routing e2e: multiple images across block boundaries."
    image_a = _make_data_uri((17, 99, 201), large.side)
    image_b = _make_data_uri((89, 210, 34), small.side)
    image_c = _make_data_uri((144, 61, 230), small.side)
    payload_text = _build_payload([], prompt)
    payload_a = _build_payload([image_a], prompt)
    payload_ab = _build_payload([image_a, image_b], prompt)
    payload_ac = _build_payload([image_a, image_c], prompt)
    payload_aa = _build_payload([image_a, image_a], prompt)

    # _send checks every request's engine token count against the blocks the
    # router hashed from the planner's expansion.
    text_only = _send(services, payload_text, "text_only")
    a_cold = _send(services, payload_a, "A_cold")
    a_repeat = _send(services, payload_a, "A_repeat")
    ab_cold = _send(services, payload_ab, "AB_cold")
    ab_repeat = _send(services, payload_ab, "AB_repeat")
    ac_cold = _send(services, payload_ac, "AC_cold")
    aa_cold = _send(services, payload_aa, "AA_cold")
    text_blocks = text_only.prompt_tokens // services.block_size

    # Each image adds its geometry's token count plus the template delimiters,
    # measured entirely on the engine side.
    large_increment = large.tokens + IMAGE_DELIMITER_TOKENS
    small_increment = small.tokens + IMAGE_DELIMITER_TOKENS
    added_a = a_cold.prompt_tokens - text_only.prompt_tokens
    added_b = ab_cold.prompt_tokens - a_cold.prompt_tokens
    added_second_a = aa_cold.prompt_tokens - a_cold.prompt_tokens
    assert added_a == large_increment, (
        f"one {large.side}px image added {added_a} prompt tokens on the worker, "
        f"expected {large_increment}"
    )
    assert added_b == small_increment, (
        f"appending a {small.side}px image added {added_b} prompt tokens on the "
        f"worker, expected {small_increment}"
    )
    assert added_second_a == large_increment, (
        f"appending a second {large.side}px image added {added_second_a} prompt "
        f"tokens on the worker, expected {large_increment}"
    )
    assert ac_cold.prompt_tokens == ab_cold.prompt_tokens, (
        f"same geometry, different content must expand identically, got "
        f"AB={ab_cold.prompt_tokens}, AC={ac_cold.prompt_tokens}"
    )

    # KV-event hash parity across both image runs and the boundary they share.
    _assert_nearly_full_repeat_overlap(a_repeat)
    _assert_nearly_full_repeat_overlap(ab_repeat)

    # A then C shares header, text, image A and C's opening delimiter with the
    # cached A-then-B prompt. Blocks wholly inside that prefix must match; the
    # block where C's tokens begin must not, and nothing after it may either.
    # Image A is longer than a block, so that prefix fills at least one block
    # past the text-only prompt's: C must keep image A's cached blocks.
    assert ac_cold.overlap_blocks > text_blocks, (
        f"AC_cold: swapping the second image must keep the first image's cached "
        f"blocks, got {ac_cold.overlap_blocks}/{ac_cold.total_blocks}, no more than "
        f"the {text_blocks} text-only block(s) (A repeat matched "
        f"{a_repeat.overlap_blocks}/{a_repeat.total_blocks}).\n"
        f"Recent frontend logs:\n{ac_cold.logs[-4000:]}"
    )
    # Image A's cached blocks bound the match: at most all of them, at least
    # all but the one that in the single-image prompt held the suffix instead.
    min_overlap_ac = a_repeat.overlap_blocks - 1
    max_overlap_ac = a_repeat.overlap_blocks
    assert min_overlap_ac <= ac_cold.overlap_blocks <= max_overlap_ac, (
        f"AC_cold: expected the overlap to stop at image A's last full block, "
        f"within [{min_overlap_ac}, {max_overlap_ac}], got "
        f"{ac_cold.overlap_blocks}/{ac_cold.total_blocks} (A repeat matched "
        f"{a_repeat.overlap_blocks}/{a_repeat.total_blocks}).\n"
        f"Recent frontend logs:\n{ac_cold.logs[-4000:]}"
    )
    # The second image is longer than a block, so the cached A-then-B prompt
    # holds at least one full block past that boundary for C to miss.
    assert ac_cold.overlap_blocks < ab_repeat.overlap_blocks, (
        f"AC_cold: swapping the second image must lose overlap versus the AB repeat, "
        f"got AC={ac_cold.overlap_blocks}/{ac_cold.total_blocks}, AB repeat="
        f"{ab_repeat.overlap_blocks}/{ab_repeat.total_blocks}; the second image's "
        f"content is not part of its blocks' routing keys"
    )
