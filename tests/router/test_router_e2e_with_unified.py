# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end router tests against the **unified-backend** entrypoints
(``python -m dynamo.vllm.unified_main`` etc.) introduced under the
``dynamo.common.backend`` abstraction.

Each backend's existing legacy entrypoint already has e2e router tests in
``test_router_e2e_with_{vllm,sglang,trtllm}.py``. This file mirrors a
focused subset of those tests against the unified entrypoint instead, so
the new ABC + ``Worker``-driven publishers are validated against the same
router/frontend stack the legacy path is gated on.

The unified entrypoints share their CLI arg parser with the legacy path
(see ``dynamo.{vllm,sglang,trtllm}.args.parse_args``). The only difference
between the legacy and unified runs is the launched Python module, so each
``Unified*Process`` is a thin subclass that swaps that one token in every
worker process's ``command`` list.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.router.e2e_harness import run_basic_router_test, run_router_decisions_test
from tests.router.test_router_e2e_with_sglang import MODEL_NAME as SGLANG_MODEL_NAME
from tests.router.test_router_e2e_with_sglang import SGLANG_ARGS, SGLangProcess
from tests.router.test_router_e2e_with_trtllm import MODEL_NAME as TRTLLM_MODEL_NAME
from tests.router.test_router_e2e_with_trtllm import (
    TRTLLM_ARGS,
    TRTLLM_BLOCK_SIZE,
    TRTLLMProcess,
)
from tests.router.test_router_e2e_with_vllm import BLOCK_SIZE as VLLM_BLOCK_SIZE
from tests.router.test_router_e2e_with_vllm import MODEL_NAME as VLLM_MODEL_NAME
from tests.router.test_router_e2e_with_vllm import VLLM_ARGS, VLLMProcess

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Process subclasses — swap the launched Python module to point at each
# backend's `*.unified_main` entrypoint. Everything else (port allocation,
# CLI args, env vars, KV-events config, NIXL plumbing, indexer wiring) is
# inherited unchanged because the unified path reuses the same arg parser
# and runtime stack as the legacy path — only the engine glue differs.
# --------------------------------------------------------------------------- #


def _swap_module(worker_processes, legacy_module: str, unified_module: str) -> None:
    """Replace `python3 -m <legacy_module>` with `python3 -m <unified_module>`
    on every worker process's command list. The token appears exactly once
    per command (the canonical `-m <module>` placement); we assert that to
    catch upstream test refactors that change the command shape."""
    for process in worker_processes:
        cmd = process.command
        found = False
        for i, tok in enumerate(cmd):
            if tok == legacy_module:
                cmd[i] = unified_module
                found = True
                break
        if not found:
            raise RuntimeError(
                f"expected to find {legacy_module!r} in worker command "
                f"to retarget at {unified_module!r}; got {cmd!r}"
            )


class UnifiedVLLMProcess(VLLMProcess):
    """vLLM workers launched via ``dynamo.vllm.unified_main``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _swap_module(self.worker_processes, "dynamo.vllm", "dynamo.vllm.unified_main")

    process_name = "Unified vLLM worker"
    cleanup_name = "Unified vLLM worker resources"


class UnifiedSGLangProcess(SGLangProcess):
    """SGLang workers launched via ``dynamo.sglang.unified_main``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _swap_module(
            self.worker_processes, "dynamo.sglang", "dynamo.sglang.unified_main"
        )

    process_name = "Unified SGLang worker"
    cleanup_name = "Unified SGLang worker resources"


class UnifiedTRTLLMProcess(TRTLLMProcess):
    """TRT-LLM workers launched via ``dynamo.trtllm.unified_main``.

    Overrides the worker component from ``tensorrt_llm`` (legacy default) to
    ``backend`` so all three unified backends share one component naming
    convention. Legacy ``test_router_e2e_with_trtllm.py`` is untouched.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _swap_module(
            self.worker_processes, "dynamo.trtllm", "dynamo.trtllm.unified_main"
        )
        new_endpoint = f"dyn://{self.namespace}.backend.generate"
        endpoint_args = ["--endpoint", new_endpoint]
        for process in self.worker_processes:
            if any(
                tok == "--endpoint" or tok.startswith("--endpoint=")
                for tok in process.command
            ):
                raise RuntimeError(
                    f"expected legacy command to omit --endpoint; got {process.command!r}"
                )
            process.command.extend(endpoint_args)
        self.component_name = "backend"
        self.endpoint = new_endpoint

    process_name = "Unified TRT-LLM worker"
    cleanup_name = "Unified TRT-LLM worker resources"


# --------------------------------------------------------------------------- #
# Shared markers
# --------------------------------------------------------------------------- #

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.router,
    pytest.mark.unified,
]


# --------------------------------------------------------------------------- #
# vLLM unified-path tests
# --------------------------------------------------------------------------- #


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.vllm
@pytest.mark.model(VLLM_MODEL_NAME)
@pytest.mark.profiled_vram_gib(6.9)
@pytest.mark.requested_vllm_kv_cache_bytes(331_801_000)
@pytest.mark.timeout(360)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_vllm_kv_router_basic(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
) -> None:
    """Two unified-vLLM workers behind the KV router complete a small
    concurrent request batch end-to-end. Exercises the new ABC's
    ``kv_event_sources`` (vLLM returns ``ZmqSource`` per dp_rank) and
    ``metrics_sources`` (vLLM stat-logger feeds the snapshot cache) via
    the ``Worker``-driven publisher dispatch."""
    run_basic_router_test(
        engine_process_cls=UnifiedVLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        num_workers=2,
        single_gpu=True,
        request=request,
        request_plane=request_plane,
        block_size=VLLM_BLOCK_SIZE,
        model_name=VLLM_MODEL_NAME,
    )


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.vllm
@pytest.mark.model(VLLM_MODEL_NAME)
@pytest.mark.profiled_vram_gib(6.9)
@pytest.mark.requested_vllm_kv_cache_bytes(331_801_000)
@pytest.mark.timeout(360)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_vllm_router_decisions_multiple_workers(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
) -> None:
    """Prefix-reuse correctness: progressive requests with overlapping
    prefixes must all land on the worker that already holds the matching
    blocks. Validates that the unified ``ZmqSource`` correctly plumbs the
    same KV events the legacy path emits, so the router's overlap scoring
    is identical."""
    run_router_decisions_test(
        engine_process_cls=UnifiedVLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=VLLM_MODEL_NAME,
        block_size=VLLM_BLOCK_SIZE,
        component_name="backend",
        num_workers=2,
        single_gpu=True,
        test_dp_rank=False,
    )


@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.vllm
@pytest.mark.model(VLLM_MODEL_NAME)
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_vllm_router_decisions_dp(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
) -> None:
    """Per-dp_rank routing: one vLLM worker hosting 2 data-parallel ranks.
    Validates that the unified path's `kv_event_sources()` per-rank
    descriptors plumb through to per-rank `worker_kv_indexer_query_dp{N}`
    registrations and per-rank scoring."""
    run_router_decisions_test(
        engine_process_cls=UnifiedVLLMProcess,
        engine_args_name="vllm_args",
        engine_args=VLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=VLLM_MODEL_NAME,
        block_size=VLLM_BLOCK_SIZE,
        component_name="backend",
        num_workers=1,
        single_gpu=False,
        test_dp_rank=True,
        extra_process_kwargs={"data_parallel_size": 2},
    )


# --------------------------------------------------------------------------- #
# SGLang unified-path tests
# --------------------------------------------------------------------------- #


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.sglang
@pytest.mark.model(SGLANG_MODEL_NAME)
@pytest.mark.timeout(360)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_sglang_kv_router_basic(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    request_plane,
) -> None:
    """SGLang's unified path returns one ``ZmqSource`` per local dp_rank
    (validates the multi-rank list-returning ABC). Basic batch run end
    to end through the KV router."""
    run_basic_router_test(
        engine_process_cls=UnifiedSGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        num_workers=2,
        single_gpu=True,
        request=request,
        request_plane=request_plane,
        block_size=SGLANG_ARGS.get("page_size", 16),
        model_name=SGLANG_MODEL_NAME,
    )


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.sglang
@pytest.mark.model(SGLANG_MODEL_NAME)
@pytest.mark.timeout(360)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_sglang_router_decisions_multiple_workers(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    request_plane,
) -> None:
    """Same prefix-reuse correctness check, against the SGLang unified
    entrypoint. Confirms the scheduler-pull metrics task and the per-rank
    ZMQ event subscriber both wire through the new ``Worker``-managed
    publishers correctly."""
    run_router_decisions_test(
        engine_process_cls=UnifiedSGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=SGLANG_MODEL_NAME,
        block_size=SGLANG_ARGS.get("page_size", 16),
        component_name="backend",
        num_workers=2,
        single_gpu=True,
        test_dp_rank=False,
    )


@pytest.mark.pre_merge
@pytest.mark.gpu_2
@pytest.mark.sglang
@pytest.mark.model(SGLANG_MODEL_NAME)
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
@pytest.mark.skip(reason="DYN-2265 (same as legacy SGLang DP test)")
def test_unified_sglang_router_decisions_dp(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    request_plane,
) -> None:
    """Per-dp_rank routing on the SGLang unified path. Exercises
    `_local_dp_rank_range`'s multi-rank slice and the per-rank `ZmqSource`
    descriptors."""
    run_router_decisions_test(
        engine_process_cls=UnifiedSGLangProcess,
        engine_args_name="sglang_args",
        engine_args=SGLANG_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=SGLANG_MODEL_NAME,
        block_size=SGLANG_ARGS.get("page_size", 16),
        component_name="backend",
        num_workers=1,
        single_gpu=False,
        test_dp_rank=True,
        extra_process_kwargs={"data_parallel_size": 2},
    )


# --------------------------------------------------------------------------- #
# TRT-LLM unified-path tests
# --------------------------------------------------------------------------- #
#
# TRT-LLM is the only backend where the unified path is a meaningful
# behavioural change rather than a refactor: the legacy path drives
# `get_kv_cache_events_async` from the asyncio loop, while the unified
# path uses a `PushSource` and runs the polling on a dedicated thread.
# These tests stand the path up end-to-end against a real engine.


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.trtllm
@pytest.mark.model(TRTLLM_MODEL_NAME)
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_trtllm_kv_router_basic(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    request_plane,
) -> None:
    """End-to-end smoke against the TRT-LLM unified entrypoint. Verifies
    that ``PushSource.on_ready`` correctly hands the publisher to the
    engine's polling thread and that ``publish_stored``/``publish_removed``
    reach the router via NATS."""
    run_basic_router_test(
        engine_process_cls=UnifiedTRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args=TRTLLM_ARGS,
        num_workers=2,
        single_gpu=True,
        request=request,
        request_plane=request_plane,
        block_size=TRTLLM_BLOCK_SIZE,
        model_name=TRTLLM_MODEL_NAME,
    )


@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.trtllm
@pytest.mark.model(TRTLLM_MODEL_NAME)
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_trtllm_router_decisions_multiple_workers(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    request_plane,
) -> None:
    """Prefix-reuse correctness on the TRT-LLM unified path. The polling
    thread + `_dispatch_kv_event` must produce the same router-visible
    events as the legacy `_publish_kv_cache_events_task` for the router's
    overlap scoring to agree."""
    run_router_decisions_test(
        engine_process_cls=UnifiedTRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args=TRTLLM_ARGS,
        request=request,
        request_plane=request_plane,
        model_name=TRTLLM_MODEL_NAME,
        block_size=TRTLLM_BLOCK_SIZE,
        component_name="backend",
        num_workers=2,
        single_gpu=True,
        test_dp_rank=False,
    )


@pytest.mark.gpu_2
@pytest.mark.nightly
@pytest.mark.trtllm
@pytest.mark.model(TRTLLM_MODEL_NAME)
@pytest.mark.timeout(600)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
def test_unified_trtllm_router_decisions_attention_dp(
    request,
    runtime_services_dynamic_ports,
    predownload_models,
    set_ucx_tls_no_mm,
    request_plane,
) -> None:
    """Attention-DP routing: one TRT-LLM worker with attention_dp_size=2.
    Exercises `EngineConfig.data_parallel_size` and per-rank `PushSource`
    + `worker_kv_indexer_query_dp{N}` registration on the unified path."""
    run_router_decisions_test(
        engine_process_cls=UnifiedTRTLLMProcess,
        engine_args_name="trtllm_args",
        engine_args={
            **TRTLLM_ARGS,
            "enable_attention_dp": True,
            "tensor_parallel_size": 2,
        },
        request=request,
        request_plane=request_plane,
        model_name=TRTLLM_MODEL_NAME,
        block_size=TRTLLM_BLOCK_SIZE,
        component_name="backend",
        num_workers=1,
        single_gpu=False,
        test_dp_rank=True,
    )
