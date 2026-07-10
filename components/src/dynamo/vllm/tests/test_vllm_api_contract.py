# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the deep, *non-public* vLLM internals dynamo's vLLM backend
relies on.

The ``InstrumentedScheduler`` (self-benchmark mode, ``--benchmark-mode``) subclasses
vLLM's V1 ``AsyncScheduler`` and constructs ``NewRequestData`` / ``SchedulerOutput``
objects directly. vLLM even warns at runtime that this scheduler interface "is not
public and compatibility may not be maintained." Because the coupling is to internal
classes/fields rather than imports alone, a vLLM version bump can break a benchmark
worker *at runtime* (e.g. an ``AssertionError`` deep in the GPU model runner) with no
import or compile error — and only on GPU CI.

These tests assert the contract up front so such breaks surface in fast unit CI.

Concrete motivating break (vLLM 0.21 -> 0.22): vLLM added the now-required
``NewRequestData.prefill_token_ids`` field — the v2 GPU model runner asserts it is not
None and uses it as the request's ``all_token_ids``. ``InstrumentedScheduler``'s
hand-built ``NewRequestData`` for synthetic decode requests had to start populating it
(``_bench_inject_fake_decode``). The first test below guards exactly that field.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

# Module-level vLLM import so the real site-packages ``vllm`` loads (matches the
# pattern in test_vllm_instrumented_scheduler.py / test_vllm_unit.py).
import vllm  # noqa: F401,E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_instrumented_scheduler_internal_imports_resolve():
    """Every deep vLLM symbol ``instrumented_scheduler`` imports must still exist.

    Catches vLLM renames/moves (the ImportError class of breakage) in unit CI.
    """
    required = {
        "vllm.sampling_params": ["SamplingParams"],
        "vllm.utils.hashing": ["get_hash_fn_by_name"],
        "vllm.v1.core.kv_cache_utils": ["get_request_block_hasher", "init_none_hash"],
        "vllm.v1.core.sched.async_scheduler": ["AsyncScheduler"],
        "vllm.v1.core.sched.output": [
            "CachedRequestData",
            "NewRequestData",
            "SchedulerOutput",
        ],
        "vllm.v1.request": ["Request", "RequestStatus"],
    }
    missing = []
    for mod_name, symbols in required.items():
        module = importlib.import_module(mod_name)
        missing += [f"{mod_name}.{s}" for s in symbols if not hasattr(module, s)]
    assert not missing, (
        "vLLM internal symbols the InstrumentedScheduler imports are gone "
        f"(likely a vLLM rename/move): {missing}"
    )


def test_new_request_data_has_fields_instrumented_scheduler_sets():
    """``InstrumentedScheduler._bench_inject_fake_decode`` builds ``NewRequestData``
    directly. Guard the fields it sets — notably ``prefill_token_ids``, which vLLM
    >=0.22's v2 GPU model runner requires (asserted non-None, used as all_token_ids).
    """
    from vllm.v1.core.sched.output import NewRequestData

    field_names = {f.name for f in dataclasses.fields(NewRequestData)}
    needed = {
        "req_id",
        "prompt_token_ids",
        "mm_features",
        "sampling_params",
        "pooling_params",
        "block_ids",
        "num_computed_tokens",
        "lora_request",
        "prefill_token_ids",
    }
    missing = needed - field_names
    assert not missing, (
        "vLLM NewRequestData no longer has fields InstrumentedScheduler sets "
        f"{sorted(missing)} — update _bench_inject_fake_decode."
    )


def test_async_scheduler_has_methods_instrumented_scheduler_overrides():
    """Guard the ``AsyncScheduler`` methods ``InstrumentedScheduler`` overrides /
    calls via ``super()``."""
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    for method in (
        "schedule",
        "update_from_output",
        "has_requests",
        "shutdown",
        "add_request",
    ):
        assert hasattr(AsyncScheduler, method), (
            f"vLLM AsyncScheduler.{method} is gone — the InstrumentedScheduler "
            "override/super() call is stale."
        )


def test_kv_cache_manager_has_methods_instrumented_scheduler_uses():
    """Guard the ``KVCacheManager`` methods the InstrumentedScheduler calls through
    ``self.kv_cache_manager`` while building synthetic benchmark batches."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    for method in ("allocate_slots", "new_step_starts", "take_new_block_ids"):
        assert hasattr(
            KVCacheManager, method
        ), f"vLLM KVCacheManager.{method} is gone — InstrumentedScheduler relies on it."


def test_request_exposes_all_token_ids():
    """``InstrumentedScheduler`` passes ``req._all_token_ids`` as
    ``NewRequestData.prefill_token_ids`` (mirroring vLLM's own scheduler). Guard that
    private attribute so a rename is caught here, not at runtime."""
    from vllm.v1.request import Request

    # vllm-omni monkeypatches vllm.v1.request.Request with its OmniRequest subclass,
    # whose body does not redeclare `_all_token_ids` (it inherits it). Walk the MRO so
    # the inherited attribute is still detected, while a real rename in vLLM still fails.
    sources = []
    for klass in Request.__mro__:
        try:
            sources.append(inspect.getsource(klass))
        except (OSError, TypeError):
            continue

    assert any("_all_token_ids" in src for src in sources), (
        "vllm.v1.request.Request no longer exposes `_all_token_ids` — "
        "InstrumentedScheduler relies on it for NewRequestData.prefill_token_ids."
    )
