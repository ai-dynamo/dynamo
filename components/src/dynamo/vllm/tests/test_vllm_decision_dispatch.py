# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary transport fast path and symmetric native publication validation."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm import decision_native_registry as registry
from dynamo.vllm.decision_dispatch import DecisionDispatcher

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize("kind", ["head", "read"])
def test_ordinary_dispatch_does_not_construct_context_metadata(kind):
    class OrdinaryContext:
        @property
        def metadata(self):
            raise AssertionError("ordinary request accessed context metadata")

    assert (
        DecisionDispatcher(kind).maybe_handle(
            {"input": "ordinary", "token_ids": [1]}, OrdinaryContext()
        )
        is None
    )


@pytest.mark.parametrize("variant", ["NativeReady", "NativeFailed"])
@pytest.mark.parametrize("change", ["unknown-name", "wrong-kind", "narrowed-limit"])
def test_publication_uses_canonical_name_and_kind_with_a_narrowed_limit(
    variant, change
):
    canonical = registry.PROFILES[registry.D3_PROFILE]
    selected = {
        "unknown-name": replace(canonical, name="unknown"),
        "wrong-kind": replace(canonical, kind="read"),
        "narrowed-limit": replace(canonical, max_candidates=2),
    }[change]
    publication = getattr(registry, variant)(selected)
    runtime = SimpleNamespace(set_engine_specific=Mock())
    written = registry.publish_native_decision_capability(
        runtime,
        publication,
        ModelInput.Text if selected.kind == "head" else ModelInput.Tokens,
        (
            ModelType.Classify | ModelType.Pooling
            if selected.kind == "head"
            else ModelType.Chat
        ),
        WorkerType.Aggregated,
    )
    if change != "narrowed-limit":
        assert written is None
        runtime.set_engine_specific.assert_not_called()
    else:
        assert written is publication
        if variant == "NativeReady":
            runtime.set_engine_specific.assert_any_call(
                "vllm_native_decision_max_candidates", "2"
            )
        else:
            runtime.set_engine_specific.assert_called_once_with(
                registry.NATIVE_QUALIFICATION_FAILURE,
                '{"profile": "vllm_d3_decision_v1"}',
            )
