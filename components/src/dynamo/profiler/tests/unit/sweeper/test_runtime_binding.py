# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for runtime binding resolution (tracking issue #13545, item 3).
Uses injected fakes throughout -- no real aiconfigurator needed, matching
common.check_support's confirmed signature and result shape exactly."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamo.profiler.sweeper.runtime_binding import (
    RuntimeBindingError,
    resolve_runtime_binding,
)

_TRTLLM_AGG_CANDIDATE = {
    "backend": "trtllm",
    "backend_version": "1.3.0rc10",
    "hardware_sku": "gb200",
    "deployment_mode": "agg",
    "strategy": "tp",
}


def _fake_support(agg=True, disagg=True, exact_match=True):
    def check_support(*, model, system, backend, version, architecture):
        return SimpleNamespace(
            agg_supported=agg, disagg_supported=disagg, exact_match=exact_match
        )

    return check_support


def _fake_gpus_per_node(value=4):
    return lambda system: value


def test_resolves_runtime_image_from_backend_and_version() -> None:
    binding = resolve_runtime_binding(
        _TRTLLM_AGG_CANDIDATE,
        model="deepseek-ai/DeepSeek-V3",
        check_support=_fake_support(),
        lookup_num_gpus_per_node=_fake_gpus_per_node(),
    )
    assert binding.runtime_image == "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:1.3.0rc10"


def test_resolves_num_gpus_per_node_from_injected_lookup() -> None:
    binding = resolve_runtime_binding(
        _TRTLLM_AGG_CANDIDATE,
        model="deepseek-ai/DeepSeek-V3",
        check_support=_fake_support(),
        lookup_num_gpus_per_node=_fake_gpus_per_node(4),
    )
    assert binding.num_gpus_per_node == 4


def test_checks_agg_supported_for_agg_candidates_not_disagg_supported() -> None:
    """The precise bug this module exists to avoid: checking the wrong
    boolean. agg_supported=False, disagg_supported=True must still fail for
    an agg candidate."""
    check_support = _fake_support(agg=False, disagg=True)
    with pytest.raises(RuntimeBindingError, match="no matching performance data"):
        resolve_runtime_binding(
            _TRTLLM_AGG_CANDIDATE,
            model="deepseek-ai/DeepSeek-V3",
            check_support=check_support,
            lookup_num_gpus_per_node=_fake_gpus_per_node(),
        )


def test_checks_disagg_supported_for_disagg_candidates_not_agg_supported() -> None:
    disagg_candidate = dict(_TRTLLM_AGG_CANDIDATE, deployment_mode="disagg")
    check_support = _fake_support(agg=True, disagg=False)
    with pytest.raises(RuntimeBindingError, match="no matching performance data"):
        resolve_runtime_binding(
            disagg_candidate,
            model="deepseek-ai/DeepSeek-V3",
            check_support=check_support,
            lookup_num_gpus_per_node=_fake_gpus_per_node(),
        )


def test_direct_renderer_selected_by_default() -> None:
    binding = resolve_runtime_binding(
        _TRTLLM_AGG_CANDIDATE,
        model="deepseek-ai/DeepSeek-V3",
        check_support=_fake_support(),
        lookup_num_gpus_per_node=_fake_gpus_per_node(),
    )
    assert binding.renderer == "direct"


@pytest.mark.parametrize("strategy", ["tep", "dep"])
def test_aic_renderer_selected_for_known_direct_unsupported_strategies(strategy) -> None:
    """Confirmed empirically this session: TrtllmConfigModifier.set_config_tep_size/
    set_config_dep_size raise NotImplementedError for the direct renderer."""
    candidate = dict(_TRTLLM_AGG_CANDIDATE, strategy=strategy)
    binding = resolve_runtime_binding(
        candidate,
        model="deepseek-ai/DeepSeek-V3",
        check_support=_fake_support(),
        lookup_num_gpus_per_node=_fake_gpus_per_node(),
    )
    assert binding.renderer == "aic"


def test_tep_dep_only_forces_aic_for_trtllm_not_other_backends() -> None:
    """The unsupported-strategy set is backend-specific -- vllm/sglang tep
    candidates (if they existed) must not be forced onto aic incorrectly."""
    candidate = dict(_TRTLLM_AGG_CANDIDATE, backend="vllm", strategy="tep")
    binding = resolve_runtime_binding(
        candidate,
        model="deepseek-ai/DeepSeek-V3",
        check_support=_fake_support(),
        lookup_num_gpus_per_node=_fake_gpus_per_node(),
    )
    assert binding.renderer == "direct"
