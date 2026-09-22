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

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

_TRTLLM_AGG_CANDIDATE = {
    "backend": "trtllm",
    "backend_version": "1.3.0rc10",
    "hardware_sku": "gb200",
    "deployment_mode": "agg",
    "strategy": "tp",
}

_DYNAMO_VERSION = "0.5.0"


def _fake_support(agg=True, disagg=True, exact_match=True):
    def check_support(*, model, system, backend, version, architecture):
        return SimpleNamespace(
            agg_supported=agg, disagg_supported=disagg, exact_match=exact_match
        )

    return check_support


def _fake_gpus_per_node(value=4):
    return lambda system: value


def _resolve(candidate, **overrides):
    kwargs = {
        "model": "deepseek-ai/DeepSeek-V3",
        "dynamo_version": _DYNAMO_VERSION,
        "check_support": _fake_support(),
        "lookup_num_gpus_per_node": _fake_gpus_per_node(),
    }
    kwargs.update(overrides)
    return resolve_runtime_binding(candidate, **kwargs)


def test_resolves_runtime_image_from_backend_and_version() -> None:
    binding = _resolve(_TRTLLM_AGG_CANDIDATE)
    assert binding.runtime_image == "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:1.3.0rc10"


def test_resolves_num_gpus_per_node_from_injected_lookup() -> None:
    binding = _resolve(
        _TRTLLM_AGG_CANDIDATE, lookup_num_gpus_per_node=_fake_gpus_per_node(4)
    )
    assert binding.num_gpus_per_node == 4


def test_lookup_num_gpus_per_node_has_no_default_and_must_be_supplied() -> None:
    """Regression test: the only known real implementation
    (aiconfigurator.sdk.task_v2._lookup_num_gpus_per_node) does not exist
    in the latest released aiconfigurator (0.11.0), confirmed by direct
    import attempt. A silent default would raise an opaque ImportError
    from inside aiconfigurator on every normal call; requiring this
    parameter makes that failure explicit and immediate instead."""
    with pytest.raises(TypeError, match="lookup_num_gpus_per_node"):
        resolve_runtime_binding(
            _TRTLLM_AGG_CANDIDATE,
            model="deepseek-ai/DeepSeek-V3",
            dynamo_version=_DYNAMO_VERSION,
            check_support=_fake_support(),
        )


def test_agg_candidate_checks_agg_supported_not_disagg_supported() -> None:
    """agg_supported=False, disagg_supported=True must still fail for an
    agg candidate -- these are independent booleans in the real result
    shape, and checking the wrong one would silently accept an
    unrenderable candidate."""
    with pytest.raises(RuntimeBindingError, match="no matching performance data"):
        _resolve(
            _TRTLLM_AGG_CANDIDATE, check_support=_fake_support(agg=False, disagg=True)
        )


def test_disagg_candidate_checks_disagg_supported_not_agg_supported() -> None:
    disagg_candidate = dict(_TRTLLM_AGG_CANDIDATE, deployment_mode="disagg")
    with pytest.raises(RuntimeBindingError, match="no matching performance data"):
        _resolve(disagg_candidate, check_support=_fake_support(agg=True, disagg=False))


def test_direct_renderer_selected_by_default() -> None:
    assert _resolve(_TRTLLM_AGG_CANDIDATE).renderer == "direct"


@pytest.mark.parametrize("strategy", ["tep", "dep"])
def test_aic_renderer_selected_for_known_direct_unsupported_strategies(
    strategy,
) -> None:
    candidate = dict(_TRTLLM_AGG_CANDIDATE, strategy=strategy)
    assert _resolve(candidate).renderer == "aic"


def test_tep_dep_only_forces_aic_for_trtllm_not_other_backends() -> None:
    candidate = dict(_TRTLLM_AGG_CANDIDATE, backend="vllm", strategy="tep")
    assert _resolve(candidate).renderer == "direct"


@pytest.mark.parametrize(
    "prefill_strategy,decode_strategy",
    [("tep", "tp"), ("tp", "dep")],
)
def test_disagg_checks_both_roles_not_just_a_missing_top_level_strategy(
    prefill_strategy, decode_strategy
) -> None:
    """Regression test: disagg candidates carry prefill_strategy/
    decode_strategy per role -- the top-level `strategy` field is absent
    (None) for every disagg candidate. Reading only `strategy` would
    silently select "direct" for every disagg candidate regardless of
    role strategy, reaching the same NotImplementedError this logic
    exists to avoid. Either role using tep/dep must force aic."""
    candidate = {
        "backend": "trtllm",
        "backend_version": "1.3.0rc10",
        "hardware_sku": "gb200",
        "deployment_mode": "disagg",
        "prefill_strategy": prefill_strategy,
        "decode_strategy": decode_strategy,
    }
    assert _resolve(candidate).renderer == "aic"


def test_disagg_with_both_roles_tp_still_selects_direct() -> None:
    candidate = {
        "backend": "trtllm",
        "backend_version": "1.3.0rc10",
        "hardware_sku": "gb200",
        "deployment_mode": "disagg",
        "prefill_strategy": "tp",
        "decode_strategy": "tp",
    }
    assert _resolve(candidate).renderer == "direct"


def test_non_canonical_backend_version_sets_runtime_version_override() -> None:
    """Regression test: DGDGenerationOptions rejects a non-MAJOR.MINOR.PATCH
    image tag unless runtime_version_override is set (confirmed against
    the real _RUNTIME_VERSION_PATTERN in renderers/base.py). backend_version
    "1.3.0rc10" is not canonical -- the resolved binding must carry the
    override so a caller can construct valid DGDGenerationOptions."""
    binding = _resolve(_TRTLLM_AGG_CANDIDATE)  # backend_version="1.3.0rc10"
    assert binding.runtime_version_override == _DYNAMO_VERSION


def test_canonical_backend_version_leaves_runtime_version_override_unset() -> None:
    candidate = dict(_TRTLLM_AGG_CANDIDATE, backend_version="1.3.0")
    binding = _resolve(candidate)
    assert binding.runtime_image == "nvcr.io/nvidia/ai-dynamo/trtllm-runtime:1.3.0"
    assert binding.runtime_version_override is None
