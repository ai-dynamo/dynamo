# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-GPU VMM ownership, admission and rollback tests with a fake CUDA driver."""

import importlib.util
import sys
from pathlib import Path

import pytest

_helper_relative = Path("lib/bindings/kvbm/python/kvbm/vllm_integration/gpu_mnnvl.py")
_helper_file = next(
    (
        parent / _helper_relative
        for parent in Path(__file__).resolve().parents
        if (parent / _helper_relative).is_file()
    ),
    None,
)
if _helper_file is None:
    _package = importlib.util.find_spec("kvbm")
    if _package is None or _package.submodule_search_locations is None:
        raise ModuleNotFoundError(
            "KVBM fabric helper source is unavailable", name="kvbm"
        )
    _locations = list(_package.submodule_search_locations)
    if len(_locations) != 1:
        raise RuntimeError("One unambiguous KVBM helper source is required")
    _helper_file = Path(_locations[0]) / "vllm_integration/gpu_mnnvl.py"
_helper_spec = importlib.util.spec_from_file_location(
    __name__ + "._gpu_mnnvl", _helper_file
)
if _helper_spec is None or _helper_spec.loader is None:
    raise RuntimeError("KVBM fabric helper source cannot be loaded")
m = importlib.util.module_from_spec(_helper_spec)
sys.modules[_helper_spec.name] = m
_helper_spec.loader.exec_module(m)
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
]


def test_effective_domain_matches_same_clique_not_gpu_uuid_and_separates_other_profiles():
    first = m.FabricIdentity("GPU-00000000-0000-0000-0000-000000000001", "1" * 32, 0)
    second = m.FabricIdentity("GPU-00000000-0000-0000-0000-000000000002", "1" * 32, 0)
    domain = m.effective_domain("layout-v1", first.domain)
    assert domain == m.effective_domain("layout-v1", second.domain)
    assert domain != m.effective_domain("other-layout", first.domain)
    assert domain != m.effective_domain("layout-v1", "mnnvl:" + "2" * 32 + ":0")
    assert domain != m.effective_domain("layout-v1", "mnnvl:" + "1" * 32 + ":1")
    assert domain != "layout-v1"


@pytest.mark.parametrize(
    "domain",
    [
        "mnnvl:" + "0" * 32 + ":0",
        "mnnvl:" + "1" * 32 + ":00",
        "mnnvl:" + "1" * 32 + ":-1",
        "mnnvl:" + "1" * 32 + ":4294967296",
        "tcp",
        None,
    ],
)
def test_uncanonical_or_missing_fabric_domain_is_not_a_capability(domain):
    with pytest.raises(ValueError):
        m.effective_domain("layout-v1", domain)


@pytest.mark.parametrize("layout_domain", ["bad\x00layout", "", " padded ", None])
def test_invalid_raw_layout_domain_cannot_be_hidden_by_compatibility_hash(
    layout_domain,
):
    with pytest.raises(ValueError):
        m.effective_domain(layout_domain, "mnnvl:" + "1" * 32 + ":0")
