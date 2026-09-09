# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Dynamo's `dgd` output adapter (DEP #14282).

Validates against the confirmed real AISimulate output-adapter ABI
(aisimulate/src/aisimulate/output_adapter.py): name/api_version attributes,
write(config, *, result, output_dir) returning relative, existing paths.
`_FakeSweepResult` mirrors the real shipped test suite's own
`_RecommendationResult([candidate])` shape -- see dgd_output_adapter.py's
module docstring for why `.candidates`/`.workload` are inferred, not yet
confirmed against the real `SweepResult` class.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

try:
    from dynamo.profiler.sweeper.dgd_output_adapter import (
        OUTPUT_ADAPTER_API_VERSION,
        DgdOutputAdapter,
        DgdOutputConfigError,
    )
except ImportError as exc:
    pytest.skip(f"Skip (missing dependency): {exc}", allow_module_level=True)


class _FakeCandidate:
    def __init__(self, config):
        self.config = config


class _FakeSweepResult:
    def __init__(self, candidates, workload=None):
        self.candidates = candidates
        self.workload = workload


_CANDIDATE_CONFIG = {
    "deployment_mode": "agg",
    "backend": "trtllm",
    "model_name": "Qwen/Qwen3-8B",
    "backend_version": "1.3.0rc10",
    "tp": 4,
    "pp": 1,
    "attention_dp": 1,
    "moe_tp": 1,
    "moe_ep": 1,
    "strategy": "tp",
    "replicas": 2,
    "used_gpus": 8,
    "agg_max_num_batched_tokens": 8192,
    "agg_max_num_seqs": 1024,
    "agg_block_size": 64,
    "agg_gpu_memory_utilization": 0.9,
    "agg_enable_prefix_caching": True,
    "concurrency": 64,
}

_DGD_CONFIG = {
    "name": "qwen",
    "renderer": "direct",
    "format": "manifest",
    "runtime_image": "my-registry/tensorrtllm-runtime:1.3.0rc10",
    "runtime_version_override": "0.5.0",
    "num_gpus_per_node": 8,
}


def test_adapter_declares_the_confirmed_name_and_api_version() -> None:
    """Matches AISimulate's real validate_output_adapter() checks exactly:
    name must equal the requested entry-point name, api_version must be an
    int equal to OUTPUT_ADAPTER_API_VERSION, write must be callable."""
    adapter = DgdOutputAdapter()
    assert adapter.name == "dgd"
    assert type(adapter.api_version) is int
    assert adapter.api_version == OUTPUT_ADAPTER_API_VERSION == 1
    assert callable(adapter.write)


def test_write_returns_relative_paths_that_exist(tmp_path: Path) -> None:
    """Matches AISimulate's real write_output_adapters() validation: every
    returned path must be relative (no leading '/', no '..') and must exist
    under output_dir immediately after write() returns."""
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult([_FakeCandidate(_CANDIDATE_CONFIG)])

    paths = adapter.write(_DGD_CONFIG, result=result, output_dir=tmp_path)

    assert len(paths) == 1
    for raw_path in paths:
        path = Path(raw_path)
        assert not path.is_absolute()
        assert ".." not in path.parts
        assert (tmp_path / path).exists()
    assert "Qwen/Qwen3-8B" in (tmp_path / paths[0]).read_text()


def test_pareto_naming_matches_the_real_name_prefix_convention(tmp_path: Path) -> None:
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult(
        [_FakeCandidate(_CANDIDATE_CONFIG), _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=2))]
    )
    config = dict(_DGD_CONFIG, name=None, name_prefix="pareto")

    paths = adapter.write(config, result=result, output_dir=tmp_path)

    assert [Path(p).stem for p in paths] == ["pareto-000", "pareto-001"]


def test_missing_dgd_config_field_raises_config_error(tmp_path: Path) -> None:
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult([_FakeCandidate(_CANDIDATE_CONFIG)])
    incomplete_config = {"name": "qwen"}  # missing runtime_image, num_gpus_per_node

    with pytest.raises(DgdOutputConfigError, match="missing required field"):
        adapter.write(incomplete_config, result=result, output_dir=tmp_path)
