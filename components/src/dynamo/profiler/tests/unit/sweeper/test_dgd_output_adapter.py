# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Dynamo's `dgd` output adapter (DEP #14282)."""

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
    from dynamo.profiler.sweeper.renderers import CandidateMaterializationError
except ImportError as exc:
    pytest.skip(f"Skip (missing dependency): {exc}", allow_module_level=True)


class _FakeCandidate:
    def __init__(self, config, score: float = 0.0, used_gpus: int | None = None):
        self.config = config
        # _best_candidate (scalar-mode selection) reads .score/.used_gpus on
        # every candidate, even when there's only one -- defaults keep the
        # single-candidate tests below unaffected; P1's test sets these
        # explicitly to prove the ranking rule itself.
        self.score = score
        self.used_gpus = (
            used_gpus if used_gpus is not None else config.get("used_gpus", 0)
        )


class _FakeProvenance:
    def __init__(self, workload):
        self.config = {"workload": workload}


class _FakeSweepResult:
    def __init__(self, candidates, workload=None):
        self.selected_candidates = candidates
        self.provenance = _FakeProvenance(workload)


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
    adapter = DgdOutputAdapter()
    assert adapter.name == "dgd"
    assert type(adapter.api_version) is int
    assert adapter.api_version == OUTPUT_ADAPTER_API_VERSION == 1
    assert callable(adapter.write)


def test_write_returns_relative_paths_that_exist(tmp_path: Path) -> None:
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult(
        [_FakeCandidate(_CANDIDATE_CONFIG)], workload="qwen-workload"
    )

    paths = adapter.write(_DGD_CONFIG, result=result, output_dir=tmp_path)

    assert len(paths) == 1
    path = Path(paths[0])
    assert path.stem == "qwen"
    assert not path.is_absolute()
    assert ".." not in path.parts
    assert (tmp_path / path).exists()
    assert "Qwen/Qwen3-8B" in (tmp_path / paths[0]).read_text()


def test_write_passes_the_result_workload_through_to_the_renderer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the value SweepResult.workload is passed through with, via a spy
    on render_dgd itself (renderer-agnostic) rather than relying on the
    "direct" renderer's render(), whose unused `_workload` parameter would
    let a mutation to workload=None pass silently."""
    from dynamo.profiler.sweeper import dgd_output_adapter as mod

    seen = []
    real = mod.render_dgd

    def spy(candidate, workload, options, **kwargs):
        seen.append(workload)
        return real(candidate, workload, options, **kwargs)

    monkeypatch.setattr(mod, "render_dgd", spy)

    adapter = DgdOutputAdapter()
    result = _FakeSweepResult(
        [_FakeCandidate(_CANDIDATE_CONFIG)], workload="qwen-workload"
    )

    adapter.write(_DGD_CONFIG, result=result, output_dir=tmp_path)

    assert seen == ["qwen-workload"]


def test_pareto_naming_matches_the_real_name_prefix_convention(tmp_path: Path) -> None:
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult(
        [
            _FakeCandidate(_CANDIDATE_CONFIG),
            _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=2)),
        ],
        workload="qwen-workload",
    )
    config = dict(_DGD_CONFIG, name=None, name_prefix="pareto")

    paths = adapter.write(config, result=result, output_dir=tmp_path)

    assert [Path(p).stem for p in paths] == ["pareto-000", "pareto-001"]


def test_missing_dgd_config_field_raises_config_error(tmp_path: Path) -> None:
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult([_FakeCandidate(_CANDIDATE_CONFIG)])
    incomplete_config = {"name": "qwen"}

    with pytest.raises(DgdOutputConfigError, match="missing required field"):
        adapter.write(incomplete_config, result=result, output_dir=tmp_path)


def test_scalar_config_picks_the_best_candidate_when_multiple_are_given(
    tmp_path: Path,
) -> None:
    """dgd.name (scalar mode) with >1 candidate must select the single
    highest-scoring one (ties broken by fewer GPUs), not raise and not
    render all of them -- restores the deleted CLI's _best_candidate rule."""
    adapter = DgdOutputAdapter()
    low = _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=2), score=10.0, used_gpus=2)
    high = _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=4), score=20.0, used_gpus=4)
    result = _FakeSweepResult([low, high], workload="qwen-workload")

    paths = adapter.write(_DGD_CONFIG, result=result, output_dir=tmp_path)

    assert len(paths) == 1
    assert Path(paths[0]).stem == "qwen"

    result = _FakeSweepResult([high, low], workload="qwen-workload")

    paths = adapter.write(_DGD_CONFIG, result=result, output_dir=tmp_path)

    assert len(paths) == 1
    assert Path(paths[0]).stem == "qwen"
    assert "nvidia.com/gpu: '4'" in (tmp_path / paths[0]).read_text()


def test_scalar_selection_breaks_score_ties_on_fewer_gpus(tmp_path: Path) -> None:
    """Tie-break rule specifically: equal score, prefer fewer GPUs."""
    adapter = DgdOutputAdapter()
    many_gpus = _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=4), score=15.0, used_gpus=8)
    few_gpus = _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=2), score=15.0, used_gpus=2)
    result = _FakeSweepResult([many_gpus, few_gpus], workload="qwen-workload")

    paths = adapter.write(_DGD_CONFIG, result=result, output_dir=tmp_path)

    assert len(paths) == 1
    # Both configs render the same model name, so len(paths) alone can't
    # tell the two candidates apart -- this asserts the manifest content
    # that actually names the winner (fewer-GPU candidate, tp=2), so a
    # wrong winner (max-instead-of-min, an inverted tie-break, or ignoring
    # score entirely) fails this test instead of passing silently.
    assert "nvidia.com/gpu: '2'" in (tmp_path / paths[0]).read_text()


def test_pareto_render_skips_unrenderable_candidate_and_continues(
    tmp_path: Path,
) -> None:
    """One CandidateMaterializationError must not lose the rest of the
    front -- restores the deleted _render_pareto's per-candidate skip."""
    adapter = DgdOutputAdapter()
    good_a = _FakeCandidate(_CANDIDATE_CONFIG, score=10.0, used_gpus=8)
    bad = _FakeCandidate(
        dict(_CANDIDATE_CONFIG, backend="unsupported"), score=9.0, used_gpus=8
    )
    good_b = _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=2), score=8.0, used_gpus=2)
    result = _FakeSweepResult([good_a, bad, good_b], workload="qwen-workload")
    config = dict(_DGD_CONFIG, name=None, name_prefix="pareto")

    paths = adapter.write(config, result=result, output_dir=tmp_path)

    assert len(paths) == 2  # the bad candidate is skipped, not fatal


def test_name_rejects_path_escaping_values(tmp_path: Path) -> None:
    """dgd.name must not be able to write outside output_dir."""
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult(
        [_FakeCandidate(_CANDIDATE_CONFIG)], workload="qwen-workload"
    )

    for escaping_name in ("../escape", "/etc/passwd", "a/b", ".."):
        config = dict(_DGD_CONFIG, name=escaping_name)
        with pytest.raises(DgdOutputConfigError, match="single path component"):
            adapter.write(config, result=result, output_dir=tmp_path)


def test_name_prefix_rejects_path_escaping_values(tmp_path: Path) -> None:
    """Same validation must apply to dgd.name_prefix, not just dgd.name."""
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult(
        [
            _FakeCandidate(_CANDIDATE_CONFIG),
            _FakeCandidate(dict(_CANDIDATE_CONFIG, tp=2)),
        ],
        workload="qwen-workload",
    )

    for escaping_prefix in ("../escape", "/etc/passwd", "a/b"):
        config = dict(_DGD_CONFIG, name=None, name_prefix=escaping_prefix)
        with pytest.raises(DgdOutputConfigError, match="single path component"):
            adapter.write(config, result=result, output_dir=tmp_path)


def test_scalar_with_no_candidates_raises_materialization_error(tmp_path: Path) -> None:
    """Empty-list guard: a scalar run with no candidates must raise
    CandidateMaterializationError, not crash inside _best_candidate's max()
    on an empty sequence."""
    adapter = DgdOutputAdapter()
    result = _FakeSweepResult([], workload="qwen-workload")

    with pytest.raises(
        CandidateMaterializationError, match="nothing to pick a scalar winner from"
    ):
        adapter.write(_DGD_CONFIG, result=result, output_dir=tmp_path)


def test_pareto_all_candidates_failing_keeps_the_real_cause(tmp_path: Path) -> None:
    """When every Pareto candidate fails to materialize, the raised
    CandidateMaterializationError must chain the real underlying cause,
    not just be visible via the stderr skip messages."""
    adapter = DgdOutputAdapter()
    bad_a = _FakeCandidate(
        dict(_CANDIDATE_CONFIG, backend="unsupported"), score=10.0, used_gpus=8
    )
    bad_b = _FakeCandidate(
        dict(_CANDIDATE_CONFIG, backend="unsupported", tp=2), score=8.0, used_gpus=2
    )
    result = _FakeSweepResult([bad_a, bad_b], workload="qwen-workload")
    config = dict(_DGD_CONFIG, name=None, name_prefix="pareto")

    with pytest.raises(
        CandidateMaterializationError, match="no candidate could be rendered"
    ) as exc_info:
        adapter.write(config, result=result, output_dir=tmp_path)
    assert exc_info.value.__cause__ is not None
