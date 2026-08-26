# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest
import yaml

from dynamo.profiler.tests.sweeper import run_cases

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _write_case(case_path, *, backend="vllm") -> None:
    case_path.mkdir()
    (case_path / "dgdr-v1beta1.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "Qwen/Qwen3-32B",
                "backend": backend,
                "image": "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.5.0",
                "hardware": {
                    "gpuSku": "h200_sxm",
                    "totalGpus": 8,
                    "numGpusPerNode": 8,
                },
                "workload": {"isl": 1024, "osl": 128},
            }
        )
    )
    (case_path / "sweeper.yaml").write_text(
        yaml.safe_dump(
            {
                "search_space": {
                    "model_name": "Qwen/Qwen3-32B",
                    "hardware_sku": "h200_sxm",
                    "gpu_budget": 8,
                    "backend": [backend],
                },
                "workload": {"isl": 1024, "osl": 128},
            }
        )
    )


@pytest.fixture(autouse=True)
def _stub_runtime_image_derivation(monkeypatch) -> None:
    image_names = {
        "vllm": "vllm-runtime",
        "sglang": "sglang-runtime",
        "trtllm": "tensorrtllm-runtime",
    }

    def derive(image, backend):
        prefix, _, suffix = image.rpartition("/")
        _, _, tag = suffix.partition(":")
        return f"{prefix}/{image_names[backend]}:{tag}"

    monkeypatch.setattr(run_cases, "_derive_runtime_image", derive)


def test_load_case_uses_conventional_files_and_derives_dgd_options(tmp_path) -> None:
    case_path = tmp_path / "qwen"
    _write_case(case_path)

    case = run_cases.load_case(case_path)

    assert case.name == "qwen"
    assert case.generation_options.runtime_image == (
        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0"
    )
    assert case.generation_options.num_gpus_per_node == 8


def test_case_environment_uses_generated_directory_and_restores_process(
    monkeypatch, tmp_path
) -> None:
    case_path = tmp_path / "qwen"
    _write_case(case_path)
    case = run_cases.load_case(case_path)
    monkeypatch.setenv("HF_HOME", "/original/huggingface")
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    with run_cases._case_environment(case):
        assert os.environ["HF_HOME"] == str(case.cache_dir / "huggingface")
        assert os.environ["MPLCONFIGDIR"] == str(case.cache_dir / "matplotlib")
        assert os.environ["XDG_CACHE_HOME"] == str(case.cache_dir)

    assert os.environ["HF_HOME"] == "/original/huggingface"
    assert "MPLCONFIGDIR" not in os.environ


def test_load_case_rejects_different_core_intent(tmp_path) -> None:
    case_path = tmp_path / "qwen"
    _write_case(case_path)
    sweeper = yaml.safe_load((case_path / "sweeper.yaml").read_text())
    sweeper["search_space"]["gpu_budget"] = 16
    (case_path / "sweeper.yaml").write_text(yaml.safe_dump(sweeper))

    with pytest.raises(ValueError, match="gpu_budget"):
        run_cases.load_case(case_path)


def test_repository_learning_case_follows_the_convention() -> None:
    case = run_cases.load_case(run_cases._CASES_ROOT / "qwen3-32b-vllm-disagg")

    assert case.dgdr_input["model"] == "Qwen/Qwen3-32B"
    assert case.sweeper_input["search_space"]["deployment_mode"] == ["disagg"]
    assert (case.path / "recipe-dgd.yaml").is_file()


def test_sweeper_runs_once_and_renders_same_candidate_twice(
    monkeypatch, tmp_path
) -> None:
    case_path = tmp_path / "qwen"
    _write_case(case_path)
    case = run_cases.load_case(case_path)
    config = SimpleNamespace(
        goal=SimpleNamespace(is_pareto=False),
        workload=object(),
    )
    candidate = SimpleNamespace(
        config={"backend": "vllm"},
        score=3.0,
        used_gpus=4,
        metrics={"throughput": 3.0},
        objectives=None,
    )
    sweep_calls = []
    render_calls = []

    monkeypatch.setattr(run_cases, "load_sweep_config", lambda _path: config)

    def fake_sweep(received_config):
        sweep_calls.append(received_config)
        return SimpleNamespace(candidates=[candidate])

    def fake_render(received, workload, options, *, dgd_name, renderer):
        render_calls.append((received, workload, options, dgd_name, renderer))
        return f"kind: DynamoGraphDeployment\nrenderer: {renderer}\n"

    monkeypatch.setattr(run_cases, "run_sweep", fake_sweep)
    monkeypatch.setattr(run_cases, "render_dgd", fake_render)

    run_cases._run_sweeper_renderers(case)

    assert sweep_calls == [config]
    assert [call[0] for call in render_calls] == [candidate, candidate]
    assert [call[-1] for call in render_calls] == ["aic", "direct"]
    assert (case.generated_dir / "sweeper-candidate.yaml").is_file()
    assert (case.generated_dir / "sweeper-aic-dgd.yaml").is_file()
    assert (case.generated_dir / "sweeper-direct-dgd.yaml").is_file()


def test_renderer_failure_does_not_hide_other_renderer(monkeypatch, tmp_path) -> None:
    case_path = tmp_path / "qwen"
    _write_case(case_path)
    case = run_cases.load_case(case_path)
    config = SimpleNamespace(
        goal=SimpleNamespace(is_pareto=False),
        workload=object(),
    )
    candidate = SimpleNamespace(
        config={"backend": "vllm"},
        score=3.0,
        used_gpus=4,
        metrics={},
        objectives=None,
    )

    monkeypatch.setattr(run_cases, "load_sweep_config", lambda _path: config)
    monkeypatch.setattr(
        run_cases,
        "run_sweep",
        lambda _config: SimpleNamespace(candidates=[candidate]),
    )

    def render(_candidate, _workload, _options, *, dgd_name, renderer):
        if renderer == "aic":
            raise RuntimeError("missing bridge")
        return f"kind: DynamoGraphDeployment\nname: {dgd_name}\n"

    monkeypatch.setattr(run_cases, "render_dgd", render)

    with pytest.raises(RuntimeError, match="aic: missing bridge"):
        run_cases._run_sweeper_renderers(case)

    assert not (case.generated_dir / "sweeper-aic-dgd.yaml").exists()
    assert (case.generated_dir / "sweeper-aic-error.txt").read_text() == (
        "missing bridge\n"
    )
    assert (case.generated_dir / "sweeper-direct-dgd.yaml").is_file()
    assert not (case.generated_dir / "sweeper-direct-error.txt").exists()
