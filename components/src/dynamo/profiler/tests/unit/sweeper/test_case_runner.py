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


def _write_case_and_hardware(tmp_path, monkeypatch, *, backend="vllm"):
    cases_root = tmp_path / "cases"
    hardware_root = tmp_path / "hardware"
    case_path = cases_root / "qwen"
    hardware_path = hardware_root / "h200-sxm-8gpu"
    case_path.mkdir(parents=True)
    hardware_path.mkdir(parents=True)
    (case_path / "dgdr-v1beta1.yaml").write_text(
        yaml.safe_dump(
            {
                "model": "Qwen/Qwen3-32B",
                "backend": backend,
                "image": "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.5.0",
                "workload": {"isl": 1024, "osl": 128},
            }
        )
    )
    (case_path / "sweeper.yaml").write_text(
        yaml.safe_dump(
            {
                "search_space": {
                    "model_name": "Qwen/Qwen3-32B",
                    "backend": [backend],
                },
                "workload": {"isl": 1024, "osl": 128},
            }
        )
    )
    (hardware_path / "dgdr-v1beta1.patch.yaml").write_text(
        yaml.safe_dump(
            {
                "hardware": {
                    "gpuSku": "h200_sxm",
                    "totalGpus": 8,
                    "numGpusPerNode": 8,
                }
            }
        )
    )
    (hardware_path / "sweeper.patch.yaml").write_text(
        yaml.safe_dump({"search_space": {"hardware_sku": "h200_sxm", "gpu_budget": 8}})
    )
    monkeypatch.setattr(run_cases, "_CASES_ROOT", cases_root)
    monkeypatch.setattr(run_cases, "_HARDWARE_ROOT", hardware_root)
    return case_path, hardware_path


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


def test_merge_patch_merges_maps_replaces_lists_and_removes_nulls() -> None:
    base = {"map": {"kept": 1, "removed": 2}, "list": [1], "scalar": "old"}
    patch = {"map": {"removed": None, "added": 3}, "list": [2], "scalar": "new"}

    assert run_cases._merge_patch(base, patch) == {
        "map": {"kept": 1, "added": 3},
        "list": [2],
        "scalar": "new",
    }
    assert base["map"]["removed"] == 2


def test_load_case_composes_hardware_and_derives_dgd_options(
    monkeypatch, tmp_path
) -> None:
    _write_case_and_hardware(tmp_path, monkeypatch)
    hardware = run_cases.load_hardware("h200-sxm-8gpu")

    case = run_cases.load_case("qwen", hardware, output_root=tmp_path / "output")

    assert case.hardware.name == "h200-sxm-8gpu"
    assert case.dgdr_input["hardware"]["gpuSku"] == "h200_sxm"
    assert case.sweeper_input["search_space"]["gpu_budget"] == 8
    assert case.generation_options.runtime_image == (
        "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0"
    )
    assert case.generation_options.num_gpus_per_node == 8
    assert case.generated_dir == tmp_path / "output/h200-sxm-8gpu/qwen"


def test_case_environment_uses_output_cache_and_restores_process(
    monkeypatch, tmp_path
) -> None:
    _write_case_and_hardware(tmp_path, monkeypatch)
    case = run_cases.load_case(
        "qwen", run_cases.load_hardware("h200-sxm-8gpu"), output_root=tmp_path / "out"
    )
    monkeypatch.setenv("HF_HOME", "/original/huggingface")
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)

    with run_cases._case_environment(case):
        assert os.environ["HF_HOME"] == str(case.cache_dir / "huggingface")
        assert os.environ["MPLCONFIGDIR"] == str(case.cache_dir / "matplotlib")
        assert os.environ["XDG_CACHE_HOME"] == str(case.cache_dir)

    assert os.environ["HF_HOME"] == "/original/huggingface"
    assert "MPLCONFIGDIR" not in os.environ


def test_load_case_rejects_hardware_disagreement(monkeypatch, tmp_path) -> None:
    _, hardware_path = _write_case_and_hardware(tmp_path, monkeypatch)
    (hardware_path / "sweeper.patch.yaml").write_text(
        yaml.safe_dump({"search_space": {"hardware_sku": "h200_sxm", "gpu_budget": 16}})
    )

    with pytest.raises(ValueError, match="gpu_budget"):
        run_cases.load_case(
            "qwen",
            run_cases.load_hardware("h200-sxm-8gpu"),
            output_root=tmp_path / "out",
        )


def test_load_suite_returns_explicit_case_hardware_pairs(tmp_path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "source": "https://github.com/ai-dynamo/dynamo/issues/8469",
                "tests": [{"case": "qwen", "hardware": "h200-sxm-8gpu"}],
            }
        )
    )

    assert run_cases.load_suite(suite) == [
        run_cases.SuiteEntry(case="qwen", hardware="h200-sxm-8gpu")
    ]
    assert run_cases.default_suite_output_root(suite) == (
        run_cases._DEFAULT_OUTPUT_ROOT / "suite"
    )


def test_repository_learning_case_composes_without_case_local_hardware() -> None:
    hardware = run_cases.load_hardware("h200-sxm-16gpu")
    case = run_cases.load_case("qwen3-32b-vllm-disagg", hardware)

    assert "hardware" not in yaml.safe_load(
        (case.path / "dgdr-v1beta1.yaml").read_text()
    )
    assert (
        "hardware_sku"
        not in yaml.safe_load((case.path / "sweeper.yaml").read_text())["search_space"]
    )
    assert case.recipe is not None
    assert case.recipe.source == "recipes/qwen3-32b/vllm/disagg-kv-router/deploy.yaml"


def test_repository_suite_resolves_every_ashna_case() -> None:
    suite_path = run_cases._ROOT / "testsuite-issue-8469.yaml"
    entries = run_cases.load_suite(suite_path)

    assert len(entries) == 8
    for entry in entries:
        case = run_cases.load_case(entry.case, run_cases.load_hardware(entry.hardware))
        assert case.recipe is not None
        assert case.recipe.path.is_file()


def test_sweeper_runs_once_and_renders_same_candidate_twice(
    monkeypatch, tmp_path
) -> None:
    _write_case_and_hardware(tmp_path, monkeypatch)
    case = run_cases.load_case(
        "qwen", run_cases.load_hardware("h200-sxm-8gpu"), output_root=tmp_path / "out"
    )
    run_cases._write_composed_inputs(case)
    config = SimpleNamespace(goal=SimpleNamespace(is_pareto=False), workload=object())
    candidate = SimpleNamespace(
        config={"backend": "vllm"},
        score=3.0,
        used_gpus=4,
        metrics={"throughput": 3.0},
        objectives=None,
    )
    sweep_calls = []
    render_calls = []
    monkeypatch.setattr(run_cases, "load_sweep_config", lambda path: config)

    def fake_sweep(received_config):
        sweep_calls.append(received_config)
        return SimpleNamespace(candidates=[candidate])

    def fake_render(received, workload, options, *, dgd_name, renderer):
        render_calls.append((received, workload, options, dgd_name, renderer))
        return f"kind: DynamoGraphDeployment\nrenderer: {renderer}\n"

    monkeypatch.setattr(run_cases, "run_sweep", fake_sweep)
    monkeypatch.setattr(run_cases, "render_dgd", fake_render)

    assert run_cases._run_sweeper_renderers(case) == []
    assert sweep_calls == [config]
    assert [call[0] for call in render_calls] == [candidate, candidate]
    assert [call[-1] for call in render_calls] == ["aic", "direct"]
    assert (case.generated_dir / "candidate-sweeper.yaml").is_file()
    assert (case.generated_dir / "dgd-sweeper-aic.yaml").is_file()
    assert (case.generated_dir / "dgd-sweeper-direct.yaml").is_file()


def test_renderer_failure_keeps_other_renderer_output(monkeypatch, tmp_path) -> None:
    _write_case_and_hardware(tmp_path, monkeypatch)
    case = run_cases.load_case(
        "qwen", run_cases.load_hardware("h200-sxm-8gpu"), output_root=tmp_path / "out"
    )
    run_cases._write_composed_inputs(case)
    config = SimpleNamespace(goal=SimpleNamespace(is_pareto=False), workload=object())
    candidate = SimpleNamespace(
        config={"backend": "vllm"}, score=3.0, used_gpus=4, metrics={}, objectives=None
    )
    monkeypatch.setattr(run_cases, "load_sweep_config", lambda path: config)
    monkeypatch.setattr(
        run_cases, "run_sweep", lambda received: SimpleNamespace(candidates=[candidate])
    )

    def render(received, workload, options, *, dgd_name, renderer):
        if renderer == "aic":
            raise RuntimeError("missing bridge")
        return f"kind: DynamoGraphDeployment\nname: {dgd_name}\n"

    monkeypatch.setattr(run_cases, "render_dgd", render)
    stale_aic_output = case.generated_dir / "dgd-sweeper-aic.yaml"
    stale_aic_output.write_text("stale\n")

    assert run_cases._run_sweeper_renderers(case) == ["aic: missing bridge"]
    assert (case.generated_dir / "error-sweeper-aic.txt").read_text() == (
        "missing bridge\n"
    )
    assert not stale_aic_output.exists()
    assert (case.generated_dir / "dgd-sweeper-direct.yaml").is_file()
