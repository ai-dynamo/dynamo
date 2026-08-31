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


def test_dgd_name_is_stable_and_leaves_room_for_component_names() -> None:
    assert run_cases._dgd_name("qwen") == "qwen"
    shortened = run_cases._dgd_name("deepseek-r1-trtllm-disagg-wide-ep")

    assert shortened == "deepseek-r1-trtllm-8c2e7a9a"
    assert len(shortened) <= 28


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


def test_main_does_not_execute_case_wide_skip(monkeypatch, tmp_path, capsys) -> None:
    cases_root = tmp_path / "cases"
    case_path = cases_root / "qwen"
    case_path.mkdir(parents=True)
    (case_path / "recipe.yaml").write_text(
        yaml.safe_dump(
            {
                "source": "recipes/qwen/deploy.yaml",
                "requirements": {"h200-sxm-8gpu": {"gpus": 8}},
            }
        )
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "tests": [
                    {
                        "case": "qwen",
                        "hardware": "h200-sxm-8gpu",
                        "status": "skipped",
                        "reason": "The historical input is not recoverable.",
                        "links": ["https://github.com/ai-dynamo/dynamo/issues/8469"],
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(run_cases, "_CASES_ROOT", cases_root)
    monkeypatch.setattr(run_cases, "_REPOSITORY_ROOT", tmp_path)

    assert run_cases.main(["--suite", str(suite)]) == 0
    output = capsys.readouterr().out
    assert "skipped: The historical input is not recoverable." in output
    assert "1 case(s) skipped" in output


def test_load_suite_parses_nested_render_and_deploy_exceptions(tmp_path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        yaml.safe_dump(
            {
                "tests": [
                    {
                        "case": "qwen",
                        "hardware": "h200-sxm-8gpu",
                        "exceptions": {
                            "render": {
                                "sweeper-direct": {
                                    "status": "broken",
                                    "reason": "Direct rendering lacks this topology.",
                                    "links": [
                                        "https://github.com/ai-dynamo/dynamo/issues/8469"
                                    ],
                                }
                            },
                            "deploy": {
                                "recipe": {
                                    "status": "skipped",
                                    "reason": "The historical recipe was removed.",
                                }
                            },
                        },
                    }
                ]
            },
            sort_keys=False,
        )
    )

    [entry] = run_cases.load_suite(suite)

    assert entry.exception_for("render", "sweeper-direct") == run_cases.SuiteException(
        status="broken",
        reason="Direct rendering lacks this topology.",
        links=("https://github.com/ai-dynamo/dynamo/issues/8469",),
    )
    assert entry.exception_for("deploy", "recipe") == run_cases.SuiteException(
        status="skipped", reason="The historical recipe was removed."
    )
    assert entry.exception_for("render", "sweeper-aic") is None


def test_recipe_hardware_discovery_writes_new_file_without_changing_source(
    monkeypatch, tmp_path
) -> None:
    cases_root = tmp_path / "cases"
    case_path = cases_root / "qwen"
    case_path.mkdir(parents=True)
    recipe_path = case_path / "recipe.yaml"
    original = {
        "source": "recipes/qwen/deploy.yaml",
        "requirements": {"h200-sxm-8gpu": {"gpus": 8}},
    }
    recipe_path.write_text(yaml.safe_dump(original, sort_keys=False))
    monkeypatch.setattr(run_cases, "_CASES_ROOT", cases_root)
    monkeypatch.setattr(run_cases, "_REPOSITORY_ROOT", tmp_path)
    recipe = run_cases.load_recipe("qwen")

    assert recipe is not None
    output = run_cases.write_discovered_recipe_requirement(recipe, "h100-sxm-4gpu", 4)

    assert yaml.safe_load(recipe_path.read_text()) == original
    assert output == case_path / "recipe.new.yaml"
    assert yaml.safe_load(output.read_text()) == {
        "source": "recipes/qwen/deploy.yaml",
        "requirements": {
            "h200-sxm-8gpu": {"gpus": 8},
            "h100-sxm-4gpu": {"gpus": 4},
        },
    }


@pytest.mark.parametrize("api_version", ["v1alpha1", "v1beta1"])
def test_recipe_gpu_count_includes_replicas(monkeypatch, tmp_path, api_version) -> None:
    cases_root = tmp_path / "cases"
    case_path = cases_root / "qwen"
    case_path.mkdir(parents=True)
    recipe_dgd = tmp_path / "deploy.yaml"
    if api_version == "v1alpha1":
        spec = {
            "services": {
                "Worker": {
                    "replicas": 3,
                    "resources": {"limits": {"gpu": "2"}},
                }
            }
        }
    else:
        spec = {
            "components": [
                {
                    "name": "worker",
                    "replicas": 3,
                    "podTemplate": {
                        "spec": {
                            "containers": [
                                {
                                    "name": "main",
                                    "resources": {"limits": {"nvidia.com/gpu": "2"}},
                                }
                            ]
                        }
                    },
                }
            ]
        }
    recipe_dgd.write_text(
        yaml.safe_dump(
            {
                "apiVersion": f"nvidia.com/{api_version}",
                "kind": "DynamoGraphDeployment",
                "metadata": {"name": "qwen"},
                "spec": spec,
            }
        )
    )
    (case_path / "recipe.yaml").write_text(
        yaml.safe_dump({"source": "deploy.yaml", "requirements": {}})
    )
    monkeypatch.setattr(run_cases, "_CASES_ROOT", cases_root)
    monkeypatch.setattr(run_cases, "_REPOSITORY_ROOT", tmp_path)
    recipe = run_cases.load_recipe("qwen")

    assert recipe is not None
    assert run_cases.recipe_gpu_count(recipe) == 6


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

    assert len(entries) == 29
    for entry in entries:
        assert run_cases.load_recipe(entry.case) is not None
        assert run_cases.missing_case_inputs(entry.case) == ()
        case = run_cases.load_case(entry.case, run_cases.load_hardware(entry.hardware))
        assert case.recipe is not None


def test_repository_gb300_suite_composes_sweeper_and_skips_v1() -> None:
    suite_path = run_cases._ROOT / "testsuite-gb300.yaml"
    [entry] = run_cases.load_suite(suite_path)

    assert entry.case == "qwen3-32b-vllm-agg"
    assert entry.hardware == "gb300-4gpu"
    assert entry.exception_for("render", "profiler-v1beta1") == (
        run_cases.SuiteException(
            status="skipped",
            reason="DGDR v1beta1 does not define gb300 in its gpuSku enum.",
        )
    )
    case = run_cases.load_case(entry.case, run_cases.load_hardware(entry.hardware))
    assert case.dgdr_input["hardware"] == {
        "gpuSku": "gb300",
        "totalGpus": 4,
        "numGpusPerNode": 4,
        "interconnect": "nvlink",
        "rdma": True,
    }
    assert case.sweeper_input["search_space"]["hardware_sku"] == "gb300"
    assert case.sweeper_input["search_space"]["gpu_budget"] == 4


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

    assert run_cases._run_sweeper_renderers(case) == ["sweeper-aic: missing bridge"]
    assert (case.generated_dir / "error-sweeper-aic.txt").read_text() == (
        "missing bridge\n"
    )
    assert not stale_aic_output.exists()
    assert (case.generated_dir / "dgd-sweeper-direct.yaml").is_file()


def test_broken_renderer_failure_does_not_fail_case(monkeypatch, tmp_path) -> None:
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
        if renderer == "direct":
            raise RuntimeError("unsupported topology")
        return f"kind: DynamoGraphDeployment\nname: {dgd_name}\n"

    monkeypatch.setattr(run_cases, "render_dgd", render)
    entry = run_cases.SuiteEntry(
        case="qwen",
        hardware="h200-sxm-8gpu",
        exceptions={
            "render": {
                "sweeper-direct": run_cases.SuiteException(
                    status="broken", reason="Direct rendering lacks this topology."
                )
            }
        },
    )

    assert run_cases._run_sweeper_renderers(case, entry) == []
    assert (case.generated_dir / "dgd-sweeper-aic.yaml").is_file()
    assert not (case.generated_dir / "dgd-sweeper-direct.yaml").exists()
