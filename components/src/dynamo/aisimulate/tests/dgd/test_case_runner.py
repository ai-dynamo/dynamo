# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest
import yaml

from dynamo.aisimulate.tests.dgd.comparison import run_cases

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
    name = "deepseek-r1-trtllm-disagg-wide-ep"
    assert run_cases._dgd_name("qwen") == "qwen"
    shortened = run_cases._dgd_name(name)

    assert shortened == run_cases._dgd_name(name)
    assert len(shortened) <= 28
    assert shortened != run_cases._dgd_name(name + "-other")


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
    monkeypatch.setattr(run_cases, "_CASES_ROOT", tmp_path / "missing-cases")

    assert run_cases.main(["--suite", str(suite)]) == 0
    output = capsys.readouterr().out
    assert "skipped: The historical input is not recoverable." in output
    assert "1 case(s) skipped" in output


def test_load_suite_parses_expected_renderer_failure(tmp_path) -> None:
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
                                    "match": "TEP.*not implemented",
                                    "links": [
                                        "https://github.com/ai-dynamo/dynamo/issues/8469"
                                    ],
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
        match="TEP.*not implemented",
    )
    assert entry.exception_for("render", "sweeper-aic") is None


@pytest.mark.parametrize(
    ("match", "expected_failures"),
    [
        (None, ["sweeper-aic: missing bridge"]),
        ("missing bridge", []),
        ("unsupported topology", ["sweeper-aic: missing bridge"]),
    ],
    ids=["unqualified-failure", "expected-failure", "unrelated-regression"],
)
def test_renderer_failure_keeps_other_renderer_output(
    monkeypatch, tmp_path, match, expected_failures
) -> None:
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

    entry = run_cases.SuiteEntry(
        case="qwen",
        hardware="h200-sxm-8gpu",
        exceptions={
            "render": {
                "sweeper-aic": run_cases.SuiteException(
                    status="broken", reason="Known renderer failure.", match=match
                )
            }
        },
    )

    assert run_cases._run_sweeper_renderers(case, entry) == expected_failures
    assert (case.generated_dir / "error-sweeper-aic.txt").read_text() == (
        "missing bridge\n"
    )
    assert not stale_aic_output.exists()
    assert (case.generated_dir / "dgd-sweeper-direct.yaml").is_file()
