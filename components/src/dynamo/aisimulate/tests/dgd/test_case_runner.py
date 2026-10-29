# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
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


@pytest.fixture
def profiler_child(monkeypatch):
    popen = subprocess.Popen
    children = []

    def substitute(script, *, stdin=None):
        def launch(command, **kwargs):
            assert command[1:3] == ["-m", "dynamo.profiler"]
            child = popen(
                [sys.executable, "-c", script, *command[3:]], stdin=stdin, **kwargs
            )
            children.append(child)
            return child

        monkeypatch.setattr(run_cases.subprocess, "Popen", launch)
        return children

    yield substitute
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


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


def test_discovered_recipe_requirements_preserve_source_and_merge_proposals(
    monkeypatch, tmp_path
) -> None:
    case_path, _ = _write_case_and_hardware(tmp_path, monkeypatch)
    monkeypatch.setattr(run_cases, "_REPOSITORY_ROOT", tmp_path)
    original = {
        "source": "recipes/qwen/deploy.yaml",
        "requirements": {
            "h200-sxm-8gpu": {"gpus": 8},
            "gb200-sxm-4gpu": {"gpus": 4, "notes": "Known working topology."},
        },
    }
    recipe_path = case_path / "recipe.yaml"
    original_text = "# Maintained recipe provenance.\n" + yaml.safe_dump(
        original, sort_keys=False
    )
    recipe_path.write_text(original_text)
    recipe = run_cases.load_recipe("qwen")
    assert recipe is not None
    proposal_path = case_path / "recipe.new.yaml"

    assert (
        run_cases.write_discovered_recipe_requirement(recipe, "h200-sxm-8gpu", 8)
        is None
    )
    assert not proposal_path.exists()

    assert (
        run_cases.write_discovered_recipe_requirement(recipe, "h100-sxm-4gpu", 4)
        == proposal_path
    )
    assert (
        run_cases.write_discovered_recipe_requirement(recipe, "a100-sxm-2gpu", 2)
        == proposal_path
    )
    assert yaml.safe_load(proposal_path.read_text()) == {
        "source": original["source"],
        "requirements": {
            **original["requirements"],
            "h100-sxm-4gpu": {"gpus": 4},
            "a100-sxm-2gpu": {"gpus": 2},
        },
    }

    proposal_text = proposal_path.read_text()
    assert (
        run_cases.write_discovered_recipe_requirement(recipe, "h200-sxm-8gpu", 8)
        is None
    )
    assert proposal_path.read_text() == proposal_text
    assert recipe_path.read_text() == original_text


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
    ("stream", "diagnostic", "match", "expected_failure"),
    [
        ("stderr", "Known profiler failure", "Known profiler failure", False),
        ("stdout", "Known profiler failure", "Known profiler failure", False),
        ("stderr", "Unrelated profiler bug", "Known profiler failure", True),
        ("stderr", "Unrelated profiler bug", "returned non-zero exit status", True),
    ],
    ids=["known-stderr", "known-stdout", "unrelated-error", "generic-exit-message"],
)
@pytest.mark.timeout(10)
def test_profiler_failure_matches_child_diagnostic(
    monkeypatch,
    tmp_path,
    capfd,
    profiler_child,
    stream,
    diagnostic,
    match,
    expected_failure,
) -> None:
    _write_case_and_hardware(tmp_path, monkeypatch)
    case = run_cases.load_case(
        "qwen", run_cases.load_hardware("h200-sxm-8gpu"), output_root=tmp_path / "out"
    )
    profiler_child(
        "import sys; print('Profiler progress', flush=True); "
        f"print({diagnostic!r}, file=sys.{stream}, flush=True); sys.exit(7)"
    )
    entry = run_cases.SuiteEntry(
        case=case.name,
        hardware=case.hardware.name,
        exceptions={
            "render": {
                "profiler-v1beta1": run_cases.SuiteException(
                    status="broken", reason="Known profiler failure.", match=match
                ),
                "sweeper": run_cases.SuiteException(
                    status="skipped", reason="This case exercises the profiler process."
                ),
            }
        },
    )

    observed = f"Profiler progress\n{diagnostic}"
    expected = [f"profiler-v1beta1: {observed}"] if expected_failure else []
    assert run_cases.run_case(case, entry) == expected
    assert (case.generated_dir / "error-profiler-v1beta1.txt").read_text() == (
        observed + "\n"
    )
    output = capfd.readouterr()
    assert "Profiler progress" in output.out + output.err
    assert diagnostic in output.out + output.err


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
@pytest.mark.timeout(10)
def test_profiler_forwards_progress_before_child_exit(
    monkeypatch, tmp_path, profiler_child, stream
) -> None:
    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    _write_case_and_hardware(tmp_path, monkeypatch)
    case = run_cases.load_case(
        "qwen", run_cases.load_hardware("h200-sxm-8gpu"), output_root=tmp_path / "out"
    )
    run_cases._write_composed_inputs(case)
    children = profiler_child(
        "import select, sys\n"
        "from pathlib import Path\n"
        f"print('Profiler progress', file=sys.{stream})\n"
        "ready, _, _ = select.select([sys.stdin], [], [], 5)\n"
        "if not ready or sys.stdin.readline() != 'continue\\n':\n"
        "    raise RuntimeError('Progress was not forwarded before profiler exit')\n"
        "output = Path(sys.argv[sys.argv.index('--output-dir') + 1])\n"
        "(output / 'final_config.yaml').write_text('kind: DynamoGraphDeployment\\n')\n",
        stdin=subprocess.PIPE,
    )
    observed = []
    original_print = print

    def acknowledge_progress(*values, **kwargs):
        original_print(*values, **kwargs)
        if values and "Profiler progress" in values[0]:
            child = children[0]
            assert child.poll() is None, "Profiler output was buffered until child exit"
            child.stdin.write("continue\n")
            child.stdin.flush()
            observed.append(values[0])

    monkeypatch.setattr(run_cases, "print", acknowledge_progress, raising=False)

    run_cases._run_v1_profiler(case)

    assert observed == ["Profiler progress\n"]
    assert (case.generated_dir / "dgd-profiler-v1beta1.yaml").is_file()


@pytest.mark.timeout(10)
def test_profiler_kills_child_if_forwarding_output_fails(
    monkeypatch, tmp_path, profiler_child
) -> None:
    _write_case_and_hardware(tmp_path, monkeypatch)
    case = run_cases.load_case(
        "qwen", run_cases.load_hardware("h200-sxm-8gpu"), output_root=tmp_path / "out"
    )
    run_cases._write_composed_inputs(case)
    children = profiler_child(
        "import select, sys\n"
        "print('Profiler progress', flush=True)\n"
        "ready, _, _ = select.select([sys.stdin], [], [], 5)\n"
        "if ready:\n"
        "    sys.stdin.readline()\n"
        "sys.exit(23)\n",
        stdin=subprocess.PIPE,
    )
    original_print = print

    def fail_forwarding(*values, **kwargs):
        if values == ("Profiler progress\n",):
            raise BrokenPipeError("Profiler output consumer closed")
        original_print(*values, **kwargs)

    monkeypatch.setattr(run_cases, "print", fail_forwarding, raising=False)

    with pytest.raises(BrokenPipeError, match="Profiler output consumer closed"):
        run_cases._run_v1_profiler(case)

    assert children[0].returncode is not None and children[0].returncode < 0


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
