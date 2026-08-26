"""Tests for changed-path to pytest-marker selection."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

try:
    import tomllib  # Python >=3.11
except ImportError:
    import tomli as tomllib  # type: ignore

sys.path.insert(0, str(Path(__file__).parent))

from codeowners_match import compute_resolution  # noqa: E402
from select_pytest import (  # noqa: E402
    build_plan,
    selected_pr_tests,
    selected_tests_by_lane,
)

from tests.report_pytest_markers import missing_categories  # noqa: E402

REPO_ROOT = Path(__file__).parents[2]


def _model():
    return compute_resolution(
        {
            "meta": {"catch_all": "@runtime"},
            "areas": [
                {
                    "label": "runtime",
                    "github_team": "@runtime",
                    "path_globs": ["lib/", "tests/"],
                },
                {
                    "label": "router",
                    "github_team": "@router",
                    "path_globs": ["lib/router/", "tests/router/"],
                    "pytest": {
                        "markers": ["router"],
                    },
                },
                {
                    "label": "backend-vllm",
                    "github_team": "@vllm",
                    "path_globs": ["components/vllm/"],
                    "pytest": {"markers": ["vllm"]},
                },
                {
                    "label": "multimodal",
                    "github_team": "@multimodal",
                    "path_globs": ["components/vllm/multimodal/"],
                    "pytest": {"markers": ["multimodal"]},
                },
                {
                    "label": "docs",
                    "github_team": "@docs",
                    "path_globs": ["docs/"],
                    "pytest": {"mode": "none"},
                },
                {
                    "label": "agents",
                    "github_team": "@agents",
                    "path_globs": [],
                    "pytest": {"mode": "none"},
                },
            ],
            "shared": [{"glob": "lib/router/agents/", "owners": ["router", "agents"]}],
        }
    )


def test_shared_router_change_uses_router_marker_for_every_backend() -> None:
    plan = build_plan(_model(), ["lib/router/scheduler.rs"])

    assert plan.mode == "markers"
    assert plan.lanes["vllm"].expression == "router"
    assert plan.lanes["sglang"].expression == "router"
    assert plan.lanes["trtllm"].expression == "router"
    assert plan.lanes["generic"].expression == "router"


def test_backend_and_feature_markers_are_intersected() -> None:
    plan = build_plan(_model(), ["components/vllm/multimodal/image.py"])

    assert plan.clauses == ("vllm and multimodal",)
    assert plan.lanes["vllm"].expression == "multimodal"
    assert plan.lanes["sglang"].mode == "none"
    assert plan.lanes["generic"].mode == "none"


def test_different_files_are_unioned_without_cross_intersection() -> None:
    plan = build_plan(
        _model(),
        ["components/vllm/worker.py", "lib/router/scheduler.rs"],
    )

    assert plan.lanes["vllm"].mode == "full"
    assert plan.lanes["sglang"].expression == "router"
    assert set(plan.clauses) == {"router", "vllm"}


def test_specific_marker_mapping_overrides_generic_fallback_area() -> None:
    plan = build_plan(_model(), ["tests/router/test_router.py"])

    assert plan.mode == "markers"
    assert not plan.fallback_reasons
    assert plan.changed_test_files == ("tests/router/test_router.py",)


def test_unmapped_executable_area_falls_back_to_full() -> None:
    plan = build_plan(_model(), ["lib/runtime.py"])

    assert plan.mode == "full"
    assert plan.lanes["vllm"].mode == "full"
    assert "no marker mapping" in plan.fallback_reasons[0]


def test_explicit_no_pytest_area_selects_no_tests() -> None:
    plan = build_plan(_model(), ["docs/overview.md"])

    assert plan.mode == "none"
    assert all(selection.mode == "none" for selection in plan.lanes.values())


def test_shared_ownership_contributes_marker_metadata() -> None:
    plan = build_plan(_model(), ["lib/router/agents/tool.py"])

    assert set(plan.areas) == {"agents", "router", "runtime"}
    assert plan.lanes["generic"].expression == "router"


def test_unknown_path_falls_back_to_full() -> None:
    plan = build_plan(_model(), ["unknown/new.py"])

    assert plan.mode == "full"
    assert plan.fallback_reasons == ("unknown/new.py: no explicit ownership area",)


def test_repository_area_fallback_is_implicit_and_markers_are_registered() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    omitted_labels = {area["label"] for area in spec["areas"] if "pytest" not in area}
    assert omitted_labels
    assert all(
        area.get("pytest", {}).get("mode") != "fallback" for area in spec["areas"]
    )

    model = compute_resolution(spec)
    resolved_by_label = {area.label: area for area in model.areas}
    assert all(
        resolved_by_label[label].pytest_mode == "fallback" for label in omitted_labels
    )
    configured = {marker for area in model.areas for marker in area.pytest_markers}
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    registered = {
        declaration.split(":", 1)[0]
        for declaration in pyproject["tool"]["pytest"]["ini_options"]["markers"]
    }

    assert configured <= registered


def test_repository_router_path_selects_router_marker() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    plan = build_plan(
        compute_resolution(spec), ["components/src/dynamo/router/router.py"]
    )

    assert plan.areas == ("router",)
    assert all(selection.expression == "router" for selection in plan.lanes.values())


def test_repository_selective_backend_jobs_keep_unit_smoke_tests() -> None:
    workflow = (REPO_ROOT / ".github/workflows/pr.yaml").read_text(encoding="utf-8")
    selective_marker_inputs = [
        line.strip()
        for line in workflow.splitlines()
        if "test_markers:" in line and "PYTEST_SELECTION_MODE == 'selective'" in line
    ]

    assert len(selective_marker_inputs) == 10
    assert all("and (unit or ({0}))" in line for line in selective_marker_inputs)


def test_unit_framework_test_does_not_require_selective_feature_marker() -> None:
    markers = {"pre_merge", "unit", "gpu_0", "vllm"}

    assert "Selective Feature" not in missing_categories(markers)


def test_non_unit_framework_test_requires_a_selective_feature_marker() -> None:
    markers = {"pre_merge", "e2e", "gpu_1", "vllm"}

    assert "Selective Feature" in missing_categories(markers)


def test_non_unit_framework_test_can_carry_multiple_selective_features() -> None:
    markers = {"pre_merge", "e2e", "gpu_1", "vllm", "router", "multimodal"}

    assert "Selective Feature" not in missing_categories(markers)


def test_exact_lane_inventory_uses_effective_collected_markers() -> None:
    plan = build_plan(_model(), ["lib/router/scheduler.rs"])
    records = [
        {"nodeid": "tests/router/test_generic.py::test_router", "markers": ["router"]},
        {
            "nodeid": "tests/serve/test_vllm.py::test_router",
            "markers": ["router", "vllm"],
        },
        {"nodeid": "tests/serve/test_vllm.py::test_other", "markers": ["vllm"]},
    ]

    selected = selected_tests_by_lane(plan, records)

    assert selected["generic"] == ["tests/router/test_generic.py::test_router"]
    assert selected["vllm"] == ["tests/serve/test_vllm.py::test_router"]


def test_exact_pr_inventory_applies_lifecycle_and_gpu_markers() -> None:
    plan = build_plan(_model(), ["lib/router/scheduler.rs"])
    records = [
        {
            "nodeid": "tests/serve/test_vllm.py::test_cpu",
            "markers": ["pre_merge", "gpu_0", "router", "vllm"],
        },
        {
            "nodeid": "tests/serve/test_vllm.py::test_gpu",
            "markers": ["pre_merge", "gpu_1", "router", "vllm"],
        },
        {
            "nodeid": "tests/serve/test_vllm.py::test_nightly",
            "markers": ["nightly", "gpu_1", "router", "vllm"],
        },
        {
            "nodeid": "components/src/vllm/tests/test_smoke.py::test_unit",
            "markers": ["pre_merge", "gpu_0", "unit", "vllm"],
        },
        {
            "nodeid": "tests/serve/test_vllm.py::test_multi_gpu",
            "markers": ["pre_merge", "gpu_2", "router", "vllm"],
        },
        {
            "nodeid": "tests/serve/test_vllm.py::test_four_gpu",
            "markers": ["post_merge", "gpu_4", "router", "vllm"],
        },
    ]

    selected = selected_pr_tests(plan, records)

    assert selected["vllm-cpu"] == [
        "components/src/vllm/tests/test_smoke.py::test_unit",
        "tests/serve/test_vllm.py::test_cpu",
    ]
    assert selected["vllm-gpu-1"] == ["tests/serve/test_vllm.py::test_gpu"]
    assert selected["vllm-multi-gpu"] == ["tests/serve/test_vllm.py::test_multi_gpu"]
    assert selected["vllm-4-gpu"] == ["tests/serve/test_vllm.py::test_four_gpu"]
