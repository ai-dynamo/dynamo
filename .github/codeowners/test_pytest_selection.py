"""Tests for changed-path to pytest-marker selection."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

try:
    import tomllib  # Python >=3.11
except ImportError:
    import tomli as tomllib  # type: ignore

sys.path.insert(0, str(Path(__file__).parent))

from codeowners_match import compute_resolution  # noqa: E402
from select_pytest import (  # noqa: E402
    _write_github_output,
    _write_summary,
    build_plan,
    selected_pr_tests,
    selected_tests_by_lane,
)

from tests.report_pytest_markers import (
    SELECTIVE_FEATURE_MARKERS,
    STUB_MODULES,
    MarkerReportPlugin,
    Report,
)
from tests.report_pytest_markers import TestRecord as MarkerTestRecord  # noqa: E402
from tests.report_pytest_markers import (
    missing_categories,
    validate_path_feature_alignment,
    validate_selective_marker_coverage,
)

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
                },
                {
                    "label": "agents",
                    "github_team": "@agents",
                    "path_globs": [],
                },
                {
                    "label": "tooling",
                    "github_team": "@runtime",
                    "path_globs": ["tools/"],
                    "pytest": {"markers": []},
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


def test_area_without_pytest_markers_falls_back_to_full() -> None:
    plan = build_plan(_model(), ["docs/config.yaml"])

    assert plan.mode == "full"
    assert all(selection.mode == "full" for selection in plan.lanes.values())


def test_explicit_empty_marker_mapping_selects_smoke_only() -> None:
    plan = build_plan(_model(), ["tools/generate.py"])

    assert plan.mode == "none"
    assert plan.smoke_only_paths == ("tools/generate.py",)
    assert not plan.fallback_reasons
    assert all(selection.mode == "none" for selection in plan.lanes.values())


def test_smoke_only_path_does_not_discard_a_feature_mapping() -> None:
    plan = build_plan(_model(), ["lib/router/scheduler.rs", "tools/generate.py"])

    assert plan.mode == "markers"
    assert plan.smoke_only_paths == ("tools/generate.py",)
    assert all(selection.expression == "router" for selection in plan.lanes.values())


def test_shared_ownership_contributes_marker_metadata() -> None:
    plan = build_plan(_model(), ["lib/router/agents/tool.py"])

    assert set(plan.areas) == {"agents", "router", "runtime"}
    assert plan.lanes["generic"].expression == "router"


def test_unknown_path_falls_back_to_full() -> None:
    plan = build_plan(_model(), ["unknown/new.py"])

    assert plan.mode == "full"
    assert plan.fallback_reasons == ("unknown/new.py: no explicit ownership area",)


def test_repository_area_policies_are_explicit_and_markers_are_registered() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    omitted_labels = {area["label"] for area in spec["areas"] if "pytest" not in area}
    assert not omitted_labels
    assert all(set(area.get("pytest", {})) <= {"markers"} for area in spec["areas"])

    model = compute_resolution(spec)
    assert all(area.pytest_configured for area in model.areas)
    assert any(
        area.pytest_configured and not area.pytest_markers for area in model.areas
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


def test_repository_area_features_are_the_audit_vocabulary() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    configured_features = {
        marker
        for area in spec["areas"]
        for marker in area.get("pytest", {}).get("markers", [])
        if marker not in {"vllm", "sglang", "trtllm"}
    }

    assert configured_features == SELECTIVE_FEATURE_MARKERS


@pytest.mark.parametrize("marker", ["pre_merge", "gpu_1", "unit", "e2e"])
def test_area_policy_rejects_non_feature_marker(marker: str) -> None:
    with pytest.raises(SystemExit, match="not a framework or selective feature"):
        compute_resolution(
            {
                "meta": {"catch_all": "@runtime"},
                "areas": [
                    {
                        "label": "invalid",
                        "github_team": "@runtime",
                        "path_globs": ["tests/"],
                        "pytest": {"markers": [marker]},
                    }
                ],
            }
        )


def test_repository_router_path_selects_router_marker() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    plan = build_plan(
        compute_resolution(spec), ["components/src/dynamo/router/router.py"]
    )

    assert plan.areas == ("router",)
    assert all(selection.expression == "router" for selection in plan.lanes.values())


def test_repository_mixed_feature_serve_test_maps_every_feature() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    plan = build_plan(compute_resolution(spec), ["tests/serve/test_vllm.py"])

    assert set(plan.areas) == {
        "backend-vllm",
        "core-tests",
        "multimodal",
        "router",
        "runtime",
    }
    assert plan.mode == "markers"
    assert plan.lanes["vllm"].expression == "core or multimodal or router"
    assert plan.lanes["sglang"].mode == "none"
    assert not plan.fallback_reasons


def test_repository_test_related_paths_have_explicit_marker_mappings() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    model = compute_resolution(spec)
    expected = {
        "components/src/dynamo/common/tests/memory/"
        "test_multimodal_embedding_cache_manager.py": {
            lane: "multimodal" for lane in ("generic", "sglang", "trtllm", "vllm")
        },
        "components/src/dynamo/common/tests/multimodal/"
        "test_async_encoder_cache.py": {
            lane: "multimodal" for lane in ("generic", "sglang", "trtllm", "vllm")
        },
        "components/src/dynamo/common/tests/multimodal/"
        "test_nvdec_decoder_gpu.py": {
            lane: "multimodal" for lane in ("generic", "sglang", "trtllm", "vllm")
        },
        "tests/report_pytest_markers.py": {
            lane: "core" for lane in ("sglang", "trtllm", "vllm")
        },
        "tests/runtime/test_sample_multimodal_smoke.py": {
            lane: "multimodal" for lane in ("generic", "sglang", "trtllm", "vllm")
        },
        "tests/serve/test_sample.py": {"vllm": "core"},
        "tests/vllm_self_benchmark/test_self_benchmark_gpu.py": {"vllm": "core"},
    }

    for path, lane_expressions in expected.items():
        plan = build_plan(model, [path])
        assert plan.mode == "markers", path
        assert not plan.fallback_reasons, path
        assert {
            lane: selection.expression
            for lane, selection in plan.lanes.items()
            if selection.mode == "markers"
        } == lane_expressions, path


def test_repository_test_readme_does_not_select_pytest() -> None:
    spec = yaml.safe_load(
        (REPO_ROOT / ".github/codeowners/areas.yaml").read_text(encoding="utf-8")
    )
    plan = build_plan(compute_resolution(spec), ["tests/README.md"])

    assert plan.mode == "none"
    assert plan.ignored_paths == ("tests/README.md",)
    assert not plan.fallback_reasons
    assert all(selection.mode == "none" for selection in plan.lanes.values())


def test_summary_reports_effective_backend_markers_not_internal_clauses(
    tmp_path: Path,
) -> None:
    summary = tmp_path / "summary"
    plan = build_plan(_model(), ["lib/router/scheduler.rs"])

    _write_summary(summary, ["lib/router/scheduler.rs"], plan)

    rendered = summary.read_text(encoding="utf-8")
    assert "Marker clauses" not in rendered
    assert "### Backend feature markers" in rendered
    assert "- `vllm`: `router`" in rendered


def test_github_outputs_only_applied_backend_features(tmp_path: Path) -> None:
    plan = build_plan(_model(), ["lib/router/scheduler.rs"])
    output = tmp_path / "github-output"

    _write_github_output(output, plan, apply_selection=True)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "sglang_mode=markers",
        "sglang_features=router",
        "trtllm_mode=markers",
        "trtllm_features=router",
        "vllm_mode=markers",
        "vllm_features=router",
    ]


def test_github_outputs_empty_features_for_shadow_and_full_fallback(
    tmp_path: Path,
) -> None:
    shadow_output = tmp_path / "shadow-output"
    _write_github_output(
        shadow_output,
        build_plan(_model(), ["lib/router/scheduler.rs"]),
        apply_selection=False,
    )
    fallback_output = tmp_path / "fallback-output"
    _write_github_output(
        fallback_output,
        build_plan(_model(), ["lib/runtime.py"]),
        apply_selection=True,
    )

    expected_full = [
        "sglang_mode=full",
        "sglang_features=",
        "trtllm_mode=full",
        "trtllm_features=",
        "vllm_mode=full",
        "vllm_features=",
    ]
    assert shadow_output.read_text(encoding="utf-8").splitlines() == expected_full
    assert fallback_output.read_text(encoding="utf-8").splitlines() == expected_full


def test_github_outputs_none_for_lanes_without_selected_features(
    tmp_path: Path,
) -> None:
    output = tmp_path / "github-output"
    plan = build_plan(_model(), ["components/vllm/worker.py"])

    _write_github_output(output, plan, apply_selection=True)

    assert output.read_text(encoding="utf-8").splitlines() == [
        "sglang_mode=none",
        "sglang_features=",
        "trtllm_mode=none",
        "trtllm_features=",
        "vllm_mode=full",
        "vllm_features=",
    ]


def test_repository_selective_backend_jobs_keep_default_marker_dimensions() -> None:
    workflow = (REPO_ROOT / ".github/workflows/pr.yaml").read_text(encoding="utf-8")
    selective_marker_inputs = [
        line.strip()
        for line in workflow.splitlines()
        if "test_markers:" in line and "_features" in line
    ]

    assert len(selective_marker_inputs) == 9
    assert all("and (unit or ({0}))" in line for line in selective_marker_inputs)
    assert all("gpu_" in line for line in selective_marker_inputs)
    assert all(
        "pre_merge" in line or "post_merge" in line for line in selective_marker_inputs
    )
    assert "pytest_selection_mode" not in workflow
    assert "pytest_selection_areas" not in workflow
    assert "pytest_generic_" not in workflow
    assert "pytest_vllm_mode" in workflow
    assert "pytest_sglang_mode" in workflow
    assert "pytest_trtllm_mode" in workflow
    assert workflow.count("_mode == 'markers'") == 9
    assert workflow.count("_mode == 'none'") == 9
    assert workflow.count("and unit' ||") == 9
    assert (
        "continue-on-error: ${{ vars.PYTEST_SELECTION_MODE != 'selective' }}"
        in workflow
    )
    assert "id: pytest-selector-dependency" in workflow
    assert "id: pytest-shadow-dependencies" in workflow
    assert "id: pytest-python-setup" in workflow
    assert "steps.pytest-python-setup.outcome == 'failure'" in workflow
    assert "steps.pytest-shadow-dependencies.outcome == 'failure'" in workflow
    assert (
        "pytest-shadow-selection-${{ github.run_id }}-${{ github.run_attempt }}"
        in workflow
    )
    assert "Pytest shadow selector failed open" in workflow


def test_unit_framework_test_does_not_require_selective_feature_marker() -> None:
    markers = {"pre_merge", "unit", "gpu_0", "vllm"}

    assert "Selective Feature" not in missing_categories(markers)


def test_non_unit_framework_test_requires_a_selective_feature_marker() -> None:
    markers = {"pre_merge", "e2e", "gpu_1", "vllm"}

    assert "Selective Feature" in missing_categories(markers)


def test_non_unit_framework_test_can_carry_multiple_selective_features() -> None:
    markers = {"pre_merge", "e2e", "gpu_1", "vllm", "router", "multimodal"}

    assert "Selective Feature" not in missing_categories(markers)


def test_marker_inventory_keeps_skip_items_and_audits_skipif() -> None:
    class Item:
        def __init__(self, nodeid: str, markers: list[str]):
            self.nodeid = nodeid
            self._markers = markers

        def iter_markers(self):
            return [SimpleNamespace(name=marker) for marker in self._markers]

    plugin = MarkerReportPlugin()
    plugin.pytest_collection_modifyitems(
        None,
        None,
        [
            Item("tests/test_run.py::test_run", ["pre_merge", "unit", "gpu_0"]),
            Item("tests/test_skip.py::test_skip", ["pre_merge", "unit", "skip"]),
            Item(
                "tests/test_skipif.py::test_skipif",
                ["pre_merge", "e2e", "gpu_0", "vllm", "skipif"],
            ),
        ],
    )

    report = plugin.build_report()
    assert [test.nodeid for test in report.collected_tests] == [
        "tests/test_run.py::test_run",
        "tests/test_skip.py::test_skip",
        "tests/test_skipif.py::test_skipif",
    ]
    assert report.total_skipped_mypy == 1
    assert report.tests[-1].missing == ["Selective Feature"]


def test_marker_audit_rejects_configured_feature_without_tests() -> None:
    core_test = MarkerTestRecord("test_core", ["core"], [])
    report = Report(
        total_checked=1,
        total_skipped_mypy=0,
        total_missing=0,
        tests=[core_test],
        collected_tests=[core_test],
    )

    validate_selective_marker_coverage(report, {"core", "router"})

    assert report.unused_selective_markers == ["router"]


def test_path_audit_requires_a_mapped_feature_marker() -> None:
    aligned = MarkerTestRecord(
        "tests/router/test_aligned.py::test_router",
        ["pre_merge", "e2e", "gpu_0", "vllm", "router"],
        [],
    )
    mismatched = MarkerTestRecord(
        "tests/router/test_mismatched.py::test_router",
        ["pre_merge", "e2e", "gpu_0", "vllm", "core"],
        [],
    )
    report = Report(
        total_checked=2,
        total_skipped_mypy=0,
        total_missing=0,
        tests=[aligned, mismatched],
        collected_tests=[aligned, mismatched],
    )

    validate_path_feature_alignment(report, _model())

    assert report.path_marker_mismatches is not None
    assert len(report.path_marker_mismatches) == 1
    assert report.path_marker_mismatches[0].nodeid == mismatched.nodeid
    assert report.path_marker_mismatches[0].mapped_features == ["router"]
    assert report.path_marker_mismatches[0].actual_features == ["core"]


def test_path_audit_requires_a_mapped_framework_marker() -> None:
    mismatched = MarkerTestRecord(
        "components/vllm/test_new.py::test_backend",
        ["pre_merge", "e2e", "gpu_0", "sglang", "core"],
        [],
    )
    report = Report(
        total_checked=1,
        total_skipped_mypy=0,
        total_missing=0,
        tests=[mismatched],
        collected_tests=[mismatched],
    )

    validate_path_feature_alignment(report, _model())

    assert report.path_marker_mismatches is not None
    assert report.path_marker_mismatches[0].mapped_frameworks == ["vllm"]
    assert report.path_marker_mismatches[0].actual_frameworks == ["sglang"]


def test_path_audit_rejects_a_non_unit_framework_test_without_mapping() -> None:
    unmapped = MarkerTestRecord(
        "tests/unmapped/test_new.py::test_backend",
        ["pre_merge", "e2e", "gpu_0", "vllm", "core"],
        [],
    )
    report = Report(
        total_checked=1,
        total_skipped_mypy=0,
        total_missing=0,
        tests=[unmapped],
        collected_tests=[unmapped],
    )

    validate_path_feature_alignment(report, _model())

    assert report.path_marker_mismatches is not None
    assert report.path_marker_mismatches[0].mapped_features == []
    assert report.path_marker_mismatches[0].mapped_frameworks == []


def test_marker_audit_collects_tool_calling_tests_deterministically() -> None:
    source = (REPO_ROOT / "tests/frontend/test_tool_calling_sglang.py").read_text(
        encoding="utf-8"
    )

    assert "jsonschema" in STUB_MODULES
    assert "pytest.mark.pre_merge" in source


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
    ]

    selected = selected_pr_tests(plan, records)

    assert selected["vllm-cpu"] == [
        "components/src/vllm/tests/test_smoke.py::test_unit",
        "tests/serve/test_vllm.py::test_cpu",
    ]
    assert selected["vllm-gpu-1"] == ["tests/serve/test_vllm.py::test_gpu"]
    assert selected["vllm-multi-gpu"] == ["tests/serve/test_vllm.py::test_multi_gpu"]
