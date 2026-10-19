# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import select_tests  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
POLICY = select_tests.load_policy()


def _registered_markers() -> set[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return {
        marker.split(":", 1)[0].split("(", 1)[0]
        for marker in config["tool"]["pytest"]["ini_options"]["markers"]
    }


def _tracked_paths() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return {path.decode() for path in result.stdout.split(b"\0") if path}


def _selection(paths: list[str], **kwargs) -> dict:
    return select_tests.select_changed_files(
        paths,
        POLICY,
        event="pull_request",
        pr=42,
        labels=[],
        **kwargs,
    )


def test_feature_path_scopes_to_matching_framework():
    result = _selection(["components/src/dynamo/vllm/multimodal_utils/cache_config.py"])
    assert result["paths"] == {
        "components/src/dynamo/vllm/multimodal_utils/cache_config.py": "feature:multimodal"
    }
    assert result["feature_markers"] == ["multimodal"]
    assert result["frameworks"]["vllm"]["markers"] == ["multimodal"]
    assert result["frameworks"]["vllm"]["decision"] == "selective"
    for framework in ("sglang", "trtllm"):
        assert result["frameworks"][framework]["markers"] == []
        assert result["frameworks"][framework]["decision"] == "selective"


def test_router_feature_reuses_codeowners_area_globs():
    result = _selection(["lib/kv-router/src/indexer/mod.rs"])
    assert result["paths"]["lib/kv-router/src/indexer/mod.rs"] == "feature:router"
    assert result["feature_markers"] == ["router"]
    assert all(item["markers"] == ["router"] for item in result["frameworks"].values())


def test_kvbm_feature_path_is_selected():
    result = _selection(["lib/kvbm-common/src/lib.rs"])
    assert result["paths"]["lib/kvbm-common/src/lib.rs"] == "feature:kvbm"
    assert result["feature_markers"] == ["kvbm"]
    assert all(item["markers"] == ["kvbm"] for item in result["frameworks"].values())


@pytest.mark.parametrize(
    "path",
    ["lib/runtime/src/lib.rs", "container/render.py"],
)
def test_runtime_and_container_changes_select_full(path):
    result = _selection([path])
    assert result["paths"][path] == "shared_full"
    assert all(item["decision"] == "full" for item in result["frameworks"].values())


def test_framework_only_change_selects_that_framework_full():
    result = _selection(["components/src/dynamo/vllm/args.py"])
    assert result["paths"]["components/src/dynamo/vllm/args.py"] == "framework:vllm"
    assert result["frameworks"]["vllm"]["decision"] == "full"
    assert result["frameworks"]["sglang"]["decision"] == "selective"
    assert result["frameworks"]["trtllm"]["decision"] == "selective"


def test_docs_are_smoke_only():
    path = "docs/fern/pages/kubernetes/kv-cache-offloading/lmcache.mdx"
    result = _selection([path])
    assert result["paths"][path] == "smoke_only"
    assert all(
        item["decision"] == "selective" for item in result["frameworks"].values()
    )
    assert result["feature_markers"] == []


def test_changed_test_file_is_shared_by_all_frameworks():
    path = "tests/serve/test_vllm.py"
    result = _selection([path])
    assert result["paths"][path] == "test_file"
    for item in result["frameworks"].values():
        assert item["changed_test_files"] == [path]


def test_tests_conftest_is_shared_full():
    result = _selection(["tests/conftest.py"])
    assert result["paths"]["tests/conftest.py"] == "shared_full"
    assert all(item["decision"] == "full" for item in result["frameworks"].values())


def test_unmapped_path_selects_full():
    path = "deploy/helm/charts/platform/Chart.yaml"
    result = _selection([path])
    assert result["paths"][path] == "unmapped"
    assert all(item["decision"] == "full" for item in result["frameworks"].values())


@pytest.mark.parametrize(
    ("paths", "event", "pr", "reason"),
    [
        ([], "pull_request", 42, "empty-changed-files"),
        (["docs/README.md"], "push", None, "non-pr-event"),
    ],
)
def test_empty_and_non_pr_inputs_select_full(paths, event, pr, reason):
    result = select_tests.select_changed_files(
        paths,
        POLICY,
        event=event,
        pr=pr,
        labels=[],
    )
    assert all(item["decision"] == "full" for item in result["frameworks"].values())
    assert all(reason in item["reasons"] for item in result["frameworks"].values())


def test_ci_full_tests_label_selects_full():
    result = select_tests.select_changed_files(
        ["docs/README.md"],
        POLICY,
        event="pull_request",
        pr=42,
        labels=["ci:full-tests"],
    )
    assert all(item["decision"] == "full" for item in result["frameworks"].values())
    assert all(
        "ci:full-tests" in item["reasons"] for item in result["frameworks"].values()
    )


def test_copy_pr_push_ref_is_a_pull_request_event():
    result = select_tests.select_changed_files(
        ["docs/README.md"],
        POLICY,
        event="push",
        ref="refs/heads/pull-request/42",
        pr=42,
        pr_event=True,
        labels=[],
    )
    assert all(
        item["decision"] == "selective" for item in result["frameworks"].values()
    )
    assert all(
        "non-pr-event" not in item["reasons"] for item in result["frameworks"].values()
    )


def test_mixed_feature_and_shared_path_escalates_every_framework():
    result = _selection(
        [
            "components/src/dynamo/vllm/multimodal_utils/cache_config.py",
            "lib/runtime/src/lib.rs",
        ]
    )
    assert result["feature_markers"] == ["multimodal"]
    assert all(item["decision"] == "full" for item in result["frameworks"].values())
    assert all(
        any(
            reason.startswith("shared_full:lib/runtime/src/lib.rs")
            for reason in item["reasons"]
        )
        for item in result["frameworks"].values()
    )


def test_label_lookup_failure_fails_closed_and_exits_zero(
    tmp_path, monkeypatch, capsys
):
    changed = tmp_path / "changed.json"
    changed.write_text('["docs/README.md"]', encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
    monkeypatch.setenv("GITHUB_REPOSITORY", "ai-dynamo/dynamo")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setattr(
        select_tests,
        "lookup_labels",
        lambda repo, pr: (_ for _ in ()).throw(select_tests.LabelLookupError()),
    )

    assert select_tests.main(["--changed-files-file", str(changed)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert all(item["decision"] == "full" for item in result["frameworks"].values())
    assert all(
        "label-lookup-failed" in item["reasons"]
        for item in result["frameworks"].values()
    )


def test_selector_exception_emits_full_json_and_exits_zero(
    tmp_path, monkeypatch, capsys
):
    changed = tmp_path / "changed.json"
    changed.write_text('["docs/README.md"]', encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setattr(
        select_tests,
        "load_policy",
        lambda path: (_ for _ in ()).throw(RuntimeError("bad policy")),
    )

    assert select_tests.main(["--changed-files-file", str(changed)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["version"] == 1
    assert all(item["decision"] == "full" for item in result["frameworks"].values())
    assert all(
        item["reasons"] == ["selector-error: bad policy"]
        for item in result["frameworks"].values()
    )


def test_config_globs_match_tracked_files_and_markers_are_registered():
    select_tests.validate_selection_config(
        POLICY,
        _tracked_paths(),
        _registered_markers(),
    )


def test_unknown_feature_marker_fails_validation():
    policy = copy.deepcopy(POLICY)
    policy["test_selection"]["features"]["router"]["markers"] = ["not_registered"]
    with pytest.raises(ValueError, match="unregistered marker"):
        select_tests.validate_selection_config(
            policy, _tracked_paths(), _registered_markers()
        )


def test_feature_without_paths_or_areas_fails_validation():
    policy = copy.deepcopy(POLICY)
    policy["test_selection"]["features"]["router"]["paths"] = []
    policy["test_selection"]["features"]["router"]["areas"] = []
    with pytest.raises(ValueError, match="must define paths or areas"):
        select_tests.validate_selection_config(
            policy, _tracked_paths(), _registered_markers()
        )
