# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for .github/scripts/bump_version.py.

Covers the engine (Version type, rule table, apply_rules) and the CLI
entry points (--check, --dry-run, full bump, post-release minimal bump).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "bump_version.py"

# Pure unit tests: no GPU, no services, run on every PR.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


@pytest.fixture(scope="module")
def bv():
    """Load bump_version.py as a module (it lives outside the python package tree).

    Must register the module in sys.modules *before* exec so that the
    dataclass decorator can resolve forward-referenced annotations like
    ``int | None`` that the runtime looks up via ``sys.modules[cls.__module__]``.
    """
    spec = importlib.util.spec_from_file_location("bump_version", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bump_version"] = mod
    loaded = False
    try:
        spec.loader.exec_module(mod)
        loaded = True
    finally:
        if not loaded:
            sys.modules.pop("bump_version", None)
    return mod


# ---------------------------------------------------------------------------
# Version type
# ---------------------------------------------------------------------------


class TestVersion:
    def test_parse_plain(self, bv):
        v = bv.Version.parse("1.2.3")
        assert (v.major, v.minor, v.patch, v.post) == (1, 2, 3, None)
        assert v.python() == "1.2.3"
        assert v.semver() == "1.2.3"
        assert v.dashed() == "1-2-3"
        assert not v.is_post

    def test_parse_post_dot(self, bv):
        v = bv.Version.parse("0.9.0.post2")
        assert v.post == 2
        assert v.python() == "0.9.0.post2"
        assert v.semver() == "0.9.0-post2"
        assert v.dashed() == "0-9-0-post2"
        assert v.is_post

    def test_parse_post_dash(self, bv):
        v = bv.Version.parse("0.9.0-post2")
        assert v == bv.Version.parse("0.9.0.post2")

    @pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "v1.2.3", "1.2.3-rc1", "abc"])
    def test_parse_rejects_invalid(self, bv, bad):
        with pytest.raises(argparse.ArgumentTypeError):
            bv.Version.parse(bad)

    def test_ordering(self, bv):
        # PEP 440 ordering: base < post-release; minor bumps order numerically.
        assert bv.Version.parse("1.0.0") < bv.Version.parse("1.0.1")
        assert bv.Version.parse("0.9.0") < bv.Version.parse("0.10.0")
        assert bv.Version.parse("0.9.0") < bv.Version.parse("0.9.0.post1")
        assert bv.Version.parse("0.9.0.post1") < bv.Version.parse("0.9.0.post2")
        assert bv.Version.parse("0.9.0.post2") > bv.Version.parse("0.9.0")
        # Sorting a mixed list must not raise (regression guard for order=True bug).
        versions = [
            bv.Version.parse("0.9.0.post1"),
            bv.Version.parse("0.9.0"),
            bv.Version.parse("1.0.0"),
        ]
        assert sorted(versions) == [
            bv.Version.parse("0.9.0"),
            bv.Version.parse("0.9.0.post1"),
            bv.Version.parse("1.0.0"),
        ]


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


class TestRules:
    def _rule(self, bv, name):
        for r in bv.RULES:
            if r.name == name:
                return r
        raise AssertionError(f"rule {name!r} not found")

    def test_pyproject_project_version(self, bv, tmp_path):
        path = tmp_path / "pyproject.toml"
        path.write_text(
            '[project]\nname = "ai-dynamo"\nversion = "0.9.0"\n',
            encoding="utf-8",
        )
        rule = self._rule(bv, "pyproject_project_version")
        out = rule.apply(path.read_text(encoding="utf-8"), bv.Version.parse("1.0.0"))
        assert 'version = "1.0.0"' in out
        assert "0.9.0" not in out

    def test_pyproject_ai_dynamo_pin_all_extras(self, bv):
        rule = self._rule(bv, "pyproject_ai_dynamo_pin")
        txt = (
            '"ai-dynamo-runtime==0.9.0",\n'
            '"ai_dynamo_runtime==0.9.0",\n'
            '"ai-dynamo[vllm]==0.9.0",\n'
            '"ai-dynamo[sglang,trtllm]==0.9.0",\n'
        )
        out = rule.apply(txt, bv.Version.parse("1.0.0"))
        assert "0.9.0" not in out
        assert out.count("1.0.0") == 4

    def test_cargo_uses_semver(self, bv):
        rule = self._rule(bv, "cargo_package_version")
        out = rule.apply(
            '[package]\nname = "x"\nversion = "0.9.0"\n',
            bv.Version.parse("0.9.0.post1"),
        )
        assert '"0.9.0-post1"' in out
        assert "0.9.0.post1" not in out

    def test_helm_chart_version(self, bv):
        rule = self._rule(bv, "helm_chart_version")
        out = rule.apply("apiVersion: v2\nversion: 0.9.0\n", bv.Version.parse("1.0.0"))
        assert "version: 1.0.0" in out

    def test_image_tag_ai_dynamo_ns_broad(self, bv):
        rule = self._rule(bv, "image_tag_ai_dynamo_ns")
        src = (
            "nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.9.0\n"
            "ai-dynamo/dynamo-frontend:0.9.0\n"
            "ai-dynamo/kubernetes-operator:0.9.0.post1\n"
        )
        out = rule.apply(src, bv.Version.parse("1.0.0"))
        assert out.count("1.0.0") == 3
        assert "0.9.0" not in out

    def test_image_tag_short_allowlist(self, bv):
        rule = self._rule(bv, "image_tag_short_dynamo")
        src = "image: vllm-runtime:0.9.0\nimage: dynamo-frontend:0.9.0\n"
        out = rule.apply(src, bv.Version.parse("1.0.0"))
        assert "vllm-runtime:1.0.0" in out
        assert "dynamo-frontend:1.0.0" in out

    def test_short_tag_does_not_rewrite_unrelated(self, bv):
        rule = self._rule(bv, "image_tag_short_dynamo")
        # "postgres:0.9.0" must NOT be rewritten
        out = rule.apply("postgres:0.9.0\n", bv.Version.parse("1.0.0"))
        assert "postgres:0.9.0" in out

    def test_operator_dynamo_version_field(self, bv):
        rule = self._rule(bv, "operator_dynamoVersion_field")
        out = rule.apply('dynamoVersion: "0.9.0"\n', bv.Version.parse("1.0.0"))
        assert 'dynamoVersion: "1.0.0"' in out

    def test_git_refs(self, bv):
        co = self._rule(bv, "git_checkout_release_branch")
        url = self._rule(bv, "git_url_release_ref")
        out = co.apply("git checkout release/0.9.0\n", bv.Version.parse("1.0.0"))
        assert "release/1.0.0" in out
        out2 = url.apply(
            "pip install git+https://x@release/0.9.0\n", bv.Version.parse("1.0.0")
        )
        assert "release/1.0.0" in out2

    def test_env_dynamo_version(self, bv):
        rule = self._rule(bv, "env_dynamo_version")
        out = rule.apply("DYNAMO_VERSION=0.9.0\n", bv.Version.parse("1.0.0"))
        assert "DYNAMO_VERSION=1.0.0" in out


# ---------------------------------------------------------------------------
# File iteration: exclusions + opt-out marker
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestIteration:
    def test_ignore_marker_skips_file(self, bv, tmp_path):
        _write(
            tmp_path / "keep_old.py",
            '# bump-version: ignore\nVERSION = "nvcr.io/nvidia/ai-dynamo/foo:0.9.0"\n',
        )
        # Only the DOCS table-row rules touch this file; no ai-dynamo image rule should fire.
        changes = bv.apply_rules(
            tmp_path,
            bv.Version.parse("1.0.0"),
            active_scopes={
                bv.Scope.CORE,
                bv.Scope.CONTAINERS,
                bv.Scope.HELM,
                bv.Scope.DOCS,
            },
            dry_run=True,
        )
        assert changes == []
        # And the file was not rewritten.
        assert "0.9.0" in (tmp_path / "keep_old.py").read_text(encoding="utf-8")

    def test_excluded_globs(self, bv, tmp_path):
        _write(tmp_path / "foo.lock", 'tag = "nvcr.io/nvidia/ai-dynamo/x:0.9.0"\n')
        changes = bv.apply_rules(
            tmp_path,
            bv.Version.parse("1.0.0"),
            active_scopes={bv.Scope.CONTAINERS},
            dry_run=True,
        )
        assert changes == []

    def test_binary_files_skipped(self, bv, tmp_path):
        # Extensionless name so the test isolates _is_binary from EXCLUDE_GLOBS
        # (which would skip *.bin via the binary-extension list anyway).
        (tmp_path / "blob").write_bytes(
            b"\x00\x01\x02nvcr.io/nvidia/ai-dynamo/x:0.9.0"
        )
        changes = bv.apply_rules(
            tmp_path,
            bv.Version.parse("1.0.0"),
            active_scopes={bv.Scope.CONTAINERS},
            dry_run=True,
        )
        assert changes == []


# ---------------------------------------------------------------------------
# End-to-end CLI scenarios
# ---------------------------------------------------------------------------


def _make_fake_repo(root: Path, version: str) -> None:
    """Seed a miniature repo with every scope represented."""
    _write(
        root / "pyproject.toml",
        f'[project]\nname = "ai-dynamo"\nversion = "{version}"\n'
        f'dependencies = ["ai-dynamo-runtime=={version}", "ai-dynamo[vllm]=={version}"]\n',
    )
    _write(
        root / "lib" / "runtime" / "Cargo.toml",
        f'[package]\nname = "dynamo-runtime"\nversion = "{version}"\n',
    )
    _write(
        root / "deploy" / "helm" / "charts" / "platform" / "Chart.yaml",
        f"apiVersion: v2\nname: dynamo-platform\nversion: {version}\n"
        f'appVersion: "{version}"\n'
        f"dependencies:\n- name: dynamo-operator\n  version: {version}\n",
    )
    _write(
        root / "deploy" / "helm" / "charts" / "snapshot" / "values.yaml",
        "image:\n"
        "  repository: nvcr.io/nvidia/ai-dynamo/snapshot-agent\n"
        f"  tag: {version}\n",
    )
    # Spec file: container image tag, git ref, pip pin, and env var. Lives in
    # a path that _is_spec_file() recognises (Dockerfile.* prefix), so the
    # broad CONTAINERS rules fire here. (Markdown prose is intentionally out
    # of scope after the _is_spec_file filter — see Patch 5.)
    _write(
        root / "container" / "Dockerfile.example",
        f"FROM nvcr.io/nvidia/ai-dynamo/vllm-runtime:{version}\n"
        f"RUN git checkout release/{version} \\\n"
        f" && pip install ai-dynamo=={version}\n"
        f"ENV DYNAMO_VERSION={version}\n",
    )
    _write(
        root / "deploy" / "operator" / "samples" / "dgd.yaml",
        f"apiVersion: nvidia.com/v1alpha1\nkind: DynamoGraphDeployment\nspec:\n"
        f'  dynamoVersion: "{version}"\n',
    )


def test_dry_run_does_not_write(bv, tmp_path):
    _make_fake_repo(tmp_path, "0.9.0")
    rc = bv.main(
        [
            "--new-version",
            "1.0.0",
            "--repo-root",
            str(tmp_path),
            "--dry-run",
            "--skip-docs",
        ]
    )
    assert rc == 0
    # pyproject still says 0.9.0 because dry_run=True.
    assert '"0.9.0"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")


def test_full_bump_writes_every_scope(bv, tmp_path):
    _make_fake_repo(tmp_path, "0.9.0")
    rc = bv.main(
        ["--new-version", "1.0.0", "--repo-root", str(tmp_path), "--skip-docs"]
    )
    assert rc == 0
    assert 'version = "1.0.0"' in (tmp_path / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    # Cargo uses semver form; 1.0.0 has no post so it's the same.
    assert 'version = "1.0.0"' in (tmp_path / "lib/runtime/Cargo.toml").read_text(
        encoding="utf-8"
    )
    chart = (tmp_path / "deploy/helm/charts/platform/Chart.yaml").read_text(
        encoding="utf-8"
    )
    assert "version: 1.0.0" in chart
    assert 'appVersion: "1.0.0"' in chart
    snap = (tmp_path / "deploy/helm/charts/snapshot/values.yaml").read_text(
        encoding="utf-8"
    )
    assert "tag: 1.0.0" in snap
    dockerfile = (tmp_path / "container/Dockerfile.example").read_text(encoding="utf-8")
    assert "vllm-runtime:1.0.0" in dockerfile
    assert "release/1.0.0" in dockerfile
    assert "ai-dynamo==1.0.0" in dockerfile
    assert "DYNAMO_VERSION=1.0.0" in dockerfile
    dgd = (tmp_path / "deploy/operator/samples/dgd.yaml").read_text(encoding="utf-8")
    assert 'dynamoVersion: "1.0.0"' in dgd
    assert "0.9.0" not in (chart + snap + dockerfile + dgd)


def test_post_release_uses_semver_in_helm_python_elsewhere(bv, tmp_path):
    _make_fake_repo(tmp_path, "0.9.0")
    rc = bv.main(
        ["--new-version", "0.9.0.post1", "--repo-root", str(tmp_path), "--skip-docs"]
    )
    assert rc == 0
    # Python scope: dotted post
    assert '"0.9.0.post1"' in (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    # Cargo: dashed post (semver)
    assert '"0.9.0-post1"' in (tmp_path / "lib/runtime/Cargo.toml").read_text(
        encoding="utf-8"
    )
    # Helm Chart.yaml: dashed post
    chart = (tmp_path / "deploy/helm/charts/platform/Chart.yaml").read_text(
        encoding="utf-8"
    )
    assert "version: 0.9.0-post1" in chart
    assert 'appVersion: "0.9.0-post1"' in chart
    # Image tag: dotted post
    dockerfile = (tmp_path / "container/Dockerfile.example").read_text(encoding="utf-8")
    assert "vllm-runtime:0.9.0.post1" in dockerfile


def test_skip_flags_combine(bv, tmp_path):
    _make_fake_repo(tmp_path, "0.9.0")
    rc = bv.main(
        [
            "--new-version",
            "1.0.0",
            "--repo-root",
            str(tmp_path),
            "--skip-core",
            "--skip-helm",
            "--skip-docs",
        ]
    )
    assert rc == 0
    # core untouched: both the [project].version line AND the ai-dynamo self-pin
    # must remain on the old version. Asserting only the version-line would miss
    # a regression where the broad CONTAINERS pip pin rule rewrites the pin.
    py = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.9.0"' in py
    assert "ai-dynamo-runtime==0.9.0" in py
    assert "ai-dynamo[vllm]==0.9.0" in py
    assert "1.0.0" not in py
    # helm untouched
    assert "version: 0.9.0" in (
        tmp_path / "deploy/helm/charts/platform/Chart.yaml"
    ).read_text(encoding="utf-8")
    # containers (outside pyproject.toml) were bumped in spec files
    assert "vllm-runtime:1.0.0" in (tmp_path / "container/Dockerfile.example").read_text(
        encoding="utf-8"
    )


def test_check_mode_stale_exits_nonzero(bv, tmp_path):
    # repo is on 0.9.0 but we "expect" it to be on 1.0.0 already => everything is stale
    _make_fake_repo(tmp_path, "0.9.0")
    rc = bv.main(
        [
            "--check",
            "--expected-version",
            "1.0.0",
            "--repo-root",
            str(tmp_path),
            "--skip-docs",
        ]
    )
    assert rc == 1


def test_check_mode_fresh_exits_zero(bv, tmp_path):
    # A repo already on 1.0.0 with expected-version 1.0.0 has nothing stale.
    _make_fake_repo(tmp_path, "1.0.0")
    rc = bv.main(
        [
            "--check",
            "--expected-version",
            "1.0.0",
            "--repo-root",
            str(tmp_path),
            "--skip-docs",
        ]
    )
    assert rc == 0


def test_check_autodetects_from_pyproject(bv, tmp_path):
    _make_fake_repo(tmp_path, "0.9.0")
    rc = bv.main(["--check", "--repo-root", str(tmp_path), "--skip-docs"])
    assert rc == 0  # everything IS at 0.9.0 (the detected current version)


def test_check_mode_detects_stale_docs(bv, tmp_path):
    """--check must catch staleness in DOCS rules (now in the unified RULES table).

    After Axis B, the support-matrix / release-artifacts / feature-matrix
    files no longer carry IGNORE_MARKER -- the bespoke specialised helpers
    are gone, replaced by narrowly-targeted rules that only match the exact
    version-bearing tokens (header tag, "At a Glance" line, etc) and leave
    historical table rows untouched. So apply_rules(active_scopes={DOCS})
    sees these files, and --check can detect a stale "*Updated for Dynamo
    vX.Y.Z*" tag through the same dry-run pass.
    """
    _make_fake_repo(tmp_path, "1.0.0")
    # Plant a stale DOCS-only reference: feature-matrix tag still on the old
    # version. The narrow `feature_matrix_tag` rule must catch this.
    fm = tmp_path / "docs" / "reference" / "feature-matrix.md"
    fm.parent.mkdir(parents=True, exist_ok=True)
    fm.write_text(
        "*Updated for Dynamo v0.9.0*\n\nSome historical content with v0.5.0 mentions.\n",
        encoding="utf-8",
    )
    # Sanity check: with --skip-docs the stale tag is invisible to --check.
    assert (
        bv.main(
            [
                "--check",
                "--expected-version",
                "1.0.0",
                "--repo-root",
                str(tmp_path),
                "--skip-docs",
            ]
        )
        == 0
    )
    # Now without --skip-docs it must be flagged.
    assert (
        bv.main(
            [
                "--check",
                "--expected-version",
                "1.0.0",
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_summary_file_is_written(bv, tmp_path):
    _make_fake_repo(tmp_path, "0.9.0")
    summary = tmp_path / "summary.md"
    rc = bv.main(
        [
            "--new-version",
            "1.0.0",
            "--repo-root",
            str(tmp_path),
            "--dry-run",
            "--skip-docs",
            "--summary-file",
            str(summary),
        ]
    )
    assert rc == 0
    body = summary.read_text(encoding="utf-8")
    assert "Version bump" in body
    assert "1.0.0" in body


def test_no_change_when_old_equals_new(bv, tmp_path):
    _make_fake_repo(tmp_path, "1.0.0")
    rc = bv.main(
        [
            "--new-version",
            "1.0.0",
            "--repo-root",
            str(tmp_path),
            "--old-version",
            "1.0.0",
            "--skip-docs",
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# DOCS specialised updates
# ---------------------------------------------------------------------------


def _bump_docs(bv, repo, new_version: str, **ctx_overrides):
    """Apply DOCS rules + table insertions in one pass — the contract used by
    --bump and --check for docs after the Axis B refactor.

    Returns the combined Change list so tests can assert which rules fired.
    """
    ctx = {
        "release_date": "Feb 15, 2026",
        "vllm": "0.19.0",
        "sglang": "0.5.7",
        "trtllm": "1.3.0",
        "nixl": "0.10.1",
        **ctx_overrides,
    }
    v = bv.Version.parse(new_version)
    return (
        bv.apply_rules(repo, v, active_scopes={bv.Scope.DOCS}, ctx=ctx, dry_run=False)
        + bv.insert_table_rows(repo, v, ctx, dry_run=False)
    )


class TestReleaseArtifacts:
    """release-artifacts.md edits go through DOCS rules + insert_table_rows.

    The bespoke update_release_artifacts() helper is gone — a missing table
    insertion point now raises from insert_table_rows().
    """

    def test_unknown_table_raises(self, bv, tmp_path):
        p = tmp_path / "docs" / "reference" / "release-artifacts.md"
        _write(p, "## Current Release: Dynamo v0.9.0\n\nNo table here.\n")
        with pytest.raises(RuntimeError, match="release_artifacts_table_row"):
            _bump_docs(bv, tmp_path, "1.0.0")

    def test_table_row_inserted(self, bv, tmp_path):
        doc = (
            "## Current Release: Dynamo v0.9.0\n\n"
            "**GitHub Release:** [v0.9.0](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0)\n\n"
            "**Docs:** [v0.9.0](https://docs.dynamo.nvidia.com/dynamo)\n\n"
            "### GitHub Releases\n\n"
            "| Version | Release Date | GitHub | Docs | Notes |\n"
            "|---------|------|---------|------|------|\n"
            "| `v0.9.0` | Dec 01, 2025 | [Release](x) | [Docs](y) | |\n"
        )
        p = tmp_path / "docs" / "reference" / "release-artifacts.md"
        _write(p, doc)
        _bump_docs(bv, tmp_path, "1.0.0")
        out = p.read_text(encoding="utf-8")
        # Header rule rewrote the "Current Release" line.
        assert "## Current Release: Dynamo v1.0.0" in out
        # GitHub-link and Docs-link rules rewrote the per-release links.
        assert "[v1.0.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.0)" in out
        assert "[v1.0.0](https://docs.dynamo.nvidia.com/dynamo)" in out
        # Table-insertion appended the new row with all 5 columns.
        assert "| `v1.0.0` | Feb 15, 2026 " in out
        assert "[Release](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.0)" in out
        # Historical row is left intact (no broad rewrite).
        assert "| `v0.9.0` | Dec 01, 2025 " in out

    def test_idempotent(self, bv, tmp_path):
        """Running the docs update twice must not insert the row twice."""
        doc = (
            "## Current Release: Dynamo v0.9.0\n\n"
            "**GitHub Release:** [v0.9.0](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0)\n\n"
            "**Docs:** [v0.9.0](https://docs.dynamo.nvidia.com/dynamo)\n\n"
            "### GitHub Releases\n\n"
            "| Version | Release Date | GitHub | Docs | Notes |\n"
            "|---------|------|---------|------|------|\n"
            "| `v0.9.0` | Dec 01, 2025 | [Release](x) | [Docs](y) | |\n"
        )
        p = tmp_path / "docs" / "reference" / "release-artifacts.md"
        _write(p, doc)
        _bump_docs(bv, tmp_path, "1.0.0")
        first = p.read_text(encoding="utf-8")
        _bump_docs(bv, tmp_path, "1.0.0")
        assert p.read_text(encoding="utf-8") == first


class TestSupportMatrix:
    def test_at_a_glance_and_row(self, bv, tmp_path):
        doc = (
            "**Latest stable release:** [v0.9.0](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0) -- "
            "SGLang `v0.5.0` | TensorRT-LLM `v1.0.0` | vLLM `v0.18.0` | NIXL `v0.10.0`\n\n"
            "| Version | SGLang | TRT-LLM | vLLM | NIXL |\n"
            "|---------|--------|---------|------|------|\n"
            "| **main (ToT)** | `latest` | `latest` | `latest` | `latest` |\n"
            "| **v0.9.0** | `0.5.0` | `1.0.0` | `0.18.0` | `0.10.0` |\n"
        )
        p = tmp_path / "docs" / "reference" / "support-matrix.md"
        _write(p, doc)
        _bump_docs(bv, tmp_path, "1.0.0")
        out = p.read_text(encoding="utf-8")
        assert "[v1.0.0](https://github.com/ai-dynamo/dynamo/releases/tag/v1.0.0)" in out
        assert "SGLang `0.5.7`" in out
        assert "TensorRT-LLM `1.3.0`" in out
        # New backend row inserted right after main(ToT) row.
        assert "| **v1.0.0** | `0.5.7` | `1.3.0` | `0.19.0` | `0.10.1` |" in out
        # Historical row preserved.
        assert "| **v0.9.0** | `0.5.0` | `1.0.0` | `0.18.0` | `0.10.0` |" in out


class TestFeatureMatrix:
    def test_updated_for_tag(self, bv, tmp_path):
        p = tmp_path / "docs" / "reference" / "feature-matrix.md"
        _write(
            p,
            "*Updated for Dynamo v0.9.0*\n\n"
            "Some content that mentions historical version v0.5.0 in prose.\n",
        )
        _bump_docs(bv, tmp_path, "1.0.0")
        out = p.read_text(encoding="utf-8")
        assert "*Updated for Dynamo v1.0.0*" in out
        # Historical version mention in prose must NOT be rewritten.
        assert "historical version v0.5.0" in out


# ---------------------------------------------------------------------------
# Tier-1 regression guards (one test per fragility patch)
# ---------------------------------------------------------------------------


class TestRegressionGuards:
    def test_release_artifacts_docs_url_uses_canonical_domain(self, bv, tmp_path):
        # FIX 1 guard: the docs link in release-artifacts.md must use the
        # canonical host ``docs.dynamo.nvidia.com/dynamo`` and the rule's
        # regex must match that exact host — not a sibling like
        # ``docs.nvidia.com/dynamo`` which a previous iteration emitted.
        doc = (
            "## Current Release: Dynamo v0.9.0\n\n"
            "**GitHub Release:** [v0.9.0](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0)\n\n"
            "**Docs:** [v0.9.0](https://docs.dynamo.nvidia.com/dynamo)\n\n"
            "### GitHub Releases\n\n"
            "| Version | Release Date | GitHub | Docs | Notes |\n"
            "|---------|------|---------|------|------|\n"
            "| `v0.9.0` | Dec 01, 2025 | [Release](x) | [Docs](y) | |\n"
        )
        p = tmp_path / "docs" / "reference" / "release-artifacts.md"
        _write(p, doc)
        _bump_docs(bv, tmp_path, "1.0.0")
        out = p.read_text(encoding="utf-8")
        # Canonical host preserved, version rewritten.
        assert "[v1.0.0](https://docs.dynamo.nvidia.com/dynamo)" in out
        # The inserted table row's Docs column must also use the canonical host.
        assert "[Docs](https://docs.dynamo.nvidia.com/dynamo)" in out
        # Non-canonical hosts must never appear.
        assert "docs.nvidia.com/dynamo" not in out

    def test_ver_does_not_strip_rc_suffix(self, bv):
        # FIX 2 guard: ``_VER`` must refuse to match the numeric portion of
        # strings that continue with additional release qualifiers — otherwise
        # rewriting the capture leaves orphaned suffixes like ``1.0.0-rc1``
        # after a bump that had nothing to say about the rc track.
        import re as _re

        pattern = _re.compile(bv._VER)
        # Word-char / dot / dash suffixes must block the match entirely.
        assert pattern.search("0.9.0-rc1") is None
        assert pattern.search("0.9.0rc1") is None
        assert pattern.search("0.9.0.dev2") is None
        assert pattern.search("0.9.0-alpha") is None
        # Plain versions and .postN / -postN still match as before.
        assert pattern.fullmatch("0.9.0") is not None
        assert pattern.fullmatch("0.9.0.post1") is not None
        assert pattern.fullmatch("0.9.0-post2") is not None

    def test_change_record_tracks_every_rule_that_fired(self, bv, tmp_path):
        # FIX 3 guard: when multiple rules rewrite the same file in one pass,
        # the resulting ``Change`` record must list every rule name — not just
        # the first or last. The summary table depends on this to attribute
        # rewrites back to rules.
        doc = (
            "## Current Release: Dynamo v0.9.0\n\n"
            "**GitHub Release:** [v0.9.0](https://github.com/ai-dynamo/dynamo/releases/tag/v0.9.0)\n\n"
            "**Docs:** [v0.9.0](https://docs.dynamo.nvidia.com/dynamo)\n\n"
        )
        p = tmp_path / "docs" / "reference" / "release-artifacts.md"
        _write(p, doc)
        changes = bv.apply_rules(
            tmp_path,
            bv.Version.parse("1.0.0"),
            active_scopes={bv.Scope.DOCS},
            dry_run=True,
        )
        # Exactly one Change record for the one rewritten file.
        matching = [c for c in changes if c.path.name == "release-artifacts.md"]
        assert len(matching) == 1
        fired = set(matching[0].rules)
        # All three DOCS rules that could fire on this content must be tracked.
        expected = {
            "release_artifacts_header",
            "release_artifacts_github_link",
            "release_artifacts_docs_link",
        }
        assert expected.issubset(fired), (
            f"missing rules in change record: {expected - fired}"
        )
