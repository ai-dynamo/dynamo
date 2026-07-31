# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the generated Rust API reference."""

from __future__ import annotations

import shutil
from pathlib import Path

import gen_rust_api
import pytest
import rust_api_discovery
import rust_api_rendering

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

REPO_ROOT = Path(__file__).resolve().parents[4]
FERN_ROOT = REPO_ROOT / "docs" / "fern"
RELEASES_DATA = FERN_ROOT / "components" / "releases.data.ts"

EXPECTED_CRATES = {
    "dynamo-async-openai",
    "dynamo-config",
    "dynamo-kv-router",
    "dynamo-llm",
    "dynamo-memory",
    "dynamo-mocker",
    "dynamo-parsers",
    "dynamo-protocols",
    "dynamo-runtime",
    "dynamo-tokenizers",
    "dynamo-tokens",
    "kvbm-logical",
}
CORE_CRATES = {
    "dynamo-kv-router",
    "dynamo-llm",
    "dynamo-memory",
    "dynamo-runtime",
    "kvbm-logical",
}
INTERNAL_CRATES = {"dynamo-rl", "dynamo-vllm-rs-backend", "kvbm-engine"}
# A workspace release does not republish every crate. dynamo-config tops out at
# 1.2.1 on crates.io, so pinning it to the workspace version emits a docs.rs URL
# that 404s and fails the link checker. Keep the release data on the newest
# version that actually shipped, and extend this map when another crate skips a
# release rather than defaulting it back to the workspace version.
LAGGING_CRATE_VERSIONS = {"dynamo-config": "1.2.1"}


@pytest.fixture(scope="session")
def reference() -> rust_api_discovery.RustReference:
    return rust_api_discovery.discover_rust_reference(REPO_ROOT, RELEASES_DATA)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    fern = tmp_path / "docs" / "fern"
    (fern / "components").mkdir(parents=True)
    (fern / "pages" / "reference" / "api" / "rust").mkdir(parents=True)
    return fern


@pytest.fixture()
def cached_reference(
    reference: rust_api_discovery.RustReference,
    monkeypatch: pytest.MonkeyPatch,
) -> rust_api_discovery.RustReference:
    monkeypatch.setattr(
        gen_rust_api,
        "discover_rust_reference",
        lambda: reference,
    )
    return reference


def test_discovery_matches_the_published_crate_inventory(
    reference: rust_api_discovery.RustReference,
) -> None:
    names = {crate.name for crate in reference.crates}
    assert names == EXPECTED_CRATES
    assert names.isdisjoint(INTERNAL_CRATES)


def test_workspace_version_matches_current_release(
    reference: rust_api_discovery.RustReference,
) -> None:
    """The published-crate inventory pins to ``reference.release_tag``.

    The workspace itself is allowed to sit ahead of that tag while the next
    development version bakes -- ``test_release_tag_may_lag_a_development_workspace``
    covers that direction -- so this test only exercises the published /
    lagging matrix, not workspace-tag equality.
    """
    assert reference.release_tag == "1.3.0"
    current = [
        crate
        for crate in reference.crates
        if crate.badge != "Deprecated" and crate.name not in LAGGING_CRATE_VERSIONS
    ]
    assert {crate.version for crate in current} == {reference.release_tag}
    by_name = {crate.name: crate for crate in reference.crates}
    for name, version in LAGGING_CRATE_VERSIONS.items():
        assert by_name[name].version == version
        assert by_name[name].docs_href == f"https://docs.rs/{name}/{version}"


def test_release_tag_may_lag_a_development_workspace() -> None:
    """main carries the next development version long before its crates ship,
    so the shipped release tag is allowed to sit behind the workspace."""
    rust_api_discovery.validate_release_tag({"CURRENT_TAG": "1.3.0"}, "1.4.0")
    rust_api_discovery.validate_release_tag({"CURRENT_TAG": "1.3.0"}, "1.3.0")


def test_release_tag_ahead_of_workspace_is_rejected() -> None:
    """Release data claiming a version the workspace has not reached would pin
    docs.rs links at crates that were never published."""
    with pytest.raises(ValueError, match="ahead of workspace version"):
        rust_api_discovery.validate_release_tag({"CURRENT_TAG": "1.5.0"}, "1.4.0")


def test_release_tag_must_be_present_and_parsable() -> None:
    with pytest.raises(ValueError, match="CURRENT_TAG"):
        rust_api_discovery.validate_release_tag({}, "1.4.0")


def test_crates_have_pinned_docs_rs_links(
    reference: rust_api_discovery.RustReference,
) -> None:
    for crate in reference.crates:
        assert crate.docs_href == f"https://docs.rs/{crate.name}/{crate.version}"
        assert "/latest" not in crate.docs_href
    deprecated = next(
        crate for crate in reference.crates if crate.name == "dynamo-async-openai"
    )
    assert deprecated.version == "1.0.2"
    assert deprecated.badge == "Deprecated"


def test_core_and_external_crates_are_classified(
    reference: rust_api_discovery.RustReference,
) -> None:
    core = {crate.name for crate in reference.crates if crate.group == "core"}
    assert core == CORE_CRATES
    by_name = {crate.name: crate for crate in reference.crates}
    for name in ("dynamo-async-openai", "dynamo-config", "dynamo-parsers"):
        assert by_name[name].member_path is None


def test_bindings_link_to_repository_source(
    reference: rust_api_discovery.RustReference,
) -> None:
    assert {binding.name for binding in reference.bindings} == {
        "dynamo-codegen",
        "libdynamo_llm",
    }
    assert all(
        binding.source_href.startswith(rust_api_discovery.SOURCE_BASE)
        for binding in reference.bindings
    )
    assert all("docs.rs" not in binding.source_href for binding in reference.bindings)


def test_rendered_page_is_complete_and_deterministic(
    reference: rust_api_discovery.RustReference,
) -> None:
    first = rust_api_rendering.render_page(reference)
    assert first == rust_api_rendering.render_page(reference)
    assert rust_api_rendering.MDX_GENERATED_MARKER in first
    for name in EXPECTED_CRATES:
        assert name in first


def test_rendered_page_is_native_mdx(
    reference: rust_api_discovery.RustReference,
) -> None:
    """Crate tables are plain Markdown, so Fern indexes them for search and
    derives the Markdown twin itself instead of a hand-built fallback."""
    page = rust_api_rendering.render_page(reference)
    assert page.startswith("---\n# SPDX-FileCopyrightText:")
    assert "title: Rust API" in page
    assert "ApiRustIndex" not in page
    assert "<llms-only>" not in page
    assert "## Core Crates" in page
    assert "cargo add dynamo-runtime@1.3.0" in page


def test_rendered_page_leads_with_native_crate_cards(
    reference: rust_api_discovery.RustReference,
) -> None:
    """Each crate group gets a card linking to its release-pinned docs.rs."""
    page = rust_api_rendering.render_page(reference)
    assert "<CardGroup" in page
    for crate in reference.crates:
        assert f'href="{crate.docs_href}"' in page


def test_generator_writes_and_checks_outputs(
    workspace: Path,
    cached_reference: rust_api_discovery.RustReference,
) -> None:
    assert gen_rust_api.main(["--fern-root", str(workspace)]) == 0
    assert (workspace / "pages" / "reference" / "api" / "rust" / "README.mdx").is_file()
    assert gen_rust_api.main(["--fern-root", str(workspace), "--check"]) == 0


def test_check_mode_detects_rust_page_drift(
    workspace: Path,
    cached_reference: rust_api_discovery.RustReference,
) -> None:
    assert gen_rust_api.main(["--fern-root", str(workspace)]) == 0
    page = workspace / "pages" / "reference" / "api" / "rust" / "README.mdx"
    page.write_text(page.read_text(encoding="utf-8") + "\n<!-- drift -->\n")
    assert gen_rust_api.main(["--fern-root", str(workspace), "--check"]) == 1


def test_rust_page_is_registered_and_linked_from_the_landing() -> None:
    index = (FERN_ROOT / "index.yml").read_text(encoding="utf-8")
    landing = (FERN_ROOT / "pages" / "reference" / "api" / "README.mdx").read_text(
        encoding="utf-8"
    )
    assert "pages/reference/api/rust/README.mdx" in index
    assert 'href="rust/README.mdx"' in landing


def test_shipped_rust_outputs_are_fresh(
    reference: rust_api_discovery.RustReference,
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    shutil.copytree(
        FERN_ROOT / "pages" / "reference" / "api",
        generated / "pages" / "reference" / "api",
    )
    assert rust_api_rendering.render_page(reference) == (
        generated / "pages" / "reference" / "api" / "rust" / "README.mdx"
    ).read_text(encoding="utf-8")
