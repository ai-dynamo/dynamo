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
    # Anchored to releases.data.ts rather than a literal: the tag moves every
    # release, and a hardcoded one turns each bump into a failing test that
    # says nothing about the property under test.
    assert reference.release_tag == rust_api_discovery.validate_release_tag(
        rust_api_discovery.parse_data_module(RELEASES_DATA),
        reference.workspace_version,
    )
    # A patch release republishes only the crates it touched, so the lagging
    # set is data, not a constant: 1.3.1 shipped 8 of 12 and left
    # dynamo-parsers and dynamo-protocols at 1.3.0. Enumerating the laggards
    # here means every partial release fails this test until someone edits the
    # list, which says nothing about whether the inventory is correct.
    #
    # The invariant that actually matters is directional. A crate pinned
    # *ahead* of the shipped tag would link docs.rs at something never
    # published; a crate behind it is the normal state of a patch release.
    version_key = rust_api_discovery._version_key
    tag = version_key(reference.release_tag)
    current = [crate for crate in reference.crates if crate.badge != "Deprecated"]
    assert current, "expected at least one non-deprecated crate"
    for crate in current:
        assert version_key(crate.version) <= tag, (
            f"{crate.name} {crate.version} is ahead of release tag "
            f"{reference.release_tag}"
        )
    # The tag has to correspond to a real publish, not just bound the set.
    assert any(crate.version == reference.release_tag for crate in current)


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
    # Pinned to the crate's own version, not a literal: the install command
    # tracks whatever shipped, so a literal here fails on every republish.
    runtime = next(c for c in reference.crates if c.name == "dynamo-runtime")
    assert f"cargo add dynamo-runtime@{runtime.version}" in page


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
