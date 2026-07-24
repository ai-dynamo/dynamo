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
    (fern / "reference" / "api" / "rust").mkdir(parents=True)
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
    assert reference.workspace_version == "1.3.0"
    current = [crate for crate in reference.crates if crate.badge != "Deprecated"]
    assert {crate.version for crate in current} == {reference.workspace_version}


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


def test_rendered_typescript_is_typed_complete_and_deterministic(
    reference: rust_api_discovery.RustReference,
) -> None:
    first = rust_api_rendering.render_ts_data(reference)
    second = rust_api_rendering.render_ts_data(reference)
    assert first == second
    assert rust_api_rendering.TS_GENERATED_MARKER in first
    assert "export interface RustCrate" in first
    assert "export const RUST_CRATES: RustCrate[]" in first
    for name in EXPECTED_CRATES:
        assert f'"{name}"' in first


def test_rendered_page_has_frontmatter_component_and_llms_twin(
    reference: rust_api_discovery.RustReference,
) -> None:
    page = rust_api_rendering.render_page(reference)
    assert page.startswith("---\n# SPDX-FileCopyrightText:")
    assert "title: Rust API" in page
    assert "import { ApiRustIndex }" in page
    assert "<llms-only>" in page and "</llms-only>" in page
    assert "### Core Crates" in page
    assert "cargo add dynamo-runtime@1.3.0" in page


def test_generator_writes_and_checks_outputs(
    workspace: Path,
    cached_reference: rust_api_discovery.RustReference,
) -> None:
    assert gen_rust_api.main(["--fern-root", str(workspace)]) == 0
    assert (workspace / "components" / "rust-api-reference.data.ts").is_file()
    assert (workspace / "reference" / "api" / "rust" / "README.mdx").is_file()
    assert gen_rust_api.main(["--fern-root", str(workspace), "--check"]) == 0


def test_check_mode_detects_rust_page_drift(
    workspace: Path,
    cached_reference: rust_api_discovery.RustReference,
) -> None:
    assert gen_rust_api.main(["--fern-root", str(workspace)]) == 0
    page = workspace / "reference" / "api" / "rust" / "README.mdx"
    page.write_text(page.read_text(encoding="utf-8") + "\n<!-- drift -->\n")
    assert gen_rust_api.main(["--fern-root", str(workspace), "--check"]) == 1


def test_rust_page_is_registered_and_linked_from_the_hero() -> None:
    index = (FERN_ROOT / "index.yml").read_text(encoding="utf-8")
    hero = (FERN_ROOT / "components" / "ApiReferenceHero.tsx").read_text(
        encoding="utf-8"
    )
    assert "reference/api/rust/README.mdx" in index
    assert 'landingHref: "api/rust"' in hero


def test_shipped_rust_outputs_are_fresh(
    reference: rust_api_discovery.RustReference,
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    shutil.copytree(FERN_ROOT / "components", generated / "components")
    shutil.copytree(FERN_ROOT / "reference", generated / "reference")
    assert rust_api_rendering.render_ts_data(reference) == (
        generated / "components" / "rust-api-reference.data.ts"
    ).read_text(encoding="utf-8")
    assert rust_api_rendering.render_page(reference) == (
        generated / "reference" / "api" / "rust" / "README.mdx"
    ).read_text(encoding="utf-8")
