# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-cutting regression tests for the generated API references."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import api_discovery
import api_rendering
import gen_python_api
import kubernetes_api_discovery
import kubernetes_api_rendering
import markdown_rendering
import pytest
import rust_api_discovery
import rust_api_rendering
import yaml
from griffe import Function, GriffeLoader

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
FERN_ROOT = REPO_ROOT / "docs" / "fern"
COMPONENTS_DIR = FERN_ROOT / "components"
K8S_DIR = FERN_ROOT / "pages" / "reference" / "kubernetes-api"
K8S_SOURCE_MD = K8S_DIR / "additional-resources" / "api-reference-k8s.md"
K8S_TARGET_MDX = K8S_DIR / "full-api-reference.mdx"
INDEX_YML = FERN_ROOT / "index.yml"
DOCS_YML = FERN_ROOT / "docs.yml"
REF_STYLES_COMPONENT = COMPONENTS_DIR / "ReferenceStyles.tsx"
API_LANDING = FERN_ROOT / "pages" / "reference" / "api" / "README.mdx"


def _reference_layout() -> list[dict[str, Any]]:
    """Return the ``layout`` list of the Reference tab."""
    nav = yaml.safe_load(INDEX_YML.read_text(encoding="utf-8"))
    reference_tab = next(
        entry for entry in nav["navigation"] if entry.get("tab") == "reference"
    )
    return reference_tab["layout"]


def _top_level(label: str) -> dict[str, Any]:
    """Return the top-level Reference entry titled ``label``.

    Python API, Rust API, and Kubernetes API are siblings directly under the
    tab, so one lookup serves all three regardless of whether an entry is a
    ``section`` or a ``page``.
    """
    for entry in _reference_layout():
        if entry.get("section") == label or entry.get("page") == label:
            return entry
    raise AssertionError(f"{label} not found at the top level of the reference tab")


def _api_landing() -> dict[str, Any]:
    """Return the API Reference landing page."""
    return _top_level("API Reference")


def _kubernetes_api_section() -> dict[str, Any]:
    """Return the Kubernetes API section."""
    return _top_level("Kubernetes API")


def _python_api_section() -> dict[str, Any]:
    """Return the Python API section."""
    return _top_level("Python API")


def _rust_api_entry() -> dict[str, Any]:
    """Return the Rust API entry."""
    return _top_level("Rust API")


_KubernetesPage = tuple[kubernetes_api_discovery.KubernetesReference, str]
_KubernetesPackagePairs = tuple[
    tuple[kubernetes_api_discovery.KubernetesPackage, ...],
    tuple[kubernetes_api_discovery.KubernetesPackage, ...],
]
_SAMPLE_METHOD = api_discovery.Method(
    name="run",
    signature="run(value: str) -> None",
    summary="Run one value.",
    source_path="sample.py",
    source_line=20,
    source_href="https://example.com/sample.py#L20",
)
_SAMPLE_SYMBOL = api_discovery.Symbol(
    name="Worker",
    kind="class",
    qualname="sample.Worker",
    import_path="sample.Worker",
    summary="A sample worker.",
    signature="Worker(name: str)",
    source_path="sample.py",
    source_line=10,
    source_href="https://example.com/sample.py#L10",
    methods=(_SAMPLE_METHOD,),
)
_SAMPLE_MODULE = api_discovery.Module(
    name="sample",
    slug="sample",
    summary="Sample module.",
    source_path="sample.py",
    source_href="https://example.com/sample.py",
    symbols=(_SAMPLE_SYMBOL,),
)


EXPECTED_PYTHON_MODULE_SLUGS = (
    "_core",
    "runtime",
    "llm",
    "frontend",
    "common",
    "health_check",
    "logits_processing",
    "planner",
    "router",
    "mocker",
    "nixl_connect",
)


def _iter_sections(node: Any) -> Iterator[dict[str, Any]]:
    """Every ``section`` node anywhere below ``node``."""
    if isinstance(node, list):
        for item in node:
            yield from _iter_sections(item)
    elif isinstance(node, dict):
        if "section" in node:
            yield node
        for value in node.values():
            yield from _iter_sections(value)


def test_api_sections_collapse_consistently() -> None:
    """Python API and Kubernetes API must agree on collapse state.

    Python API shipped expanded while Kubernetes API collapsed, which read as
    an accident rather than a decision once the two became peers: eleven
    ``dynamo.*`` entries stood open beside a Kubernetes section folded to one
    line. Both collapse now, so the four API entries occupy four rows and the
    reader opens the surface they came for.
    """
    for section in (_python_api_section(), _kubernetes_api_section()):
        assert section.get("collapsed") is True, (
            f"{section.get('section')} must set 'collapsed: true' so the API "
            "entries present consistently in the sidebar"
        )


def test_machine_readable_releases_page_stays_hidden() -> None:
    """The agent-facing releases mirror is not a sidebar entry.

    ``pages/reference/general/releases-machine-readable.mdx`` is a generated
    plain-markdown mirror of ``releases.data.ts`` for agents, and it says so
    in its own subtitle. Its human-facing content -- the CUDA toolkit and
    minimum driver history -- belongs on Compatibility, which carries that
    table directly. Unhiding this page too would publish the same matrix
    twice under a name that means nothing to a reader.
    """
    entries = [
        entry
        for entry in _walk_nav(_reference_layout())
        if entry.get("path") == "pages/reference/general/releases-machine-readable.mdx"
    ]
    assert len(entries) == 1
    assert entries[0].get("hidden") is True


def _walk_nav(node: Any) -> Iterator[dict[str, Any]]:
    """Yield every mapping node anywhere below ``node``."""
    if isinstance(node, list):
        for item in node:
            yield from _walk_nav(item)
    elif isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_nav(value)


def test_python_module_pages_are_visible_in_sidebar() -> None:
    """Every generated Python module page must be a visible sidebar entry."""
    python_section = _python_api_section()
    child_pages = [item for item in python_section["contents"] if "page" in item]
    slugs = {page["slug"] for page in child_pages}

    assert slugs == set(EXPECTED_PYTHON_MODULE_SLUGS)
    hidden = [page["slug"] for page in child_pages if page.get("hidden") is True]
    assert hidden == [], f"Python module pages must not be hidden: {hidden}"


def test_reference_tab_lists_python_rust_and_kubernetes_as_peers() -> None:
    """The three language surfaces must be top-level siblings in the tab.

    Python, Rust, and Kubernetes document one public surface each, so none of
    them belongs a level below the others. Nesting Python and Rust under a
    generic wrapper led the tab with the wrapper rather than with a language.
    """
    for entry in (_python_api_section(), _rust_api_entry(), _kubernetes_api_section()):
        assert entry.get("icon"), (
            f"{entry.get('section') or entry.get('page')} must carry an icon so "
            "the three language surfaces read as peers in the sidebar"
        )

    _api_landing()


def test_every_api_surface_shares_the_api_slug_prefix() -> None:
    """One route family covers all four API entries.

    Fern derives a URL from nav depth, so these slugs are explicit for two
    different reasons. Python and Rust are pinned so that promoting them to the
    top level did not rewrite the ``/reference/api/...`` routes they were
    already published under -- ``docs.yml`` points twenty anchored redirects at
    ``reference/api/python/nixl_connect`` alone, the retired NIXL Connect class
    pages, and each would otherwise 404. Kubernetes is pinned so it joins that
    family instead of sitting beside it at ``/reference/kubernetes-api``.

    A later edit that "tidies" any of these down to a bare slug would silently
    move published routes, so all four are asserted together.
    """
    assert _api_landing().get("slug") == "api"
    assert _python_api_section().get("slug") == "api/python"
    assert _rust_api_entry().get("slug") == "api/rust"
    assert _kubernetes_api_section().get("slug") == "api/kubernetes"

    redirects = yaml.safe_load(DOCS_YML.read_text(encoding="utf-8"))["redirects"]
    destinations = {r["destination"].split("#", 1)[0] for r in redirects}
    sources = {r["source"] for r in redirects}

    assert "/dynamo/dev/reference/api/python/nixl_connect" in destinations, (
        "the NIXL Connect redirects must keep resolving to the generated "
        "module page under its pinned slug"
    )
    assert "/dynamo/dev/reference/kubernetes-api/full-api-reference" in sources, (
        "the retired /reference/kubernetes-api routes were published, so each "
        "must redirect into the api/kubernetes family"
    )
    assert not any(
        d.startswith("/dynamo/dev/reference/kubernetes-api") for d in destinations
    ), "no redirect may still point at the retired kubernetes-api route"


def test_api_surfaces_are_contiguous_in_the_sidebar() -> None:
    """The four API entries must sit together, with nothing wedged between.

    Making them peers was only half of it. Kubernetes API previously rendered
    eight entries below Rust -- past Compatibility, Examples, Releases and
    Glossary -- so the three language surfaces never read as one group even
    though they were siblings.
    """
    labels = [
        entry.get("section") or entry.get("page") for entry in _reference_layout()
    ]
    positions = [
        labels.index(name)
        for name in ("API Reference", "Python API", "Rust API", "Kubernetes API")
    ]
    assert positions == list(
        range(positions[0], positions[0] + 4)
    ), f"API surfaces must be contiguous; found them at {positions} in {labels}"


def test_api_landing_points_kubernetes_at_colocated_route() -> None:
    """The landing card group must point Kubernetes at the colocated route."""
    source = API_LANDING.read_text(encoding="utf-8")
    card = re.search(
        r'<Card title="Kubernetes"[^>]*?href="([^"]+)"',
        source,
    )

    assert card is not None, "Kubernetes card not found on the API landing page"
    assert (
        card.group(1) == "../kubernetes-api/full-api-reference.mdx"
    ), f"Kubernetes card must point at the colocated page, got {card.group(1)!r}"


def test_api_landing_cards_carry_the_sidebar_icons() -> None:
    """Each landing card must show the brand icon its sidebar entry shows.

    The cards are the first thing on the tab and were the only place the three
    surfaces appeared without their icons, so a reader arriving at the landing
    page saw a different visual vocabulary than the sidebar beside it.
    """
    source = API_LANDING.read_text(encoding="utf-8")
    for title, icon in (
        ("Python", "fa-brands fa-python"),
        ("Rust", "fa-brands fa-rust"),
        ("Kubernetes", "fa-brands fa-kubernetes"),
    ):
        card = re.search(rf'<Card title="{title}"[^>]*>', source)
        assert card is not None, f"{title} card not found on the API landing page"
        assert f'icon="{icon}"' in card.group(0), (
            f"{title} card must carry {icon!r} to match its sidebar entry, "
            f"got {card.group(0)!r}"
        )


_UNMERGED_DOCS_LINK_RE = re.compile(
    r"https://github\.com/ai-dynamo/dynamo/(?:blob|tree)/main/docs/fern/\S*"
)


@pytest.fixture(scope="session")
def discovered_python_modules() -> list[api_discovery.Module]:
    """One griffe pass for every check that inspects freshly rendered pages.

    The Python/Rust pages are publish-time artifacts (not committed), so
    content regressions must be asserted against a fresh render; reading the
    tree would silently iterate zero files and pass vacuously."""
    loader = api_discovery.build_loader()
    return [
        api_discovery.discover_module(loader, spec) for spec in api_discovery.MODULES
    ]


def _api_reference_sources(
    modules: list[api_discovery.Module],
) -> dict[str, str]:
    """name -> page text for every page this reference owns.

    Committed pages (the hand-written API landing, the Kubernetes output)
    read from disk; the publish-time Python/Rust pages come from a fresh
    render."""
    sources = {
        str(page.relative_to(REPO_ROOT)): page.read_text(encoding="utf-8")
        for page in sorted((FERN_ROOT / "pages" / "reference" / "api").rglob("*.mdx"))
    }
    assert sources, "the committed API landing page has gone missing"
    sources[str(K8S_TARGET_MDX.relative_to(REPO_ROOT))] = K8S_TARGET_MDX.read_text(
        encoding="utf-8"
    )
    sources["<rendered> python/README.mdx"] = api_rendering.render_landing_page(modules)
    for module in modules:
        sources[
            f"<rendered> python/{module.slug}.mdx"
        ] = api_rendering.render_module_page(module)
    rust_reference = rust_api_discovery.discover_rust_reference(
        REPO_ROOT, FERN_ROOT / "components" / "releases.data.ts"
    )
    sources["<rendered> rust/README.mdx"] = rust_api_rendering.render_page(
        rust_reference
    )
    return sources


def test_api_pages_never_link_to_docs_paths_through_main(
    discovered_python_modules: list[api_discovery.Module],
) -> None:
    """These pages, their generator scripts, and the raw Kubernetes Markdown
    all arrive in the same change. A ``blob/main`` deep link to any of them
    resolves to a 404 until that change merges, so the link checker fails on
    exactly the commits that introduce the pages. Reference the repo path as
    inline code instead, or link the sibling page relatively."""
    offenders: dict[str, list[str]] = {}
    for name, text in _api_reference_sources(discovered_python_modules).items():
        found = _UNMERGED_DOCS_LINK_RE.findall(text)
        if found:
            offenders[name] = found

    assert not offenders, f"self-referential main links: {offenders}"


def test_api_reference_ships_no_bespoke_render_components() -> None:
    """Every API surface renders through native Fern MDX, so the reference owns
    no React render components at all."""
    retired = (
        "ApiSurfaceBrowser.tsx",
        "ApiPythonIndex.tsx",
        "ApiRustIndex.tsx",
        "ApiKubernetesReference.tsx",
        "KubernetesSchemaDetails.tsx",
        "KubernetesApiTypes.ts",
        "api-reference.data.ts",
        "rust-api-reference.data.ts",
        "ApiReferenceHero.tsx",
    )
    for name in retired:
        assert not (
            COMPONENTS_DIR / name
        ).exists(), f"{name} should be gone after the native-MDX migration"


def test_shared_index_page_title_lives_in_reference_styles() -> None:
    """Landing / index components share a single 20px title style."""
    styles = REF_STYLES_COMPONENT.read_text(encoding="utf-8")

    assert (
        ".dynref-index-title" in styles
    ), "shared index title class missing from ReferenceStyles"


def test_python_anchors_use_qualnames_so_duplicate_names_stay_distinct() -> None:
    """Two submodules can expose the same symbol name; anchoring on the bare
    name would collide and send both deep links to the first one."""
    shared_name = api_discovery.Symbol(
        name="Client",
        kind="class",
        qualname="dynamo._core.Client",
        import_path="dynamo._core.Client",
        summary="",
        signature="",
        source_path="lib/x.py",
        source_line=1,
        source_href="https://example.invalid",
    )
    other = replace(shared_name, qualname="dynamo.llm.Client")

    assert api_rendering.symbol_anchor(shared_name) != api_rendering.symbol_anchor(
        other
    )


def test_python_imports_use_the_public_alias_path() -> None:
    """Griffe resolves symbols to their defining module; importing from the
    canonical path breaks when the public surface re-exports under an alias."""
    symbol = api_discovery.Symbol(
        name="PyRuntimeMetrics",
        kind="class",
        qualname="dynamo._core.internal.PyRuntimeMetrics",
        import_path="dynamo._core.PyRuntimeMetrics",
        summary="",
        signature="",
        source_path="lib/x.py",
        source_line=1,
        source_href="https://example.invalid",
    )

    assert (
        api_rendering.import_statement(symbol)
        == "from dynamo._core import PyRuntimeMetrics"
    )


@pytest.fixture(scope="module")
def kubernetes_page() -> _KubernetesPage:
    source = K8S_SOURCE_MD.read_text(encoding="utf-8")
    reference = kubernetes_api_discovery.parse_reference(source)
    return reference, kubernetes_api_rendering.render_mdx(reference)


def test_kubernetes_page_is_self_contained(kubernetes_page: _KubernetesPage) -> None:
    """Release snapshots copy the page. Inlining the schema as MDX means a
    snapshot cannot drift from a shared component or data module the way an
    imported ``.data.ts`` could."""
    _, mdx = kubernetes_page

    assert "import {" not in mdx
    assert "api-reference.data" not in mdx
    assert not (K8S_DIR / "api-reference.data.ts").exists()


def test_kubernetes_source_path_note_points_at_a_real_file() -> None:
    """The rendered page tells readers where the raw crd-ref-docs Markdown
    lives. That path is emitted as inline code rather than a link, so the
    broken-link checker cannot catch it when the source file moves -- as it
    did when the reference was relocated under ``pages/reference/``."""
    quoted = kubernetes_api_rendering.SOURCE_PATH_MD
    missing = f"SOURCE_PATH_MD does not exist on disk: {quoted}"

    assert (REPO_ROOT / quoted).is_file(), missing
    assert quoted == str(K8S_SOURCE_MD.relative_to(REPO_ROOT))


def test_kubernetes_prose_carries_no_literal_br_markers(
    kubernetes_page: _KubernetesPage,
) -> None:
    """crd-ref-docs encodes the Go comments' hard line wraps as ``<br />``.
    Escaping that marker for MDX turns it into visible ``&lt;br /&gt;`` text
    in enum and field descriptions, so it must be collapsed first."""
    _, mdx = kubernetes_page

    assert "&lt;br" not in mdx
    assert "<br" not in mdx


def test_kubernetes_field_type_links_resolve(
    kubernetes_page: _KubernetesPage,
) -> None:
    """Every local fragment link must land on an anchor the page renders,
    otherwise a field type deep-links into nothing."""
    _, mdx = kubernetes_page
    rendered_anchors = set(re.findall(r'<(?:Accordion|div) id="([^"]+)"', mdx))
    local_link_targets = set(re.findall(r"\]\(#([^)]+)\)", mdx))

    assert local_link_targets <= rendered_anchors


def test_kubernetes_external_type_links_leave_the_type_attribute(
    kubernetes_page: _KubernetesPage,
) -> None:
    """Field types like ``metadata`` carry an absolute Markdown link to the
    upstream Kubernetes API. An MDX attribute renders no Markdown, so the raw
    ``[label](url)`` leaks through as mangled text -- the label belongs in
    ``type`` and the link in the body, where Markdown is processed.
    """
    _, mdx = kubernetes_page
    attributes = re.findall(r'\stype="([^"]*)"', mdx)

    assert attributes, "no ParamField type attributes rendered"
    leaked = [value for value in attributes if "](" in value]
    assert not leaked, f"Markdown links leaked into type attributes: {leaked[:3]}"
    assert 'type="ObjectMeta"' in mdx
    assert "https://kubernetes.io/docs/reference/generated/kubernetes-api" in mdx


def test_kubernetes_page_carries_full_field_semantics(
    kubernetes_page: _KubernetesPage,
) -> None:
    """Fern derives the Markdown and llms.txt twins from MDX, so the field
    schema must be in the page rather than a hand-built fallback block."""
    reference, mdx = kubernetes_page
    field = next(
        field
        for package in reference.packages
        for type_ in package.types
        for field in type_.fields
        if field.default and field.validation
    )

    assert f'<ParamField path="{field.name}"' in mdx
    assert f'default="{field.default}"' in mdx


def test_kubernetes_page_carries_enum_semantics(
    kubernetes_page: _KubernetesPage,
) -> None:
    """Enum values render as badges beside their descriptions, not as a
    hover-only title attribute."""
    reference, mdx = kubernetes_page
    enum_type = next(
        type_
        for package in reference.packages
        for type_ in package.types
        if type_.enum_values
    )

    assert f'title="{enum_type.display_name}">' in mdx
    for value in enum_type.enum_values:
        assert f'<Badge intent="note" minimal>{value.name}</Badge>' in mdx


def test_pre_merge_gates_every_api_generator_input() -> None:
    filters = (REPO_ROOT / ".github" / "filters.yaml").read_text(encoding="utf-8")
    action = (
        REPO_ROOT / ".github" / "actions" / "changed-files" / "action.yml"
    ).read_text(encoding="utf-8")

    assert "\napi_docs:\n" in filters
    for source_path in (
        "lib/bindings/python/src/**",
        "components/src/dynamo/**",
        "**/Cargo.toml",
        "docs/fern/pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md",
        # Navigation and the operator footer are read by the tests this job
        # runs, so an edit that touches only them still has to gate the job.
        "docs/fern/index.yml",
        "docs/fern/docs.yml",
        "deploy/operator/docs/footer.md",
    ):
        assert source_path in filters
    assert "api_docs:" in action
    assert "steps.filter.outputs.api_docs_any_modified" in action


def test_pre_merge_runs_all_api_generators_hermetically() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "pre-merge.yml").read_text(
        encoding="utf-8"
    )
    publish = (REPO_ROOT / ".github" / "workflows" / "fern-docs.yml").read_text(
        encoding="utf-8"
    )
    project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "api_docs: ${{ steps.changes.outputs.api_docs }}" in workflow
    assert "api-docs:" in workflow
    assert "needs.changed-files.outputs.api_docs == 'true'" in workflow
    assert "docs/fern/scripts/tests/test_gen_python_api.py" in workflow
    assert "docs/fern/scripts/tests/test_gen_rust_api.py" in workflow
    assert "docs/fern/scripts/tests/test_gen_kubernetes_api.py" in workflow
    assert "-c /dev/null" not in workflow
    # Python/Rust references are publish-time artifacts: pre-merge must run
    # both generators in WRITE mode (proving a source PR cannot break
    # generation) and never as a freshness diff against committed pages,
    # which no longer exist. Kubernetes output stays committed, so its
    # freshness gate stays.
    for generator in ("python", "rust"):
        assert f"gen_{generator}_api.py\n" in workflow
        assert f"gen_{generator}_api.py --check" not in workflow
    assert "gen_kubernetes_api.py --check" in workflow
    # The publish and preview paths must GENERATE the pages before syncing
    # them to the docs-website branch (dev sync and version snapshots both).
    for generator in ("python", "rust"):
        assert f"gen_{generator}_api.py --check" not in publish
    assert "gen_kubernetes_api.py --check" in publish
    assert "Generate API references" in publish
    assert "Generate API references at the tag" in publish
    # fern check validates nav paths, so the fern-check job must materialize
    # the generated pages first.
    assert "Generate API reference pages" in workflow
    # Step names existing is not enough: generation must PRECEDE each
    # consumer step, or a reorder ships snapshots (and runs fern check)
    # against a tree with no pages.
    assert publish.index("Generate API references") < publish.index(
        "Sync dev content from main"
    )
    assert publish.index("Generate API references at the tag") < publish.index(
        "Build versioned pages from tagged commit"
    )
    assert workflow.index("Generate API reference pages") < workflow.index(
        "Validate Fern configuration"
    )
    assert "griffe==2.1.0" in workflow
    assert "griffe==2.1.0" in publish
    assert '"griffe==2.1.0"' in project


def test_pre_merge_runs_fern_from_docs_root() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "pre-merge.yml").read_text(
        encoding="utf-8"
    )

    for step_name, command in (
        ("Validate Fern configuration", "fern check"),
        ("Check for broken links", "fern docs broken-links"),
    ):
        step = workflow.split(f"- name: {step_name}", maxsplit=1)[1].split(
            "\n\n", maxsplit=1
        )[0]
        assert "working-directory: docs/fern" in step
        assert f"run: {command}" in step


@pytest.fixture(scope="module")
def kubernetes_package_pairs() -> _KubernetesPackagePairs:
    source = K8S_SOURCE_MD.read_text(encoding="utf-8")
    reference = kubernetes_api_discovery.parse_reference(source)
    package_text, _ = kubernetes_api_discovery._split_defaults_section(source)
    raw_packages = tuple(kubernetes_api_discovery._iter_packages(package_text))
    return raw_packages, reference.packages


def _field_anchor_pairs(
    raw_package: kubernetes_api_discovery.KubernetesPackage,
    package: kubernetes_api_discovery.KubernetesPackage,
) -> Iterator[tuple[str, str]]:
    remap = {
        raw_type.anchor: type_.anchor
        for raw_type, type_ in zip(raw_package.types, package.types, strict=True)
    }
    for raw_type, type_ in zip(raw_package.types, package.types, strict=True):
        for raw_field, field in zip(raw_type.fields, type_.fields, strict=True):
            raw_match = re.search(r"\]\(#([^)]+)\)", raw_field.type)
            if raw_match is None or raw_match.group(1) not in remap:
                continue
            match = re.search(r"\]\(#([^)]+)\)", field.type)
            assert match is not None
            yield match.group(1), remap[raw_match.group(1)]


def test_kubernetes_type_anchors_are_globally_unique(
    kubernetes_package_pairs: _KubernetesPackagePairs,
) -> None:
    _, packages = kubernetes_package_pairs
    all_anchors = [type_.anchor for package in packages for type_ in package.types]
    assert len(all_anchors) == len(set(all_anchors))


def test_kubernetes_type_references_stay_package_local(
    kubernetes_package_pairs: _KubernetesPackagePairs,
) -> None:
    _, packages = kubernetes_package_pairs
    all_anchors = {type_.anchor for package in packages for type_ in package.types}
    for package in packages:
        package_anchors = {type_.anchor for type_ in package.types}
        assert all(ref.anchor in package_anchors for ref in package.resource_types)
        appears_in = [ref for type_ in package.types for ref in type_.appears_in]
        assert all(ref.anchor in all_anchors for ref in appears_in)


def test_kubernetes_field_links_follow_package_remaps(
    kubernetes_package_pairs: _KubernetesPackagePairs,
) -> None:
    raw_packages, packages = kubernetes_package_pairs
    for raw_package, package in zip(raw_packages, packages, strict=True):
        for actual, expected in _field_anchor_pairs(raw_package, package):
            assert actual == expected


def test_python_signature_preserves_all_parameter_kinds(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text(
        "def kinds(pos_only, /, positional: int = 1, *args: str, "
        "keyword: bool, **kwargs: object):\n"
        "    pass\n\n"
        "def keyword_only(value, *, flag: bool = False):\n"
        "    pass\n",
        encoding="utf-8",
    )
    loader = GriffeLoader(search_paths=[str(tmp_path)])
    module = loader.load("sample")
    kinds = module.members["kinds"]
    keyword_only = module.members["keyword_only"]

    assert isinstance(kinds, Function)
    assert isinstance(keyword_only, Function)
    kinds_signature = api_discovery._function_signature(kinds)
    keyword_signature = api_discovery._function_signature(keyword_only)
    assert "pos_only, /, positional: int = 1" in kinds_signature
    assert "*args: str, keyword: bool, **kwargs: object" in kinds_signature
    assert "args: str =" not in kinds_signature
    assert "kwargs: object =" not in kinds_signature
    assert "value, *, flag: bool = False" in keyword_signature


def test_python_page_includes_signatures_and_methods() -> None:
    """Signatures and public methods must be in the page itself, since Fern
    derives the Markdown and llms.txt twins from it."""
    rendered = api_rendering.render_module_page(_SAMPLE_MODULE)

    assert _SAMPLE_SYMBOL.signature in rendered
    assert _SAMPLE_METHOD.signature in rendered
    assert _SAMPLE_METHOD.summary in rendered


def test_kubernetes_sources_use_supported_admonitions() -> None:
    footer = REPO_ROOT / "deploy" / "operator" / "docs" / "footer.md"
    source_paths = (footer, K8S_SOURCE_MD)
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        assert ":::{note}" not in text
        assert "> [!NOTE]" in text
    rendered = K8S_TARGET_MDX.read_text(encoding="utf-8")
    assert "> [!WARNING]" in rendered


def test_kubernetes_table_parser_preserves_literal_pipes() -> None:
    row = r"| `field` _string_ | uses `a|b` and x \| y |  | Required: {} |"

    cells = kubernetes_api_discovery._split_table_row(row, 4)

    assert cells[1].strip() == "uses `a|b` and x | y"
    with pytest.raises(ValueError, match="expected 4 cells"):
        kubernetes_api_discovery._split_table_row("| too | few |", 4)


def test_python_generator_detects_and_removes_orphaned_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fern = tmp_path / "fern"
    pages = fern / "pages" / "reference" / "api" / "python"
    pages.mkdir(parents=True)
    module = api_discovery.Module(
        name="sample",
        slug="sample",
        summary="Sample module.",
        source_path="sample.py",
        source_href="https://example.com/sample.py",
        symbols=(),
    )
    monkeypatch.setattr(gen_python_api, "discover_all_modules", lambda: [module])
    assert gen_python_api.main(["--fern-root", str(fern)]) == 0
    orphan = pages / "obsolete.mdx"
    orphan.write_text("stale", encoding="utf-8")

    assert gen_python_api.main(["--fern-root", str(fern), "--check"]) == 1
    assert orphan.is_file()
    assert gen_python_api.main(["--fern-root", str(fern)]) == 0
    assert not orphan.exists()


@pytest.mark.parametrize("cell_renderer", (rust_api_rendering._cell,))
def test_mdx_table_cells_escape_source_metacharacters(
    cell_renderer: Callable[[str], str],
) -> None:
    rendered = cell_renderer("Value {item} <Widget> | next\nline")

    assert "{" not in rendered and "}" not in rendered
    assert "<Widget>" not in rendered
    assert "&#123;item&#125;" in rendered
    assert "&lt;Widget&gt;" in rendered
    assert "\\|" in rendered


def test_mdx_prose_escapes_jsx_but_spares_inline_code() -> None:
    """Entities are not decoded inside code spans, so escaping there would
    surface a literal ``&lt;`` to the reader."""
    rendered = markdown_rendering.escape_mdx_prose(
        "Takes a `map<string, int>` and {opts} for <Widget>"
    )

    assert "`map<string, int>`" in rendered
    assert "&#123;opts&#125;" in rendered
    assert "&lt;Widget&gt;" in rendered


def test_mdx_prose_drops_sphinx_role_prefixes() -> None:
    """Markdown has no cross-reference roles, so a surviving ``:class:``
    prefix would reach the reader as literal text before the code span."""
    rendered = markdown_rendering.escape_mdx_prose(
        "See :class:`RawEngine`, :meth:`generate`, :func:`f`, "
        ":attr:`Cfg.llm`, and :mod:`dynamo.common`."
    )

    assert ":class:" not in rendered
    assert ":meth:" not in rendered
    assert "`RawEngine`" in rendered
    assert "`dynamo.common`" in rendered


def test_mdx_prose_keeps_unknown_colon_pairs_intact() -> None:
    """Only the known Sphinx roles are stripped; ordinary prose that happens
    to sit next to a code span is left exactly as authored."""
    rendered = markdown_rendering.escape_mdx_prose("Timeout:30: `seconds` applies")

    assert "Timeout:30:" in rendered


def test_generated_python_pages_carry_no_sphinx_roles(
    discovered_python_modules: list[api_discovery.Module],
) -> None:
    """Guards the published output, not just the helper: every curated page
    renders from docstrings that mix Google and Sphinx styles. The pages are
    publish-time artifacts, so the guard runs on a fresh render -- globbing
    the (empty) tree would pass vacuously."""
    assert discovered_python_modules
    for module in discovered_python_modules:
        text = api_rendering.render_module_page(module)
        for role in (":class:`", ":meth:`", ":func:`", ":attr:`", ":mod:`"):
            assert role not in text, f"{module.slug}.mdx still carries {role}"


def test_kubernetes_attributes_escape_source_metacharacters() -> None:
    """The Kubernetes surface renders MDX attributes, not Markdown table cells."""
    rendered = kubernetes_api_rendering._attr('Scale "up" & down\nnow')

    assert '"' not in rendered.replace("&quot;", "")
    assert "&quot;up&quot;" in rendered
    assert "&amp;" in rendered
    assert "\n" not in rendered


def test_kubernetes_prose_escapes_jsx_outside_code_spans() -> None:
    rendered = kubernetes_api_rendering._prose(
        "Accepts <T> and {opt} but `map[string]<T>` stays literal"
    )

    assert "&lt;T&gt;" in rendered
    assert "&#123;opt&#125;" in rendered
    assert "`map[string]<T>`" in rendered
