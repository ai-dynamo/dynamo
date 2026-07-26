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
import rust_api_rendering
import yaml
from griffe import Function, GriffeLoader

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
FERN_ROOT = REPO_ROOT / "docs" / "fern"
COMPONENTS_DIR = FERN_ROOT / "components"
K8S_DIR = FERN_ROOT / "kubernetes"
INDEX_YML = FERN_ROOT / "index.yml"
DOCS_YML = FERN_ROOT / "docs.yml"
REF_STYLES_COMPONENT = COMPONENTS_DIR / "ReferenceStyles.tsx"
API_LANDING = FERN_ROOT / "reference" / "api" / "README.mdx"


def _reference_general_layout() -> list[dict[str, Any]]:
    """Return the ``layout`` list of the Reference tab's General variant."""
    nav = yaml.safe_load(INDEX_YML.read_text(encoding="utf-8"))
    reference_tab = next(
        entry for entry in nav["navigation"] if entry.get("tab") == "reference"
    )
    general = next(
        variant
        for variant in reference_tab.get("variants", [])
        if variant.get("title") == "General"
    )
    return general["layout"]


def _api_reference_section() -> dict[str, Any]:
    """Return the API Reference section from the General variant."""
    for entry in _reference_general_layout():
        if entry.get("section") == "API Reference":
            return entry
    raise AssertionError("API Reference section not found in General variant")


def _python_api_section() -> dict[str, Any]:
    """Return the Python API section (nested under API Reference)."""
    for entry in _api_reference_section()["contents"]:
        if entry.get("section") == "Python API":
            return entry
    raise AssertionError("Python API section not found under API Reference")


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


def test_python_module_pages_are_visible_in_sidebar() -> None:
    """Every generated Python module page must be a visible sidebar entry."""
    python_section = _python_api_section()
    child_pages = [item for item in python_section["contents"] if "page" in item]
    slugs = {page["slug"] for page in child_pages}

    assert slugs == set(EXPECTED_PYTHON_MODULE_SLUGS)
    hidden = [page["slug"] for page in child_pages if page.get("hidden") is True]
    assert hidden == [], f"Python module pages must not be hidden: {hidden}"


def test_api_reference_colocates_python_rust_kubernetes() -> None:
    """Python, Rust, and Kubernetes must appear as siblings under API Reference."""
    api_reference = _api_reference_section()
    languages: list[str] = []
    for entry in api_reference["contents"]:
        title = entry.get("section") or entry.get("page")
        if title in ("Python API", "Rust API", "Kubernetes API"):
            languages.append(title)

    assert set(languages) == {
        "Python API",
        "Rust API",
        "Kubernetes API",
    }, f"missing languages under API Reference: {sorted(set(languages))}"


def test_reference_tab_no_longer_has_kubernetes_api_variant() -> None:
    """The stand-alone Kubernetes API variant is removed once colocated."""
    nav = yaml.safe_load(INDEX_YML.read_text(encoding="utf-8"))
    reference_tab = next(
        entry for entry in nav["navigation"] if entry.get("tab") == "reference"
    )
    variant_titles = [
        variant.get("title") for variant in reference_tab.get("variants", [])
    ]

    assert (
        "Kubernetes API" not in variant_titles
    ), f"Kubernetes API variant should be removed from reference tab, got {variant_titles}"


# The section landing consumes the old "full-api-reference" slug because
# it now owns kubernetes/api-reference-fern.mdx; the trimmed per-CRD
# references keep their slugs as sibling pages inside the section.
_EXPECTED_K8S_REDIRECTS: dict[str, str] = {
    "/dynamo/dev/reference/kubernetes-api/full-api-reference": (
        "/dynamo/dev/reference/api/kubernetes"
    ),
    "/dynamo/dev/reference/kubernetes-api/dynamographdeployment": (
        "/dynamo/dev/reference/api/kubernetes/dynamographdeployment"
    ),
    "/dynamo/dev/reference/kubernetes-api/dynamographdeploymentrequest": (
        "/dynamo/dev/reference/api/kubernetes/dynamographdeploymentrequest"
    ),
    "/dynamo/dev/reference/kubernetes-api/dynamocomponentdeployment": (
        "/dynamo/dev/reference/api/kubernetes/dynamocomponentdeployment"
    ),
}


def test_kubernetes_api_url_redirects_present() -> None:
    """Legacy /reference/kubernetes-api/* URLs must redirect to /reference/api/kubernetes/*."""
    docs = yaml.safe_load(DOCS_YML.read_text(encoding="utf-8"))
    redirects = {r["source"]: r["destination"] for r in docs.get("redirects", [])}
    for source, destination in _EXPECTED_K8S_REDIRECTS.items():
        assert source in redirects, f"missing redirect for {source}"
        assert redirects[source] == destination, (
            f"unexpected redirect target for {source}: "
            f"got {redirects[source]}, want {destination}"
        )


def test_api_landing_points_kubernetes_at_colocated_route() -> None:
    """The landing card group must point Kubernetes at the colocated route."""
    source = API_LANDING.read_text(encoding="utf-8")
    card = re.search(
        r'<Card title="Kubernetes" href="([^"]+)"',
        source,
    )

    assert card is not None, "Kubernetes card not found on the API landing page"
    assert (
        card.group(1) == "../../kubernetes/api-reference-fern.mdx"
    ), f"Kubernetes card must point at the colocated page, got {card.group(1)!r}"
    assert (
        "kubernetes-api/full-api-reference" not in source
    ), "landing must not reference the removed kubernetes-api variant"


_UNMERGED_DOCS_LINK_RE = re.compile(
    r"https://github\.com/ai-dynamo/dynamo/(?:blob|tree)/main/docs/fern/\S*"
)


def _api_reference_pages() -> list[Path]:
    """Every committed page this reference owns."""
    pages = sorted((FERN_ROOT / "reference" / "api").rglob("*.mdx"))
    pages.append(K8S_DIR / "api-reference-fern.mdx")
    return pages


def test_api_pages_never_link_to_docs_paths_through_main() -> None:
    """These pages, their generator scripts, and the raw Kubernetes Markdown
    all arrive in the same change. A ``blob/main`` deep link to any of them
    resolves to a 404 until that change merges, so the link checker fails on
    exactly the commits that introduce the pages. Reference the repo path as
    inline code instead, or link the sibling page relatively."""
    offenders: dict[str, list[str]] = {}
    for page in _api_reference_pages():
        found = _UNMERGED_DOCS_LINK_RE.findall(page.read_text(encoding="utf-8"))
        if found:
            offenders[str(page.relative_to(REPO_ROOT))] = found

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
    source = (K8S_DIR / "api-reference.md").read_text(encoding="utf-8")
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
        "docs/fern/kubernetes/api-reference.md",
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
    assert "pytest -c /dev/null -q docs/fern/scripts/tests" in workflow
    for generator in ("python", "rust", "kubernetes"):
        assert f"gen_{generator}_api.py --check" in workflow
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
    source = (K8S_DIR / "api-reference.md").read_text(encoding="utf-8")
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
    for package in packages:
        package_anchors = {type_.anchor for type_ in package.types}
        refs = list(package.resource_types)
        refs.extend(ref for type_ in package.types for ref in type_.appears_in)
        assert all(ref.anchor in package_anchors for ref in refs)


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
    source_paths = (footer, K8S_DIR / "api-reference.md")
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        assert ":::{note}" not in text
        assert "> [!NOTE]" in text
    rendered = (K8S_DIR / "api-reference-fern.mdx").read_text(encoding="utf-8")
    assert "<Warning>" not in rendered
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
    pages = fern / "reference" / "api" / "python"
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
