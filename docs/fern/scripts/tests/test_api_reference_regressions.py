# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-cutting regression tests for the generated API references."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import api_discovery
import api_rendering
import gen_python_api
import kubernetes_api_discovery
import kubernetes_api_rendering
import pytest
import rust_api_rendering
from griffe import Function, GriffeLoader

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
FERN_ROOT = REPO_ROOT / "docs" / "fern"
COMPONENTS_DIR = FERN_ROOT / "components"
K8S_DIR = FERN_ROOT / "kubernetes"
K8S_SCHEMA_COMPONENT = COMPONENTS_DIR / "KubernetesSchemaDetails.tsx"


def test_python_component_uses_qualnames_for_identity_and_imports() -> None:
    source = (COMPONENTS_DIR / "ApiSurfaceBrowser.tsx").read_text(encoding="utf-8")
    anchor_helper = re.search(r"function symbolAnchorId.*?\n}", source, flags=re.DOTALL)
    import_helper = re.search(
        r"function symbolImportStatement.*?\n}", source, flags=re.DOTALL
    )

    assert anchor_helper is not None
    assert "symbol.qualname" in anchor_helper.group()
    assert source.count("key={s.qualname}") == 2
    assert import_helper is not None
    assert "symbol.importPath.lastIndexOf" in import_helper.group()
    assert "data-dynref-copy={symbolImportStatement(symbol)}" in source
    assert "from ${mod.name} import" not in source
    data = (COMPONENTS_DIR / "api-reference.data.ts").read_text(encoding="utf-8")
    assert 'name: "PyRuntimeMetrics"' in data
    assert 'importPath: "dynamo._core.PyRuntimeMetrics"' in data


def test_kubernetes_page_snapshots_its_generated_data() -> None:
    data_path = K8S_DIR / "api-reference.data.ts"
    mdx = (K8S_DIR / "api-reference-fern.mdx").read_text(encoding="utf-8")
    component = (COMPONENTS_DIR / "ApiKubernetesReference.tsx").read_text(
        encoding="utf-8"
    )

    assert data_path.is_file()
    assert 'from "./api-reference.data"' in mdx
    assert "reference={KUBERNETES_REFERENCE}" in mdx
    assert 'from "./kubernetes-api-reference.data"' not in component
    assert "reference: KubernetesReference" in component


def test_kubernetes_component_preserves_safe_field_type_links() -> None:
    source = K8S_SCHEMA_COMPONENT.read_text(encoding="utf-8")

    assert "function FieldType" in source
    assert "<FieldType value={field.type}" in source
    assert 'href.startsWith("#") || href.startsWith("https://")' in source
    assert "validAnchors.has(href.slice(1))" in source
    assert "dangerouslySetInnerHTML" not in source


def test_kubernetes_llms_fallback_contains_complete_schema_details() -> None:
    source = (K8S_DIR / "api-reference.md").read_text(encoding="utf-8")
    reference = kubernetes_api_discovery.parse_reference(source)
    rendered = kubernetes_api_rendering.render_mdx(reference)
    llms_body = rendered.split("<llms-only>", 1)[1].split("</llms-only>", 1)[0]
    field = next(
        field
        for package in reference.packages
        for type_ in package.types
        for field in type_.fields
        if field.default and field.validation
    )
    enum_type = next(
        type_
        for package in reference.packages
        for type_ in package.types
        if type_.enum_values
    )

    assert (
        "| Field | Type | Required | Default | Description | Validation |" in llms_body
    )
    for value in (field.name, field.type, field.default, field.validation):
        assert kubernetes_api_rendering._md_cell(value) in llms_body
    assert f"#### {enum_type.display_name}" in llms_body
    for value in enum_type.enum_values:
        assert value.name in llms_body
        assert value.description in llms_body
    rendered_anchors = set(re.findall(r'<a id="([^"]+)"></a>', llms_body))
    local_link_targets = set(re.findall(r"\]\(#([^)]+)\)", llms_body))
    assert local_link_targets <= rendered_anchors


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


def test_kubernetes_type_anchors_are_globally_unique_and_package_local() -> None:
    source = (K8S_DIR / "api-reference.md").read_text(encoding="utf-8")
    reference = kubernetes_api_discovery.parse_reference(source)
    package_text, _ = kubernetes_api_discovery._split_defaults_section(source)
    raw_packages = tuple(kubernetes_api_discovery._iter_packages(package_text))
    all_anchors = [
        type_.anchor for package in reference.packages for type_ in package.types
    ]

    assert len(all_anchors) == len(set(all_anchors))
    for raw_package, package in zip(raw_packages, reference.packages, strict=True):
        package_anchors = {type_.anchor for type_ in package.types}
        remap = {
            raw_type.anchor: type_.anchor
            for raw_type, type_ in zip(raw_package.types, package.types, strict=True)
        }
        refs = list(package.resource_types)
        refs.extend(ref for type_ in package.types for ref in type_.appears_in)
        assert all(ref.anchor in package_anchors for ref in refs)
        for raw_type, type_ in zip(raw_package.types, package.types, strict=True):
            for raw_field, field in zip(raw_type.fields, type_.fields, strict=True):
                raw_match = re.search(r"\]\(#([^)]+)\)", raw_field.type)
                match = re.search(r"\]\(#([^)]+)\)", field.type)
                if raw_match is not None and raw_match.group(1) in remap:
                    assert match is not None
                    assert match.group(1) == remap[raw_match.group(1)]


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


def test_python_llms_fallback_includes_signatures_and_methods() -> None:
    method = api_discovery.Method(
        name="run",
        signature="run(value: str) -> None",
        summary="Run one value.",
        source_path="sample.py",
        source_line=20,
        source_href="https://example.com/sample.py#L20",
    )
    symbol = api_discovery.Symbol(
        name="Worker",
        kind="class",
        qualname="sample.Worker",
        import_path="sample.Worker",
        summary="A sample worker.",
        signature="Worker(name: str)",
        source_path="sample.py",
        source_line=10,
        source_href="https://example.com/sample.py#L10",
        methods=(method,),
    )
    module = api_discovery.Module(
        name="sample",
        slug="sample",
        summary="Sample module.",
        source_path="sample.py",
        source_href="https://example.com/sample.py",
        symbols=(symbol,),
    )
    rendered = api_rendering.render_module_page(module)
    llms_body = rendered.split("<llms-only>", 1)[1].split("</llms-only>", 1)[0]

    assert symbol.signature in llms_body
    assert method.signature in llms_body
    assert method.summary in llms_body


def test_kubernetes_fields_and_enums_have_visible_semantics() -> None:
    source = K8S_SCHEMA_COMPONENT.read_text(encoding="utf-8")

    assert '<table className="dynref-k8s-fields">' in source
    assert '<th scope="row"' in source
    assert '<dl className="dynref-k8s-enum-values">' in source
    assert '{value.description || "No description."}' in source
    assert "title={value.description" not in source


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


@pytest.mark.parametrize(
    "cell_renderer",
    (api_rendering._cell, rust_api_rendering._cell, kubernetes_api_rendering._md_cell),
)
def test_mdx_table_cells_escape_source_metacharacters(
    cell_renderer: Callable[[str], str],
) -> None:
    rendered = cell_renderer("Value {item} <Widget> | next\nline")

    assert "{" not in rendered and "}" not in rendered
    assert "<Widget>" not in rendered
    assert "&#123;item&#125;" in rendered
    assert "&lt;Widget&gt;" in rendered
    assert "\\|" in rendered
