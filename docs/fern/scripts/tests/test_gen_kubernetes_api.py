# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the Dynamo Kubernetes API docs generator.

Exercises the deterministic parser + renderer that turn the upstream
``docs/fern/pages/reference/kubernetes-api/additional-resources/
api-reference-k8s.md`` (crd-ref-docs output + header/footer +
fix-api-anchors.py) into a typed model and thin MDX shell. The generator
writes into a scratch workspace so a failing test can never mutate the
tracked docs tree. Hermetic; no network, no Dynamo runtime. Invocation::

    uv run --no-project --python 3.13 --with pytest --with pyyaml \\
        python3 -m pytest docs/fern/scripts/tests -v
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import gen_kubernetes_api
import kubernetes_api_discovery
import kubernetes_api_rendering
import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
FERN_ROOT = REPO_ROOT / "docs" / "fern"
K8S_DIR = FERN_ROOT / "pages" / "reference" / "kubernetes-api"
SOURCE_MD = K8S_DIR / "additional-resources" / "api-reference-k8s.md"
TARGET_MDX = K8S_DIR / "full-api-reference.mdx"
# Stops at ``&`` as well as ``<``: the shipped MDX escapes the ``<br />``
# that follows this URL, and the entity would otherwise be read as part of
# it and defeat the comparison against the unescaped source.
_DEV_URL_RE = re.compile(r"[^\s<>()\[\]&\"']*/dynamo/dev/[^\s<>()\[\]&,;\"']*")

# Content baseline pinned by the plan.
EXPECTED_PACKAGES = (
    "nvidia.com/v1alpha1",
    "nvidia.com/v1beta1",
    "operator.config.dynamo.nvidia.com/v1alpha1",
)
EXPECTED_TYPE_COUNTS = {
    # PodSnapshot, PodSnapshotContent, and their 8 related sub-types
    # (PodReference, PodSnapshotSource/Spec/Status,
    # PodSnapshotContentSource/Spec/Status, PodSnapshotReference) moved to
    # github.com/ai-dynamo/snapshot; was 83.
    "nvidia.com/v1alpha1": 73,
    "nvidia.com/v1beta1": 66,
    "operator.config.dynamo.nvidia.com/v1alpha1": 32,
}
EXPECTED_OPERATOR_DEFAULT_SECTIONS = (
    "Pod Specification Defaults",
    "Security Context",
    "Shared Memory Configuration",
    "Health Probes by Component Type",
    "Environment Variables",
    "Service Accounts",
    "Image Pull Secrets",
    "Autoscaling Defaults",
    "Port Configurations",
    "Backend-Specific Configurations",
    "Implementation Reference",
    "Notes",
)
# Types renamed by fix-api-anchors.py so their v1beta1 anchors stay unique.
V1BETA1_DEDUP_TYPES = (
    "DynamoGraphDeploymentRequest",
    "DynamoGraphDeploymentRequestSpec",
    "DynamoGraphDeploymentRequestStatus",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def source_text() -> str:
    return SOURCE_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def reference(source_text: str) -> kubernetes_api_discovery.KubernetesReference:
    return kubernetes_api_discovery.parse_reference(source_text)


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Clone the four generator I/O paths into ``tmp_path``.

    The parse input is copied from the source tree so workspace tests
    exercise the same 3-package / 195-heading baseline as the model tests."""
    ws = tmp_path / "repo"
    dst = ws / "docs" / "fern"
    dst.mkdir(parents=True)
    (dst / "components").mkdir()
    k8s_dst = dst / "pages" / "reference" / "kubernetes-api" / "additional-resources"
    k8s_dst.mkdir(parents=True)
    shutil.copy2(SOURCE_MD, k8s_dst / "api-reference-k8s.md")
    return ws


# ---------------------------------------------------------------------------
# Discovery / model integrity
# ---------------------------------------------------------------------------


def test_reference_lists_the_three_agreed_packages(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """The upstream Markdown carries exactly three CRD/config API packages;
    the parser must surface each of them so the compact index has three
    filterable groups."""
    assert tuple(pkg.name for pkg in reference.packages) == EXPECTED_PACKAGES


@pytest.mark.parametrize("package_name,expected", sorted(EXPECTED_TYPE_COUNTS.items()))
def test_each_package_type_count_matches_the_baseline(
    reference: kubernetes_api_discovery.KubernetesReference,
    package_name: str,
    expected: int,
) -> None:
    """The compact index pins the exact per-package type counts (83 / 66 /
    32). Any drift from the tracked upstream API surface is a scope change
    and must be reviewed as one."""
    by_name = {pkg.name: pkg for pkg in reference.packages}
    package = by_name[package_name]
    assert (
        len(package.types) == expected
    ), f"{package_name}: expected {expected} types, got {len(package.types)}"


def test_total_type_count_matches_the_baseline(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """The typed sections in the compact index sum to the per-package
    baseline. Combined with the three Resource Types pseudo-headings and the
    twelve operator-default subsections, that is the heading parity
    ``test_total_heading_parity_matches_the_baseline`` guards."""
    total = sum(len(pkg.types) for pkg in reference.packages)
    assert total == sum(EXPECTED_TYPE_COUNTS.values())


def test_operator_defaults_carries_exactly_twelve_subsections(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """The operator-defaults section owns twelve `###` subsections in the
    compact-index rendering. The intro bullet list is preserved separately
    on the reference."""
    titles = tuple(sub.title for sub in reference.operator_defaults.subsections)
    assert titles == EXPECTED_OPERATOR_DEFAULT_SECTIONS


def test_total_heading_parity_matches_the_baseline(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """The compact index preserves every section: the typed schemas, one
    Resource Types index per package, and the operator-default subsections.
    Derived from the same baselines rather than restating a total, so a
    reviewed scope change lands in one place."""
    typed = sum(len(pkg.types) for pkg in reference.packages)
    resource_indexes = len(reference.packages)  # one per package
    operator_defaults = len(reference.operator_defaults.subsections)
    expected = (
        sum(EXPECTED_TYPE_COUNTS.values())
        + len(EXPECTED_PACKAGES)
        + len(EXPECTED_OPERATOR_DEFAULT_SECTIONS)
    )
    assert typed + resource_indexes + operator_defaults == expected


def test_each_package_has_a_resource_types_index(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """Every package renders a "Resource Types" jump list. An empty list
    indicates a parser regression."""
    for package in reference.packages:
        assert (
            package.resource_types
        ), f"{package.name}: parser dropped the Resource Types list"
        for ref in package.resource_types:
            assert (
                ref.name and ref.anchor
            ), f"{package.name}: malformed Resource Types entry {ref!r}"


def test_resource_types_link_to_real_types_on_the_same_page(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """The Resource Types jump list must point at anchors the reference
    actually exposes. A broken link here is a Fern build error."""
    all_anchors = {t.anchor for pkg in reference.packages for t in pkg.types}
    for package in reference.packages:
        for ref in package.resource_types:
            assert ref.anchor in all_anchors, (
                f"{package.name}: Resource Types link '{ref.name}' -> "
                f"#{ref.anchor} has no matching type section"
            )


def test_v1beta1_shared_type_names_use_deduplicated_anchors(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """``fix-api-anchors.py`` prefixes duplicate v1beta1 types with ``v1beta1-``."""
    v1beta1_pkg = next(p for p in reference.packages if p.name == "nvidia.com/v1beta1")
    by_display = {t.display_name: t for t in v1beta1_pkg.types}
    for type_name in V1BETA1_DEDUP_TYPES:
        display = f"v1beta1 {type_name}"
        assert display in by_display, f"v1beta1 missing renamed type '{display}'"
        expected_anchor = f"v1beta1-{type_name.lower()}"
        assert by_display[display].anchor == expected_anchor


def test_dynamocheckpoint_carries_the_expected_field_set(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """DynamoCheckpoint (resource) surfaces apiVersion/kind/metadata/spec/status."""
    v1alpha1 = next(p for p in reference.packages if p.name == "nvidia.com/v1alpha1")
    by_name = {t.name: t for t in v1alpha1.types}
    dyn_checkpoint = by_name["DynamoCheckpoint"]
    assert dyn_checkpoint.kind == "resource"
    field_names = [f.name for f in dyn_checkpoint.fields]
    assert field_names == ["apiVersion", "kind", "metadata", "spec", "status"]


def test_enum_types_expose_their_enum_values_not_fields(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """Enums carry values with descriptions; the model classifies them apart from schemas."""
    all_types = [t for pkg in reference.packages for t in pkg.types]
    by_name = {t.name: t for t in all_types}
    dgd_state = by_name["DGDState"]
    assert dgd_state.kind == "enum"
    assert dgd_state.underlying_type == "string"
    values = tuple(v.name for v in dgd_state.enum_values)
    assert values == ("initializing", "pending", "successful", "failed")
    assert dgd_state.fields == ()


def test_schema_types_expose_typed_fields(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """Schema types expose fields with type, default, and required flag."""
    v1alpha1 = next(p for p in reference.packages if p.name == "nvidia.com/v1alpha1")
    by_name = {t.name: t for t in v1alpha1.types}
    selector = by_name["ConfigMapKeySelector"]
    assert selector.kind == "type"
    by_field = {f.name: f for f in selector.fields}
    assert set(by_field) == {"name", "key"}
    assert by_field["key"].default == "disagg.yaml"
    assert by_field["name"].required is True


def test_types_are_sorted_alphabetically_within_a_package(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """Types stay in the crd-ref-docs case-sensitive canonical-name order."""
    for package in reference.packages:
        canonical = [t.name for t in package.types]
        assert canonical == sorted(canonical), f"{package.name}: parser reordered types"


def test_operator_defaults_intro_preserves_summary_bullets(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """Intro bullets summarising the twelve operator-defaults topics survive parsing."""
    intro = reference.operator_defaults.intro_markdown
    for needle in ("Health Probes", "Security Context", "Shared Memory"):
        assert (
            f"**{needle}**" in intro
        ), f"operator-defaults intro missing bullet for '{needle}'"


def test_operator_defaults_subsections_carry_their_body_markdown(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """Each subsection stores raw Markdown so the MDX embeds it verbatim.

    The `### Overriding Security Context` in the source body sits at `###` in the
    stored text; ``_demote_headings`` shifts it to `####` at MDX render time so
    the twelve source-``##`` subsections land at ``###`` on the rendered page.
    """
    by_title = {s.title: s for s in reference.operator_defaults.subsections}
    security = by_title["Security Context"]
    assert "fsGroup" in security.body_markdown
    assert "### Overriding Security Context" in security.body_markdown
    ports = by_title["Port Configurations"]
    assert "### Frontend Components" in ports.body_markdown


# ---------------------------------------------------------------------------
# MDX rendering
# ---------------------------------------------------------------------------


def test_render_mdx_carries_frontmatter_and_no_body_h1(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """Fern renders the page title from the navigation, so a body H1 would
    duplicate it."""
    text = kubernetes_api_rendering.render_mdx(reference)
    assert text.startswith("---\n# SPDX-FileCopyrightText:")
    assert "title: API Reference" in text
    body = text.split("---\n", 2)[-1]
    assert not _has_body_h1(body), (
        "body must not contain a Markdown H1 outside code fences "
        "(Fern renders the title from the nav)"
    )


def test_render_mdx_preserves_dedup_anchors_for_deep_links(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """v1beta1-dedup anchors are the deep-link contract for shared type
    names that collide across packages."""
    text = kubernetes_api_rendering.render_mdx(reference)
    for type_name in V1BETA1_DEDUP_TYPES:
        anchor = f"v1beta1-{type_name.lower()}"
        assert (
            f'<Accordion id="{anchor}"' in text
        ), f"MDX missing deduped anchor '{anchor}' for v1beta1 {type_name}"


def _has_body_h1(body: str) -> bool:
    """Track ```-code-fence state so YAML/shell comments aren't mistaken."""
    in_fence = False
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.startswith("# ") and not stripped.startswith("# SPDX"):
            return True
    return False


def test_render_mdx_embeds_the_twelve_operator_default_subsections(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    """The twelve operator-default subsections appear as demoted `###`
    headings after the compact component. This is the observable content
    baseline the plan pins and what the llms-only twin also indexes."""
    text = kubernetes_api_rendering.render_mdx(reference)
    for title in EXPECTED_OPERATOR_DEFAULT_SECTIONS:
        assert (
            f"### {title}" in text
        ), f"operator-defaults subsection '### {title}' missing from MDX"
    # Sub-subsections must demote correctly (### Frontend Components in
    # source-body -> #### Frontend Components under Port Configurations).
    assert "#### Frontend Components" in text


def test_render_mdx_is_deterministic(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> None:
    a = kubernetes_api_rendering.render_mdx(reference)
    b = kubernetes_api_rendering.render_mdx(reference)
    assert a == b


# ---------------------------------------------------------------------------
# Generator I/O + --check
# ---------------------------------------------------------------------------


def test_generator_writes_the_mdx_page(workspace: Path) -> None:
    fern = workspace / "docs" / "fern"
    rc = gen_kubernetes_api.main(["--fern-root", str(fern)])
    assert rc == 0
    assert (
        fern / "pages" / "reference" / "kubernetes-api" / "full-api-reference.mdx"
    ).is_file()


def test_check_mode_returns_zero_on_fresh_outputs(workspace: Path) -> None:
    fern = workspace / "docs" / "fern"
    assert gen_kubernetes_api.main(["--fern-root", str(fern)]) == 0
    assert gen_kubernetes_api.main(["--fern-root", str(fern), "--check"]) == 0


def test_check_mode_flags_mdx_shell_drift(workspace: Path) -> None:
    fern = workspace / "docs" / "fern"
    assert gen_kubernetes_api.main(["--fern-root", str(fern)]) == 0
    mdx = fern / "pages" / "reference" / "kubernetes-api" / "full-api-reference.mdx"
    mdx.write_text(
        mdx.read_text(encoding="utf-8") + "\n<!-- drift -->\n", encoding="utf-8"
    )
    assert gen_kubernetes_api.main(["--fern-root", str(fern), "--check"]) == 1


# ---------------------------------------------------------------------------
# Cross-checks against the shipped tree
# ---------------------------------------------------------------------------


def test_shipped_mdx_matches_regeneration_output() -> None:
    """The shipped ``full-api-reference.mdx`` is generator output; running
    the generator against the same source must produce byte-identical
    text. Any drift here means the tracked file is stale."""
    text = SOURCE_MD.read_text(encoding="utf-8")
    reference = kubernetes_api_discovery.parse_reference(text)
    rendered = kubernetes_api_rendering.render_mdx(reference)
    shipped = TARGET_MDX.read_text(encoding="utf-8")
    assert rendered == shipped


def test_generator_adds_no_hardcoded_dev_paths() -> None:
    """Fern serves each doc version under its own path prefix, so a
    ``/dynamo/dev/...`` link sends a versioned snapshot back to dev.

    The generator must never author one. Operator Go comments sometimes do,
    to give ``kubectl explain`` readers a followable URL, and rewriting
    those is not this page's call to make: the site path carries navigation
    tab prefixes that have no counterpart on disk, so there is no sound
    mapping back to a relative link. Hold the generator to what it controls
    by allowing through only the dev URLs the CRD source already carries.
    """
    shipped = set(_DEV_URL_RE.findall(TARGET_MDX.read_text(encoding="utf-8")))
    upstream = set(_DEV_URL_RE.findall(SOURCE_MD.read_text(encoding="utf-8")))
    assert not (shipped - upstream), (
        "generator authored dev-pinned links absent from the CRD source: "
        f"{sorted(shipped - upstream)}"
    )


def test_index_yml_still_registers_the_compact_mdx() -> None:
    """The compact-index MDX must remain the Full API Reference page and
    the hidden agent Markdown source stays reachable by direct URL."""
    doc = yaml.safe_load((FERN_ROOT / "index.yml").read_text(encoding="utf-8"))
    paths = _collect_all_paths(doc)
    assert "pages/reference/kubernetes-api/full-api-reference.mdx" in paths
    assert (
        "pages/reference/kubernetes-api/additional-resources/api-reference-k8s.md"
        in paths
    )


# ---------------------------------------------------------------------------
# Native Fern component rendering
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def native_mdx(reference: kubernetes_api_discovery.KubernetesReference) -> str:
    return kubernetes_api_rendering.render_mdx(reference)


def test_render_mdx_carries_no_react_component_or_data_import(native_mdx: str) -> None:
    """Content lives in MDX so Fern's own components, search indexing, and
    Markdown extraction apply. A React mount would hide all of it."""
    assert "ApiKubernetesReference" not in native_mdx
    assert "api-reference.data" not in native_mdx


def test_render_mdx_drops_the_hand_built_llms_only_twin(native_mdx: str) -> None:
    """Fern serves any page as Markdown and builds llms.txt from MDX, so the
    hand-maintained twin is duplicated content once the page is native."""
    assert "<llms-only>" not in native_mdx


def test_render_mdx_renders_every_type_in_an_accordion(
    reference: kubernetes_api_discovery.KubernetesReference, native_mdx: str
) -> None:
    """Accordion content stays searchable and SEO-indexed while collapsed,
    unlike raw ``<details>``."""
    total_types = sum(len(pkg.types) for pkg in reference.packages)
    assert native_mdx.count("<Accordion ") == total_types


def test_render_mdx_gives_each_accordion_an_explicit_id(
    reference: kubernetes_api_discovery.KubernetesReference, native_mdx: str
) -> None:
    """Cross-package field links target ``#anchor``, and Accordion takes that
    id natively -- no empty ``<a id>``, which renders as a link with no text."""
    assert "<a id=" not in native_mdx
    for type_ in _iter_types(reference):
        assert f'<Accordion id="{type_.anchor}"' in native_mdx


def test_render_mdx_renders_schema_fields_as_param_fields(
    reference: kubernetes_api_discovery.KubernetesReference, native_mdx: str
) -> None:
    """``<ParamField>`` is the house pattern for reference fields and is
    already used in 29 pages across this site."""
    field_total = sum(len(t.fields) for t in _iter_types(reference))
    assert native_mdx.count("<ParamField path=") == field_total


def test_render_mdx_marks_required_fields_natively(
    reference: kubernetes_api_discovery.KubernetesReference, native_mdx: str
) -> None:
    """Required is a first-class ParamField attribute, not a text badge."""
    required_total = sum(
        1 for t in _iter_types(reference) for f in t.fields if f.required
    )
    assert native_mdx.count("required={true}") == required_total


def test_render_mdx_renders_enum_values_as_badges(
    reference: kubernetes_api_discovery.KubernetesReference, native_mdx: str
) -> None:
    """Enum values were previously a description list; Badge matches how the
    backend config references render allowed values."""
    enum_total = sum(len(t.enum_values) for t in _iter_types(reference))
    assert native_mdx.count('<Badge intent="note" minimal>') == enum_total


def test_render_mdx_lists_resource_types_in_a_card_group(
    reference: kubernetes_api_discovery.KubernetesReference, native_mdx: str
) -> None:
    """Resource-type entry points render as native cards rather than a
    hand-styled CSS grid."""
    assert "<CardGroup" in native_mdx
    for ref in reference.packages[0].resource_types:
        assert f'<Card title="{ref.name}" href="#{ref.anchor}"' in native_mdx


def test_render_mdx_keeps_the_operator_defaults_trailer(native_mdx: str) -> None:
    """The twelve operator-default subsections are prose, not schema, and
    must survive the migration unchanged."""
    assert "## Operator Default Values Injection" in native_mdx
    for title in EXPECTED_OPERATOR_DEFAULT_SECTIONS:
        assert f"### {title}" in native_mdx


def test_native_mdx_is_deterministic(
    reference: kubernetes_api_discovery.KubernetesReference, native_mdx: str
) -> None:
    """``--check`` and diff review depend on byte-identical regeneration."""
    assert kubernetes_api_rendering.render_mdx(reference) == native_mdx


def _iter_types(
    reference: kubernetes_api_discovery.KubernetesReference,
) -> list[kubernetes_api_discovery.KubernetesType]:
    return [type_ for pkg in reference.packages for type_ in pkg.types]


def _collect_all_paths(node: object) -> set[str]:
    found: set[str] = set()
    if isinstance(node, list):
        for item in node:
            found |= _collect_all_paths(item)
    elif isinstance(node, dict):
        path = node.get("path")
        if isinstance(path, str):
            found.add(path)
        for value in node.values():
            found |= _collect_all_paths(value)
    return found
