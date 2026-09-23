# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

ROOT = Path(__file__).resolve().parents[2]
FERN = ROOT / "docs/fern"
EXAMPLES = FERN / "pages/recipes/examples"
EMBED = re.compile(r'<Code\s+src="([^"]+)"')
CARD = re.compile(r'<div className="dynamo-example-card" ([^>]+)>')
ATTRIBUTE = re.compile(r'([\w-]+)="([^"]*)"')


def cards() -> dict[str, dict[str, str]]:
    result = {}
    for match in CARD.finditer((EXAMPLES / "overview.mdx").read_text()):
        attributes = dict(ATTRIBUTE.findall(match[1]))
        key = attributes["data-example"]
        assert key not in result, f"duplicate catalog card: {key}"
        result[key] = attributes
    assert result, "empty examples catalog"
    return result


def embeds(page: Path) -> set[Path]:
    return {(page.parent / src).resolve() for src in EMBED.findall(page.read_text())}


def all_embeds() -> set[Path]:
    return {asset for page in EXAMPLES.glob("*.mdx") for asset in embeds(page)}


def navigation_paths(node: object) -> set[str]:
    if isinstance(node, dict):
        own = {node["path"]} if isinstance(node.get("path"), str) else set()
        return own | {
            path for value in node.values() for path in navigation_paths(value)
        }
    if isinstance(node, list):
        return {path for value in node for path in navigation_paths(value)}
    return set()


def test_catalog_pages_and_navigation_agree() -> None:
    catalog = cards()
    pages = {page.stem for page in EXAMPLES.glob("*.mdx")} - {"overview"}
    assert set(catalog) == pages
    nav = yaml.safe_load((FERN / "index.yml").read_text())
    paths = navigation_paths(nav)
    overview = (EXAMPLES / "overview.mdx").read_text()
    for name in pages:
        assert f"pages/recipes/examples/{name}.mdx" in paths
        assert re.search(rf"\]\({re.escape(name)}\.mdx\)", overview)
    assert "pages/recipes/examples/overview.mdx" in paths
    assert not any(
        "cli-templates/" in path or "kubernetes-templates/" in path for path in paths
    )


def variants(page: Path) -> list[dict[str, str]]:
    match = re.search(
        r"<ExampleSelector\s+variants=\{(\[.*?\])\}\s*>", page.read_text(), re.DOTALL
    )
    assert match, f"missing unified selector: {page}"
    return json.loads(match[1])


def test_card_targets_match_available_page_variants() -> None:
    for name, metadata in cards().items():
        page = EXAMPLES / f"{name}.mdx"
        targets = metadata["data-targets"].split()
        assert len(targets) == len(set(targets)), name
        actual = {variant["target"] for variant in variants(page)}
        assert set(targets) == actual, name
        assert all(
            re.fullmatch(r"(local|kubernetes):[a-z]+", target) for target in targets
        ), name
        assert metadata["data-topic"] and metadata["data-topic-label"], name


def test_each_selector_option_has_one_native_source_bundle() -> None:
    for name in cards():
        page = EXAMPLES / f"{name}.mdx"
        text = page.read_text()
        options = variants(page)
        ids = [option["id"] for option in options]
        blocks = re.findall(
            r'<div className="dynamo-example-variant" data-example-variant="([^"]+)">(.*?)</div>',
            text,
            re.DOTALL,
        )
        assert len(ids) == len(set(ids)), name
        assert len(blocks) == len(ids), name
        assert {key for key, _ in blocks} == set(ids), name
        for key, body in blocks:
            assert EMBED.search(body), (name, key)
        assert all(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", key) for key in ids), name
        assert all(option["label"] and option["description"] for option in options), (
            name
        )
        assert not re.search(r"</?(?:Tabs?|Accordion(?:Group)?)\b", text), name
        assert text.count("<ExampleSelector") == 1, name
        assert "hide-toc: true" in text, name


def test_unified_selector_options_are_distinguishable_per_target() -> None:
    for name in cards():
        options = variants(EXAMPLES / f"{name}.mdx")
        keys = [(option["target"], option["label"]) for option in options]
        assert len(keys) == len(set(keys)), name


def test_example_pages_share_the_toc_less_article_alignment_fix() -> None:
    styles = (FERN / "components/examples-layout.ts").read_text()
    assert ".fern-layout-guide > article" in styles
    assert "margin-left: 0 !important" in styles
    assert "margin-right: auto !important" in styles
    assert "--content-width:" not in styles
    assert "--page-width:" not in styles
    for component in ["ExamplesCatalog.tsx", "ExampleSelector.tsx"]:
        assert "EXAMPLES_LAYOUT_CSS" in (FERN / "components" / component).read_text()


def test_sparse_target_combinations_are_not_invented() -> None:
    catalog = cards()
    assert set(catalog["lora"]["data-targets"].split()) == {
        "local:sglang",
        "local:vllm",
        "kubernetes:vllm",
    }
    assert (
        "kubernetes:sglang"
        not in catalog["disaggregated-kv-routing"]["data-targets"].split()
    )
    assert set(catalog["embeddings"]["data-targets"].split()) == {
        "local:sglang",
        "local:vllm",
    }
    six_targets = {
        f"{platform}:{backend}"
        for platform in ["local", "kubernetes"]
        for backend in ["vllm", "sglang", "trtllm"]
    }
    for name in ["aggregated", "disaggregated", "aggregated-kv-routing"]:
        assert set(catalog[name]["data-targets"].split()) == six_targets


def test_embedded_assets_exist_and_remain_repo_sources() -> None:
    for page in EXAMPLES.glob("*.mdx"):
        if page.stem == "overview":
            continue
        sources = embeds(page)
        assert sources, f"example without source files: {page}"
        for source in sources:
            assert source.is_file(), f"{page}: missing {source}"
            assert source.is_relative_to(ROOT / "examples"), source


LAUNCH_HELPERS = {
    "setup_minio.sh",
    "_test_agg.sh",
    "validate_lora_agg.sh",
    "validate_omni_lora_agg.sh",
}
DEFERRED_PAGES = {"cancellation", "chat-templates", "custom-routing", "power-budget"}


def manifest_documents(source: Path) -> list[dict]:
    if source.suffix not in {".yaml", ".yml"}:
        return []
    try:
        return [
            doc
            for doc in yaml.safe_load_all(source.read_text())
            if isinstance(doc, dict)
        ]
    except yaml.YAMLError:
        # Infrastructure templates can require envsubst before parsing. They are
        # not deployments eligible for this catalog.
        return []


def is_deployment_source(source: Path) -> bool:
    if source == ROOT / "examples/power-aware-budget/dgd.yaml":
        # Explicitly documented as a contract fragment, not a complete DGD.
        return False
    if source.suffix == ".sh":
        return (
            "launch" in source.relative_to(ROOT).parts
            and source.name not in LAUNCH_HELPERS
        ) or source == ROOT / "examples/diffusers/local/run_local.sh"
    return any(
        doc.get("kind") in {"DynamoGraphDeployment", "DynamoGraphDeploymentRequest"}
        for doc in manifest_documents(source)
    )


def test_only_deployments_and_launch_scripts_are_embedded() -> None:
    for page in EXAMPLES.glob("*.mdx"):
        for source in embeds(page):
            assert is_deployment_source(source), (page, source)


def test_backend_deployments_and_launch_inventory_is_preserved() -> None:
    actual = all_embeds()
    for backend in ["vllm", "sglang", "trtllm"]:
        base = ROOT / "examples/backends" / backend
        candidates = set(base.joinpath("deploy").rglob("*.yaml"))
        candidates |= set(base.joinpath("launch").rglob("*.sh"))
        required = {
            p.resolve() for p in candidates if is_deployment_source(p.resolve())
        }
        assert required <= actual, sorted(str(p) for p in required - actual)


def test_batch_gateway_exposes_only_the_dgd() -> None:
    page = EXAMPLES / "batch-gateway.mdx"
    assert len(variants(page)) == 1
    assert embeds(page) == {
        ROOT / "examples/deployments/llm-d-batch-gateway/dynamo.yaml"
    }
    card = cards()["batch-gateway"]
    assert card["data-targets"] == "kubernetes:vllm"
    assert card["data-keywords"] == "dynamo.yaml"


def test_cloud_picker_has_only_kubernetes_deployments() -> None:
    page = EXAMPLES / "cloud-deployments.mdx"
    assert {v["target"] for v in variants(page)} == {
        "kubernetes:vllm",
        "kubernetes:sglang",
    }
    for source in embeds(page):
        assert source.is_relative_to(
            ROOT / "examples/deployments/EKS/manifests"
        ) or source.is_relative_to(ROOT / "examples/deployments/GKE")
        assert any(
            doc.get("kind") == "DynamoGraphDeployment"
            for doc in manifest_documents(source)
        )
    labels = (FERN / "components/example-options.ts").read_text()
    assert 'ecs: "Amazon ECS"' not in labels
    assert 'platform: "Platform configuration"' not in labels


def test_dgdr_backend_labels_match_the_manifest() -> None:
    page = EXAMPLES / "dgdr.mdx"
    text = page.read_text()
    blocks = dict(
        re.findall(
            r'<div className="dynamo-example-variant" data-example-variant="([^"]+)">(.*?)</div>',
            text,
            re.DOTALL,
        )
    )
    for option in variants(page):
        sources = [
            (page.parent / src).resolve() for src in EMBED.findall(blocks[option["id"]])
        ]
        for source in sources:
            (doc,) = manifest_documents(source)
            assert doc["kind"] == "DynamoGraphDeploymentRequest"
            assert option["target"] == f"kubernetes:{doc['spec']['backend']}"
    assert len(variants(page)) == len(
        list((ROOT / "examples/deployments/dgdr").glob("*.yaml"))
    )


def test_single_options_keep_the_selected_button_style() -> None:
    component = (FERN / "components/ExampleSelector.tsx").read_text()
    assert "lqs-static" not in component
    assert "choices.length === 1" not in component
    assert "aria-pressed={selected === choice.id}" in component
    assert (
        '.lqs-chip[aria-pressed="true"] { background-color: var(--lqs-stable); }'
        in component
    )
    assert "if (next === platform) return;" in component
    assert "if (next === backend) return;" in component


def test_multimodal_topology_mapping_uses_semantics_not_filenames() -> None:
    epd = embeds(EXAMPLES / "encode-prefill-decode.mdx")
    e_pd = embeds(EXAMPLES / "encode-pd.mdx")
    assert ROOT / "examples/backends/sglang/launch/multimodal_disagg.sh" in epd
    assert ROOT / "examples/backends/sglang/launch/multimodal_epd.sh" in e_pd
    assert (
        ROOT / "examples/backends/trtllm/launch/epd_multimodal_image_and_embeddings.sh"
        in epd
    )
    assert ROOT / "examples/backends/sglang/launch/multimodal_epd.sh" not in epd


def test_tracing_uses_existing_backend_flags() -> None:
    page = EXAMPLES / "tracing.mdx"
    for backend in ["sglang", "trtllm"]:
        for topology in ["agg", "disagg"]:
            source = ROOT / f"examples/backends/{backend}/launch/{topology}.sh"
            assert source in embeds(page)
            assert "--enable-otel" in source.read_text()
            assert (
                f"cd examples/backends/{backend} && bash launch/{topology}.sh --enable-otel"
                in page.read_text()
            )


def test_old_template_redirects_are_dev_scoped_and_do_not_chain() -> None:
    config = yaml.safe_load((FERN / "docs.yml").read_text())
    assert all(isinstance(css, str) for css in config["css"])
    redirects = {entry["source"]: entry["destination"] for entry in config["redirects"]}
    old_paths = [
        "cli-templates/v-llm",
        "cli-templates/sg-lang",
        "cli-templates/tensor-rt-llm",
        "cli-templates/additional-backends",
        "kubernetes-templates/dgd/v-llm",
        "kubernetes-templates/dgd/sg-lang",
        "kubernetes-templates/dgd/tensor-rt-llm",
        "kubernetes-templates/dgdr",
        "kubernetes-templates/cloud-deployments",
        "kubernetes-templates/integration-deployments",
    ]
    for old in old_paths:
        destination = redirects[f"/dynamo/dev/recipes/{old}"]
        assert destination.startswith("/dynamo/dev/recipes/examples/")
        assert destination not in redirects
        assert (EXAMPLES / f"{destination.rsplit('/', 1)[1]}.mdx").is_file()
        assert f"/dynamo/latest/recipes/{old}" not in redirects
    assert not any(
        "recipes/cli-templates/" in destination
        or "recipes/kubernetes-templates/" in destination
        for destination in redirects.values()
    )


def test_supporting_only_pages_are_deferred_without_deleting_sources() -> None:
    catalog = cards()
    redirects = {
        entry["source"]: entry["destination"]
        for entry in yaml.safe_load((FERN / "docs.yml").read_text())["redirects"]
    }
    for name in DEFERRED_PAGES:
        assert name not in catalog
        assert not (EXAMPLES / f"{name}.mdx").exists()
        assert (
            redirects[f"/dynamo/dev/recipes/examples/{name}"]
            == "/dynamo/dev/recipes/examples/overview"
        )
    for path in [
        "examples/power-aware-budget/dgd.yaml",
        "examples/custom_backend/cancellation/server.py",
        "examples/chat_templates/gemma4_tool.jinja",
        "examples/router/policy-class-queues.yaml",
        "examples/deployments/llm-d-batch-gateway/run_example.py",
        "examples/deployments/llm-d-batch-gateway/batch-gateway-values.yaml",
    ]:
        assert (ROOT / path).is_file()


def test_source_filenames_are_searchable_in_the_catalog() -> None:
    for name, metadata in cards().items():
        source_names = {source.name for source in embeds(EXAMPLES / f"{name}.mdx")}
        assert source_names == set(metadata["data-keywords"].split()), name


def test_custom_encoder_choices_name_the_actual_inference_backend() -> None:
    page = EXAMPLES / "custom-encoders.mdx"
    assert {option["target"] for option in variants(page)} == {"local:vllm"}
    for source in embeds(page):
        assert "python -m dynamo.vllm" in source.read_text()
