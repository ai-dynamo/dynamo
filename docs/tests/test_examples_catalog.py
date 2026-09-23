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
LANDING_PAGES = {"overview", "diffusion-overview"}
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
    pages = {page.stem for page in EXAMPLES.glob("*.mdx")} - LANDING_PAGES
    assert set(catalog) == pages
    nav = yaml.safe_load((FERN / "index.yml").read_text())
    paths = navigation_paths(nav)
    overview = (EXAMPLES / "overview.mdx").read_text()
    for name in pages:
        assert f"pages/recipes/examples/{name}.mdx" in paths
        assert re.search(rf"\]\({re.escape(name)}\.mdx\)", overview)
    for name in LANDING_PAGES:
        assert f"pages/recipes/examples/{name}.mdx" in paths
    assert "[Diffusion Overview](diffusion-overview.mdx)" in overview
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
        if page.stem in LANDING_PAGES:
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


def test_diffusion_section_matches_catalog_without_changing_existing_slugs() -> None:
    nav = yaml.safe_load((FERN / "index.yml").read_text())
    recipes = next(item for item in nav["navigation"] if item.get("tab") == "recipes")
    examples = next(
        item for item in recipes["layout"] if item.get("section") == "Examples"
    )
    diffusion = next(
        item for item in examples["contents"] if item.get("section") == "Diffusion"
    )
    assert diffusion["skip-slug"] is True
    expected = {
        "image-generation",
        "video-generation",
        "audio-generation",
        "text-diffusion",
        "fastvideo",
    }
    overview, *detail_pages = diffusion["contents"]
    assert overview == {
        "page": "Overview",
        "path": "pages/recipes/examples/diffusion-overview.mdx",
        "slug": "diffusion",
    }
    assert {Path(item["path"]).stem for item in detail_pages} == expected
    catalog = cards()
    assert {
        name for name, card in catalog.items() if card["data-topic"] == "diffusion"
    } == expected
    for item in detail_pages:
        assert item["slug"] == Path(item["path"]).stem
        assert catalog[item["slug"]]["data-topic-label"] == "Diffusion"
    assert catalog["audio"]["data-topic"] == "workloads"


@pytest.mark.parametrize(
    ("page", "sources"),
    [
        (
            "image-generation",
            [
                "sglang/deploy/agg_image_diffusion.yaml",
                "trtllm/deploy/agg_image_diffusion.yaml",
                "vllm/deploy/agg_omni_image.yaml",
            ],
        ),
        ("text-diffusion", ["sglang/deploy/agg_llm_diffusion.yaml"]),
        (
            "audio-generation",
            [
                "vllm/deploy/agg_omni_audio.yaml",
                "vllm/deploy/experimental/agg_omni_audio.yaml",
                "vllm/launch/agg_omni_audio.sh",
            ],
        ),
        (
            "video-generation",
            [
                "vllm/deploy/experimental/agg_omni_video.yaml",
                "vllm/deploy/experimental/agg_omni_i2v.yaml",
            ],
        ),
    ],
)
def test_diffusion_sources_are_on_the_matching_workload_page(
    page: str, sources: list[str]
) -> None:
    expected = {ROOT / "examples/backends" / source for source in sources}
    assert expected <= embeds(EXAMPLES / f"{page}.mdx")


@pytest.mark.parametrize(
    "filename", ["agg_omni_audio.yaml", "agg_omni_video.yaml", "agg_omni_i2v.yaml"]
)
def test_experimental_diffusion_preserves_image_and_cache_requirements(
    filename: str,
) -> None:
    source = ROOT / "examples/backends/vllm/deploy/experimental" / filename
    (manifest,) = manifest_documents(source)
    assert "Experimental" in source.read_text()
    assert manifest["apiVersion"] == "nvidia.com/v1beta1"
    assert "namespace" not in manifest["metadata"]
    assert manifest["spec"]["backendFramework"] == "vllm"
    components = manifest["spec"]["components"]
    frontend = next(c for c in components if c["type"] == "frontend")
    worker = next(c for c in components if c["type"] == "worker")
    for component in components:
        pod = component["podTemplate"]["spec"]
        assert pod["nodeSelector"] == {"kubernetes.io/arch": "amd64"}
    pod = worker["podTemplate"]["spec"]
    container = pod["containers"][0]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert "--enforce-eager" in container["args"]
    env = {item["name"]: item["value"] for item in container["env"]}
    (cache,) = [
        v["persistentVolumeClaim"]
        for v in pod["volumes"]
        if "persistentVolumeClaim" in v
    ]
    assert cache["claimName"] == "model-cache-pvc"
    if filename == "agg_omni_audio.yaml":
        for component in components:
            assert component["runtimeVersionOverride"] == "1.6.0"
            image = component["podTemplate"]["spec"]["containers"][0]["image"]
            assert image.endswith(":20260918-a9792db")
        assert env["HF_HOME"] == "/model-cache"
        assert not cache.get("readOnly", False)
    else:
        for component in components:
            assert component["runtimeVersionOverride"] == "1.4.2"
        assert frontend["podTemplate"]["spec"]["containers"][0]["image"] == (
            "nvcr.io/nvidia/ai-dynamo/dynamo-frontend@sha256:"
            "4d6435ad3893487e7e5a8751eb2381199464f80248284bb98c5e0998f09a91c0"
        )
        assert container["image"] == (
            "docker.io/jont828/dynamo-ffmpeg-color@sha256:"
            "1b72cc6dfbbf610ba6e1135832c9cb702f7e6ddca2d57ddfc01e7e67e066980c"
        )
        assert env["HF_HUB_OFFLINE"] == "1"
        assert cache["readOnly"] is True
        (mount,) = [
            m for m in container["volumeMounts"] if m["name"] == "shared-model-cache"
        ]
        assert mount["readOnly"] is True
        assert mount["subPath"] == "model-cache"
        assert env["HF_HUB_CACHE"] == mount["mountPath"] + "/hub"



def diffusion_cards() -> list[tuple[dict[str, str], str]]:
    text = (EXAMPLES / "diffusion-overview.mdx").read_text()
    return [
        (dict(ATTRIBUTE.findall(attributes)), body)
        for attributes, body in re.findall(
            r'<article className="dynamo-diffusion-card" ([^>]+)>(.*?)</article>',
            text,
            re.DOTALL,
        )
    ]


def test_diffusion_overview_cards_match_the_dgd_models_and_backends() -> None:
    page = EXAMPLES / "diffusion-overview.mdx"
    text = page.read_text()
    detail_pages = [
        "image-generation",
        "video-generation",
        "audio-generation",
        "text-diffusion",
    ]
    source_pages = {
        source: name
        for name in detail_pages
        for source in embeds(EXAMPLES / f"{name}.mdx")
        if source.suffix == ".yaml"
    }
    assert len(source_pages) == 8
    assert not embeds(page)
    assert "<Accordion" not in text
    assert "<CardGroup" not in text
    entries = diffusion_cards()
    assert len(entries) == len(source_pages)
    seen = set()
    models = set()
    for attributes, body in entries:
        source = ROOT / attributes["data-source"]
        assert source in source_pages
        assert source not in seen
        seen.add(source)
        (manifest,) = manifest_documents(source)
        backend = manifest["spec"]["backendFramework"]
        assert attributes["data-backend"] == backend
        assert {"sglang": "SGLang", "trtllm": "TensorRT-LLM", "vllm": "vLLM-Omni"}[
            backend
        ] in body
        worker = next(
            c for c in manifest["spec"]["components"] if c["type"] == "worker"
        )
        args = worker["podTemplate"]["spec"]["containers"][0]["args"]
        flag = "--model" if "--model" in args else "--model-path"
        model = args[args.index(flag) + 1]
        models.add(model)
        assert attributes["data-model"] == model
        assert model in body
        (href,) = re.findall(r'href="([^"]+)"', body)
        assert href == f"{source_pages[source]}.mdx"
        assert body.count("<a ") == 1
        assert "aria-label=" in body
        assert 'className="dynamo-diffusion-name"' in body
        (footer,) = re.findall(
            r'<div className="dynamo-diffusion-card-footer" aria-hidden="true">(.*?)</div>',
            body,
        )
        assert footer == "<span>↗</span>"
        (image,) = re.findall(r'<img[^>]+src="([^"]+)"', body)
        assert (page.parent / image).resolve().is_file()
        experimental = "agg_omni_audio" in source.name or "experimental" in source.parts
        assert attributes["data-experimental"] == str(experimental).lower()
        assert ('className="dynamo-diffusion-experimental"' in body) == experimental
        assert "One GPU" not in body
        assert "Dynamo 1." not in body
        assert "validated" not in body.lower()
        assert "tested" not in body.lower()
    assert len(models) == 7
    assert seen == set(source_pages)


def test_diffusion_weight_chips_match_the_recorded_checkpoint_files() -> None:
    snapshot = yaml.safe_load(
        (EXAMPLES / "_catalog/diffusion-model-sizes.yaml").read_text()
    )
    referenced = set()
    for attributes, body in diffusion_cards():
        model = attributes["data-model"]
        referenced.add(model)
        recorded = snapshot["models"][model]
        assert re.fullmatch(r"[0-9a-f]{40}", recorded["revision"])
        assert (
            recorded["source"]
            == f"https://huggingface.co/{model}/tree/{recorded['revision']}"
        )
        assert recorded["size_bytes"] == sum(recorded["files"].values())
        size = f"{recorded['size_bytes'] / 1e9:.1f}"
        assert float(size) == recorded["size_gb"]
        assert attributes["data-size-gb"] == size
        assert f"{size} GB weights" in body
        assert all(name.endswith(".safetensors") for name in recorded["files"])
        if any(name.startswith("transformer/") for name in recorded["files"]):
            assert all("/" in name for name in recorded["files"])
    assert referenced == set(snapshot["models"])
    text = (EXAMPLES / "diffusion-overview.mdx").read_text()
    assert "not GPU memory requirements" in text


def test_diffusion_catalog_is_server_styled_accessible_and_responsive() -> None:
    component = (FERN / "components/DiffusionCatalog.tsx").read_text()
    assert '"use client"' not in component
    assert "EXAMPLES_LAYOUT_CSS" in component
    assert "dangerouslySetInnerHTML" in component
    assert ".dark .dynamo-diffusion" in component
    assert ":focus-visible" in component
    assert "prefers-reduced-motion" in component
    assert "@media (max-width:" in component
    assert ".dynamo-diffusion-card[hidden] { display: none !important; }" in component
    assert "<DiffusionCatalogControls>{children}</DiffusionCatalogControls>" in component
    styles = (FERN / "components/examples-layout.ts").read_text()
    assert ".dynamo-diffusion" in styles


def test_diffusion_catalog_controls_preserve_native_mdx_cards() -> None:
    component = (FERN / "components/DiffusionCatalogControls.tsx").read_text()
    assert '"use client"' in component
    assert 'type="search"' in component
    assert 'role="status"' in component
    assert 'aria-atomic="true"' in component
    assert "useId()" in component
    for field in ("type", "backend", "provider", "size", "status"):
        assert f'key: "{field}"' in component
    assert "a.sizeGB - b.sizeGB" in component
    assert "b.sizeGB - a.sizeGB" in component
    assert 'setQuery("")' in component
    assert "setFilters(EMPTY_FILTERS)" in component
    assert 'setSort("default")' in component
    assert 'className="dynamo-diffusion-grid" ref={grid}>{children}</div>' in component
    assert "appendChild(node)" in component
    assert "fetch(" not in component


def test_diffusion_overview_keeps_local_only_cases_discoverable() -> None:
    text = (EXAMPLES / "diffusion-overview.mdx").read_text()
    for name in [
        "text-to-video-diffusion.sh",
        "agg_video_diffusion.sh",
        "disagg_omni_glm_image.sh",
        "disagg_omni_glm_image_nixl.sh",
    ]:
        assert name in text
        assert any(name == source.name for source in all_embeds())
    assert "black-forest-labs/FLUX.1-dev" in text
    assert "FastVideo/FastWan2.1-T2V-1.3B-Diffusers" in text
    assert "[Local script and two existing DGDs](fastvideo.mdx)" in text
