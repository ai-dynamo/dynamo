# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the DeploymentCustomizer manifest engine against every embedded example manifest.

The engine (docs/fern/components/deployment-customizer.ts) is import-free TypeScript,
so Node runs it directly with type stripping (Node 22.6 or later). The test skips when
no such Node is available. PyYAML checks the results, so the engine's own parser is
never the judge of its output.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

ROOT = Path(__file__).resolve().parents[2]
FERN = ROOT / "docs/fern"
ENGINE = FERN / "components/deployment-customizer.ts"
EXAMPLES = FERN / "pages/recipes/examples"
EMBED = re.compile(r'<Code\s+src="([^"]+\.ya?ml)"')
MOUNT = "/home/dynamo/.cache/huggingface"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REGISTRY = "registry.example.com/team"
PUBLISHED = [
    "vllm-runtime",
    "sglang-runtime",
    "tensorrtllm-runtime",
    "dynamo-frontend",
    "dynamo-planner",
    "kubernetes-operator",
    "snapshot-agent",
]

# Customizes each manifest three ways and prints the results as JSON.
RUNNER = r"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [enginePath, inputPath] = process.argv.slice(2);
const engine = await import(pathToFileURL(enginePath).href);
const input = JSON.parse(readFileSync(inputPath, "utf8"));

function settings(info, fileName) {
  return {
    fileName,
    model: null,
    image: { defaultRegistry: "nvcr.io/nvidia/ai-dynamo", publishedRepos: input.published, registry: null, tagMode: "keep", tag: "1.4.2" },
    names: {},
    components: {},
    env: engine.initialEnvRows(info),
    storage: { mode: "download", pvcName: "model-cache", size: "100Gi", storageClass: "", accessMode: "ReadWriteMany", mountPath: input.mount, createPvc: true, downloadJob: true },
    secret: { mode: "existing", name: info.secret ?? "hf-token-secret" },
    templateVars: {},
    namespace: "dynamo-system",
    bundle: false,
    chatModel: true,
  };
}

const results = [];
for (const path of input.sources) {
  const source = readFileSync(path, "utf8");
  const fileName = path.split("/").pop();
  const info = engine.analyzeManifest(source);
  const identity = engine.customizeManifest(source, info, settings(info, fileName)).manifest;

  const heavy = settings(info, fileName);
  heavy.model = input.model;
  heavy.image = { ...heavy.image, registry: input.registry, tagMode: "release", tag: "1.4.1" };
  for (const c of info.components) {
    heavy.components[c.key] = {
      replicas: (c.replicas ?? 1) + 1,
      ...(c.gpus !== null ? { gpus: c.gpus + 1 } : {}),
      ...(c.nodes !== null ? { nodes: c.nodes + 1 } : {}),
    };
  }
  const rows = engine.initialEnvRows(info);
  const removed = rows.find((row) => !row.from) ?? null;
  heavy.env = rows.filter((row) => row !== removed);
  const scope = info.envScopes[0];
  if (scope) heavy.env.push({ id: "added", scope: scope.id, name: "ADDED_VAR", value: "1", from: null, orig: null });
  heavy.storage = { ...heavy.storage, mode: "pvc" };
  heavy.secret = { mode: "kubectl", name: "renamed-secret" };
  const heavyResult = engine.customizeManifest(source, info, heavy);

  const none = settings(info, fileName);
  none.secret = { mode: "none", name: info.secret ?? "hf-token-secret" };

  const lines = engine.splitLines(source).lines;
  const highlighted = engine.highlightYaml(source);
  const highlightExact = highlighted.length === lines.length && highlighted.every((tokens, i) => tokens.map((t) => t.t).join("") === lines[i]);

  results.push({
    path,
    info,
    identity,
    highlightExact,
    removed: removed && { scope: removed.scope, name: removed.name },
    heavy: Object.fromEntries(heavyResult.files.map((file) => [file.id, file.content])),
    none: engine.customizeManifest(source, info, none).manifest,
  });
}
process.stdout.write(JSON.stringify(results));
"""


def node_command() -> list[str] | None:
    node = shutil.which("node")
    if not node:
        return None
    probe = subprocess.run(
        [
            node,
            "--experimental-strip-types",
            "-e",
            "process.exit(process.features.typescript ? 0 : 1)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        [node, "--experimental-strip-types", "--no-warnings"]
        if probe.returncode == 0
        else None
    )


def embedded_manifests() -> list[Path]:
    sources = {
        (page.parent / src).resolve()
        for page in EXAMPLES.glob("*.mdx")
        for src in EMBED.findall(page.read_text())
    }
    assert sources, "no embedded YAML manifests found"
    return sorted(sources)


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    command = node_command()
    if command is None:
        pytest.skip("needs Node with TypeScript type stripping (22.6 or later)")
    work = tmp_path_factory.mktemp("customizer")
    runner = work / "runner.mjs"
    runner.write_text(RUNNER)
    payload = work / "input.json"
    payload.write_text(
        json.dumps(
            {
                "sources": [str(path) for path in embedded_manifests()],
                "published": PUBLISHED,
                "mount": MOUNT,
                "model": MODEL,
                "registry": REGISTRY,
            }
        )
    )
    completed = subprocess.run(
        [*command, str(runner), str(ENGINE), str(payload)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def documents(text: str) -> list[dict]:
    return [doc for doc in yaml.safe_load_all(text) if doc is not None]


def walk(node: object):
    yield node
    if isinstance(node, dict):
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


def main_container(component: dict) -> dict:
    containers = component["podTemplate"]["spec"]["containers"]
    return next((c for c in containers if c.get("name") == "main"), containers[0])


def scope_env(docs: list[dict], scope: str) -> list[dict]:
    kind, doc, *rest = scope.split(":")
    spec = docs[int(doc)]["spec"]
    if kind == "g":
        return spec.get("env") or []
    return main_container(spec["components"][int(rest[0])]).get("env") or []


def test_engine_has_no_imports() -> None:
    # Node runs the engine without a bundler, so it must stay self-contained.
    assert not re.search(r"^\s*import\s", ENGINE.read_text(), re.MULTILINE)


def test_untouched_settings_reproduce_each_manifest(results: list[dict]) -> None:
    for result in results:
        source = Path(result["path"]).read_text()
        expected = source if source.endswith("\n") else source + "\n"
        assert result["identity"] == expected, result["path"]
        assert result["highlightExact"], result["path"]


def test_customized_manifests_parse_and_carry_every_edit(results: list[dict]) -> None:
    for result in results:
        path, info = result["path"], result["info"]
        original = documents(Path(path).read_text())
        docs = documents(result["heavy"]["manifest"])
        assert len(docs) == len(original), path

        for component in info["components"]:
            spec = docs[component["doc"]]["spec"]["components"][component["index"]]
            assert spec["replicas"] == (component["replicas"] or 1) + 1, (
                path,
                component["name"],
            )
            if component["gpus"] is not None:
                limits = main_container(spec)["resources"].get("limits") or {}
                if "nvidia.com/gpu" in limits:
                    assert str(limits["nvidia.com/gpu"]) == str(
                        component["gpus"] + 1
                    ), path
            if component["nodes"] is not None:
                assert spec["multinode"]["nodeCount"] == component["nodes"] + 1, path

        if result["removed"]:
            removed = result["removed"]
            before = [
                e
                for e in scope_env(original, removed["scope"])
                if e.get("name") == removed["name"]
            ]
            after = [
                e
                for e in scope_env(docs, removed["scope"])
                if e.get("name") == removed["name"]
            ]
            assert len(after) == len(before) - 1 or removed["name"] == "HF_HOME", (
                path,
                removed,
            )
        if info["envScopes"]:
            added = [
                e
                for e in scope_env(docs, info["envScopes"][0]["id"])
                if e.get("name") == "ADDED_VAR"
            ]
            assert added and added[0]["value"] == "1", path

        for node in walk(docs):
            if isinstance(node, dict) and isinstance(node.get("image"), str):
                image = node["image"]
                assert (
                    not image.startswith(("nvcr.io/nvidia/ai-dynamo/", "my-registry/"))
                    or "@" in image
                ), (path, image)

        if info["secret"]:
            for node in walk(docs):
                if isinstance(node, dict) and isinstance(node.get("secretRef"), dict):
                    assert node["secretRef"]["name"] != info["secret"], path
            for node in walk(documents(result["none"])):
                if isinstance(node, dict):
                    for ref in ("secretRef", "secretKeyRef"):
                        assert (node.get(ref) or {}).get("name") != info["secret"], path
                    assert (
                        node.get("envFrom", []) is not None
                        and node.get("env", []) is not None
                    ), path

        if info["models"]:
            text = result["heavy"]["manifest"]
            assert MODEL in text, path
            assert not re.search(
                rf"(?<![\w./-]){re.escape(info['models'][0])}(?![\w./-])", text
            ), path


def test_model_cache_mounts_match_what_workers_read(results: list[dict]) -> None:
    for result in results:
        path, info = result["path"], result["info"]
        docs = documents(result["heavy"]["manifest"])
        if not info["modelPvc"]:
            for component in info["components"]:
                if not component["loadsModel"]:
                    continue
                spec = docs[component["doc"]]["spec"]["components"][component["index"]]
                main = main_container(spec)
                assert any(
                    m["mountPath"] == MOUNT for m in main.get("volumeMounts") or []
                ), path
                assert {"name": "HF_HOME", "value": MOUNT} in (
                    main.get("env") or []
                ), path
                claims = [
                    (v.get("persistentVolumeClaim") or {}).get("claimName")
                    for v in spec["podTemplate"]["spec"].get("volumes") or []
                ]
                assert "model-cache" in claims, path

        cache = result["heavy"].get("model-cache")
        if not info["models"]:
            assert cache is None or "kind: Job" not in cache, path
            continue
        jobs = [doc for doc in documents(cache) if doc["kind"] == "Job"]
        assert len(jobs) == 1, path
        container = jobs[0]["spec"]["template"]["spec"]["containers"][0]
        env = {e["name"]: e["value"] for e in container["env"]}
        mount = container["volumeMounts"][0]
        assert env["MODEL_NAME"] == MODEL, path
        if info["modelPvc"]:
            layout = info["modelPvc"]
            assert mount["mountPath"] == layout["mountPath"], path
            assert mount.get("subPath") == layout["subPath"], path
            assert all(env.get(name) == value for name, value in layout["env"]), path
        else:
            assert mount["mountPath"] == MOUNT and env["HF_HOME"] == MOUNT, path
        pvcs = [
            doc for doc in documents(cache) if doc["kind"] == "PersistentVolumeClaim"
        ]
        claim = (info["modelPvc"] or {}).get("claim") or "model-cache"
        assert len(pvcs) == (0 if claim in info["claims"] else 1), path


def test_deploy_steps_follow_the_generated_files(results: list[dict]) -> None:
    for result in results:
        steps = result["heavy"]["steps"]
        name = Path(result["path"]).name
        assert steps.startswith("export NAMESPACE=dynamo-system\n"), name
        assert f"-f {name}" in steps or f"< {name}" in steps, name
        assert (
            "kubectl create secret generic renamed-secret" in steps
            or not result["info"]["secret"]
        ), name
        if "model-cache" in result["heavy"]:
            assert "-f model-cache.yaml" in steps, name
        # Bare placeholders such as <value> would be shell redirections.
        assert not re.search(r"=<[^>\"']+>", steps), name
