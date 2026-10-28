# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the rendered nightly cache-consumer contract for both Kimi topologies."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.integration]


@pytest.mark.parametrize(
    "variant,workers", [("agg", "agg"), ("disagg", "prefill,decode")]
)
def test_nightly_consumers_share_pinned_read_only_snapshots(tmp_path, variant, workers):
    renderer = os.environ.get("KUSTOMIZE_BIN")
    command = [renderer, "build"] if renderer else ["kubectl", "kustomize"]
    rendered = subprocess.run(
        command
        + [
            f".github/ci/kimi-k25-{variant}",
            "--load-restrictor",
            "LoadRestrictionsNone",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    original = list(yaml.safe_load_all(rendered))
    manifest = tmp_path / "rendered.yaml"
    manifest.write_text(rendered)
    dgd_name = f"kimi-k25-{variant}-kv-eagle"
    benchmark_name = dgd_name + "-bench"
    subprocess.run(
        ["bash", ".github/ci/prepare-kimi-cache.sh", str(manifest)],
        cwd=ROOT,
        env={
            **os.environ,
            "DGD": dgd_name,
            "BENCHMARK": benchmark_name,
            "WORKERS": workers,
        },
        check=True,
        timeout=30,
    )
    documents = list(yaml.safe_load_all(manifest.read_text()))
    assert len(documents) == len(original) + 1
    resources = {doc["metadata"]["name"]: doc for doc in documents}
    benchmark = resources[benchmark_name]["spec"]["template"]["spec"]["containers"][0]
    env = {entry["name"]: entry.get("value") for entry in benchmark["env"]}
    model_path = env["TOKENIZER_PATH"]
    assert "/snapshots/" in model_path
    assert env["TARGET_MODEL"] == "nvidia/Kimi-K2.5-NVFP4"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_MODULES_CACHE"].startswith("/tmp/")
    assert benchmark["command"][2].count("${TOKENIZER_PATH:-${TARGET_MODEL}}") == 2
    assert next(m for m in benchmark["volumeMounts"] if m["name"] == "model-cache")[
        "readOnly"
    ]

    dgd = resources[dgd_name]
    for worker in workers.split(","):
        container = dgd["spec"]["services"][worker]["extraPodSpec"]["mainContainer"]
        args = container["args"]
        assert args[args.index("--model-path") + 1] == model_path
        assert args[args.index("--served-model-name") + 1] == env["TARGET_MODEL"]
        assert next(m for m in container["volumeMounts"] if m["name"] == "model-cache")[
            "readOnly"
        ]

    preflight = resources[dgd_name + "-cache-preflight"]["spec"]
    pod = preflight["template"]["spec"]
    checker = pod["containers"][0]
    assert checker["command"][-2] == model_path
    eagle_path = checker["command"][-1]
    assert "/snapshots/" in eagle_path
    source_pod = dgd["spec"]["services"][workers.split(",")[0]]["extraPodSpec"]
    assert checker["image"] == source_pod["mainContainer"]["image"]
    assert pod["affinity"]["nodeAffinity"] == source_pod["affinity"]["nodeAffinity"]
    assert checker["volumeMounts"][0]["readOnly"]
    assert "gpu" not in str(checker["resources"])
    assert preflight["backoffLimit"] == 0
    assert preflight["activeDeadlineSeconds"] <= 600

    original_config = next(doc for doc in original if doc["kind"] == "ConfigMap")
    config = resources[original_config["metadata"]["name"]]
    for name, engine_yaml in config["data"].items():
        engine = yaml.safe_load(engine_yaml)
        before = yaml.safe_load(original_config["data"][name])
        assert engine["speculative_config"]["speculative_model_dir"] == eagle_path
        before["speculative_config"]["speculative_model_dir"] = eagle_path
        assert engine == before
