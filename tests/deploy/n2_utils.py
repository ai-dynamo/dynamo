# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Version selection and example patching for component compatibility tests."""

import re
from dataclasses import dataclass
from pathlib import Path

from tests.deploy.dgd_utils import DeploymentSpec

MODELS = {
    "chat": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "embedding": (
        "Qwen/Qwen3-Embedding-0.6B",
        "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    ),
}


@dataclass(frozen=True)
class VersionPair:
    name: str
    frontend: str
    worker: str


def version_matrix(
    releases: dict, line: str, frontend: str, worker: str
) -> list[VersionPair]:
    major, minor = map(int, line.split("."))
    if minor < 2:
        raise ValueError("N-2 requires two preceding minor versions in the same major")
    pairs = []
    for age in (1, 2):
        previous = releases[f"{major}.{minor - age}"]
        pairs.extend(
            [
                VersionPair(f"old-frontend-{age}", previous["frontend"], worker),
                VersionPair(f"old-worker-{age}", frontend, previous["worker"]),
            ]
        )
    return pairs


def runtime_version(image: str) -> str:
    match = re.search(r":(\d+\.\d+\.\d+)(?:[-.@]|$)", image)
    if match is None:
        raise ValueError(f"Image tag must start with its runtime version: {image}")
    return match[1]


def compatibility_spec(
    root: Path, pair: VersionPair, scenario: str, name: str, pvc: str, mount: str
) -> DeploymentSpec:
    if not pvc:
        raise ValueError("N-2 requires --model-cache-pvc with both models prepared")
    profile = "agg_embed" if scenario == "embedding" else "agg"
    spec = DeploymentSpec(str(root / f"examples/backends/sglang/deploy/{profile}.yaml"))
    spec.name = name
    spec.disable_grove()
    spec.spec()["metadata"].setdefault("annotations", {})[
        "nvidia.com/dynamo-discovery-backend"
    ] = "kubernetes"
    spec.mount_model_cache_pvc(pvc, mount)
    model, revision = MODELS[scenario]
    snapshot = f"{mount}/hub/models--{model.replace('/', '--')}/snapshots/{revision}"
    spec.add_arg_to_service("decode", "--model-path", snapshot)
    spec.add_arg_to_service("decode", "--served-model-name", model)
    for component in spec.spec()["spec"]["components"]:
        component["podTemplate"]["spec"].setdefault("nodeSelector", {})[
            "kubernetes.io/arch"
        ] = "amd64"

    for service, image in (("Frontend", pair.frontend), ("decode", pair.worker)):
        spec.set_image(image, service_name=service)
        spec.set_runtime_version(runtime_version(image), service)
        spec.set_service_env_var(service, "HF_HUB_OFFLINE", "1")
        spec.set_service_env_var(service, "TRANSFORMERS_OFFLINE", "1")
        spec.set_service_env_var(service, "DYN_REQUEST_PLANE", "tcp")
    # Check the mounted snapshot before starting the worker. Reading a byte also
    # catches missing blob targets behind Hugging Face snapshot symlinks.
    worker = next(c for c in spec.spec()["spec"]["components"] if c["name"] == "decode")
    worker["podTemplate"]["spec"].setdefault("initContainers", []).append(
        {
            "name": "check-model-snapshot",
            "image": pair.worker,
            "command": ["python3", "-c"],
            "args": [
                "import pathlib, sys\n"
                "p = pathlib.Path(sys.argv[1])\n"
                "for name in ('config.json', 'tokenizer.json', 'model.safetensors'):\n"
                "    print(f'Checking {p / name}', flush=True)\n"
                "    with (p / name).open('rb') as f:\n"
                "        if not f.read(1):\n"
                "            raise RuntimeError(f'Empty model file: {p / name}')\n",
                snapshot,
            ],
            "volumeMounts": [{"name": pvc, "mountPath": mount, "readOnly": True}],
        }
    )
    return spec
