# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml
from aisim.sweeper.result import SweepResult

from dynamo.aisimulate.tests.dgd.golden import assert_yaml_golden, repository_root

_YAML_SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & "
    "AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n\n"
)


def run_dgd_recommend(
    config_path: Path,
    output_dir: Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[list[str], subprocess.CompletedProcess[str]]:
    root = repository_root()
    try:
        config_argument = config_path.relative_to(root)
    except ValueError:
        config_argument = config_path
    command = [
        sys.executable,
        "-m",
        "aisim",
        "recommend",
        "--stack",
        "dynamo",
        "--config",
        str(config_argument),
        "--output-dir",
        str(output_dir),
        "--format",
        "json",
        "--output",
        "dgd",
    ]
    return command, subprocess.run(
        command,
        cwd=root,
        env={**os.environ, **(environment or {})},
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


def assert_recommend_succeeded(
    command: list[str], result: subprocess.CompletedProcess[str]
) -> None:
    assert result.returncode == 0, (
        f"{' '.join(command)} failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert json.loads(result.stdout)
    assert "router AIC payload requires" not in result.stderr


def assert_dgd_outputs(
    *,
    case_name: str,
    input_config: dict[str, Any],
    options: dict[str, Any],
    output_dir: Path,
    golden_path: Path | None = None,
) -> None:
    result = SweepResult.model_validate_json(
        (output_dir / "recommendation.json").read_text()
    )
    pareto = bool(result.views.pareto_front)
    selected = result.selected_candidates if pareto else result.selected_candidates[:1]
    assert selected, "recommendation produced no selected Candidates"
    output_format = options.get("format", "manifest")
    target = options["output_file" if output_format == "manifest" else "output_dir"]
    paths = [target.format(index=f"{index:03d}") for index in range(len(selected))]
    assert json.loads((output_dir / "index.json").read_text()) == {
        "renderer": options.get("generator", "aic"),
        "output": "dgd" if output_format == "manifest" else "kustomize",
        "artifacts": [{"path": path} for path in paths],
    }

    manifests = []
    for destination in paths:
        artifact = output_dir / destination
        if output_format == "kustomize":
            assert yaml.safe_load((artifact / "kustomization.yaml").read_text()) == {
                "apiVersion": "kustomize.config.k8s.io/v1beta1",
                "kind": "Kustomization",
                "resources": ["deploy.yaml"],
            }
            artifact /= "deploy.yaml"
        manifests.append(yaml.safe_load(artifact.read_text()))

    if golden_path is not None:
        assert_yaml_golden(
            golden_path,
            _YAML_SPDX_HEADER + yaml.safe_dump_all(manifests, sort_keys=False),
        )

    for index, (candidate, dgd) in enumerate(zip(selected, manifests, strict=True)):
        assert dgd["apiVersion"] == "nvidia.com/v1beta1"
        assert dgd["kind"] == "DynamoGraphDeployment"
        assert dgd["metadata"]["name"] == options["name"].format(index=f"{index:03d}")
        if options.get("namespace"):
            assert dgd["metadata"]["namespace"] == options["namespace"]
        else:
            assert "namespace" not in dgd["metadata"]

        config = candidate.config
        roles = (
            {"worker": "agg"}
            if config["deployment_mode"] == "agg"
            else {"prefill": "prefill", "decode": "decode"}
        )
        workers = {
            component["type"]: component
            for component in dgd["spec"]["components"]
            if component["type"] in {"worker", "prefill", "decode"}
        }
        assert workers.keys() == roles.keys()
        router = config.get("adapters", {}).get("dynamo.router")
        if case_name == "router-kustomize":
            assert router is not None
        if router is not None:
            assert (
                sum(
                    component["type"] == "frontend"
                    for component in dgd["spec"]["components"]
                )
                == 1
            )
        used_gpus = 0
        for component in dgd["spec"]["components"]:
            if "runtime_version_override" in options:
                assert (
                    component["runtimeVersionOverride"]
                    == options["runtime_version_override"]
                )
            else:
                assert "runtimeVersionOverride" not in component
            container = next(
                container
                for container in component["podTemplate"]["spec"]["containers"]
                if container["name"] == "main"
            )
            assert container["image"] == options["runtime_image"]
            if component["type"] == "frontend" and router is not None:
                env = {entry["name"]: entry["value"] for entry in container["env"]}
                assert (
                    env["DYN_ROUTER_MODE"]
                    == {
                        "round_robin": "round-robin",
                        "kv_router": "kv",
                    }[router["policy"]]
                )
                if router["policy"] == "kv_router":
                    assert (
                        env["DYN_ROUTER_PREFILL_LOAD_MODEL"]
                        == router["prefill_load_model"]["type"]
                    )
                    for field, env_name in (
                        ("temperature", "DYN_ROUTER_TEMPERATURE"),
                        ("overlap_score_credit", "DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT"),
                        ("prefill_load_scale", "DYN_ROUTER_PREFILL_LOAD_SCALE"),
                    ):
                        assert float(env[env_name]) == pytest.approx(
                            router[field]
                        ), env_name
            if component["type"] not in roles:
                continue
            role = roles[component["type"]]
            prefix = "" if role == "agg" else f"{role}_"
            assert component["replicas"] == config[f"{prefix}replicas"]
            worker_config = input_config["engine"]["workers"][
                "aggregated" if role == "agg" else role
            ]
            block_size = worker_config["kv_cache"].get("block_size")
            assert config.get(f"{role}_block_size") == block_size
            args = container["args"]
            for flag, expected in {
                "--model": config["model_name"],
                "--tensor-parallel-size": config[f"{prefix}tp"],
                "--pipeline-parallel-size": config[f"{prefix}pp"],
                "--data-parallel-size": config[f"{prefix}attention_dp"],
                "--max-model-len": config["context_length"],
                "--max-num-seqs": config[f"{role}_max_num_seqs"],
                "--max-num-batched-tokens": config[f"{role}_max_num_batched_tokens"],
                "--block-size": block_size,
                "--num-gpu-blocks-override": config[f"{role}_num_gpu_blocks"],
            }.items():
                if expected is None:
                    assert flag not in args
                else:
                    assert args.count(flag) == 1
                    assert args[args.index(flag) + 1] == str(expected)
            assert ("--no-enable-prefix-caching" in args) != config[
                f"{role}_enable_prefix_caching"
            ]
            used_gpus += component["replicas"] * int(
                container["resources"]["limits"]["nvidia.com/gpu"]
            )
        assert used_gpus == candidate.used_gpus

    if case_name == "pareto":
        assert len(manifests) >= 2
        assert (
            len(
                {json.dumps(manifest["spec"], sort_keys=True) for manifest in manifests}
            )
            >= 2
        )
