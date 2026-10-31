# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path
from typing import Any

import pytest
import yaml
from aisimulate.sweeper.result import SweepResult

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.timeout(240),
]

_E2E_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[7]
_CASES = tuple(sorted((_E2E_ROOT / "configs").glob("*.yaml")))
assert _CASES, f"no Recommend-to-DGD cases found under {_E2E_ROOT / 'configs'}"
_DGD_GOLDEN_CASES = {"aggregated", "disaggregated"}


def _assert_dgd_golden(golden: Path, manifests: list[dict[str, Any]]) -> None:
    expected = golden.read_text() if golden.exists() else None
    if expected is not None and list(yaml.safe_load_all(expected)) == manifests:
        return

    proposal = golden.with_name(f"{golden.stem}.new{golden.suffix}")
    proposal.parent.mkdir(parents=True, exist_ok=True)
    proposal.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
        + yaml.safe_dump_all(manifests, sort_keys=False)
    )
    paths = [golden, proposal]
    golden_arg, proposal_arg = (
        shlex.quote(
            str(
                path.relative_to(_REPO_ROOT)
                if path.is_relative_to(_REPO_ROOT)
                else path
            )
        )
        for path in paths
    )
    review_source = golden_arg if expected is not None else "/dev/null"
    diff = "\n".join(
        unified_diff(
            (expected or "").splitlines(),
            proposal.read_text().splitlines(),
            fromfile=review_source,
            tofile=proposal_arg,
            lineterm="",
        )
    )
    pytest.fail(
        f"DGD golden is missing or differs: {golden}\n"
        f"Actual CLI output was written to {proposal}; the golden was not changed.\n"
        f"Review this proposal: diff -u {review_source} {proposal_arg}\n"
        f"Accept only after reviewing this file: mv {proposal_arg} {golden_arg}\n"
        f"Then rerun this case.\n\n{diff}",
        pytrace=False,
    )


def _assert_dgd_outputs(
    config_path: Path, options: dict[str, Any], output: Path
) -> None:
    result = SweepResult.model_validate_json(
        (output / "recommendation.json").read_text()
    )
    pareto = bool(result.views.pareto_front)
    selected = result.selected_candidates if pareto else result.selected_candidates[:1]
    assert selected, "recommendation produced no selected Candidates"
    output_format = options.get("format", "manifest")
    target = options["output_file" if output_format == "manifest" else "output_dir"]
    paths = [target.format(index=f"{index:03d}") for index in range(len(selected))]
    assert json.loads((output / "index.json").read_text()) == {
        "renderer": options.get("generator", "aic"),
        "output": "dgd" if output_format == "manifest" else "kustomize",
        "artifacts": [{"path": path} for path in paths],
    }

    manifests = []
    for destination in paths:
        artifact = output / destination
        if output_format == "kustomize":
            assert yaml.safe_load((artifact / "kustomization.yaml").read_text()) == {
                "apiVersion": "kustomize.config.k8s.io/v1beta1",
                "kind": "Kustomization",
                "resources": ["deploy.yaml"],
            }
            artifact /= "deploy.yaml"
        manifests.append(yaml.safe_load(artifact.read_text()))

    if config_path.stem in _DGD_GOLDEN_CASES:
        golden = _E2E_ROOT / "goldens" / config_path.name
        _assert_dgd_golden(golden, manifests)

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
        if config_path.stem == "router-kustomize":
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
                    # "none" keeps static load accounting and its sampled scale.
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
            args = container["args"]
            for flag, expected in {
                "--model": config["model_name"],
                "--tensor-parallel-size": config[f"{prefix}tp"],
                "--pipeline-parallel-size": config[f"{prefix}pp"],
                "--data-parallel-size": config[f"{prefix}attention_dp"],
                "--max-model-len": config["context_length"],
                "--max-num-seqs": config[f"{role}_max_num_seqs"],
                "--max-num-batched-tokens": config[f"{role}_max_num_batched_tokens"],
                "--block-size": config[f"{role}_block_size"],
                "--num-gpu-blocks-override": config[f"{role}_num_gpu_blocks"],
            }.items():
                assert args.count(flag) == 1
                assert args[args.index(flag) + 1] == str(expected)
            assert ("--no-enable-prefix-caching" in args) != config[
                f"{role}_enable_prefix_caching"
            ]
            used_gpus += component["replicas"] * int(
                container["resources"]["limits"]["nvidia.com/gpu"]
            )
        assert used_gpus == candidate.used_gpus

    if config_path.stem == "pareto":
        assert len(manifests) >= 2
        assert (
            len(
                {json.dumps(manifest["spec"], sort_keys=True) for manifest in manifests}
            )
            >= 2
        )


@pytest.mark.parametrize("config_path", _CASES, ids=lambda path: path.stem)
def test_dgd_cli_cases(config_path: Path, tmp_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text())
    output = tmp_path / config_path.stem
    command = [
        sys.executable,
        "-m",
        "aisimulate",
        "recommend",
        "--stack",
        "dynamo",
        "--config",
        str(config_path.relative_to(_REPO_ROOT)),
        "--output-dir",
        str(output),
        "--format",
        "json",
        "--output",
        "dgd",
    ]
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"{' '.join(command)} failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert json.loads(result.stdout)
    assert "router AIC payload requires" not in result.stderr
    _assert_dgd_outputs(config_path, config["dgd"], output)
