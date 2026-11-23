# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dynamo.aisimulate.tests.dgd import cli

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [
        (Path("/workspace/config.yaml"), "config.yaml"),
        (Path("/tmp/config.yaml"), "/tmp/config.yaml"),
    ],
)
def test_run_dgd_recommend_uses_workspace_relative_config(
    monkeypatch, config_path: Path, expected: str
) -> None:
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli, "repository_root", lambda: Path("/workspace"))
    monkeypatch.setattr(cli.subprocess, "run", run)

    command, result = cli.run_dgd_recommend(config_path, Path("/tmp/output"))

    assert result is not None
    assert command[command.index("--config") + 1] == expected
    assert observed["command"] == command
    assert observed["kwargs"]["cwd"] == Path("/workspace")


def test_assert_dgd_outputs_allows_omitted_block_size(
    monkeypatch, tmp_path: Path
) -> None:
    config = {
        "deployment_mode": "agg",
        "model_name": "example/model",
        "context_length": 4096,
        "agg_replicas": 1,
        "agg_tp": 1,
        "agg_pp": 1,
        "agg_attention_dp": 1,
        "agg_max_num_seqs": 8,
        "agg_max_num_batched_tokens": 4096,
        "agg_num_gpu_blocks": 128,
        "agg_enable_prefix_caching": True,
    }
    monkeypatch.setattr(
        cli.SweepResult,
        "model_validate_json",
        lambda _: SimpleNamespace(
            views=SimpleNamespace(pareto_front=[]),
            selected_candidates=[SimpleNamespace(config=config, used_gpus=1)],
        ),
    )
    (tmp_path / "recommendation.json").write_text("{}")
    (tmp_path / "index.json").write_text(
        json.dumps(
            {
                "renderer": "aic",
                "output": "dgd",
                "artifacts": [{"path": "deployment.yaml"}],
            }
        )
    )
    args = [
        "--model",
        "example/model",
        "--tensor-parallel-size",
        "1",
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        "4096",
        "--num-gpu-blocks-override",
        "128",
    ]
    (tmp_path / "deployment.yaml").write_text(
        yaml.safe_dump(
            {
                "apiVersion": "nvidia.com/v1beta1",
                "kind": "DynamoGraphDeployment",
                "metadata": {"name": "example"},
                "spec": {
                    "components": [
                        {
                            "name": "Worker",
                            "type": "worker",
                            "replicas": 1,
                            "podTemplate": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "main",
                                            "image": "example:latest",
                                            "args": args,
                                            "resources": {
                                                "limits": {"nvidia.com/gpu": "1"}
                                            },
                                        }
                                    ]
                                }
                            },
                        }
                    ]
                },
            },
            sort_keys=False,
        )
    )

    cli.assert_dgd_outputs(
        case_name="example",
        input_config={"engine": {"workers": {"aggregated": {}}}},
        options={
            "name": "example",
            "output_file": "deployment.yaml",
            "runtime_image": "example:latest",
        },
        output_dir=tmp_path,
    )
