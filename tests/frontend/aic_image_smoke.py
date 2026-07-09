# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke the AIC payload and native engine in the shipped frontend image."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import aiconfigurator
import aiconfigurator_core
from aiconfigurator.sdk.engine import compile_engine
from aiconfigurator.sdk.memory import estimate_num_gpu_blocks
from aiconfigurator.sdk.task_v2 import Task

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.mocker import AicEngineConfig, EnginePerfLimits, RustEnginePerfModel


def _assert_installed_aic_matches_frontend_manifest() -> None:
    """Validate the candidate that installs AIC into the frontend image."""
    root = Path(__file__).resolve().parents[2]
    with (root / "benchmarks/pyproject.toml").open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]

    matches = [
        Requirement(value)
        for value in dependencies
        if Requirement(value).name == "aiconfigurator"
    ]
    assert len(matches) == 1, matches
    requirement = matches[0]
    installed_version = Version(metadata.version("aiconfigurator"))

    if not requirement.url:
        assert requirement.specifier
        assert installed_version in requirement.specifier
        return

    expected_repo, separator, expected_ref = requirement.url.removeprefix(
        "git+"
    ).rpartition("@")
    assert separator and expected_repo and expected_ref, requirement.url

    direct_url_text = metadata.distribution("aiconfigurator").read_text(
        "direct_url.json"
    )
    assert direct_url_text
    direct_url = json.loads(direct_url_text)
    assert str(direct_url["url"]).removesuffix(".git") == expected_repo.removesuffix(
        ".git"
    )
    vcs_info = direct_url["vcs_info"]
    assert vcs_info["requested_revision"] == expected_ref
    commit = str(vcs_info["commit_id"])
    assert re.fullmatch(r"[0-9a-fA-F]{40}", commit), vcs_info
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", expected_ref):
        assert commit.lower().startswith(expected_ref.lower())


def main() -> None:
    _assert_installed_aic_matches_frontend_manifest()
    assert aiconfigurator_core and compile_engine and estimate_num_gpu_blocks and Task

    package_root = Path(aiconfigurator.__file__).resolve().parent
    assert (package_root / "model_configs/Qwen--Qwen3-32B_config.json").is_file()
    assert (package_root / "systems/h200_sxm.yaml").is_file()
    parquet_files = list(
        (package_root / "systems/data/h200_sxm/vllm/0.14.0").glob("*.parquet")
    )
    assert parquet_files
    for path in parquet_files:
        with path.open("rb") as handle:
            assert handle.read(4) == b"PAR1"

    model = RustEnginePerfModel.from_native(
        aic_config=AicEngineConfig(
            model_name="Qwen/Qwen3-32B",
            backend="vllm",
            system_name="h200_sxm",
            backend_version="0.14.0",
            tp_size=1,
            attention_dp_size=1,
        ),
        worker_type="prefill",
        limits=EnginePerfLimits(
            max_num_batched_tokens=4096,
            max_num_seqs=128,
            max_kv_tokens=1_000_000,
        ),
    )
    estimate = model.estimate_forward_pass_time(
        [
            ForwardPassMetrics(
                scheduled_requests=ScheduledRequestMetrics(
                    num_prefill_requests=1,
                    sum_prefill_tokens=1024,
                )
            )
        ]
    )
    assert estimate is not None and estimate > 0.0
    diagnostics = json.loads(model.diagnostics())
    assert diagnostics["source"] == "aic"
    assert diagnostics["readiness"] == "ready"
    assert diagnostics["last_warning"] is None


if __name__ == "__main__":
    main()
