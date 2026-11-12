# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ThunderAgent router CLI configuration."""

from __future__ import annotations

import json
import os
import sys
from types import ModuleType

import pytest
from aisimulate_core.sdk import ForwardPassPerfModelConfig

import dynamo.router.args as router_args
import dynamo.thunderagent_router.args as thunderagent_router_args
from dynamo.thunderagent_router.__main__ import _publish_sglang_generate_capability
from dynamo.thunderagent_router.args import build_ais_perf_config, parse_args

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

AIS_IDENTITY = {"model": "Qwen/Qwen3-0.6B", "system": "h200_sxm", "backend": "vllm"}
AIS_ARGS = [
    "--endpoint",
    "dynamo.prefill.generate",
    "--router-prefill-load-model",
    "ais",
]


@pytest.fixture
def ais_perf_config_arguments(monkeypatch) -> None:
    """Make ``build_ais_perf_config`` return the arguments of the native class.

    AIS inputs of the shell that runs the tests are removed too.
    """
    for name in os.environ:
        if name.startswith(("DYN_AIS_", "DYN_AIC_", "DYN_ROUTER_PREFILL_LOAD_")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(router_args, "AisPerfConfig", dict)


def test_sglang_generate_capability_is_opt_in() -> None:
    base = [
        "--endpoint",
        "dynamo.sglang.generate",
        "--model-name",
        "Qwen/Qwen3-0.6B",
    ]
    assert parse_args(base).publish_sglang_generate is False
    assert (
        parse_args([*base, "--publish-sglang-generate"]).publish_sglang_generate is True
    )
    assert (
        parse_args([*base, "--no-publish-sglang-generate"]).publish_sglang_generate
        is False
    )


def test_sglang_generate_capability_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("DYN_THUNDERAGENT_PUBLISH_SGLANG_GENERATE", "true")
    assert (
        parse_args(
            [
                "--endpoint",
                "dynamo.sglang.generate",
                "--model-name",
                "Qwen/Qwen3-0.6B",
            ]
        ).publish_sglang_generate
        is True
    )


def test_sglang_generate_capability_requires_model_name() -> None:
    with pytest.raises(
        ValueError, match="--publish-sglang-generate requires --model-name"
    ):
        parse_args(
            [
                "--endpoint",
                "dynamo.sglang.generate",
                "--publish-sglang-generate",
            ]
        )


def test_publish_sglang_generate_capability_uses_backend_contract(monkeypatch) -> None:
    module = ModuleType("dynamo.sglang.engine_generate")
    module.SGLANG_GENERATE_CAPABILITY = "sglang_generate"
    monkeypatch.setitem(sys.modules, module.__name__, module)

    published: list[tuple[str, str]] = []

    class RuntimeConfig:
        def set_engine_specific(self, key: str, value: str) -> None:
            published.append((key, value))

    runtime_config = RuntimeConfig()

    _publish_sglang_generate_capability(runtime_config)  # type: ignore[arg-type]
    assert published == [("sglang_generate", "true")]


def test_new_flag_reaches_thunderagent_config(ais_perf_config_arguments) -> None:
    value = {**AIS_IDENTITY, "worker_type": "prefill"}

    config = parse_args([*AIS_ARGS, "--ais-perf-model", json.dumps(value)])

    assert build_ais_perf_config(config) == {
        "config": ForwardPassPerfModelConfig(**value).to_dict()
    }


def test_start_warns_once_for_old_ais_inputs(ais_perf_config_arguments) -> None:
    value = {**AIS_IDENTITY, "worker_type": "prefill"}

    # The router reads the config when it parses and again when it starts.
    with pytest.warns(FutureWarning) as records:
        config = parse_args([*AIS_ARGS, "--ais-perf-config", json.dumps(value)])
        built = build_ais_perf_config(config)

    assert [str(record.message) for record in records] == [
        "--ais-perf-config is deprecated and will be removed in v1.9; "
        "use --ais-perf-model."
    ]
    assert records[0].filename == thunderagent_router_args.__file__
    assert built == {"config": ForwardPassPerfModelConfig(**value).to_dict()}
