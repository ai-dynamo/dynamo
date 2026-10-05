# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit legacy policy crosses configuration/binding boundaries without env writes."""

import argparse
import json
import os

import pytest

from dynamo._core import EngineType, EntrypointArgs
from dynamo.common.legacy_vllm import LegacyVllmRelease, LegacyVllmTargets
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def declaration_json():
    return json.dumps(
        [
            {
                "namespace": "legacy",
                "component": "worker",
                "endpoint": "generate",
                "model": "model-a",
                "worker_type": "aggregated",
                "dynamo_release": "1.5.0",
            }
        ]
    )


def parse_config(args):
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


@pytest.mark.parametrize("processor", ["dynamo", "vllm"])
def test_cli_legacy_target_reaches_binding_without_env_write(
    declaration_json, processor
):
    before = dict(os.environ)
    config = parse_config(
        ["--dyn-chat-processor", processor, "--legacy-vllm-targets", declaration_json]
    )
    assert config.legacy_vllm_targets == declaration_json
    targets = LegacyVllmTargets.from_json(config.legacy_vllm_targets)
    assert (
        targets.resolve(
            "legacy", "worker", "generate", "model-a", "aggregated", "tokens"
        )
        is LegacyVllmRelease.DYNAMO_15
    )
    EntrypointArgs(EngineType.Dynamic, legacy_vllm_targets=config.legacy_vllm_targets)
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "mode",
    [["--interactive"], ["--kserve-grpc-server"], ["--dyn-chat-processor", "sglang"]],
)
def test_legacy_policy_rejects_unsupported_frontend_mode(declaration_json, mode):
    with pytest.raises(ValueError, match="requires the HTTP frontend"):
        parse_config([*mode, "--legacy-vllm-targets", declaration_json])


@pytest.mark.parametrize("source", ["{}", "null", '[{"namespace":"*"}]'])
def test_cli_and_binding_reject_invalid_legacy_policy(source):
    with pytest.raises(ValueError):
        parse_config(["--legacy-vllm-targets", source])
    with pytest.raises(ValueError):
        EntrypointArgs(EngineType.Dynamic, legacy_vllm_targets=source)


def test_duplicate_policy_rejected_at_both_boundaries(declaration_json):
    source = json.dumps(json.loads(declaration_json) * 2)
    with pytest.raises(ValueError, match="duplicate"):
        parse_config(["--legacy-vllm-targets", source])
    with pytest.raises(ValueError, match="duplicate"):
        EntrypointArgs(EngineType.Dynamic, legacy_vllm_targets=source)


def test_binding_rejects_legacy_policy_for_mocker(declaration_json):
    with pytest.raises(ValueError, match="dynamic"):
        EntrypointArgs(EngineType.Mocker, legacy_vllm_targets=declaration_json)


def test_legacy_policy_is_opt_in():
    assert parse_config([]).legacy_vllm_targets is None
    EntrypointArgs(EngineType.Dynamic)
