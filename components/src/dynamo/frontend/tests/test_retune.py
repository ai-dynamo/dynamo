# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

import pytest

from dynamo.common.configuration.groups.router_args import build_router_config
from dynamo.common.configuration.utils import ArgEnv
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.llm import EngineType, EntrypointArgs, KvRouterConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(autouse=True)
def builtin_default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DYN_ROUTER_WORKER_SELECTION_POLICY", raising=False)


@pytest.fixture
def capability() -> dict:
    return {
        "version": 1,
        "configHash": "a" * 64,
        "frontendComponentName": "gateway",
        "adminPort": 8002,
        "traceStorage": {
            "claimName": "traces",
            "mountPath": "/trace",
            "segmentsDir": "segments",
            "windowsDir": "windows",
        },
        "router": {
            "profile": "kv-v1",
            "overlapScoreCredit": 2.0,
            "prefillLoadScale": 0.5,
            "temperature": 0.0,
        },
    }


def _startup(capability: dict, argv: list[str], env: dict[str, str]) -> EntrypointArgs:
    value = json.dumps(capability)
    parser = argparse.ArgumentParser()
    FrontendArgGroup(
        env=ArgEnv({**env, "DYN_FRONTEND_RETUNE_CAPABILITY": value})
    ).add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return EntrypointArgs(
        EngineType.Dynamic,
        router_config=build_router_config(config),
        frontend_retune_capability=config.retune_capability,
    )


def test_profile_conformance_survives_full_replay_serialization() -> None:
    fixture_path = Path(__file__).parents[5] / "lib/llm/tests/data/retune-kv-v1.json"
    with fixture_path.open() as source:
        fixture = json.load(source)
    profile = KvRouterConfig.retune_profile("kv-v1", 1.0, 1.0, 0.0)
    assert json.loads(profile.to_json()) == fixture["router_config"]
    assert profile.frontend_kwargs() == fixture["frontend_kwargs"]
    replay = KvRouterConfig.from_json(json.dumps(profile.frontend_kwargs()))
    assert replay.frontend_kwargs() == profile.frontend_kwargs()
    for role, parameters in fixture["policy_parameters"].items():
        assert replay.policy_parameters(role) == parameters


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        ([], {}),
        (["--router-mode", "kv", "--router-kv-overlap-score-credit", "2.0"], {}),
        ([], {"DYN_ROUTER_MODE": "kv", "DYN_ROUTER_TEMPERATURE": "0.0"}),
    ],
)
def test_equivalent_launch_settings_are_accepted(
    capability: dict, argv: list[str], env: dict[str, str]
) -> None:
    _startup(capability, argv, env)


@pytest.mark.parametrize(
    ("argv", "env"),
    [
        (["--router-mode", "round-robin"], {}),
        (["--router-kv-overlap-score-credit", "1.0"], {}),
        ([], {"DYN_ROUTER_TEMPERATURE": "1.0"}),
        (["--no-router-replica-sync"], {}),
    ],
)
def test_conflicting_actual_launch_settings_are_rejected(
    capability: dict, argv: list[str], env: dict[str, str]
) -> None:
    with pytest.raises(ValueError, match="Retune|retune"):
        _startup(capability, argv, env)


def test_resolved_policy_file_is_validated_semantically(
    capability: dict, tmp_path: Path
) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text("router:\n  router_replica_sync: true\n")
    _startup(capability, ["--router-policy-config", str(policy)], {})
    policy.write_text("router:\n  router_replica_sync: false\n")
    with pytest.raises(ValueError, match="Retune|retune"):
        _startup(capability, ["--router-policy-config", str(policy)], {})


def test_unsupported_contract_version_does_not_fall_back(capability: dict) -> None:
    capability["version"] = 2
    with pytest.raises(ValueError, match="unsupported.*Retune"):
        _startup(capability, [], {})
