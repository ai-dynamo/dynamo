# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest
import yaml
from autoscaling_arena.match_config import (
    MatchConfigError,
    feasible_worker_targets,
    load_match_config,
)
from autoscaling_arena.match_runner import (
    MatchPublishError,
    preflight_publish_destinations,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def load_example(tmp_path, changes=None, edit=None):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/match.jev.example.yaml").read_text())
    config["backend"]["planner_config"] = str(root / "configs/planner.sim.example.yaml")
    config["backend"]["autoscalers"][0]["config"].update(changes or {})
    if edit is not None:
        edit(config)
    path = tmp_path / "match.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_match_config(path)


def test_jev_pilot_defaults_and_matrix(tmp_path):
    config = load_example(tmp_path)
    assert config.backend.autoscalers[0].type == "jev"
    assert config.backend.autoscalers[0].config["max_prefill"] == 16
    assert config.backend.autoscalers[0].config["min_confidence"] == 0


def test_cognitive_agent_example_selects_the_self_hosted_model():
    root = Path(__file__).resolve().parents[1]
    config = load_match_config(root / "configs/match.cognitive-agent.example.yaml")
    agent = config.backend.autoscalers[0]
    assert (agent.name, agent.type) == ("cognitive-agent", "jev")
    assert agent.config["provider"] == "self_hosted"
    assert agent.config["model"] == "PelaAI/KnowLine-31B-Gen1"
    assert config.backend.replay.telemetry_sample_interval_s == 5


def test_self_hosted_endpoint_survives_match_resolution(tmp_path):
    config = load_example(
        tmp_path,
        {
            "provider": "self_hosted",
            "base_url": "http://localhost:8123",
            "model": "PelaAI/KnowLine-31B-Gen1",
            "max_calls": 40400,
        },
    )
    parameters = config.backend.autoscalers[0].config
    assert parameters["provider"] == "self_hosted"
    assert parameters["base_url"] == "http://localhost:8123"
    assert parameters["max_calls"] == 40400


def test_self_hosted_jev_needs_only_a_starting_fleet_within_the_budget(tmp_path):
    def asymmetric_workers(config):
        # Four-GPU prefill and eight-GPU decode workers under a 16-GPU budget.
        backend = config["backend"]
        backend["gpu_budget"] = 16
        backend["autoscalers"] = backend["autoscalers"][:1]
        for key in ("tp_size", "moe_tp_size", "moe_ep_size"):
            del backend["engines"]["common"][key]
        backend["engines"]["prefill"]["tp_size"] = 4
        backend["engines"]["decode"]["tp_size"] = 8

    bounds = {"max_prefill": 2, "max_decode": 2}
    self_hosted = {
        "provider": "self_hosted",
        "base_url": "http://localhost:8123",
        "model": "PelaAI/KnowLine-31B-Gen1",
    }
    backend = load_example(
        tmp_path, {**bounds, **self_hosted}, edit=asymmetric_workers
    ).backend
    # Both pools at two workers would need 24 GPUs; the joint choices stop at 16.
    assert set(feasible_worker_targets(backend, backend.autoscalers[0])) == {
        (1, 1),
        (2, 1),
    }
    # Hosted Jev sizes each pool on its own, so both ceilings must fit together.
    with pytest.raises(MatchConfigError, match="maximum fleet requires 24 GPUs"):
        load_example(tmp_path, bounds, edit=asymmetric_workers)


@pytest.mark.parametrize(
    "change",
    [
        {"min_confidence": 1.1},
        {"min_confidence": -1},
        {"timeout_s": 0},
        {"max_calls": 0},
        {"history_ticks": True},
        {"failure_mode": "planner"},
        {"api_key": "do-not-accept-secrets-in-config"},
        {"max_prefill": 33},
        {"min_decode": 2},
        {"model": ""},
        {"provider": "self_hosted"},
        {"base_url": "http://localhost:8123"},
        {"timeout_s": float("nan")},
        {"failure_mode": ["hold"]},
        {"prior_calls": -1},
        {"prior_calls": True},
        {"prior_calls": 601},
        {"replay_decisions": "missing.jsonl"},
        {"replay_decisions_sha256": "0" * 64},
    ],
)
def test_jev_rejects_invalid_or_unsafe_configuration(tmp_path, change):
    with pytest.raises(MatchConfigError):
        load_example(tmp_path, change)


def test_recovery_log_is_resolved_and_budget_is_retained(tmp_path):
    source = tmp_path / "previous.jsonl"
    source.write_text("")
    config = load_example(
        tmp_path,
        {
            "prior_calls": 10,
            "max_calls": 600,
            "replay_decisions": "previous.jsonl",
            "replay_decisions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )
    parameters = config.backend.autoscalers[0].config
    assert parameters["replay_decisions"] == str(source)
    assert parameters["max_calls"] == 600
    assert parameters["prior_calls"] == 10


def test_recovery_log_must_exist(tmp_path):
    with pytest.raises(MatchConfigError, match="file does not exist"):
        load_example(
            tmp_path,
            {
                "replay_decisions": "missing.jsonl",
                "replay_decisions_sha256": "0" * 64,
            },
        )


SELF_HOSTED = {
    "provider": "self_hosted",
    "base_url": "http://localhost:8123",
    "model": "PelaAI/KnowLine-31B-Gen1",
}


@pytest.mark.parametrize("interval", [None, 10])
def test_self_hosted_requires_five_second_telemetry(tmp_path, interval):
    def change_cadence(config):
        config["backend"]["replay"]["telemetry_sample_interval_s"] = interval

    with pytest.raises(MatchConfigError, match="5-second replay telemetry"):
        load_example(tmp_path, SELF_HOSTED, edit=change_cadence)
    # Hosted Jev does not read replay telemetry, so its cadence stays optional.
    load_example(tmp_path, edit=change_cadence)


def test_self_hosted_token_requires_https_beyond_loopback(tmp_path):
    remote = {**SELF_HOSTED, "api_key_env": "MODEL_TOKEN"}
    with pytest.raises(MatchConfigError, match="https"):
        load_example(tmp_path, {**remote, "base_url": "http://decision.internal:8123"})
    load_example(tmp_path, {**remote, "base_url": "https://decision.internal:8123"})
    load_example(tmp_path, remote)


def _recovery(source):
    source.write_text("")
    return {
        "replay_decisions": source.name,
        "replay_decisions_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def test_recovery_log_cannot_be_a_publish_destination(tmp_path):
    source = tmp_path / "previous.jsonl"

    def publish_over_the_log(config):
        config["publish"]["destinations"] = [
            {"type": "json", "path": str(source), "overwrite": True}
        ]

    with pytest.raises(MatchConfigError, match="must not overwrite an input"):
        load_example(tmp_path, _recovery(source), edit=publish_over_the_log)


def test_publication_preflight_protects_the_recovery_log(tmp_path):
    source = tmp_path / "previous.jsonl"
    results = tmp_path / "results.json"

    def publish_beside_the_log(config):
        config["publish"] = {
            "artifact_root": str(tmp_path / "artifacts"),
            "destinations": [{"type": "json", "path": str(results), "overwrite": True}],
        }

    config = load_example(tmp_path, _recovery(source), edit=publish_beside_the_log)
    preflight_publish_destinations(config)
    # The destination now resolves to the log a recovery verifies by checksum.
    results.symlink_to(source)
    with pytest.raises(MatchPublishError, match="refusing to overwrite input"):
        preflight_publish_destinations(config)
