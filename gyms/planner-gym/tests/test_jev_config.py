# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest
import yaml
from autoscaling_arena.match_config import MatchConfigError, load_match_config


def load_example(tmp_path, changes=None):
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/match.jev.example.yaml").read_text())
    config["backend"]["planner_config"] = str(root / "configs/planner.sim.example.yaml")
    config["backend"]["autoscalers"][0]["config"].update(changes or {})
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
