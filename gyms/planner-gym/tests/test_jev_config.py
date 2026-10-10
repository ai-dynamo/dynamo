# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        {"timeout_s": float("nan")},
        {"failure_mode": ["hold"]},
    ],
)
def test_jev_rejects_invalid_or_unsafe_configuration(tmp_path, change):
    with pytest.raises(MatchConfigError):
        load_example(tmp_path, change)
