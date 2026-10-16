# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for reading EPD's power_decision.json."""

import json
import os

import pytest

from dynamo.power.decision import (
    DecisionFormatError,
    decision_file_path,
    node_watts,
    parse_decision,
    per_gpu_watts,
    read_decision_file,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

EPD_PAYLOAD = {
    "version": 1,
    "sequence": 7,
    "timestamp": 1760000000.25,
    "per_node_gpu_power_limit_w": {"hecate0001": 1350, "hecate0002": 1420.0},
}


def test_decision_file_path_follows_epd_layout(tmp_path):
    path = decision_file_path(str(tmp_path))
    assert path == os.path.join(
        str(tmp_path), "epd", "actuation", "power_decision.json"
    )
    assert os.path.isabs(decision_file_path("relative/log/dir"))


def test_read_decision_file_parses_epd_schema(tmp_path):
    path = tmp_path / "power_decision.json"
    path.write_text(json.dumps(EPD_PAYLOAD))
    decision = read_decision_file(str(path))
    assert decision.version == 1
    assert decision.sequence == 7
    assert decision.timestamp == pytest.approx(1760000000.25)
    assert decision.per_node_gpu_power_limit_w == {
        "hecate0001": 1350,
        "hecate0002": 1420,
    }
    assert decision.nodes() == ("hecate0001", "hecate0002")
    assert node_watts(decision, "hecate0002") == 1420
    assert node_watts(decision, "hecate0009") is None
    assert per_gpu_watts(1350, (2, 0)) == {2: 1350, 0: 1350}


def test_read_decision_file_missing_and_invalid_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_decision_file(str(tmp_path / "missing.json"))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        read_decision_file(str(broken))


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {**EPD_PAYLOAD, "version": 2},
        {key: value for key, value in EPD_PAYLOAD.items() if key != "sequence"},
        {**EPD_PAYLOAD, "timestamp": "yesterday"},
        {**EPD_PAYLOAD, "per_node_gpu_power_limit_w": [1350]},
        {**EPD_PAYLOAD, "per_node_gpu_power_limit_w": {"n": "1350"}},
        {**EPD_PAYLOAD, "per_node_gpu_power_limit_w": {"n": 0}},
        {**EPD_PAYLOAD, "per_node_gpu_power_limit_w": {"n": True}},
    ],
)
def test_parse_decision_rejects_schema_violations(payload):
    with pytest.raises(DecisionFormatError):
        parse_decision(payload)
