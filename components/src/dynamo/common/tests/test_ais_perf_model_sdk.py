# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the AISimulate imports.

"""The configs of ``dynamo.common.ais_perf_model`` next to the AISimulate config class.

These tests import AISimulate. The tests of the same module that must also be
collected without it are in ``test_ais_perf_model_shared.py``.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "aisimulate_core.sdk",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate_core.sdk import ForwardPassPerfModelConfig

from dynamo.common import ais_perf_model
from tests.utils.ais_perf_model import IDENTITY

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_router_perf_config_equals_forward_pass_config_dict(tmp_path: Path) -> None:
    entry = {**IDENTITY, "worker_type": "prefill", "tp": 2, "backend_version": "1.0"}
    expected = ForwardPassPerfModelConfig(**entry).to_dict()
    # JSON is YAML, so the config file needs no YAML writer here.
    (tmp_path / "cfg.yaml").write_text(
        json.dumps([{**IDENTITY, "worker_type": "decode"}, entry])
    )
    untyped = {key: value for key, value in entry.items() if key != "worker_type"}

    assert ais_perf_model.router_perf_config(entry) == expected
    assert ais_perf_model.router_perf_config(entry, "prefill") == expected
    assert ais_perf_model.router_perf_config(untyped, "prefill") == expected
    assert ais_perf_model.router_perf_config("cfg.yaml", base_dir=tmp_path) == expected
    assert expected["worker_type"] == "prefill"
    assert expected["tp"] == 2


def test_normalized_config_checks_a_resolved_config() -> None:
    resolved = {**IDENTITY, "worker_type": "decode", "tp": 2}

    assert (
        ais_perf_model.normalized_config(resolved)
        == ForwardPassPerfModelConfig(**resolved).to_dict()
    )
    assert resolved == {**IDENTITY, "worker_type": "decode", "tp": 2}
    with pytest.raises(ValueError, match="^ais_perf_model requires worker_type$"):
        ais_perf_model.normalized_config(IDENTITY)
    with pytest.raises(ValueError, match="^ais_perf_model: .*not_a_field"):
        ais_perf_model.normalized_config({**resolved, "not_a_field": 1})
