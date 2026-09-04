# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GlobalPlannerConfig and ``--config`` resolution.

Every setting has exactly one home, so the interesting cases are loading
(inline JSON, JSON file, YAML file) and validation -- not precedence.
"""

import json

import pytest
from pydantic import ValidationError

from dynamo.global_planner.argparse_config import (
    create_global_planner_parser,
    resolve_config,
)
from dynamo.global_planner.config import GlobalPlannerConfig

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _resolve(*argv: str) -> GlobalPlannerConfig:
    """Parse ``argv`` through the real parser and merge into a config."""
    return resolve_config(create_global_planner_parser().parse_args(list(argv)))


# ---------------------------------------------------------------------------- #
# Defaults and validation                                                      #
# ---------------------------------------------------------------------------- #


def test_defaults_match_previous_flag_defaults():
    config = GlobalPlannerConfig()
    assert config.managed_namespaces is None
    assert config.environment == "kubernetes"
    assert config.no_operation is False
    assert config.max_total_gpus == -1
    assert config.min_total_gpus == -1
    assert config.intent_cache_ttl_seconds == 360.0
    assert config.budget_enforcement_enabled() is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_total_gpus": 8},
        {"min_total_gpus": 8},
        {"min_total_gpus": 8, "max_total_gpus": 16},
    ],
)
def test_budget_enforcement_enabled_when_either_bound_active(kwargs):
    assert GlobalPlannerConfig(**kwargs).budget_enforcement_enabled() is True


def test_rejects_unsatisfiable_band():
    with pytest.raises(ValidationError, match="no total GPU count satisfies"):
        GlobalPlannerConfig(min_total_gpus=32, max_total_gpus=16)


def test_equal_bounds_are_valid_fixed_total_mode():
    config = GlobalPlannerConfig(min_total_gpus=16, max_total_gpus=16)
    assert config.min_total_gpus == config.max_total_gpus == 16


def test_disabled_floor_does_not_trip_band_check():
    # min=-1 disables the floor; it must not be compared against max.
    config = GlobalPlannerConfig(min_total_gpus=-1, max_total_gpus=0)
    assert config.budget_enforcement_enabled() is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intent_cache_ttl_seconds": 0},
        {"intent_cache_ttl_seconds": -1},
        {"max_total_gpus": -2},
        {"min_total_gpus": -2},
        {"environment": "slurm"},
    ],
)
def test_rejects_invalid_fields(kwargs):
    with pytest.raises(ValidationError):
        GlobalPlannerConfig(**kwargs)


def test_rejects_unknown_field():
    with pytest.raises(ValidationError):
        GlobalPlannerConfig(max_total_gpu=16)  # typo: missing trailing 's'


# ---------------------------------------------------------------------------- #
# from_config_arg                                                              #
# ---------------------------------------------------------------------------- #


def test_from_inline_json():
    config = GlobalPlannerConfig.from_config_arg(
        json.dumps({"max_total_gpus": 16, "min_total_gpus": 16})
    )
    assert (config.min_total_gpus, config.max_total_gpus) == (16, 16)


def test_from_json_file(tmp_path):
    path = tmp_path / "gp.json"
    path.write_text(json.dumps({"max_total_gpus": 8, "no_operation": True}))
    config = GlobalPlannerConfig.from_config_arg(str(path))
    assert config.max_total_gpus == 8
    assert config.no_operation is True


def test_from_yaml_file(tmp_path):
    path = tmp_path / "gp.yaml"
    path.write_text(
        "max_total_gpus: 64\n"
        "min_total_gpus: 32\n"
        "managed_namespaces:\n"
        "  - app-ns-1\n"
        "  - app-ns-2\n"
    )
    config = GlobalPlannerConfig.from_config_arg(str(path))
    assert (config.min_total_gpus, config.max_total_gpus) == (32, 64)
    assert config.managed_namespaces == ["app-ns-1", "app-ns-2"]


def test_from_suffixless_file_falls_back_to_yaml(tmp_path):
    path = tmp_path / "gp-config"
    path.write_text("max_total_gpus: 4\n")
    assert GlobalPlannerConfig.from_config_arg(str(path)).max_total_gpus == 4


def test_rejects_arg_that_is_neither_path_nor_json():
    with pytest.raises(ValueError, match="neither a valid file path nor valid JSON"):
        GlobalPlannerConfig.from_config_arg("/no/such/file.yaml")


def test_file_contents_are_validated(tmp_path):
    path = tmp_path / "gp.yaml"
    path.write_text("min_total_gpus: 32\nmax_total_gpus: 16\n")
    with pytest.raises(ValidationError):
        GlobalPlannerConfig.from_config_arg(str(path))


# ---------------------------------------------------------------------------- #
# CLI resolution                                                               #
# ---------------------------------------------------------------------------- #


def test_no_args_yields_defaults():
    assert _resolve() == GlobalPlannerConfig()


def test_config_is_loaded_from_inline_json():
    config = _resolve("--config", json.dumps({"max_total_gpus": 24}))
    assert config.max_total_gpus == 24


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-total-gpus", "16"],
        ["--min-total-gpus", "16"],
        ["--managed-namespaces", "app-ns-1"],
        ["--no-operation"],
        ["--intent-cache-ttl-seconds", "120"],
        ["--environment", "kubernetes"],
    ],
)
def test_removed_per_setting_flags_are_rejected(argv, capsys):
    # These settings moved into --config. A deployment still passing them must
    # fail loudly at startup rather than have them silently ignored, which would
    # drop a GPU budget or a namespace allowlist without any signal.
    with pytest.raises(SystemExit):
        create_global_planner_parser().parse_args(argv)
    assert "unrecognized arguments" in capsys.readouterr().err


def test_config_file_path_accepted_by_parser(tmp_path):
    path = tmp_path / "gp.yaml"
    path.write_text("intent_cache_ttl_seconds: 720\n")
    config = _resolve("--config", str(path))
    assert config.intent_cache_ttl_seconds == 720
