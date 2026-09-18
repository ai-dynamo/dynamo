# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.common.configuration.groups.ais_perf_args import (
    AisPerfArgGroup,
    AisPerfConfigBase,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _parse(args):
    parser = argparse.ArgumentParser()
    AisPerfArgGroup().add_arguments(parser)
    return AisPerfConfigBase.from_cli_args(parser.parse_args(args))


@pytest.mark.parametrize("prefix", ["ais", "aic"])
def test_flat_input_uses_canonical_defaults(prefix):
    config = _parse(
        [
            f"--{prefix}-backend",
            "vllm",
            f"--{prefix}-system",
            "h200_sxm",
            f"--{prefix}-model-path",
            "example/model",
        ]
    )
    payload = config.ais_perf_kwargs()["config"]
    assert payload["worker_type"] == "aggregated"
    assert payload["estimation_mode"] == "auto"
    assert payload["fallback_policy"] == "deny"
    assert not any(name.startswith("aic_") for name in payload)


def test_full_config_preserves_tuning_roots_and_worker_identity(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    path = tmp_path / "perf.yaml"
    path.write_text(
        f"""model: example/model
system: h200_sxm
backend: vllm
worker_type: prefill
estimation_mode: fpm_regression
systems_paths: [{first}, {second}]
estimator_config:
  fpm_regression:
    sampling:
      bins_per_axis: [4, 8]
"""
    )
    payload = _parse(["--ais-perf-config", str(path)]).ais_perf_kwargs()["config"]
    assert payload["worker_type"] == "prefill"
    assert payload["systems_paths"] == [str(first), str(second)]
    assert payload["estimator_config"]["fpm_regression"]["sampling"][
        "bins_per_axis"
    ] == [4, 8]


def test_conflicting_aliases_and_config_are_rejected():
    with pytest.raises(SystemExit):
        _parse(["--ais-tp-size", "2", "--aic-tp-size", "2"])
    with pytest.raises(ValueError, match="cannot be combined"):
        _parse(
            ["--ais-perf-config", '{"model":"m"}', "--ais-tp-size", "1"]
        ).ais_perf_kwargs()


def test_legacy_environment_and_conflict(monkeypatch):
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    assert _parse([]).ais_backend == "vllm"
    monkeypatch.setenv("DYN_AIS_BACKEND", "vllm")
    with pytest.raises(ValueError, match="cannot be combined"):
        _parse([])


def test_unknown_canonical_field_rejected():
    with pytest.raises(TypeError, match="unexpected keyword"):
        _parse(
            [
                "--ais-perf-config",
                '{"model":"m","system":"s","backend":"vllm","worker_type":"prefill","typo":true}',
            ]
        ).ais_perf_kwargs()
