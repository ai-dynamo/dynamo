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


@pytest.mark.parametrize("prefix", ["ais", "aic"])
def test_repeated_flag_with_same_spelling_uses_last_value(prefix):
    assert (
        _parse([f"--{prefix}-tp-size", "2", f"--{prefix}-tp-size", "4"]).ais_tp_size
        == 4
    )


@pytest.mark.parametrize("legacy_name", ["DYN_AIC_TP_SIZE", "DYN_AIC_BACKEND"])
def test_retired_environment_fails_before_constructing_ais_config(
    monkeypatch, legacy_name
):
    monkeypatch.setenv(legacy_name, "2" if legacy_name.endswith("TP_SIZE") else "vllm")
    config = _parse(
        [
            "--aic-backend",
            "vllm",
            "--aic-system",
            "h200_sxm",
            "--aic-model-path",
            "example/model",
        ]
    )
    with pytest.raises(ValueError, match=rf"{legacy_name}.*DYN_AIS_"):
        config.ais_perf_kwargs()
    monkeypatch.delenv(legacy_name)
    monkeypatch.setenv("DYN_AIS_TP_SIZE", "2")
    config = _parse(
        [
            "--ais-backend",
            "vllm",
            "--ais-system",
            "h200_sxm",
            "--ais-model-path",
            "example/model",
        ]
    )
    assert config.ais_perf_kwargs()["config"]["tp"] == 2


def test_only_ais_environment_is_read(monkeypatch):
    monkeypatch.delenv("DYN_AIS_BACKEND", raising=False)
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    assert _parse([]).ais_backend is None
    monkeypatch.setenv("DYN_AIS_BACKEND", "sglang")
    assert _parse([]).ais_backend == "sglang"


@pytest.mark.parametrize("source", ["cli", "env"])
@pytest.mark.parametrize("contents", [None, "model: [unterminated"])
def test_invalid_config_file_reports_usage_error(
    tmp_path, monkeypatch, capsys, source, contents
):
    path = tmp_path / "invalid.yaml"
    if contents is not None:
        path.write_text(contents)
    if source == "env":
        monkeypatch.setenv("DYN_AIS_PERF_CONFIG", str(path))
        args = []
    else:
        monkeypatch.delenv("DYN_AIS_PERF_CONFIG", raising=False)
        args = ["--ais-perf-config", str(path)]
    with pytest.raises(SystemExit) as error:
        _parse(args)
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--ais-perf-config" in stderr
    assert "Traceback" not in stderr


def test_bad_environment_config_allows_help_and_explicit_override(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("DYN_AIS_PERF_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(SystemExit) as error:
        _parse(["--help"])
    assert error.value.code == 0
    assert "--ais-perf-config" in capsys.readouterr().out
    assert _parse(["--ais-perf-config", '{"model":"explicit"}']).ais_perf_config == {
        "model": "explicit"
    }


def test_unknown_canonical_field_rejected():
    with pytest.raises(TypeError, match="unexpected keyword"):
        _parse(
            [
                "--ais-perf-config",
                '{"model":"m","system":"s","backend":"vllm","worker_type":"prefill","typo":true}',
            ]
        ).ais_perf_kwargs()


def test_legacy_router_selector_normalizes_at_cli_boundary():
    from dynamo.common.configuration.groups.kv_router_args import KvRouterArgGroup

    parser = argparse.ArgumentParser()
    KvRouterArgGroup().add_arguments(parser)
    assert (
        parser.parse_args(
            ["--router-prefill-load-model", "aic"]
        ).router_prefill_load_model
        == "ais"
    )
