# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIS perf model inputs of squeeze_evolve: one config for each tier."""

from __future__ import annotations

import argparse
import json

import pytest
import yaml

import dynamo.squeeze_evolve.args as squeeze_evolve_args
from dynamo.common.tests.configuration.ais_perf_inputs_helpers import (  # noqa: F401 -- no_ais_environment is a fixture
    AIS,
    IDENTITY,
    MODEL,
    NOT_AIS,
    NOTHING_GIVEN,
    OLD_FLAT,
    exactly,
    expected,
    no_ais_environment,
    no_warning,
)
from dynamo.squeeze_evolve.args import (
    SqueezeEvolveArgGroup,
    SqueezeEvolveRunConfig,
    parse_args,
)
from dynamo.squeeze_evolve.orchestrator import parse_tiers

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.usefixtures("no_ais_environment"),
]

MODEL_NAME = ["--model-name", "squeeze/test"]
PREFILL = {**IDENTITY, "worker_type": "prefill"}
AGGREGATED = {**IDENTITY, "worker_type": "aggregated"}

OLD_FLAT_WARNING = (
    "--ais-backend, --ais-system and --ais-model-path are deprecated and will be "
    "removed in v1.9; use ais_perf_model in each --tiers entry (or "
    "--ais-perf-model with one tier)."
)


def _tiers(*ais_perf_models, count=None):
    """Return ``--tiers`` text; a tier gets ``ais_perf_model`` when one is given."""
    names = ["small", "large", "largest"]
    total = len(ais_perf_models) if count is None else count
    tiers = []
    for index in range(total):
        tier = {"endpoint": f"dynamo.{names[index]}.generate", "model": MODEL}
        if index < len(ais_perf_models) and ais_perf_models[index] is not None:
            tier["ais_perf_model"] = ais_perf_models[index]
        tiers.append(tier)
    return json.dumps(tiers)


def _config(args):
    """Parse with the real parser and return the config before validate() ran."""
    parser = argparse.ArgumentParser()
    SqueezeEvolveArgGroup().add_arguments(parser)
    return SqueezeEvolveRunConfig.from_cli_args(parser.parse_args(args))


# Old inputs only: results and errors stay as they are.


def test_squeeze_old_global_env_passes_validation(monkeypatch):
    monkeypatch.setenv("DYN_AIS_BACKEND", "vllm")
    monkeypatch.setenv("DYN_AIS_SYSTEM", "h200_sxm")
    monkeypatch.setenv("DYN_AIS_MODEL_PATH", MODEL)

    with pytest.warns(FutureWarning, match="^DYN_AIS_BACKEND, DYN_AIS_SYSTEM and "):
        config = parse_args(["--tiers", _tiers(count=2), *MODEL_NAME, *AIS])

    assert config.ais_perf_kwargs() == expected()


def test_old_input_errors_come_before_the_tiers_error():
    config = _config([*AIS, "--ais-backend", "vllm"])

    with pytest.raises(ValueError, match="^AIS perf config requires system, model$"):
        with pytest.warns(FutureWarning, match="^--ais-backend is deprecated "):
            config.validate()


@pytest.mark.parametrize(
    "tiers_and_inputs",
    [
        pytest.param([_tiers(count=2)], id="several-tiers-without-keys"),
        pytest.param([_tiers(PREFILL, PREFILL)], id="several-tiers-with-keys"),
        pytest.param([_tiers(PREFILL, count=3)], id="three-tiers-one-key"),
        pytest.param([_tiers(count=1)], id="one-tier"),
        pytest.param([_tiers(PREFILL)], id="one-tier-with-key"),
        pytest.param([_tiers(count=2), *OLD_FLAT], id="several-tiers-old-input"),
    ],
)
def test_renamed_environment_variable_is_reported_before_the_tiers(
    tiers_and_inputs, monkeypatch
):
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    monkeypatch.setenv("DYN_AIC_SYSTEM", "h200_sxm")
    config = _config(["--tiers", *tiers_and_inputs, *MODEL_NAME, *AIS])

    with pytest.raises(
        ValueError,
        match=exactly(
            "renamed environment variables are no longer read: DYN_AIC_BACKEND, "
            "DYN_AIC_SYSTEM; use the DYN_AIS_* names"
        ),
    ) as error, no_warning():
        config.validate()

    assert type(error.value) is ValueError


@pytest.mark.parametrize(
    "tiers",
    [
        pytest.param(_tiers(count=2), id="several-tiers"),
        pytest.param(_tiers(count=1), id="one-tier"),
    ],
)
def test_renamed_environment_variable_is_not_read_without_ais_load_model(
    tiers, monkeypatch
):
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    config = _config(["--tiers", tiers, *MODEL_NAME, *NOT_AIS])

    with no_warning():
        config.validate()

    assert set(config.ais_perf_kwargs_by_tier()) == {None}


def test_old_global_warning_names_tier_replacement():
    config = _config(["--tiers", _tiers(count=2), *MODEL_NAME, *AIS, *OLD_FLAT])

    with pytest.warns(FutureWarning) as records:
        config.validate()

    assert [str(record.message) for record in records] == [OLD_FLAT_WARNING]
    assert records[0].filename == __file__


def test_start_warns_once_for_old_inputs():
    with pytest.warns(FutureWarning) as records:
        config = parse_args(["--tiers", _tiers(count=2), *MODEL_NAME, *AIS, *OLD_FLAT])
        by_tier = config.ais_perf_kwargs_by_tier()

    assert [str(record.message) for record in records] == [OLD_FLAT_WARNING]
    assert records[0].filename == squeeze_evolve_args.__file__
    assert by_tier == [expected(), expected()]


# The key in a tier.


def test_parse_tiers_accepts_ais_perf_model_key():
    several = [PREFILL, AGGREGATED]

    tiers = parse_tiers(_tiers(PREFILL, several, "cfg.yaml"))

    assert [tier.ais_perf_model for tier in tiers] == [PREFILL, several, "cfg.yaml"]
    assert parse_tiers(_tiers(count=1))[0].ais_perf_model is None


def test_each_tier_reads_its_own_key():
    config = parse_args(
        [
            "--tiers",
            _tiers({**PREFILL, "tp": 2}, json.dumps(AGGREGATED)),
            *MODEL_NAME,
            *AIS,
        ]
    )

    with no_warning():
        by_tier = config.ais_perf_kwargs_by_tier()

    assert by_tier == [expected(worker_type="prefill", tp=2), expected()]


def test_no_tier_config_without_ais_load_model():
    config = parse_args(["--tiers", _tiers(count=2), *MODEL_NAME, *NOT_AIS])

    assert config.ais_perf_kwargs_by_tier() == [None, None]


@pytest.mark.parametrize(
    ("tiers", "error"),
    [
        pytest.param(
            _tiers(count=2),
            "--tiers[0] and --tiers[1] have no ais_perf_model; with "
            "--router-prefill-load-model=ais every tier needs one",
            id="no-tier-has-it",
        ),
        pytest.param(
            _tiers(PREFILL, count=2),
            "--tiers[1] has no ais_perf_model; with "
            "--router-prefill-load-model=ais every tier needs one",
            id="one-of-two",
        ),
        pytest.param(
            _tiers(None, PREFILL, count=3),
            "--tiers[0] and --tiers[2] have no ais_perf_model; with "
            "--router-prefill-load-model=ais every tier needs one",
            id="one-of-three",
        ),
    ],
)
def test_several_tiers_need_a_key_each(tiers, error):
    config = _config(["--tiers", tiers, *MODEL_NAME, *AIS])

    with pytest.raises(ValueError, match=exactly(error)):
        config.validate()


@pytest.mark.parametrize("tiers", [_tiers(count=2), _tiers(PREFILL, PREFILL)])
def test_global_new_value_with_several_tiers_is_rejected(tiers, monkeypatch):
    value = json.dumps(AGGREGATED)
    typed = _config(["--tiers", tiers, *MODEL_NAME, *AIS, "--ais-perf-model", value])

    with pytest.raises(
        ValueError,
        match=exactly(
            "--ais-perf-model applies to one tier only; write ais_perf_model in "
            "each --tiers entry"
        ),
    ):
        typed.validate()

    monkeypatch.setenv("DYN_AIS_PERF_MODEL", value)
    from_env = _config(["--tiers", tiers, *MODEL_NAME, *AIS])
    with pytest.raises(
        ValueError,
        match=exactly(
            "DYN_AIS_PERF_MODEL applies to one tier only; write ais_perf_model in "
            "each --tiers entry"
        ),
    ):
        from_env.validate()


def test_one_tier_takes_global_value_or_key_not_both(monkeypatch):
    value = json.dumps(AGGREGATED)
    global_only = ["--tiers", _tiers(count=1), "--ais-perf-model", value]
    key_only = ["--tiers", _tiers(PREFILL)]
    both = ["--tiers", _tiers(PREFILL), "--ais-perf-model", value]

    with no_warning():
        assert parse_args(
            [*global_only, *MODEL_NAME, *AIS]
        ).ais_perf_kwargs_by_tier() == [expected()]
        assert parse_args([*key_only, *MODEL_NAME, *AIS]).ais_perf_kwargs_by_tier() == [
            expected(worker_type="prefill")
        ]
    with pytest.raises(
        ValueError,
        match=exactly(
            "--ais-perf-model cannot be combined with ais_perf_model in --tiers[0]"
        ),
    ):
        _config([*both, *MODEL_NAME, *AIS]).validate()

    monkeypatch.setenv("DYN_AIS_PERF_MODEL", value)
    with pytest.raises(
        ValueError,
        match=exactly(
            "DYN_AIS_PERF_MODEL cannot be combined with ais_perf_model in --tiers[0]"
        ),
    ):
        _config([*key_only, *MODEL_NAME, *AIS]).validate()
    with no_warning():
        assert parse_args(
            ["--tiers", _tiers(count=1), *MODEL_NAME, *AIS]
        ).ais_perf_kwargs_by_tier() == [expected()]


def test_old_global_input_with_tier_key_is_rejected():
    config = _config(
        [
            "--tiers",
            _tiers(None, PREFILL),
            *MODEL_NAME,
            *AIS,
            "--ais-backend",
            "vllm",
            "--ais-system",
            "h200_sxm",
        ]
    )

    with pytest.raises(
        ValueError,
        match=exactly(
            "ais_perf_model in --tiers[1] cannot be combined with --ais-backend "
            "and --ais-system"
        ),
    ), no_warning():
        config.validate()


@pytest.mark.parametrize("load_model", [NOT_AIS, []])
def test_tier_key_without_ais_load_model_is_rejected(load_model):
    config = _config(["--tiers", _tiers(None, PREFILL), *MODEL_NAME, *load_model])

    with pytest.raises(
        ValueError,
        match=exactly(
            "ais_perf_model in --tiers[1] requires --router-prefill-load-model=ais"
        ),
    ):
        config.validate()


def test_worker_type_flag_applies_to_every_tier():
    several = [PREFILL, AGGREGATED]
    args = [
        "--tiers",
        _tiers(several, [{**PREFILL, "tp": 2}, AGGREGATED]),
        *MODEL_NAME,
        *AIS,
    ]

    config = parse_args([*args, "--router-prefill-load-worker-type", "prefill"])

    assert config.ais_perf_kwargs_by_tier() == [
        expected(worker_type="prefill"),
        expected(worker_type="prefill", tp=2),
    ]
    with pytest.raises(
        ValueError,
        match=exactly(
            "--tiers[0]: ais_perf_model: it has entries for prefill and "
            "aggregated; set --router-prefill-load-worker-type to choose one"
        ),
    ):
        _config(args).validate()


def test_bad_worker_type_env_with_tier_keys_is_rejected(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    config = _config(["--tiers", _tiers(PREFILL, PREFILL), *MODEL_NAME, *AIS])

    with pytest.raises(
        ValueError,
        match=exactly(
            "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE is 'decode'; use prefill or "
            "aggregated"
        ),
    ):
        config.validate()


def test_error_in_a_tier_value_names_the_tier():
    config = _config(
        ["--tiers", _tiers(PREFILL, {**PREFILL, "model": ""}), *MODEL_NAME, *AIS]
    )

    with pytest.raises(ValueError) as error:
        config.validate()

    assert str(error.value) == "--tiers[1]: ais_perf_model requires model"
    assert type(error.value) is ValueError
    assert type(error.value.__cause__) is ValueError
    assert str(error.value.__cause__) == "ais_perf_model requires model"


def test_errors_outside_a_tier_value_keep_their_text(monkeypatch):
    tiers = ["--tiers", _tiers(PREFILL, {**PREFILL, "nextn": 2}), *MODEL_NAME, *AIS]
    rates = _config([*tiers, "--ais-nextn-accept-rates", "abc"])

    with pytest.raises(
        ValueError,
        match=exactly(
            "ais_nextn_accept_rates must be comma-separated floats, got 'abc'"
        ),
    ):
        rates.validate()

    # A config that was not validated reads the variable with the tiers.
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    with pytest.raises(
        ValueError,
        match=exactly(
            "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE is 'decode'; use prefill or "
            "aggregated"
        ),
    ):
        _config(tiers).ais_perf_kwargs_by_tier()


def test_tiers_are_read_from_parsed_objects(capsys):
    config = _config(["--tiers", _tiers(PREFILL, AGGREGATED), *MODEL_NAME, *AIS])

    assert [tier.ais_perf_model for tier in config.tiers] == [PREFILL, AGGREGATED]
    config.validate()
    assert config.ais_perf_kwargs_by_tier() == [
        expected(worker_type="prefill"),
        expected(),
    ]

    with pytest.raises(SystemExit) as exit_info:
        _config(["--tiers", "not json", *MODEL_NAME, *AIS])
    assert exit_info.value.code == 2
    assert "argument --tiers: invalid parse_tiers value" in capsys.readouterr().err

    # The AIS checks come before the check for tiers, as they did.
    with pytest.raises(ValueError, match=exactly(NOTHING_GIVEN)):
        _config([*MODEL_NAME, *AIS]).validate()
    with pytest.raises(
        ValueError, match=exactly("--tiers is required (JSON array, cheapest first)")
    ):
        _config([*MODEL_NAME, *NOT_AIS]).validate()


def test_tier_relative_path_starts_at_working_directory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    other_dir = tmp_path / "other"
    run_dir.mkdir()
    other_dir.mkdir()
    (run_dir / "cfg.yaml").write_text(yaml.safe_dump(PREFILL))
    config = _config(["--tiers", _tiers("cfg.yaml"), *MODEL_NAME, *AIS])

    monkeypatch.chdir(run_dir)
    config.validate()
    assert config.ais_perf_kwargs_by_tier() == [expected(worker_type="prefill")]

    monkeypatch.chdir(other_dir)
    with pytest.raises(ValueError, match=r"^--tiers\[0\]: cannot read .*cfg.yaml: "):
        config.validate()
