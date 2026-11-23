# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIS perf model inputs of squeeze_evolve: one config for each tier."""

from __future__ import annotations

import argparse
import json

import pytest

import dynamo.squeeze_evolve.args as squeeze_evolve_args
from dynamo.squeeze_evolve.args import (
    SqueezeEvolveArgGroup,
    SqueezeEvolveRunConfig,
    parse_args,
)
from tests.utils.ais_perf_model import (
    AGGREGATED_ENTRY,
    AIS,
    IDENTITY,
    MODEL,
    NOT_AIS,
    OLD_FLAT,
    exactly,
    expected,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.usefixtures("no_ais_environment"),
]

MODEL_NAME = ["--model-name", "squeeze/test"]
PREFILL = {**IDENTITY, "worker_type": "prefill"}

OLD_FLAT_WARNING = (
    "--ais-backend, --ais-system and --ais-model-path are deprecated and will be "
    "removed in v1.9; use ais_perf_model in each --tiers entry."
)
TYPED_FLAG_ERROR = (
    "--ais-perf-model is not read by squeeze_evolve; write ais_perf_model in "
    "each --tiers entry"
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


def test_old_input_errors_come_before_the_tiers_error():
    config = _config([*AIS, "--ais-backend", "vllm"])

    with pytest.raises(ValueError, match="^AIS perf config requires "):
        with pytest.warns(FutureWarning, match="^--ais-backend is "):
            config.validate()


@pytest.mark.parametrize(
    "tiers",
    [
        pytest.param(_tiers(PREFILL, PREFILL), id="several-tiers-with-keys"),
        pytest.param(_tiers(count=2), id="several-tiers-without-keys"),
    ],
)
def test_renamed_environment_variable_is_reported_before_the_tiers(tiers, monkeypatch):
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    config = _config(["--tiers", tiers, *MODEL_NAME, *AIS])

    with pytest.raises(ValueError, match="DYN_AIC_BACKEND"):
        config.validate()


def test_start_warns_once_for_old_inputs():
    with pytest.warns(FutureWarning) as records:
        config = parse_args(["--tiers", _tiers(count=2), *MODEL_NAME, *AIS, *OLD_FLAT])
        by_tier = config.ais_perf_kwargs_by_tier()

    assert [str(record.message) for record in records] == [OLD_FLAT_WARNING]
    assert records[0].filename == squeeze_evolve_args.__file__
    assert by_tier == [expected(), expected()]


# The key in a tier.


def test_each_tier_reads_its_own_key():
    tiers = _tiers({**PREFILL, "tp": 2}, json.dumps(AGGREGATED_ENTRY))

    config = parse_args(["--tiers", tiers, *MODEL_NAME, *AIS])

    assert config.ais_perf_kwargs_by_tier() == [
        expected(worker_type="prefill", tp=2),
        expected(),
    ]


def test_no_tier_config_without_ais_load_model(monkeypatch):
    # A renamed variable is not read either.
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")

    config = parse_args(["--tiers", _tiers(count=2), *MODEL_NAME, *NOT_AIS])

    assert config.ais_perf_kwargs_by_tier() == [None, None]


@pytest.mark.parametrize(
    ("tiers", "missing"),
    [
        pytest.param(_tiers(PREFILL, count=2), r"--tiers\[1\] has", id="one-of-two"),
        pytest.param(
            _tiers(None, PREFILL, count=3),
            r"--tiers\[0\] and --tiers\[2\] have",
            id="one-of-three",
        ),
    ],
)
def test_every_tier_needs_a_key(tiers, missing):
    config = _config(["--tiers", tiers, *MODEL_NAME, *AIS])

    with pytest.raises(ValueError, match=f"^{missing} no ais_perf_model"):
        config.validate()


@pytest.mark.parametrize(
    "tiers_and_inputs",
    [
        pytest.param([_tiers(PREFILL, PREFILL), *AIS], id="a-key-in-each-tier"),
        pytest.param([_tiers(count=2), *AIS], id="several-tiers-without-keys"),
        pytest.param([_tiers(count=2), *AIS, *OLD_FLAT], id="old-inputs"),
        pytest.param([_tiers(count=2), *NOT_AIS], id="no-ais-load-model"),
    ],
)
def test_typed_global_flag_is_rejected(tiers_and_inputs):
    value = json.dumps(AGGREGATED_ENTRY)
    config = _config(
        ["--tiers", *tiers_and_inputs, *MODEL_NAME, "--ais-perf-model", value]
    )

    with pytest.raises(ValueError, match=exactly(TYPED_FLAG_ERROR)):
        config.validate()


def test_help_offers_only_the_tier_key(capsys):
    with pytest.raises(SystemExit):
        parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())

    assert "Not read here; write ais_perf_model in each --tiers entry." in text
    # No text offers the flag, which squeeze_evolve rejects.
    assert "use --ais-perf-model" not in text
    assert "from --ais-perf-model" not in text


def test_global_environment_variable_is_not_read(monkeypatch):
    # Other processes may read the variable from the same environment.
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "/not/a/local/file.yaml")
    inputs = [*MODEL_NAME, *AIS]

    new = parse_args(["--tiers", _tiers(PREFILL), *inputs])
    assert new.ais_perf_kwargs_by_tier() == [expected(worker_type="prefill")]

    with pytest.raises(ValueError, match=r"^--tiers\[0\] has no ais_perf_model"):
        _config(["--tiers", _tiers(count=1), *inputs]).validate()

    with pytest.warns(FutureWarning, match="^--ais-backend, "):
        old = parse_args(["--tiers", _tiers(count=2), *inputs, *OLD_FLAT])
    assert old.ais_perf_kwargs_by_tier() == [expected(), expected()]


def test_old_global_input_with_tier_key_is_rejected():
    config = _config(["--tiers", _tiers(None, PREFILL), *MODEL_NAME, *AIS, *OLD_FLAT])

    with pytest.raises(
        ValueError,
        match=r"^ais_perf_model in --tiers\[1\] cannot be combined with --ais-backend",
    ):
        config.validate()


def test_worker_type_flag_with_old_inputs_names_the_tier_key():
    inputs = [*AIS, *OLD_FLAT, "--router-prefill-load-worker-type", "prefill"]
    config = _config(["--tiers", _tiers(count=2), *MODEL_NAME, *inputs])

    with pytest.raises(ValueError, match="requires ais_perf_model in each --tiers"):
        config.validate()


def test_tier_key_without_ais_load_model_is_rejected():
    config = _config(["--tiers", _tiers(None, PREFILL), *MODEL_NAME])

    with pytest.raises(ValueError, match=r"^ais_perf_model in --tiers\[1\] requires "):
        config.validate()


def test_worker_type_flag_applies_to_every_tier():
    several = [PREFILL, AGGREGATED_ENTRY]
    tiers = _tiers(several, [{**PREFILL, "tp": 2}, AGGREGATED_ENTRY])
    args = ["--tiers", tiers, *MODEL_NAME, *AIS]

    config = parse_args([*args, "--router-prefill-load-worker-type", "prefill"])

    assert config.ais_perf_kwargs_by_tier() == [
        expected(worker_type="prefill"),
        expected(worker_type="prefill", tp=2),
    ]


def test_error_in_a_tier_value_names_the_tier():
    config = _config(
        ["--tiers", _tiers(PREFILL, {**PREFILL, "model": ""}), *MODEL_NAME, *AIS]
    )

    with pytest.raises(ValueError, match=r"^--tiers\[1\]: ais_perf_model "):
        config.validate()


def test_errors_outside_a_tier_value_keep_their_text(monkeypatch):
    tiers = ["--tiers", _tiers(PREFILL, {**PREFILL, "nextn": 2}), *MODEL_NAME, *AIS]

    with pytest.raises(ValueError, match="^ais_nextn_accept_rates must be "):
        _config([*tiers, "--ais-nextn-accept-rates", "abc"]).validate()

    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    with pytest.raises(ValueError, match="^DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE is "):
        _config(tiers).validate()


def test_missing_tiers_is_reported_with_ais_load_model():
    with pytest.raises(ValueError, match="^--tiers is required "):
        _config([*MODEL_NAME, *AIS]).validate()
