# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIS perf model inputs of the frontend, the routers and squeeze_evolve.

They share one argument group, so each rule is checked once, through the
frontend parser. Two tests run in every component: a new value builds the
config, and old inputs warn once. The suite turns a warning into an error
(``filterwarnings`` in pyproject.toml), so a call outside ``pytest.warns``
also checks that nothing warns.
"""

import argparse
import json

import pytest

import dynamo.router.args as router_args
import dynamo.thunderagent_router.args as thunderagent_router_args
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.router.args import DynamoRouterArgGroup, DynamoRouterConfig
from dynamo.router.args import parse_args as parse_router_args
from dynamo.squeeze_evolve.args import SqueezeEvolveArgGroup, SqueezeEvolveRunConfig
from dynamo.thunderagent_router.args import (
    ThunderAgentArgGroup,
    ThunderAgentRouterConfig,
)
from tests.utils.ais_perf_model import (
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

OLD_PERF_CONFIG = json.dumps({**IDENTITY, "worker_type": "prefill"})
ENDPOINT = ["--endpoint", "dynamo.prefill.generate"]

# The names a warning lists for OLD_FLAT, with their verb.
OLD_FLAT_ARE = "--ais-backend, --ais-system and --ais-model-path are"


def _frontend(args):
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return FrontendConfig.from_cli_args(
        parser.parse_args(["--router-mode", "kv", *args])
    )


def _router(args):
    parser = argparse.ArgumentParser()
    DynamoRouterArgGroup().add_arguments(parser)
    return DynamoRouterConfig.from_cli_args(parser.parse_args([*ENDPOINT, *args]))


def _thunderagent_router(args):
    parser = argparse.ArgumentParser()
    ThunderAgentArgGroup().add_arguments(parser)
    return ThunderAgentRouterConfig.from_cli_args(parser.parse_args([*ENDPOINT, *args]))


def _squeeze_evolve(args, ais_perf_model=None):
    """Parse with one tier; ``ais_perf_model`` is written into that tier."""
    tier = {"endpoint": "dynamo.small.generate", "model": MODEL}
    if ais_perf_model is not None:
        tier["ais_perf_model"] = ais_perf_model
    parser = argparse.ArgumentParser()
    SqueezeEvolveArgGroup().add_arguments(parser)
    return SqueezeEvolveRunConfig.from_cli_args(
        parser.parse_args(
            ["--tiers", json.dumps([tier]), "--model-name", "squeeze/test", *args]
        )
    )


# Each entry parses the arguments with the component's own parser and returns
# the config before validate() ran. Only the two tests that check how a
# component is wired to the shared group run in every component.
COMPONENTS = {
    "frontend": _frontend,
    "router": _router,
    "thunderagent_router": _thunderagent_router,
    "squeeze_evolve": _squeeze_evolve,
}
every_component = pytest.mark.parametrize("component", sorted(COMPONENTS))

# What each component's warning tells the user to write.
REPLACEMENT = {
    "frontend": "--ais-perf-model",
    "router": "--ais-perf-model",
    "thunderagent_router": "--ais-perf-model",
    "squeeze_evolve": "ais_perf_model in each --tiers entry",
}


def _warning_text(names, replacement="--ais-perf-model"):
    """Return the warning; ``names`` ends with ``is`` or ``are``."""
    return f"{names} deprecated and will be removed in v1.9; use {replacement}."


def _warns(names):
    return pytest.warns(FutureWarning, match=exactly(_warning_text(names)))


def _one(**keys):
    """Return inline JSON of one AIS perf model config."""
    return json.dumps({**IDENTITY, **keys})


NEW = ["--ais-perf-model", _one(worker_type="aggregated")]


# Old inputs only: results and errors stay as they are.


@pytest.mark.parametrize(
    ("flag", "names", "keys"),
    [
        pytest.param(
            ["--ais-tp-size", "2"],
            "--ais-backend, --ais-system, --ais-tp-size and --ais-model-path are",
            {"tp": 2},
            id="tp-size",
        ),
        # Here --ais-nextn writes nextn into the config, so it is an old input.
        pytest.param(
            ["--ais-nextn", "2"],
            "--ais-backend, --ais-system, --ais-model-path and --ais-nextn are",
            {"nextn": 2},
            id="nextn",
        ),
    ],
)
def test_old_flat_flags_build_same_config(flag, names, keys):
    config = _frontend([*AIS, *OLD_FLAT, *flag])

    with _warns(names):
        config.validate()

    assert config.ais_perf_kwargs() == expected(**keys)


def test_old_flat_flags_without_ais_load_model_are_ignored():
    config = _frontend([*NOT_AIS, *OLD_FLAT])

    with _warns(OLD_FLAT_ARE):
        config.validate()

    assert config.router_prefill_load_model == "none"


@pytest.mark.parametrize(
    ("args", "names", "missing"),
    [
        (["--ais-backend", "vllm"], "--ais-backend is", "system, model"),
        (
            ["--ais-perf-config", _one(model="", worker_type="prefill")],
            "--ais-perf-config is",
            "model",
        ),
    ],
    ids=["flat-flag", "perf-config"],
)
def test_old_inputs_without_a_model_keep_their_message(args, names, missing):
    config = _frontend([*AIS, *args])

    with pytest.raises(
        ValueError, match=exactly(f"AIS perf config requires {missing}")
    ):
        with _warns(names):
            config.validate()


# The new input.


@every_component
def test_new_value_builds_config_in_every_component(component):
    if component == "squeeze_evolve":
        # squeeze_evolve reads the value from the tier, not from the flag.
        config = _squeeze_evolve(AIS, ais_perf_model=NEW[1])
    else:
        config = COMPONENTS[component]([*AIS, *NEW])

    config.validate()
    if component == "squeeze_evolve":
        (built,) = config.ais_perf_kwargs_by_tier()
    else:
        built = config.ais_perf_kwargs()

    assert built == expected()


def test_typed_value_wins_over_env(monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", _one(worker_type="prefill", tp=2))
    from_env = _frontend(AIS)
    typed = _frontend([*AIS, *NEW])

    from_env.validate()
    typed.validate()

    assert from_env.ais_perf_kwargs() == expected(worker_type="prefill", tp=2)
    assert typed.ais_perf_kwargs() == expected()


# The rules that select an entry are tested with dynamo.common.ais_perf_model;
# here the flag and the variable reach them.
def test_worker_type_env_is_used_and_checked(monkeypatch):
    several = json.dumps(
        [{**IDENTITY, "worker_type": name} for name in ("prefill", "aggregated")]
    )
    args = [*AIS, "--ais-perf-model", several]

    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "prefill")
    config = _frontend(args)
    config.validate()
    assert config.ais_perf_kwargs() == expected(worker_type="prefill")
    # It is also the worker type of a config that names none.
    unnamed = _frontend([*AIS, "--ais-perf-model", _one()])
    assert unnamed.ais_perf_kwargs() == expected(worker_type="prefill")

    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    with pytest.raises(ValueError, match="^DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE is "):
        _frontend(args).validate()
    # A typed worker type is used and the variable is not read.
    typed = _frontend([*args, "--router-prefill-load-worker-type", "aggregated"])
    typed.validate()
    assert typed.ais_perf_kwargs() == expected(worker_type="aggregated")

    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "")
    with pytest.raises(ValueError, match="it has entries for prefill and aggregated"):
        _frontend(args).validate()


def test_env_only_new_value_is_ignored_without_ais_load_model(monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "/not/a/local/file.yaml")
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    config = _frontend(NOT_AIS)

    config.validate()

    assert config.router_prefill_load_model == "none"


def test_old_perf_config_env_without_ais_load_model_is_rejected(monkeypatch):
    # Unlike DYN_AIS_PERF_MODEL, the old variable alone is still an error.
    monkeypatch.setenv("DYN_AIS_PERF_CONFIG", OLD_PERF_CONFIG)
    config = _frontend(NOT_AIS)

    with pytest.raises(
        ValueError,
        match=exactly("--ais-perf-config requires --router-prefill-load-model=ais"),
    ):
        config.validate()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--ais-perf-config", OLD_PERF_CONFIG),
        ("--ais-perf-model", NEW[1]),
        ("--router-prefill-load-worker-type", "prefill"),
    ],
    ids=["perf-config", "perf-model", "worker-type"],
)
def test_flag_without_ais_load_model_is_rejected(flag, value):
    config = _frontend([*NOT_AIS, flag, value])

    with pytest.raises(
        ValueError, match=exactly(f"{flag} requires --router-prefill-load-model=ais")
    ):
        config.validate()


def test_accept_rates_checked_with_new_input():
    value = _one(worker_type="aggregated", nextn=2)
    config = _frontend(
        [*AIS, "--ais-perf-model", value, "--ais-nextn-accept-rates", "abc"]
    )

    with pytest.raises(ValueError, match="^ais_nextn_accept_rates must be "):
        config.validate()


# Old and new inputs together: an error, raised before any warning.


@pytest.mark.parametrize(
    ("args", "env", "error"),
    [
        pytest.param(
            [*NEW, "--ais-backend", "vllm"],
            {},
            "--ais-perf-model cannot be combined with --ais-backend",
            id="flag-with-flag",
        ),
        pytest.param(
            ["--ais-backend", "vllm", "--ais-tp-size", "2"],
            {"DYN_AIS_PERF_MODEL": NEW[1]},
            "DYN_AIS_PERF_MODEL cannot be combined with --ais-backend and "
            "--ais-tp-size",
            id="env-with-flags",
        ),
        pytest.param(
            [*NEW, "--ais-perf-config", OLD_PERF_CONFIG],
            {},
            "--ais-perf-model cannot be combined with --ais-perf-config",
            id="flag-with-perf-config",
        ),
        # The worker type flag belongs to the new input.
        pytest.param(
            [*OLD_FLAT, "--router-prefill-load-worker-type", "prefill"],
            {},
            "--router-prefill-load-worker-type requires --ais-perf-model",
            id="worker-type-flag-with-old",
        ),
    ],
)
def test_new_with_old_is_rejected_and_names_both(args, env, error, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = _frontend([*AIS, *args])

    with pytest.raises(ValueError, match=exactly(error)):
        config.validate()


# The warning for old inputs.


@pytest.mark.parametrize(
    ("args", "env", "names", "keys"),
    [
        # Each input is named as it was given. The worker type variable is not
        # read with old inputs: its value here is not a router worker type.
        pytest.param(
            ["--aic-backend", "vllm", "--ais-model-path", MODEL],
            {
                "DYN_AIS_SYSTEM": "h200_sxm",
                "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE": "decode",
            },
            "--aic-backend, DYN_AIS_SYSTEM and --ais-model-path are",
            {},
            id="flags-and-env",
        ),
        # The acceptance rates and the seed are not old inputs.
        pytest.param(
            [],
            {
                "DYN_AIS_PERF_CONFIG": OLD_PERF_CONFIG,
                "DYN_AIS_NEXTN_ACCEPT_RATES": "0.5",
                "DYN_AIS_MTP_SEED": "7",
            },
            "DYN_AIS_PERF_CONFIG is",
            {"worker_type": "prefill"},
            id="perf-config-env",
        ),
        pytest.param(
            ["--ais-perf-config", OLD_PERF_CONFIG],
            {},
            "--ais-perf-config is",
            {"worker_type": "prefill"},
            id="perf-config-flag",
        ),
    ],
)
def test_old_inputs_warn_once_with_exact_text(args, env, names, keys, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = _frontend([*AIS, *args])

    with pytest.warns(FutureWarning) as records:
        config.validate()

    assert [str(record.message) for record in records] == [_warning_text(names)]
    assert config.ais_perf_kwargs() == expected(**keys)


@every_component
def test_old_inputs_warn_once_at_the_caller_in_every_component(component):
    config = COMPONENTS[component]([*AIS, *OLD_FLAT])

    with pytest.warns(FutureWarning) as records:
        config.validate()
        built = config.ais_perf_kwargs()

    assert [str(record.message) for record in records] == [
        _warning_text(OLD_FLAT_ARE, REPLACEMENT[component])
    ]
    # The warning points at the code that called the validate() with the check;
    # for thunderagent_router that is its own validate().
    caller = (
        thunderagent_router_args.__file__
        if component == "thunderagent_router"
        else __file__
    )
    assert records[0].filename == caller
    assert built == expected()


def test_router_start_passes_new_config(monkeypatch):
    # The native class is replaced: only its arguments matter here.
    monkeypatch.setattr(router_args, "AisPerfConfig", dict)

    config = parse_router_args([*ENDPOINT, *AIS, *NEW])

    assert router_args.build_ais_perf_config(config) == expected()
