# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIS perf model inputs of the frontend, the routers and squeeze_evolve."""

import argparse
import json
import re
from pathlib import Path

import pytest
import yaml

import dynamo.router.args as router_args
import dynamo.thunderagent_router.args as thunderagent_router_args
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
from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig
from dynamo.router.args import DynamoRouterArgGroup, DynamoRouterConfig
from dynamo.router.args import parse_args as parse_router_args
from dynamo.squeeze_evolve.args import SqueezeEvolveArgGroup, SqueezeEvolveRunConfig
from dynamo.thunderagent_router.args import (
    ThunderAgentArgGroup,
    ThunderAgentRouterConfig,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.usefixtures("no_ais_environment"),
]

OLD_FLAT_ENV = {
    "DYN_AIS_BACKEND": "vllm",
    "DYN_AIS_SYSTEM": "h200_sxm",
    "DYN_AIS_MODEL_PATH": MODEL,
}
OLD_PERF_CONFIG = json.dumps({**IDENTITY, "worker_type": "prefill"})
ENDPOINT = ["--endpoint", "dynamo.prefill.generate"]
ONE_TIER = json.dumps([{"endpoint": "dynamo.small.generate", "model": MODEL}])

# The names a warning lists for OLD_FLAT and OLD_FLAT_ENV, with their verb.
OLD_FLAT_ARE = "--ais-backend, --ais-system and --ais-model-path are"
OLD_FLAT_ENV_ARE = "DYN_AIS_BACKEND, DYN_AIS_SYSTEM and DYN_AIS_MODEL_PATH are"
BAD_WORKER_TYPE_ENV = (
    "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE is 'decode'; use prefill or aggregated"
)


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


def _squeeze_evolve(args):
    parser = argparse.ArgumentParser()
    SqueezeEvolveArgGroup().add_arguments(parser)
    return SqueezeEvolveRunConfig.from_cli_args(
        parser.parse_args(["--tiers", ONE_TIER, "--model-name", "squeeze/test", *args])
    )


# Each entry parses the arguments with the component's own parser and returns
# the config before validate() ran.
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
    "squeeze_evolve": (
        "ais_perf_model in each --tiers entry (or --ais-perf-model with one tier)"
    ),
}


def _warning_text(component, names):
    """Return the warning of ``component``; ``names`` ends with ``is`` or ``are``."""
    return (
        f"{names} deprecated and will be removed in v1.9; "
        f"use {REPLACEMENT[component]}."
    )


def _warns(component, names):
    return pytest.warns(FutureWarning, match=exactly(_warning_text(component, names)))


def _one(**keys):
    """Return inline JSON of one AIS perf model config."""
    return json.dumps({**IDENTITY, **keys})


def _several(*worker_types):
    """Return inline JSON of a list with one config for each worker type."""
    return json.dumps(
        [{**IDENTITY, "worker_type": worker_type} for worker_type in worker_types]
    )


NEW = ["--ais-perf-model", _one(worker_type="aggregated")]


# Old inputs only: results and errors stay as they are.


def test_router_old_perf_config_builds_same_config():
    with _warns("router", "--ais-perf-config is"):
        config = parse_router_args(
            [*ENDPOINT, *AIS, "--ais-perf-config", OLD_PERF_CONFIG]
        )

    assert config.ais_perf_kwargs() == expected(worker_type="prefill")


@every_component
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
def test_old_flat_flags_build_same_config_in_every_component(
    component, flag, names, keys
):
    config = COMPONENTS[component]([*AIS, *OLD_FLAT, *flag])

    with _warns(component, names):
        config.validate()

    assert config.ais_perf_kwargs() == expected(**keys)


@every_component
def test_old_perf_config_without_ais_load_model_is_rejected(component):
    config = COMPONENTS[component]([*NOT_AIS, "--ais-perf-config", OLD_PERF_CONFIG])

    with pytest.raises(
        ValueError,
        match=exactly("--ais-perf-config requires --router-prefill-load-model=ais"),
    ):
        config.validate()


@every_component
def test_old_perf_config_env_without_ais_load_model_is_rejected(component, monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_CONFIG", OLD_PERF_CONFIG)
    config = COMPONENTS[component](NOT_AIS)

    with pytest.raises(
        ValueError,
        match=exactly("--ais-perf-config requires --router-prefill-load-model=ais"),
    ):
        config.validate()


@every_component
def test_old_perf_config_env_builds_same_config(component, monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_CONFIG", OLD_PERF_CONFIG)
    config = COMPONENTS[component](AIS)

    with _warns(component, "DYN_AIS_PERF_CONFIG is"):
        config.validate()

    assert config.ais_perf_kwargs() == expected(worker_type="prefill")


@every_component
def test_old_flat_flags_without_ais_load_model_are_ignored(component):
    config = COMPONENTS[component]([*NOT_AIS, *OLD_FLAT])

    with _warns(component, OLD_FLAT_ARE):
        config.validate()

    assert config.router_prefill_load_model == "none"


@every_component
def test_old_perf_config_with_flat_flag_is_rejected(component):
    config = COMPONENTS[component](
        [*AIS, "--ais-perf-config", OLD_PERF_CONFIG, "--ais-backend", "vllm"]
    )

    with pytest.raises(
        ValueError,
        match=exactly(
            "--ais-perf-config cannot be combined with flat --ais-* / --aic-* "
            "identity flags"
        ),
    ):
        with _warns(component, "--ais-perf-config and --ais-backend are"):
            config.validate()


@every_component
def test_old_flat_env_only_builds_same_config(component, monkeypatch):
    for name, value in OLD_FLAT_ENV.items():
        monkeypatch.setenv(name, value)
    config = COMPONENTS[component](AIS)

    with _warns(component, OLD_FLAT_ENV_ARE):
        config.validate()

    assert config.ais_perf_kwargs() == expected()


def test_old_aic_spelling_builds_same_config(capsys):
    config = _frontend(
        [
            *AIS,
            "--aic-backend",
            "vllm",
            "--aic-system",
            "h200_sxm",
            "--aic-model-path",
            MODEL,
        ]
    )

    with _warns("frontend", "--aic-backend, --aic-system and --aic-model-path are"):
        config.validate()

    assert config.ais_perf_kwargs() == expected()
    with pytest.raises(SystemExit) as exit_info:
        _frontend([*AIS, *OLD_FLAT, "--aic-backend", "vllm"])
    assert exit_info.value.code == 2
    assert "new and legacy spellings cannot be combined" in capsys.readouterr().err


@every_component
def test_old_flat_flags_without_a_model_keep_their_message(component):
    config = COMPONENTS[component]([*AIS, "--ais-backend", "vllm"])

    with pytest.raises(
        ValueError, match=exactly("AIS perf config requires system, model")
    ):
        with _warns(component, "--ais-backend is"):
            config.validate()


@every_component
def test_old_perf_config_with_empty_model_keeps_its_message(component):
    config = COMPONENTS[component](
        [*AIS, "--ais-perf-config", _one(model="", worker_type="prefill")]
    )

    with pytest.raises(
        ValueError, match=exactly("AIS perf config requires model")
    ) as error:
        with _warns(component, "--ais-perf-config is"):
            config.validate()

    assert type(error.value) is ValueError


@every_component
def test_renamed_environment_variable_is_still_rejected(component, monkeypatch):
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    config = COMPONENTS[component]([*AIS, *OLD_FLAT])
    # squeeze_evolve checks the variable first, the others after the warning.
    warning = (
        no_warning()
        if component == "squeeze_evolve"
        else _warns(component, OLD_FLAT_ARE)
    )

    with pytest.raises(
        ValueError,
        match=exactly(
            "renamed environment variables are no longer read: DYN_AIC_BACKEND; "
            "use the DYN_AIS_* names"
        ),
    ):
        with warning:
            config.validate()


# The new input.


@every_component
def test_new_flag_inline_json_builds_config(component):
    config = COMPONENTS[component]([*AIS, *NEW])

    with no_warning():
        config.validate()

    assert config.ais_perf_kwargs() == expected()


@every_component
def test_new_env_builds_config(component, monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", _one(worker_type="prefill"))
    config = COMPONENTS[component](AIS)

    with no_warning():
        config.validate()

    assert config.ais_perf_kwargs() == expected(worker_type="prefill")


def test_typed_value_wins_over_env(monkeypatch):
    monkeypatch.setenv(
        "DYN_AIS_PERF_MODEL", _one(model="other/model", worker_type="prefill")
    )
    config = _frontend([*AIS, *NEW])

    config.validate()

    assert config.ais_perf_kwargs() == expected()


@every_component
def test_empty_env_is_unset(component, monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "")
    config = COMPONENTS[component](AIS)

    with pytest.raises(ValueError, match=exactly(NOTHING_GIVEN)):
        config.validate()


@every_component
def test_empty_typed_value_reports_loader_error(component):
    config = COMPONENTS[component]([*AIS, "--ais-perf-model="])

    with pytest.raises(
        ValueError,
        match=exactly(
            "ais_perf_model: the value is empty; write a file path or inline JSON"
        ),
    ):
        config.validate()


@every_component
@pytest.mark.parametrize(
    ("value", "error"),
    [
        pytest.param(
            _one(model="", worker_type="prefill"),
            "ais_perf_model requires model",
            id="empty-model",
        ),
        pytest.param(
            json.dumps(
                {"system": "h200_sxm", "backend": "vllm", "worker_type": "prefill"}
            ),
            "ais_perf_model requires model",
            id="no-model",
        ),
        pytest.param(
            _one(model="", backend="", worker_type="prefill"),
            "ais_perf_model requires backend, model",
            id="empty-model-and-backend",
        ),
    ],
)
def test_new_value_without_backend_or_model_is_rejected(component, value, error):
    config = COMPONENTS[component]([*AIS, "--ais-perf-model", value])

    with pytest.raises(ValueError, match=exactly(error)) as raised, no_warning():
        config.validate()

    assert type(raised.value) is ValueError


def test_new_equals_old_for_same_identity():
    old = _frontend([*AIS, *OLD_FLAT, "--ais-tp-size", "2"])
    new = _frontend([*AIS, "--ais-perf-model", _one(worker_type="aggregated", tp=2)])

    with _warns(
        "frontend",
        "--ais-backend, --ais-system, --ais-tp-size and --ais-model-path are",
    ):
        old.validate()
    new.validate()

    assert new.ais_perf_kwargs() == old.ais_perf_kwargs()


# The selection rules have their own table in the tests of
# dynamo.common.ais_perf_model.
@pytest.mark.parametrize(
    ("value", "typed", "worker_type", "error"),
    [
        pytest.param(
            _one(), "prefill", "prefill", None, id="one-config-without-type-prefill"
        ),
        pytest.param(
            _one(worker_type="prefill"),
            "aggregated",
            None,
            "ais_perf_model entry: worker_type is 'prefill' but worker type "
            "'aggregated' was requested",
            id="one-prefill-config-aggregated",
        ),
        pytest.param(
            _one(worker_type="decode"),
            "prefill",
            None,
            "ais_perf_model entry: worker_type is 'decode' but worker type "
            "'prefill' was requested",
            id="one-decode-config-prefill",
        ),
        pytest.param(
            _several("decode", "aggregated"),
            None,
            "aggregated",
            None,
            id="list-with-aggregated",
        ),
    ],
)
def test_router_worker_type_table(value, typed, worker_type, error):
    args = [*AIS, "--ais-perf-model", value]
    if typed is not None:
        args += ["--router-prefill-load-worker-type", typed]
    config = _frontend(args)

    if error is None:
        assert config.ais_perf_kwargs() == expected(worker_type=worker_type)
    else:
        with pytest.raises(ValueError, match=exactly(error)):
            config.ais_perf_kwargs()


def test_worker_type_env_is_used_and_checked(monkeypatch):
    several = ["--ais-perf-model", _several("prefill", "aggregated")]
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "prefill")

    config = _frontend([*AIS, *several])
    config.validate()
    assert config.ais_perf_kwargs() == expected(worker_type="prefill")

    typed = _frontend(
        [*AIS, *several, "--router-prefill-load-worker-type", "aggregated"]
    )
    typed.validate()
    assert typed.ais_perf_kwargs() == expected(worker_type="aggregated")

    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    with pytest.raises(ValueError, match=exactly(BAD_WORKER_TYPE_ENV)):
        _frontend([*AIS, *several]).validate()
    # A config that was not validated reports it too.
    with pytest.raises(ValueError, match=exactly(BAD_WORKER_TYPE_ENV)):
        _frontend([*AIS, *several]).ais_perf_kwargs()
    # A typed worker type is used and the variable is not read.
    typed.validate()
    assert typed.ais_perf_kwargs() == expected(worker_type="aggregated")

    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "")
    with pytest.raises(ValueError, match="it has entries for prefill and aggregated"):
        _frontend([*AIS, *several]).validate()


@every_component
def test_nothing_given_reports_new_message(component):
    config = COMPONENTS[component](AIS)

    with pytest.raises(ValueError, match=exactly(NOTHING_GIVEN)):
        config.validate()


@every_component
def test_env_only_new_value_is_ignored_without_ais_load_model(component, monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "/not/a/local/file.yaml")
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    config = COMPONENTS[component](NOT_AIS)

    with no_warning():
        config.validate()

    assert config.router_prefill_load_model == "none"


@every_component
def test_new_flag_without_ais_load_model_is_rejected(component):
    config = COMPONENTS[component]([*NOT_AIS, *NEW])

    with pytest.raises(
        ValueError,
        match=exactly("--ais-perf-model requires --router-prefill-load-model=ais"),
    ):
        config.validate()


@every_component
def test_worker_type_flag_without_ais_load_model_is_rejected(component):
    config = COMPONENTS[component](
        [*NOT_AIS, "--router-prefill-load-worker-type", "prefill"]
    )

    with pytest.raises(
        ValueError,
        match=exactly(
            "--router-prefill-load-worker-type requires "
            "--router-prefill-load-model=ais"
        ),
    ):
        config.validate()


def test_ais_perf_kwargs_twice_is_equal():
    config = _frontend([*AIS, *NEW])
    config.validate()

    first = config.ais_perf_kwargs()
    second = config.ais_perf_kwargs()

    assert first == second == expected()
    assert first["config"] is not second["config"]


@every_component
def test_new_attributes_do_not_reach_kv_router_kwargs(component):
    config = COMPONENTS[component](
        [*AIS, *NEW, "--router-prefill-load-worker-type", "aggregated"]
    )
    config.validate()

    kwargs = config.kv_router_kwargs()

    assert "ais_perf_model" not in kwargs
    assert "router_prefill_load_worker_type" not in kwargs


def test_accept_rates_checked_with_new_input():
    value = _one(worker_type="aggregated", nextn=2)

    config = _frontend(
        [*AIS, "--ais-perf-model", value, "--ais-nextn-accept-rates", "0.5,0.5"]
    )
    config.validate()
    assert config.ais_perf_kwargs() == expected(nextn=2)

    bad = _frontend(
        [*AIS, "--ais-perf-model", value, "--ais-nextn-accept-rates", "abc"]
    )
    with pytest.raises(
        ValueError,
        match="^ais_nextn_accept_rates must be comma-separated floats, got 'abc'$",
    ):
        bad.validate()


def test_unknown_key_in_new_value_is_a_value_error():
    config = _frontend(
        [*AIS, "--ais-perf-model", _one(worker_type="aggregated", not_a_field=2)]
    )

    with pytest.raises(
        ValueError,
        match="^ais_perf_model: .*unexpected keyword argument 'not_a_field'$",
    ) as error:
        config.validate()

    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "args",
    [
        ["--ais-p", "x"],
        ["--ais-perf", "x"],
        ["--ais-perf-", "x"],
        # Ambiguous before the new flags too; kept as a guard.
        ["--router-prefill-load", "ais"],
    ],
)
def test_abbreviations_are_ambiguous(args, capsys):
    with pytest.raises(SystemExit) as exit_info:
        _frontend(args)

    assert exit_info.value.code == 2
    assert "ambiguous option" in capsys.readouterr().err


def test_frontend_config_has_two_new_fields_and_no_cache():
    config = _frontend([*AIS, *NEW])
    plain = _frontend([])

    config.validate()
    plain.validate()
    after_validate = dict(vars(config))
    kwargs = config.ais_perf_kwargs()

    fields = vars(config)
    assert fields == after_validate
    assert fields["ais_perf_model"] == NEW[1]
    assert fields["router_prefill_load_worker_type"] is None
    # Nothing on the config holds the config that was read.
    assert not any(value in (kwargs, kwargs["config"]) for value in fields.values())
    # The same fields as a config with no AIS input.
    assert set(fields) == set(vars(plain))
    assert vars(plain)["ais_perf_model"] is None


# Old and new inputs together.


@every_component
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
            NEW,
            OLD_FLAT_ENV,
            "--ais-perf-model cannot be combined with DYN_AIS_BACKEND, "
            "DYN_AIS_SYSTEM and DYN_AIS_MODEL_PATH",
            id="flag-with-env",
        ),
        pytest.param(
            [*NEW, "--ais-perf-config", OLD_PERF_CONFIG],
            {},
            "--ais-perf-model cannot be combined with --ais-perf-config",
            id="flag-with-perf-config",
        ),
        pytest.param(
            NEW,
            {"DYN_AIS_PERF_CONFIG": OLD_PERF_CONFIG},
            "--ais-perf-model cannot be combined with DYN_AIS_PERF_CONFIG",
            id="flag-with-perf-config-env",
        ),
        # Unlike the mocker, these components take no --ais-nextn with the new input.
        pytest.param(
            [*NEW, "--ais-nextn", "2"],
            {},
            "--ais-perf-model cannot be combined with --ais-nextn",
            id="flag-with-nextn",
        ),
        pytest.param(
            [],
            {"DYN_AIS_PERF_MODEL": NEW[1], "DYN_AIS_BACKEND": "vllm"},
            "DYN_AIS_PERF_MODEL cannot be combined with DYN_AIS_BACKEND",
            id="env-with-env",
        ),
        pytest.param(
            [],
            {"DYN_AIS_PERF_MODEL": NEW[1], "DYN_AIS_PERF_CONFIG": OLD_PERF_CONFIG},
            "DYN_AIS_PERF_MODEL cannot be combined with DYN_AIS_PERF_CONFIG",
            id="env-with-perf-config-env",
        ),
    ],
)
def test_new_with_old_is_rejected_and_names_both(
    component, args, env, error, monkeypatch
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = COMPONENTS[component]([*AIS, *args])

    with pytest.raises(ValueError, match=exactly(error)), no_warning():
        config.validate()


@every_component
def test_worker_type_flag_with_old_inputs_is_rejected(component):
    config = COMPONENTS[component](
        [*AIS, *OLD_FLAT, "--router-prefill-load-worker-type", "prefill"]
    )

    with pytest.raises(
        ValueError,
        match=exactly("--router-prefill-load-worker-type requires --ais-perf-model"),
    ), no_warning():
        config.validate()


def test_worker_type_env_with_old_inputs_is_ignored(monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    config = _frontend([*AIS, *OLD_FLAT])

    with _warns("frontend", OLD_FLAT_ARE):
        config.validate()

    assert config.ais_perf_kwargs() == expected()


# The warning for old inputs.


@every_component
@pytest.mark.parametrize(
    ("args", "env", "names"),
    [
        pytest.param(
            [
                "--aic-backend",
                "vllm",
                "--ais-model-path",
                MODEL,
                "--ais-nextn-accept-rates",
                "0.5",
                "--ais-mtp-seed",
                "7",
            ],
            {"DYN_AIS_SYSTEM": "h200_sxm"},
            "--aic-backend, DYN_AIS_SYSTEM and --ais-model-path are",
            id="flags-and-env",
        ),
        pytest.param(
            [],
            {
                "DYN_AIS_PERF_CONFIG": OLD_PERF_CONFIG,
                "DYN_AIS_NEXTN_ACCEPT_RATES": "0.5",
                "DYN_AIS_MTP_SEED": "7",
            },
            "DYN_AIS_PERF_CONFIG is",
            id="perf-config-env",
        ),
        pytest.param(
            ["--ais-perf-config", OLD_PERF_CONFIG],
            {},
            "--ais-perf-config is",
            id="perf-config-flag",
        ),
    ],
)
def test_old_inputs_warn_once_with_exact_text(component, args, env, names, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = COMPONENTS[component]([*AIS, *args])

    with pytest.warns(FutureWarning) as records:
        config.validate()

    assert [str(record.message) for record in records] == [
        _warning_text(component, names)
    ]
    assert records[0].category is FutureWarning
    # The warning points at the code that called the validate() with the check;
    # for thunderagent_router that is its own validate().
    caller = (
        thunderagent_router_args.__file__
        if component == "thunderagent_router"
        else __file__
    )
    assert records[0].filename == caller


@every_component
def test_second_ais_perf_kwargs_call_does_not_warn(component):
    config = COMPONENTS[component]([*AIS, *OLD_FLAT])
    with _warns(component, OLD_FLAT_ARE):
        config.validate()

    with no_warning():
        first = config.ais_perf_kwargs()
        second = config.ais_perf_kwargs()

    assert first == second == expected()


def test_router_start_warns_once_for_old_inputs(monkeypatch):
    # The native class is replaced: only its arguments matter here.
    monkeypatch.setattr(router_args, "AisPerfConfig", dict)

    with pytest.warns(FutureWarning) as records:
        config = parse_router_args([*ENDPOINT, *AIS, *OLD_FLAT])
        built = router_args.build_ais_perf_config(config)

    assert [str(record.message) for record in records] == [
        _warning_text("router", OLD_FLAT_ARE)
    ]
    assert records[0].filename == router_args.__file__
    assert built == expected()


def test_router_start_passes_new_config(monkeypatch):
    # The native class is replaced: only its arguments matter here.
    monkeypatch.setattr(router_args, "AisPerfConfig", dict)

    with no_warning():
        config = parse_router_args([*ENDPOINT, *AIS, *NEW])
        built = router_args.build_ais_perf_config(config)

    assert built == expected()


# Relative paths.


def test_relative_file_value_starts_at_working_directory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    other_dir = tmp_path / "other"
    run_dir.mkdir()
    other_dir.mkdir()
    (run_dir / "cfg.yaml").write_text(
        yaml.safe_dump({**IDENTITY, "worker_type": "prefill"})
    )
    config = _frontend([*AIS, "--ais-perf-model", "cfg.yaml"])

    monkeypatch.chdir(run_dir)
    config.validate()
    assert config.ais_perf_kwargs() == expected(worker_type="prefill")

    monkeypatch.chdir(other_dir)
    missing = Path.cwd() / "cfg.yaml"
    with pytest.raises(ValueError, match="^" + re.escape(f"cannot read {missing}: ")):
        config.ais_perf_kwargs()


def test_relative_paths_inside_file_start_at_file_folder(tmp_path, monkeypatch):
    conf = tmp_path / "conf"
    run_dir = tmp_path / "run"
    (conf / "sys").mkdir(parents=True)
    run_dir.mkdir()
    (conf / "cfg.yaml").write_text(
        yaml.safe_dump({**IDENTITY, "worker_type": "prefill", "systems_paths": ["sys"]})
    )
    monkeypatch.chdir(run_dir)
    config = _frontend([*AIS, "--ais-perf-model", str(conf / "cfg.yaml")])

    config.validate()

    assert config.ais_perf_kwargs()["config"]["systems_paths"] == [str(conf / "sys")]
