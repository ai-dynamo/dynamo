# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIS perf model inputs of the frontend, the routers and squeeze_evolve.

They share one argument group, so each rule is checked once, through the
frontend parser. Two tests run in every component: a new value builds the
config, and old inputs warn once.
"""

import argparse
import json

import pytest
import yaml

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
    no_warning,
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

# The names a warning lists for OLD_FLAT and OLD_FLAT_ENV, with their verb.
OLD_FLAT_ARE = "--ais-backend, --ais-system and --ais-model-path are"
OLD_FLAT_ENV_ARE = "DYN_AIS_BACKEND, DYN_AIS_SYSTEM and DYN_AIS_MODEL_PATH are"
BAD_WORKER_TYPE_ENV = (
    "DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE is 'decode'; use prefill or aggregated"
)
NOTHING_GIVEN = (
    "--router-prefill-load-model=ais requires an AIS perf model: "
    "set --ais-perf-model or DYN_AIS_PERF_MODEL"
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


def _several(*worker_types):
    """Return inline JSON of a list with one config for each worker type."""
    return json.dumps(
        [{**IDENTITY, "worker_type": worker_type} for worker_type in worker_types]
    )


NEW = ["--ais-perf-model", _one(worker_type="aggregated")]


# Old inputs only: results and errors stay as they are.


def test_router_old_perf_config_builds_same_config():
    with _warns("--ais-perf-config is"):
        config = parse_router_args(
            [*ENDPOINT, *AIS, "--ais-perf-config", OLD_PERF_CONFIG]
        )

    assert config.ais_perf_kwargs() == expected(worker_type="prefill")


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


def test_old_perf_config_without_ais_load_model_is_rejected():
    config = _frontend([*NOT_AIS, "--ais-perf-config", OLD_PERF_CONFIG])

    with pytest.raises(
        ValueError,
        match=exactly("--ais-perf-config requires --router-prefill-load-model=ais"),
    ):
        config.validate()


def test_old_perf_config_env_without_ais_load_model_is_rejected(monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_CONFIG", OLD_PERF_CONFIG)
    config = _frontend(NOT_AIS)

    with pytest.raises(
        ValueError,
        match=exactly("--ais-perf-config requires --router-prefill-load-model=ais"),
    ):
        config.validate()


def test_old_perf_config_env_builds_same_config(monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_CONFIG", OLD_PERF_CONFIG)
    config = _frontend(AIS)

    with _warns("DYN_AIS_PERF_CONFIG is"):
        config.validate()

    assert config.ais_perf_kwargs() == expected(worker_type="prefill")


def test_old_flat_flags_without_ais_load_model_are_ignored():
    config = _frontend([*NOT_AIS, *OLD_FLAT])

    with _warns(OLD_FLAT_ARE):
        config.validate()

    assert config.router_prefill_load_model == "none"


def test_old_perf_config_with_flat_flag_is_rejected():
    config = _frontend(
        [*AIS, "--ais-perf-config", OLD_PERF_CONFIG, "--ais-backend", "vllm"]
    )

    with pytest.raises(
        ValueError,
        match=exactly(
            "--ais-perf-config cannot be combined with flat --ais-* / --aic-* "
            "identity flags"
        ),
    ):
        with _warns("--ais-perf-config and --ais-backend are"):
            config.validate()


def test_old_flat_env_only_builds_same_config(monkeypatch):
    for name, value in OLD_FLAT_ENV.items():
        monkeypatch.setenv(name, value)
    config = _frontend(AIS)

    with _warns(OLD_FLAT_ENV_ARE):
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

    with _warns("--aic-backend, --aic-system and --aic-model-path are"):
        config.validate()

    assert config.ais_perf_kwargs() == expected()
    with pytest.raises(SystemExit) as exit_info:
        _frontend([*AIS, *OLD_FLAT, "--aic-backend", "vllm"])
    assert exit_info.value.code == 2
    assert "new and legacy spellings cannot be combined" in capsys.readouterr().err


def test_old_flat_flags_without_a_model_keep_their_message():
    config = _frontend([*AIS, "--ais-backend", "vllm"])

    with pytest.raises(
        ValueError, match=exactly("AIS perf config requires system, model")
    ):
        with _warns("--ais-backend is"):
            config.validate()


def test_old_perf_config_with_empty_model_keeps_its_message():
    config = _frontend(
        [*AIS, "--ais-perf-config", _one(model="", worker_type="prefill")]
    )

    with pytest.raises(
        ValueError, match=exactly("AIS perf config requires model")
    ) as error:
        with _warns("--ais-perf-config is"):
            config.validate()

    assert type(error.value) is ValueError


def test_renamed_environment_variable_is_still_rejected(monkeypatch):
    monkeypatch.setenv("DYN_AIC_BACKEND", "vllm")
    config = _frontend([*AIS, *OLD_FLAT])

    with pytest.raises(
        ValueError,
        match=exactly(
            "renamed environment variables are no longer read: DYN_AIC_BACKEND; "
            "use the DYN_AIS_* names"
        ),
    ):
        with _warns(OLD_FLAT_ARE):
            config.validate()


# The new input.


@every_component
def test_new_value_builds_config_in_every_component(component):
    if component == "squeeze_evolve":
        # squeeze_evolve reads the value from the tier, not from the flag.
        config = _squeeze_evolve(AIS, ais_perf_model=NEW[1])
    else:
        config = COMPONENTS[component]([*AIS, *NEW])

    with no_warning():
        config.validate()
        if component == "squeeze_evolve":
            (built,) = config.ais_perf_kwargs_by_tier()
        else:
            built = config.ais_perf_kwargs()

    assert built == expected()


def test_new_env_builds_config(monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", _one(worker_type="prefill"))
    config = _frontend(AIS)

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


def test_empty_env_is_unset(monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "")
    config = _frontend(AIS)

    with pytest.raises(ValueError, match=exactly(NOTHING_GIVEN)):
        config.validate()


def test_empty_typed_value_reports_loader_error():
    config = _frontend([*AIS, "--ais-perf-model="])

    with pytest.raises(ValueError) as error:
        config.validate()

    # The loader's own error class: an empty value is read, not taken as unset,
    # which would be the plain ValueError of NOTHING_GIVEN.
    assert type(error.value) is not ValueError


def test_new_equals_old_for_same_identity():
    old = _frontend([*AIS, *OLD_FLAT, "--ais-tp-size", "2"])
    new = _frontend([*AIS, "--ais-perf-model", _one(worker_type="aggregated", tp=2)])

    with _warns("--ais-backend, --ais-system, --ais-tp-size and --ais-model-path are"):
        old.validate()
    new.validate()

    assert new.ais_perf_kwargs() == old.ais_perf_kwargs()


# The rules that select an entry are tested with dynamo.common.ais_perf_model;
# here the flag and the variable reach them.
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


def test_nothing_given_reports_new_message():
    config = _frontend(AIS)

    with pytest.raises(ValueError, match=exactly(NOTHING_GIVEN)):
        config.validate()


def test_env_only_new_value_is_ignored_without_ais_load_model(monkeypatch):
    monkeypatch.setenv("DYN_AIS_PERF_MODEL", "/not/a/local/file.yaml")
    monkeypatch.setenv("DYN_ROUTER_PREFILL_LOAD_WORKER_TYPE", "decode")
    config = _frontend(NOT_AIS)

    with no_warning():
        config.validate()

    assert config.router_prefill_load_model == "none"


def test_new_flag_without_ais_load_model_is_rejected():
    config = _frontend([*NOT_AIS, *NEW])

    with pytest.raises(
        ValueError,
        match=exactly("--ais-perf-model requires --router-prefill-load-model=ais"),
    ):
        config.validate()


def test_worker_type_flag_without_ais_load_model_is_rejected():
    config = _frontend([*NOT_AIS, "--router-prefill-load-worker-type", "prefill"])

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


def test_new_attributes_do_not_reach_kv_router_kwargs():
    config = _frontend([*AIS, *NEW, "--router-prefill-load-worker-type", "aggregated"])
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


@pytest.mark.parametrize(
    ("args", "env", "error"),
    [
        pytest.param(
            [*NEW, "--ais-backend", "vllm"],
            {},
            "--ais-perf-model cannot be combined with --ais-backend",
            id="flag-with-flag",
        ),
        # A flag typed with the --aic- spelling is named as it was typed.
        pytest.param(
            [*NEW, "--aic-backend", "vllm"],
            {},
            "--ais-perf-model cannot be combined with --aic-backend",
            id="flag-with-aic-spelling",
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
        # Unlike the mocker, this group takes no --ais-nextn with the new input.
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
def test_new_with_old_is_rejected_and_names_both(args, env, error, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = _frontend([*AIS, *args])

    with pytest.raises(ValueError, match=exactly(error)), no_warning():
        config.validate()


def test_worker_type_flag_with_old_inputs_is_rejected():
    config = _frontend(
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

    with _warns(OLD_FLAT_ARE):
        config.validate()

    assert config.ais_perf_kwargs() == expected()


# The warning for old inputs.


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
def test_old_inputs_warn_once_with_exact_text(args, env, names, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    config = _frontend([*AIS, *args])

    with pytest.warns(FutureWarning) as records:
        config.validate()

    assert [str(record.message) for record in records] == [_warning_text(names)]
    assert records[0].category is FutureWarning
    assert records[0].filename == __file__


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


def test_second_ais_perf_kwargs_call_does_not_warn():
    config = _frontend([*AIS, *OLD_FLAT])
    with _warns(OLD_FLAT_ARE):
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

    assert [str(record.message) for record in records] == [_warning_text(OLD_FLAT_ARE)]
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

    # The same config in a folder without the file: it is looked up again.
    monkeypatch.chdir(other_dir)
    with pytest.raises(ValueError):
        config.ais_perf_kwargs()
