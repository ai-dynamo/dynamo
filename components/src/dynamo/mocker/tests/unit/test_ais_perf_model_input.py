# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import json
import sys
import warnings

import numpy as np
import pytest
import yaml

from dynamo.mocker import config as CONFIG
from dynamo.mocker.args import parse_args
from tests.utils.ais_perf_model import ENV_VAR

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
    pytest.mark.usefixtures("no_ais_environment"),
]

# The mocker checks an AIS config against the installed timing data. This
# model's metadata and tables ship with AISimulate, so nothing is downloaded.
MODEL = "Qwen/Qwen3-32B"
# (--disaggregation-mode value, worker type of the built config)
WORKER_TYPES = [("agg", "aggregated"), ("prefill", "prefill"), ("decode", "decode")]
# Every flat identity flag (without its --ais/--aic prefix) with a value the
# installed timing data accepts, and the config key it sets.
FLAT_FLAGS = [
    ("backend", "sglang", "backend", "sglang"),
    ("system", "h100_sxm", "system", "h100_sxm"),
    ("backend-version", "current", "backend_version", "current"),
    ("tp-size", "2", "tp", 2),
    ("moe-tp-size", "2", "moe_tp_size", 2),
    ("moe-ep-size", "1", "moe_ep_size", 1),
    ("attention-dp-size", "1", "attention_dp", 1),
]
FLAT_CONFIG = {key: value for _, _, key, value in FLAT_FLAGS}
OLD_CONFIG = {"model": MODEL, "system": "h100_sxm", "backend": "vllm", "tp": 2}
OLD_CONFIG_CONFLICT = (
    "--ais-perf-config cannot be combined with flat AIS/AIC identity flags"
)
PLANNER_DATA_CONFLICT = "choose one timing provider: AIS or planner_profile_data"

# How the warning for old inputs ends, after the names of the inputs.
REMOVAL = " deprecated and will be removed in v1.9; use --ais-perf-model VALUE."
OLD_INPUT_WARNING = (
    r"deprecated and will be removed in v1\.9; use --ais-perf-model VALUE\.$"
)
# One perf model config for each worker type, as a value writes it. They
# differ, so a test sees which one the mocker took.
ENTRIES = {
    "aggregated": {"model": MODEL, "system": "h200_sxm", "backend": "vllm", "tp": 2},
    "prefill": {"model": MODEL, "system": "h100_sxm", "backend": "vllm", "tp": 2},
    "decode": {"model": MODEL, "system": "h200_sxm", "backend": "vllm", "tp": 1},
}
ENTRY = ENTRIES["aggregated"]
TIMING_MODEL_CONFLICT = (
    "--extra-engine-args already sets engine.timing_model; it cannot be combined with"
)


def flat_flags(prefix="--ais"):
    return [
        token for name, text, _, _ in FLAT_FLAGS for token in (f"{prefix}-{name}", text)
    ]


def build(argv):
    return CONFIG.build_mocker_engine_args(parse_args(argv))


def parse_old(argv):
    """Parse a command line that holds old inputs, which warns."""
    with pytest.warns(FutureWarning, match=OLD_INPUT_WARNING):
        return parse_args(argv)


def build_old(argv):
    return CONFIG.build_mocker_engine_args(parse_old(argv))


def written_keys(config, expected):
    """Return the keys of ``expected`` as the built perf config holds them.

    The mocker returns the config with every default AISimulate adds, so a
    test compares the keys its input wrote.
    """
    return {key: config[key] for key in expected}


def assert_perf_config(result, expected):
    config = CONFIG.performance_config(result)
    assert written_keys(config, expected) == expected


def entries(*worker_types):
    """Return a list value with one config for each of ``worker_types``."""
    return [
        {"worker_type": worker_type, **ENTRIES[worker_type]}
        for worker_type in worker_types
    ]


def assert_refused_key_error(error, key):
    """Check the mocker's error for a ``key`` the perf model config class refuses."""
    assert type(error) is ValueError
    assert isinstance(error.__cause__, TypeError)
    # The text of that class's TypeError follows the prefix.
    assert str(error) == f"ais_perf_model: {error.__cause__}"
    assert key in str(error.__cause__)


def parser_error(argv, capsys):
    """Return the message the parser ends the process with for ``argv``."""
    with pytest.raises(SystemExit) as error:
        parse_args(argv)
    assert error.value.code == 2
    return capsys.readouterr().err.rsplit(": error: ", 1)[1].strip()


def engine_args_file(tmp_path, engine):
    path = tmp_path / "engine.json"
    path.write_text(json.dumps({"engine": engine}))
    return str(path)


def value_argv(source, monkeypatch):
    """Return the arguments that give ``ENTRY`` through the flag or the variable."""
    if source == ENV_VAR:
        monkeypatch.setenv(ENV_VAR, json.dumps(ENTRY))
        return []
    return [source, json.dumps(ENTRY)]


@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_switch_builds_default_identity(mode, worker_type):
    result = build_old(
        [
            "--ais-perf-model",
            "--model-path",
            MODEL,
            "--engine-type",
            "vllm",
            "--disaggregation-mode",
            mode,
        ]
    )
    assert_perf_config(
        result,
        {
            "model": MODEL,
            "backend": "vllm",
            "system": "h200_sxm",
            "worker_type": worker_type,
        },
    )


@pytest.mark.parametrize("prefix", ["--ais", "--aic"])
@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_switch_with_flat_flags_overrides_identity(prefix, mode, worker_type):
    result = build_old(
        [
            f"{prefix}-perf-model",
            "--model-path",
            MODEL,
            *flat_flags(prefix),
            f"{prefix}-nextn",
            "3",
            "--disaggregation-mode",
            mode,
        ]
    )
    assert_perf_config(
        result,
        {"model": MODEL, **FLAT_CONFIG, "nextn": 3, "worker_type": worker_type},
    )


@pytest.mark.parametrize("form", ["json", "file"])
@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_old_perf_config_builds_config(form, mode, worker_type, tmp_path):
    value = json.dumps(OLD_CONFIG)
    if form == "file":
        path = tmp_path / "perf.yaml"
        path.write_text(value)
        value = str(path)
    result = build_old(["--ais-perf-config", value, "--disaggregation-mode", mode])
    assert_perf_config(result, {**OLD_CONFIG, "worker_type": worker_type})


def test_old_perf_config_errors_are_unchanged():
    other_type = json.dumps({**OLD_CONFIG, "worker_type": "prefill"})
    args = parse_old(
        ["--ais-perf-config", other_type, "--disaggregation-mode", "decode"]
    )
    with pytest.raises(ValueError, match="^AIS worker_type must match engine role$"):
        CONFIG.build_mocker_engine_args(args)

    args = parse_old(["--ais-perf-config", json.dumps({**OLD_CONFIG, "tp_size": 3})])
    with pytest.raises(TypeError, match="tp_size"):
        CONFIG.build_mocker_engine_args(args)


@pytest.mark.parametrize(
    "extra",
    [
        ["--ais-perf-model"],
        *([f"--ais-{name}", text] for name, text, _, _ in FLAT_FLAGS),
        ["--ais-nextn", "2"],
    ],
    ids=lambda extra: extra[0],
)
def test_old_perf_config_with_switch_or_flat_flag_is_rejected(extra):
    args = parse_old(["--ais-perf-config", json.dumps(OLD_CONFIG), *extra])
    with pytest.raises(ValueError, match=f"^{OLD_CONFIG_CONFLICT}$"):
        CONFIG.build_mocker_engine_args(args)


@pytest.mark.parametrize("flags", [["--ais-backend", "vllm"], flat_flags()])
@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_flat_flags_without_switch_change_nothing(flags, mode, worker_type):
    base = ["--model-path", MODEL, "--disaggregation-mode", mode]
    result = build_old([*base, *flags])
    assert CONFIG.performance_config(result) is None
    assert result["engine"]["worker_type"] == worker_type
    assert result == build(base)


@pytest.mark.parametrize(
    "old_input",
    [["--ais-perf-model"], ["--ais-perf-config", json.dumps(OLD_CONFIG)]],
    ids=["switch", "perf-config"],
)
def test_switch_with_planner_profile_data_is_rejected(old_input, tmp_path):
    args = parse_old(
        [
            *old_input,
            "--model-path",
            MODEL,
            "--planner-profile-data",
            str(tmp_path / "profile.npz"),
        ]
    )
    with pytest.raises(ValueError, match=f"^{PLANNER_DATA_CONFLICT}$"):
        CONFIG.build_mocker_engine_args(args)


def test_mixed_spellings_are_rejected(capsys):
    with pytest.raises(SystemExit, match="2"):
        parse_args(["--ais-perf-model", "--aic-perf-model"])
    assert (
        "--ais-perf-model and its legacy --aic spelling cannot be combined"
        in capsys.readouterr().err
    )


@pytest.mark.parametrize(
    "old_inputs",
    [
        ["--ais-perf-model", "--ais-backend", "vllm"],
        ["--ais-perf-model", *flat_flags()],
        ["--ais-perf-config", json.dumps(OLD_CONFIG)],
    ],
    ids=["switch-and-backend", "switch-and-every-flat-flag", "perf-config"],
)
@pytest.mark.parametrize("worker_type", ["aggregated", "prefill", "decode"])
@pytest.mark.parametrize(
    "file_timing_model",
    [None, {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}],
    ids=["no-timing-model", "own-timing-model"],
)
def test_old_inputs_with_extra_engine_args_are_ignored(
    old_inputs, worker_type, file_timing_model, tmp_path
):
    engine = {"worker_type": worker_type, "max_num_seqs": 7}
    if file_timing_model is not None:
        engine["timing_model"] = file_timing_model
    path = tmp_path / "engine.json"
    path.write_text(json.dumps({"engine": engine}))
    base = ["--model-path", MODEL, "--extra-engine-args", str(path)]

    result = CONFIG.load_mocker_engine_args(parse_old([*base, *old_inputs]))

    assert result == CONFIG.load_mocker_engine_args(parse_args(base))
    assert result["engine"]["worker_type"] == worker_type
    assert result["engine"]["max_num_seqs"] == 7
    assert CONFIG.performance_config(result) is None


def test_repeated_switch_is_accepted():
    base = ["--model-path", MODEL, "--engine-type", "vllm"]
    result = build_old(["--ais-perf-model", "--ais-perf-model", *base])
    assert result == build_old(["--ais-perf-model", *base])
    assert_perf_config(
        result,
        {
            "model": MODEL,
            "backend": "vllm",
            "system": "h200_sxm",
            "worker_type": "aggregated",
        },
    )


@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_value_as_inline_json(mode, worker_type):
    args = parse_args(
        ["--ais-perf-model", json.dumps(ENTRY), "--disaggregation-mode", mode]
    )
    assert args.ais_perf_model_without_value is False
    assert args.ais_perf_model_value == json.dumps(ENTRY)
    assert args.ais_perf_model_source == "--ais-perf-model"
    assert not hasattr(args, "ais_perf_model")
    assert not hasattr(args, "ais_perf_model_occurrences")

    result = CONFIG.build_mocker_engine_args(args)

    assert_perf_config(result, {**ENTRY, "worker_type": worker_type})
    assert result["engine"]["timing_model"]["type"] == "external"
    assert result["engine"]["timing_model"]["provider"] == "ais"


@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES[1:])
def test_value_as_file(mode, worker_type, tmp_path):
    path = tmp_path / "perf.yaml"
    path.write_text(yaml.safe_dump(entries("prefill", "decode")))
    result = build(["--ais-perf-model", str(path), "--disaggregation-mode", mode])
    assert_perf_config(result, {**ENTRIES[worker_type], "worker_type": worker_type})


def test_value_from_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, json.dumps(ENTRY))
    args = parse_args([])
    assert args.ais_perf_model_without_value is False
    assert args.ais_perf_model_value == json.dumps(ENTRY)
    assert args.ais_perf_model_source == ENV_VAR
    assert_perf_config(
        CONFIG.build_mocker_engine_args(args), {**ENTRY, "worker_type": "aggregated"}
    )


def test_empty_env_value_is_no_value(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "")
    args = parse_args([])
    assert args.ais_perf_model_value is None
    assert args.ais_perf_model_source is None
    assert CONFIG.performance_config(CONFIG.build_mocker_engine_args(args)) is None


def test_typed_value_wins_over_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, json.dumps(ENTRIES["decode"]))
    args = parse_args(["--ais-perf-model", json.dumps(ENTRY)])
    assert args.ais_perf_model_value == json.dumps(ENTRY)
    assert args.ais_perf_model_source == "--ais-perf-model"
    assert_perf_config(
        CONFIG.build_mocker_engine_args(args), {**ENTRY, "worker_type": "aggregated"}
    )


def test_nextn_flag_fills_entry():
    result = build(["--ais-perf-model", json.dumps(ENTRY), "--ais-nextn", "3"])
    assert_perf_config(result, {**ENTRY, "nextn": 3})

    same = json.dumps({**ENTRY, "nextn": 3})
    assert result == build(["--ais-perf-model", same, "--ais-nextn", "3"])

    other = json.dumps({**ENTRIES["decode"], "nextn": 2})
    args = parse_args(
        [
            "--ais-perf-model",
            other,
            "--ais-nextn",
            "3",
            "--disaggregation-mode",
            "decode",
        ]
    )
    # A written nextn that differs from the flag: the loader refuses it, with
    # its own error class, before the mocker compares the two itself.
    with pytest.raises(ValueError) as error:
        CONFIG.build_mocker_engine_args(args)
    assert type(error.value) is not ValueError


@pytest.mark.parametrize("source", ["--ais-perf-model", ENV_VAR])
def test_value_with_planner_profile_data_is_rejected(
    source, tmp_path, monkeypatch, capsys
):
    profile = ["--planner-profile-data", str(tmp_path / "profile.npz")]
    # A deprecated alias warns only after this check: a warning here fails the test.
    alias = ["--sglang-schedule-policy", "fcfs"]
    assert (
        parser_error([*value_argv(source, monkeypatch), *profile, *alias], capsys)
        == f"{source} cannot be combined with --planner-profile-data"
    )


@pytest.mark.parametrize("source", ["--ais-perf-model", ENV_VAR])
def test_value_with_planner_profile_data_and_engine_args_file_is_accepted(
    source, tmp_path, monkeypatch
):
    # With an engine args file the profile data is not read, so nothing conflicts.
    argv = [
        *value_argv(source, monkeypatch),
        "--planner-profile-data",
        str(tmp_path / "profile.npz"),
        "--extra-engine-args",
        engine_args_file(tmp_path, {"worker_type": "aggregated"}),
    ]
    assert_perf_config(
        CONFIG.load_mocker_engine_args(parse_args(argv)),
        {**ENTRY, "worker_type": "aggregated"},
    )


def test_missing_worker_type_entry_is_a_value_error(tmp_path):
    # The loader reports the missing entry, in its own words.
    decode_only = json.dumps(entries("decode"))
    args = parse_args(["--ais-perf-model", decode_only])
    with pytest.raises(ValueError):
        CONFIG.build_mocker_engine_args(args)

    # One config that names its worker type is for that type only.
    one_config = json.dumps(entries("decode")[0])
    args = parse_args(
        ["--ais-perf-model", one_config, "--disaggregation-mode", "prefill"]
    )
    with pytest.raises(ValueError):
        CONFIG.build_mocker_engine_args(args)

    path = tmp_path / "perf.yaml"
    path.write_text(yaml.safe_dump(entries("prefill", "decode")))
    args = parse_args(["--ais-perf-model", str(path)])
    with pytest.raises(ValueError):
        CONFIG.build_mocker_engine_args(args)


def test_unknown_key_in_value_is_a_value_error(tmp_path):
    value = json.dumps({**ENTRY, "not_a_field": 3})

    args = parse_args(["--ais-perf-model", value])
    with pytest.raises(ValueError) as error:
        CONFIG.build_mocker_engine_args(args)
    assert_refused_key_error(error.value, "not_a_field")

    engine_args = engine_args_file(tmp_path, {"worker_type": "aggregated"})
    args = parse_args(["--ais-perf-model", value, "--extra-engine-args", engine_args])
    with pytest.raises(ValueError) as error:
        CONFIG.load_mocker_engine_args(args)
    assert_refused_key_error(error.value, "not_a_field")


def test_value_without_model_is_a_value_error():
    no_model = {key: value for key, value in ENTRY.items() if key != "model"}
    # --model-path names the served model; the perf model has to name its own.
    args = parse_args(["--ais-perf-model", json.dumps(no_model), "--model-path", MODEL])
    with pytest.raises(ValueError) as error:
        CONFIG.build_mocker_engine_args(args)
    assert_refused_key_error(error.value, "model")


@pytest.mark.parametrize("worker_type", ["aggregated", "prefill", "decode"])
def test_extra_engine_args_gets_timing_model(worker_type, tmp_path):
    engine_args = engine_args_file(
        tmp_path, {"worker_type": worker_type, "max_num_seqs": 7}
    )
    argv = [
        "--ais-perf-model",
        json.dumps(entries("aggregated", "prefill", "decode")),
        "--extra-engine-args",
        engine_args,
    ]

    result = CONFIG.load_mocker_engine_args(parse_args(argv))

    assert_perf_config(result, {**ENTRIES[worker_type], "worker_type": worker_type})
    assert result["engine"]["worker_type"] == worker_type
    assert result["engine"]["max_num_seqs"] == 7
    # With an engine args file every engine flag of the command line is
    # ignored, and so is this one.
    with_nextn = parse_args([*argv, "--ais-nextn", "3"])
    assert CONFIG.load_mocker_engine_args(with_nextn) == result


def test_extra_engine_args_without_engine_section_is_an_aggregated_worker(tmp_path):
    path = tmp_path / "engine.json"
    path.write_text("{}")
    args = parse_args(
        [
            "--ais-perf-model",
            json.dumps(entries("aggregated", "decode")),
            "--extra-engine-args",
            str(path),
        ]
    )
    assert_perf_config(
        CONFIG.load_mocker_engine_args(args), {**ENTRY, "worker_type": "aggregated"}
    )


def test_extra_engine_args_with_timing_model_is_rejected(tmp_path, monkeypatch):
    engine_args = engine_args_file(
        tmp_path,
        {
            "worker_type": "decode",
            "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0},
        },
    )
    value = json.dumps(entries("decode"))

    args = parse_args(["--ais-perf-model", value, "--extra-engine-args", engine_args])
    with pytest.raises(ValueError, match=f"^{TIMING_MODEL_CONFLICT} --ais-perf-model$"):
        CONFIG.load_mocker_engine_args(args)

    monkeypatch.setenv(ENV_VAR, value)
    args = parse_args(["--extra-engine-args", engine_args])
    with pytest.raises(ValueError, match=f"^{TIMING_MODEL_CONFLICT} {ENV_VAR}$"):
        CONFIG.load_mocker_engine_args(args)


@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_new_equals_old_for_same_identity(mode, worker_type):
    base = ["--model-path", MODEL, "--ais-nextn", "3", "--disaggregation-mode", mode]
    old = build_old(["--ais-perf-model", *flat_flags(), *base])
    new = build(
        ["--ais-perf-model", json.dumps({"model": MODEL, **FLAT_CONFIG}), *base]
    )
    assert CONFIG.performance_config(new) == CONFIG.performance_config(old)
    assert new == old


@pytest.mark.parametrize(
    "old_inputs, names",
    [
        (["--ais-perf-config", json.dumps(OLD_CONFIG)], "--ais-perf-config"),
        *(
            ([f"--ais-{name}", text], f"--ais-{name}")
            for name, text, _, _ in FLAT_FLAGS
        ),
        (["--aic-backend", "vllm"], "--ais-backend"),
        (
            ["--ais-tp-size", "2", "--ais-system", "h100_sxm", "--ais-backend", "vllm"],
            "--ais-backend, --ais-system and --ais-tp-size",
        ),
    ],
    ids=lambda value: value[0] if isinstance(value, list) else "",
)
def test_value_with_old_input_is_rejected(old_inputs, names, capsys):
    argv = ["--ais-perf-model", json.dumps(ENTRY), *old_inputs]
    assert (
        parser_error(argv, capsys)
        == f"--ais-perf-model cannot be combined with {names}"
    )


def test_env_value_with_old_flat_flag_is_rejected(monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, json.dumps(ENTRY))
    assert (
        parser_error(["--ais-backend", "vllm"], capsys)
        == f"{ENV_VAR} cannot be combined with --ais-backend"
    )


def test_hand_built_namespace_with_value_builds_config():
    # The attributes a caller without the parser has to set.
    base = {
        "is_prefill_worker": False,
        "is_decode_worker": False,
        "engine_type": "vllm",
    }
    value = json.dumps(ENTRY)
    expected = {**ENTRY, "worker_type": "aggregated"}

    args = argparse.Namespace(**base, ais_perf_model_value=value)
    assert_perf_config(CONFIG.build_mocker_engine_args(args), expected)

    # Old inputs next to a value are an error of parse_args, not of the builder.
    args = argparse.Namespace(
        **base,
        ais_perf_model_value=value,
        ais_perf_model_without_value=True,
        ais_backend="sglang",
    )
    assert_perf_config(CONFIG.build_mocker_engine_args(args), expected)

    # No parser rejected profile data next to the value, so the builder does.
    args = argparse.Namespace(
        **base, ais_perf_model_value=value, planner_profile_data="profile.npz"
    )
    with pytest.raises(ValueError, match=f"^{PLANNER_DATA_CONFLICT}$"):
        CONFIG.build_mocker_engine_args(args)


def test_hand_built_namespace_without_value_builds_as_before():
    args = argparse.Namespace(
        is_prefill_worker=False,
        is_decode_worker=False,
        engine_type="vllm",
        ais_perf_model_without_value=False,
    )
    assert CONFIG.performance_config(CONFIG.build_mocker_engine_args(args)) is None


def test_bare_then_option_is_the_switch():
    args = parse_old(["--ais-perf-model", "--ais-backend", "vllm"])
    assert args.ais_perf_model_without_value is True
    assert args.ais_perf_model_value is None
    assert args.ais_perf_model_source is None
    assert args.ais_backend == "vllm"


@pytest.mark.parametrize(
    "argv, value",
    [
        (["--ais-perf-model", "cfg.yaml"], "cfg.yaml"),
        (["--ais-perf-model=cfg.yaml"], "cfg.yaml"),
        # A value that starts with "-" is read as an option without "=".
        (["--ais-perf-model=-cfg.yaml"], "-cfg.yaml"),
    ],
)
def test_value_forms(argv, value):
    args = parse_args(argv)
    assert args.ais_perf_model_value == value
    assert args.ais_perf_model_without_value is False


def test_empty_typed_value_is_a_wrong_value():
    args = parse_args(["--ais-perf-model="])
    assert args.ais_perf_model_value == ""
    assert args.ais_perf_model_without_value is False
    with pytest.raises(ValueError) as error:
        CONFIG.build_mocker_engine_args(args)
    # The loader's own error class: the empty value was read.
    assert type(error.value) is not ValueError


def test_aic_spelling_takes_no_value(capsys):
    assert (
        parser_error(["--aic-perf-model", "cfg.yaml"], capsys)
        == "--aic-perf-model takes no value; write --ais-perf-model VALUE"
    )

    args = parse_old(["--aic-perf-model"])
    assert args.ais_perf_model_without_value is True
    assert args.ais_perf_model_value is None


@pytest.mark.parametrize(
    "argv",
    [
        ["--ais-perf-model", "--ais-perf-model", "cfg.yaml"],
        ["--ais-perf-model", "cfg.yaml", "--ais-perf-model"],
        # An abbreviation gets past the check for mixed spellings.
        ["--ais-perf-m", "cfg.yaml", "--aic-perf-model"],
    ],
)
def test_bare_and_valued_is_rejected(argv, capsys):
    assert parser_error(argv, capsys) == (
        "--ais-perf-model is given with and without a value; write it once, "
        "with the value"
    )


def test_bare_with_env_is_rejected(monkeypatch, capsys):
    monkeypatch.setenv(ENV_VAR, json.dumps(ENTRY))
    assert (
        parser_error(["--ais-perf-model"], capsys)
        == f"--ais-perf-model without a value cannot be combined with {ENV_VAR}"
    )


def test_last_value_wins():
    args = parse_args(["--ais-perf-model", "a.yaml", "--ais-perf-model", "b.yaml"])
    assert args.ais_perf_model_value == "b.yaml"
    assert args.ais_perf_model_without_value is False


def test_abbreviation_reaches_the_same_option(capsys):
    args = parse_args(["--ais-perf-m", "cfg.yaml"])
    assert args.ais_perf_model_value == "cfg.yaml"
    assert args.ais_perf_model_source == "--ais-perf-model"

    assert (
        parser_error(["--aic-perf", "cfg.yaml"], capsys)
        == "--aic-perf-model takes no value; write --ais-perf-model VALUE"
    )


def test_last_token_bare_switch():
    args = parse_old(["--model-path", MODEL, "--ais-perf-model"])
    assert args.ais_perf_model_without_value is True
    assert args.ais_perf_model_value is None


@pytest.mark.parametrize(
    "argv, names",
    [
        (["--ais-perf-model"], "--ais-perf-model without a value is"),
        (
            ["--ais-perf-model", "--ais-backend", "vllm", "--ais-system", "h100_sxm"],
            "--ais-perf-model without a value, --ais-backend and --ais-system are",
        ),
        (
            ["--ais-backend", "vllm", "--ais-system", "h100_sxm"],
            "--ais-backend and --ais-system are",
        ),
        (["--ais-perf-config", json.dumps(OLD_CONFIG)], "--ais-perf-config is"),
        (
            ["--aic-tp-size", "2", "--aic-perf-model"],
            "--ais-perf-model without a value and --ais-tp-size are",
        ),
        (
            # Typed in the opposite order: the names keep one order.
            [
                *(
                    token
                    for name, text, _, _ in reversed(FLAT_FLAGS)
                    for token in (f"--ais-{name}", text)
                ),
                "--ais-perf-config",
                json.dumps(OLD_CONFIG),
                "--ais-perf-model",
            ],
            "--ais-perf-model without a value, --ais-perf-config, --ais-backend, "
            "--ais-system, --ais-backend-version, --ais-tp-size, --ais-moe-tp-size, "
            "--ais-moe-ep-size and --ais-attention-dp-size are",
        ),
    ],
    ids=[
        "switch",
        "switch-and-two-flat-flags",
        "flat-flags-alone",
        "perf-config",
        "aic-spelling",
        "every-old-input",
    ],
)
def test_old_inputs_warn_once_with_exact_text(argv, names):
    with pytest.warns(FutureWarning) as records:
        parse_args(argv)
    assert [str(record.message) for record in records] == [names + REMOVAL]
    # The warning points at the line that called parse_args.
    assert records[0].filename == __file__


def test_old_inputs_with_extra_engine_args_warn_once(tmp_path):
    engine_args = engine_args_file(tmp_path, {"worker_type": "decode"})
    argv = ["--ais-perf-model", "--ais-backend", "vllm", "--extra-engine-args"]
    with pytest.warns(FutureWarning) as records:
        CONFIG.load_mocker_engine_args(parse_args([*argv, engine_args]))
    assert [str(record.message) for record in records] == [
        "--ais-perf-model without a value and --ais-backend are" + REMOVAL
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["--ais-nextn", "2"],
        ["--ais-nextn-accept-rates", "0.5,0.5", "--ais-mtp-seed", "7"],
        ["--ais-perf-model", json.dumps(ENTRY), "--ais-nextn", "2"],
    ],
    ids=["nextn", "accept-rates-and-seed", "value-and-nextn"],
)
def test_mtp_flags_are_not_old_inputs(argv):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        args = parse_args(argv)
    assert CONFIG.old_ais_perf_inputs(args) == []


def test_error_wins_over_warning():
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError, match="^--num-workers must be at least 1, got 0$"
        ):
            parse_args(["--ais-perf-model", "--num-workers", "0"])
    assert records == []


def test_relative_file_value_starts_at_working_directory(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    other_dir = tmp_path / "other"
    run_dir.mkdir()
    other_dir.mkdir()
    (run_dir / "cfg.yaml").write_text(yaml.safe_dump(ENTRY))

    monkeypatch.chdir(run_dir)
    assert_perf_config(
        build(["--ais-perf-model", "cfg.yaml"]), {**ENTRY, "worker_type": "aggregated"}
    )

    # The same arguments in a folder without the file.
    monkeypatch.chdir(other_dir)
    args = parse_args(["--ais-perf-model", "cfg.yaml"])
    with pytest.raises(ValueError):
        CONFIG.build_mocker_engine_args(args)


async def run_worker(monkeypatch, argv, fetched=None):
    """Run the mocker's entry coroutine on ``argv`` up to the worker launch.

    Returns the engine config of every launch. The model lookup needs a
    network and the launch a runtime, so both are replaced. Each model the
    entry looks up is appended to ``fetched`` when it is given.
    """
    # The CLI module sets this process default on import; keep it test-scoped.
    monkeypatch.setenv("DYN_COMPUTE_THREADS", "0")
    mocker_main = importlib.import_module("dynamo.mocker.main")
    launched = []

    async def prefetch_model(model):
        if fetched is not None:
            fetched.append(model)
        return model

    async def launch_workers(_args, config):
        launched.append(config)

    monkeypatch.setattr(sys, "argv", ["dynamo.mocker", *argv])
    monkeypatch.setattr(mocker_main, "prefetch_model", prefetch_model)
    monkeypatch.setattr(mocker_main, "compute_kv_bytes_per_token", lambda *_: 2048)
    monkeypatch.setattr(mocker_main, "launch_workers", launch_workers)
    await mocker_main.worker()
    return launched


@pytest.mark.asyncio
async def test_worker_entry_warns_once_for_old_inputs(monkeypatch):
    argv = [
        "--ais-perf-model",
        "--ais-backend",
        "vllm",
        "--model-path",
        MODEL,
        "--num-workers",
        "2",
    ]
    with pytest.warns(FutureWarning) as records:
        launched = await run_worker(monkeypatch, argv)

    assert [str(record.message) for record in records] == [
        "--ais-perf-model without a value and --ais-backend are" + REMOVAL
    ]
    assert len(launched) == 1
    assert_perf_config(
        launched[0],
        {
            "model": MODEL,
            "backend": "vllm",
            "system": "h200_sxm",
            "worker_type": "aggregated",
        },
    )


@pytest.mark.asyncio
async def test_worker_entry_takes_value_without_warning(monkeypatch):
    argv = [
        "--ais-perf-model",
        json.dumps(entries("prefill", "decode")),
        "--model-path",
        MODEL,
        "--disaggregation-mode",
        "decode",
        "--num-workers",
        "2",
    ]
    launched = await run_worker(monkeypatch, argv)

    assert len(launched) == 1
    assert_perf_config(launched[0], {**ENTRIES["decode"], "worker_type": "decode"})


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["--ais-perf-model", ENV_VAR])
async def test_worker_entry_rejects_planner_profile_data_before_model_lookup(
    source, tmp_path, monkeypatch, capsys
):
    # Profile data the mocker takes as it is, so only the combination is wrong.
    profile = tmp_path / "profile.npz"
    np.savez(
        profile,
        prefill_isl=np.array([1.0, 100.0]),
        prefill_ttft_ms=np.array([100.0, 100.0]),
        decode_active_kv_tokens=np.array([1.0, 100.0]),
        decode_context_length=np.array([1.0, 100.0]),
        decode_itl=np.array([[200.0, 200.0], [200.0, 200.0]]),
    )
    argv = [
        *value_argv(source, monkeypatch),
        "--model-path",
        MODEL,
        "--planner-profile-data",
        str(profile),
    ]
    fetched = []

    with pytest.raises(SystemExit) as error:
        await run_worker(monkeypatch, argv, fetched)

    assert error.value.code == 2
    message = capsys.readouterr().err.rsplit(": error: ", 1)[1].strip()
    assert message == f"{source} cannot be combined with --planner-profile-data"
    assert fetched == []
