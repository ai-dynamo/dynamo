# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import importlib
import json
import sys

import numpy as np
import pytest

from dynamo.mocker import config as CONFIG
from dynamo.mocker.args import parse_args
from tests.utils.ais_perf_model import AIS_PERF_MODEL_ENV

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
# Every flat identity flag with a value the installed timing data accepts, and
# the config key it sets.
FLAT_FLAGS = [
    ("--ais-backend", "sglang", "backend", "sglang"),
    ("--ais-system", "h100_sxm", "system", "h100_sxm"),
    ("--ais-backend-version", "current", "backend_version", "current"),
    ("--ais-tp-size", "2", "tp", 2),
    ("--ais-moe-tp-size", "2", "moe_tp_size", 2),
    ("--ais-moe-ep-size", "1", "moe_ep_size", 1),
    ("--ais-attention-dp-size", "1", "attention_dp", 1),
]
FLAT_ARGV = [token for flag, text, _, _ in FLAT_FLAGS for token in (flag, text)]
FLAT_CONFIG = {key: value for _, _, key, value in FLAT_FLAGS}
OLD_CONFIG = {"model": MODEL, "system": "h100_sxm", "backend": "vllm", "tp": 2}
# What the option without a value builds from --model-path and --engine-type.
SWITCH_CONFIG = {"model": MODEL, "backend": "vllm", "system": "h200_sxm"}

# How the warning for old inputs ends, after the names of the inputs.
REMOVAL = " deprecated and will be removed in v1.9; use --ais-perf-model VALUE."
OLD_INPUT_WARNING = r"; use --ais-perf-model VALUE\.$"
# One perf model config for each worker type, as a value writes it. They
# differ, so a test sees which one the mocker took.
ENTRIES = {
    "aggregated": {"model": MODEL, "system": "h200_sxm", "backend": "vllm", "tp": 2},
    "prefill": {"model": MODEL, "system": "h100_sxm", "backend": "vllm", "tp": 2},
    "decode": {"model": MODEL, "system": "h200_sxm", "backend": "vllm", "tp": 1},
}
ENTRY = ENTRIES["aggregated"]
# A list value with the config of every worker type.
EVERY_ENTRY = json.dumps(
    [{"worker_type": worker_type, **entry} for worker_type, entry in ENTRIES.items()]
)


def build(argv):
    return CONFIG.build_mocker_engine_args(parse_args(argv))


def parse_old(argv):
    """Parse a command line that holds old inputs, which warns."""
    with pytest.warns(FutureWarning, match=OLD_INPUT_WARNING):
        return parse_args(argv)


def build_old(argv):
    return CONFIG.build_mocker_engine_args(parse_old(argv))


def assert_perf_config(result, expected):
    """Check the keys of ``expected`` in the perf config ``result`` holds.

    The mocker returns the config with every default AISimulate adds, so a
    test compares the keys its input wrote.
    """
    config = CONFIG.performance_config(result)
    assert {key: config[key] for key in expected} == expected


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
    if source == AIS_PERF_MODEL_ENV:
        monkeypatch.setenv(AIS_PERF_MODEL_ENV, json.dumps(ENTRY))
        return []
    return [source, json.dumps(ENTRY)]


@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_value_as_inline_json(mode, worker_type):
    args = parse_args(["--ais-perf-model", EVERY_ENTRY, "--disaggregation-mode", mode])
    assert args.ais_perf_model_without_value is False
    assert args.ais_perf_model_value == EVERY_ENTRY
    assert args.ais_perf_model_source == "--ais-perf-model"
    assert not hasattr(args, "ais_perf_model")
    assert not hasattr(args, "ais_perf_model_occurrences")

    result = CONFIG.build_mocker_engine_args(args)

    assert_perf_config(result, {**ENTRIES[worker_type], "worker_type": worker_type})
    assert result["engine"]["timing_model"]["provider"] == "ais"


def test_value_from_env(monkeypatch):
    monkeypatch.setenv(AIS_PERF_MODEL_ENV, json.dumps(ENTRY))
    args = parse_args([])
    assert args.ais_perf_model_without_value is False
    assert args.ais_perf_model_value == json.dumps(ENTRY)
    assert args.ais_perf_model_source == AIS_PERF_MODEL_ENV
    assert_perf_config(
        CONFIG.build_mocker_engine_args(args), {**ENTRY, "worker_type": "aggregated"}
    )


def test_typed_value_wins_over_env(monkeypatch):
    monkeypatch.setenv(AIS_PERF_MODEL_ENV, json.dumps(ENTRIES["decode"]))
    args = parse_args(["--ais-perf-model", json.dumps(ENTRY)])
    assert args.ais_perf_model_value == json.dumps(ENTRY)
    assert args.ais_perf_model_source == "--ais-perf-model"
    assert_perf_config(
        CONFIG.build_mocker_engine_args(args), {**ENTRY, "worker_type": "aggregated"}
    )


def test_nextn_flag_fills_entry():
    result = build(["--ais-perf-model", json.dumps(ENTRY), "--ais-nextn", "3"])
    assert_perf_config(result, {**ENTRY, "nextn": 3})

    # A config that writes a different nextn is refused, not replaced.
    other = json.dumps({**ENTRY, "nextn": 2})
    args = parse_args(["--ais-perf-model", other, "--ais-nextn", "3"])
    with pytest.raises(ValueError) as error:
        CONFIG.build_mocker_engine_args(args)
    # The loader's own error class: the launch check of nextn against
    # aic_nextn would be a plain ValueError.
    assert type(error.value) is not ValueError


def test_value_with_planner_profile_data_is_rejected(tmp_path, capsys):
    profile = ["--planner-profile-data", str(tmp_path / "profile.npz")]
    # A deprecated alias warns only after this check: a warning here fails the test.
    alias = ["--sglang-schedule-policy", "fcfs"]
    argv = ["--ais-perf-model", json.dumps(ENTRY), *profile, *alias]
    assert "--planner-profile-data" in parser_error(argv, capsys)


def test_value_with_planner_profile_data_and_engine_args_file_is_accepted(tmp_path):
    # With an engine args file the profile data is not read, so nothing conflicts.
    profile = ["--planner-profile-data", str(tmp_path / "profile.npz")]
    # A file without an engine section is an aggregated worker.
    path = tmp_path / "engine.json"
    path.write_text("{}")
    engine_args = ["--extra-engine-args", str(path)]
    args = parse_args(["--ais-perf-model", json.dumps(ENTRY), *profile, *engine_args])
    assert_perf_config(
        CONFIG.load_mocker_engine_args(args), {**ENTRY, "worker_type": "aggregated"}
    )


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


@pytest.mark.parametrize(
    "written, worker_type",
    [({}, "aggregated"), ({"worker_type": "decode"}, "decode")],
    ids=["no-worker-type", "decode"],
)
def test_extra_engine_args_gets_timing_model(written, worker_type, tmp_path):
    engine_args = engine_args_file(tmp_path, {**written, "max_num_seqs": 7})
    argv = ["--ais-perf-model", EVERY_ENTRY, "--extra-engine-args", engine_args]

    result = CONFIG.load_mocker_engine_args(parse_args(argv))

    assert_perf_config(result, {**ENTRIES[worker_type], "worker_type": worker_type})
    assert result["engine"]["worker_type"] == worker_type
    assert result["engine"]["max_num_seqs"] == 7
    # With an engine args file every engine flag of the command line is
    # ignored, and so is this one.
    with_nextn = parse_args([*argv, "--ais-nextn", "3"])
    assert CONFIG.load_mocker_engine_args(with_nextn) == result


def test_extra_engine_args_with_timing_model_is_rejected(tmp_path, monkeypatch):
    timing_model = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
    engine_args = engine_args_file(tmp_path, {"timing_model": timing_model})
    argv = ["--extra-engine-args", engine_args]

    args = parse_args(["--ais-perf-model", json.dumps(ENTRY), *argv])
    with pytest.raises(ValueError) as error:
        CONFIG.load_mocker_engine_args(args)
    assert str(error.value) == (
        "--extra-engine-args already sets engine.timing_model; it cannot be "
        "combined with --ais-perf-model"
    )

    # A value from the variable is named as the variable.
    monkeypatch.setenv(AIS_PERF_MODEL_ENV, json.dumps(ENTRY))
    with pytest.raises(ValueError, match=f"{AIS_PERF_MODEL_ENV}$"):
        CONFIG.load_mocker_engine_args(parse_args(argv))


@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_new_equals_old_for_same_identity(mode, worker_type):
    base = ["--model-path", MODEL, "--ais-nextn", "3", "--disaggregation-mode", mode]
    old = build_old(["--ais-perf-model", *FLAT_ARGV, *base])
    new = build(
        ["--ais-perf-model", json.dumps({"model": MODEL, **FLAT_CONFIG}), *base]
    )
    assert_perf_config(
        old, {"model": MODEL, **FLAT_CONFIG, "nextn": 3, "worker_type": worker_type}
    )
    assert new == old


@pytest.mark.parametrize(
    "source, old_inputs, names",
    [
        # The two sentences the documentation gives as examples.
        ("--ais-perf-model", ["--ais-system", "h100_sxm"], "--ais-system"),
        ("--ais-perf-model", ["--ais-tp-size", "2"], "--ais-tp-size"),
        (AIS_PERF_MODEL_ENV, ["--ais-backend", "vllm"], "--ais-backend"),
        (
            "--ais-perf-model",
            ["--ais-tp-size", "2", "--ais-system", "h100_sxm", "--ais-backend", "vllm"],
            "--ais-backend, --ais-system and --ais-tp-size",
        ),
    ],
    ids=["system", "tp-size", "value-from-env", "several-in-one-order"],
)
def test_value_with_old_input_is_rejected(
    source, old_inputs, names, monkeypatch, capsys
):
    argv = [*value_argv(source, monkeypatch), *old_inputs]
    assert parser_error(argv, capsys) == f"{source} cannot be combined with {names}"


def test_hand_built_namespace_with_value_builds_config():
    # The attributes a caller without the parser has to set.
    args = argparse.Namespace(
        is_prefill_worker=False,
        is_decode_worker=False,
        engine_type="vllm",
        ais_perf_model_value=json.dumps(ENTRY),
    )
    assert_perf_config(
        CONFIG.build_mocker_engine_args(args), {**ENTRY, "worker_type": "aggregated"}
    )


# A value that starts with "-" is read as an option without "=".
@pytest.mark.parametrize("value", ["cfg.yaml", "-cfg.yaml"])
def test_value_forms(value):
    args = parse_args([f"--ais-perf-model={value}"])
    assert args.ais_perf_model_value == value
    assert args.ais_perf_model_without_value is False


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
        # An abbreviation gets past the check for mixed spellings.
        ["--ais-perf-m", "cfg.yaml", "--aic-perf-model"],
    ],
)
def test_bare_and_valued_is_rejected(argv, capsys):
    assert parser_error(argv, capsys).startswith(
        "--ais-perf-model is given with and without a value"
    )


def test_bare_with_env_is_rejected(monkeypatch, capsys):
    monkeypatch.setenv(AIS_PERF_MODEL_ENV, json.dumps(ENTRY))
    assert (
        parser_error(["--ais-perf-model"], capsys)
        == f"--ais-perf-model without a value cannot be combined with {AIS_PERF_MODEL_ENV}"
    )


def test_last_value_wins():
    args = parse_args(["--ais-perf-model", "a.yaml", "--ais-perf-model", "b.yaml"])
    assert args.ais_perf_model_value == "b.yaml"
    assert args.ais_perf_model_without_value is False


def test_abbreviation_reaches_the_same_option(capsys):
    args = parse_args(["--ais-perf-m", "cfg.yaml"])
    assert args.ais_perf_model_value == "cfg.yaml"
    assert args.ais_perf_model_source == "--ais-perf-model"

    assert parser_error(["--aic-perf", "cfg.yaml"], capsys).startswith(
        "--aic-perf-model takes no value"
    )


@pytest.mark.parametrize(
    "argv, names",
    [
        (
            ["--ais-perf-model", "--ais-backend", "vllm", "--ais-system", "h100_sxm"],
            "--ais-perf-model without a value, --ais-backend and --ais-system are",
        ),
        (
            # Typed with the --aic spellings and in another order: the names
            # are the --ais ones, in one order.
            [
                *(token.replace("--ais-", "--aic-") for token in FLAT_ARGV),
                "--ais-perf-config",
                json.dumps(OLD_CONFIG),
                "--aic-perf-model",
            ],
            "--ais-perf-model without a value, --ais-perf-config, --ais-backend, "
            "--ais-system, --ais-backend-version, --ais-tp-size, --ais-moe-tp-size, "
            "--ais-moe-ep-size and --ais-attention-dp-size are",
        ),
    ],
    ids=["switch-and-two-flat-flags", "every-old-input"],
)
def test_old_inputs_warn_once_with_exact_text(argv, names):
    with pytest.warns(FutureWarning) as records:
        parse_args(argv)
    assert [str(record.message) for record in records] == [names + REMOVAL]
    # The warning points at the line that called parse_args.
    assert records[0].filename == __file__


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
    argv = ["--ais-perf-model", "--ais-backend", "vllm", "--num-workers", "2"]
    with pytest.warns(FutureWarning, match=OLD_INPUT_WARNING) as records:
        launched = await run_worker(monkeypatch, [*argv, "--model-path", MODEL])

    assert len(records) == 1
    assert len(launched) == 1
    assert_perf_config(launched[0], SWITCH_CONFIG)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["--ais-perf-model", AIS_PERF_MODEL_ENV])
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
