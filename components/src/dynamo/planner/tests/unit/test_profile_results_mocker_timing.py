# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A profiler results directory as mocker timing, through an AIS perf model."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dynamo.common.ais_perf_model import engine_perf_config
from dynamo.mocker.args import ProfileDataResult, parse_args, use_planner_profile_data
from dynamo.mocker.config import load_mocker_engine_args
from dynamo.mocker.utils.planner_profiler_perf_data_converter import (
    write_ais_perf_model,
)
from dynamo.replay import run_synthetic_trace_replay

# The type marker is on each test: the ones that run the native replay or a
# command are integration tests.
pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.planner,
    pytest.mark.usefixtures("no_ais_environment"),
]

# Points that the profiler measured on one H200: 17 for prefill and 41 for
# decode, with a decode KV capacity of 945,029 tokens.
PROFILE = Path(__file__).resolve().parents[1] / "data/profiling_results/H200_TP1P_TP1D"
MAX_KV_TOKENS = 945_029
# Input lengths of that profile with the measured time to first token in ms.
# Each fits in one prefill pass of the default 8,192 tokens.
MEASURED_TTFT_MS = {100: 19.525209, 4168: 87.342026, 7219: 143.294598}
MODEL = "profiled-model"
WORKER_TYPES = [("agg", "aggregated"), ("prefill", "prefill"), ("decode", "decode")]
DIRECTORY = ["--planner-profile-data", str(PROFILE)]


def started(argv: list[str]) -> tuple[dict, ProfileDataResult]:
    """Return the engine args a mocker started with ``argv`` runs with.

    The second value owns the folder of a tuned model: an engine built from
    the engine args reads that folder, so keep the value until then.
    """
    args = parse_args(argv)
    resolved = use_planner_profile_data(args)
    return load_mocker_engine_args(args), resolved


def write_points(directory: Path, prefill: list, decode: list) -> Path:
    """Write ``(isl, ttft_ms)`` and ``(kv_usage, context, itl_ms)`` points as a
    profiler results directory with a KV capacity of 100,000 tokens."""
    directory.mkdir()
    (directory / "prefill_raw_data.json").write_text(
        json.dumps(
            {
                "prefill_isl": [point[0] for point in prefill],
                "prefill_ttft": [point[1] for point in prefill],
            }
        )
    )
    (directory / "decode_raw_data.json").write_text(
        json.dumps(
            {
                "x_kv_usage": [point[0] for point in decode],
                "y_context_length": [point[1] for point in decode],
                "z_itl": [point[2] for point in decode],
                "max_kv_tokens": 100_000,
            }
        )
    )
    return directory


@pytest.fixture
def too_few_points(tmp_path) -> Path:
    """A profiler results directory with three points for each phase."""
    return write_points(
        tmp_path / "profile",
        [(1000, 30.0), (4000, 90.0), (8000, 170.0)],
        [(0.1, 500.0, 7.0), (0.4, 2000.0, 15.0), (0.8, 4000.0, 30.0)],
    )


@pytest.mark.unit
@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_mocker_tunes_an_ais_perf_model_with_a_directory(mode, worker_type):
    argv = ["--model-path", MODEL, "--disaggregation-mode", mode, *DIRECTORY]

    engine = started(argv)[0]["engine"]

    assert engine["timing_model"]["provider"] == "ais"
    config = engine["timing_model"]["config"]
    recorded = {
        "model": MODEL,
        "backend": "vllm",
        "system": "unknown",
        "worker_type": worker_type,
        "estimation_mode": "fpm_regression",
        "nextn": 0,
    }
    assert {key: config[key] for key in recorded} == recorded
    # The KV capacity of the profiled engine, in blocks of 64 tokens.
    assert engine["num_gpu_blocks"] == MAX_KV_TOKENS // 64

    engine = started(argv + ["--num-gpu-blocks-override", "4096"])[0]["engine"]
    assert engine["num_gpu_blocks"] == 4096


@pytest.mark.unit
def test_directory_model_follows_engine_type_and_draft_depth():
    argv = ["--engine-type", "sglang", "--ais-nextn", "2", *DIRECTORY]

    config = started(argv)[0]["engine"]["timing_model"]["config"]

    assert (config["backend"], config["nextn"]) == ("sglang", 2)
    # Without --model-path the directory names the model.
    assert config["model"] == PROFILE.name


@pytest.mark.integration
@pytest.mark.parametrize("input_tokens", sorted(MEASURED_TTFT_MS))
def test_mocker_engine_keeps_measured_prefill_time(input_tokens):
    """One request at a time through the engine a mocker with the directory runs."""
    engine_args, _tuned_model = started(["--model-path", MODEL, *DIRECTORY])

    report = run_synthetic_trace_replay(
        input_tokens,
        50,
        3,
        replay_mode="offline",
        replay_concurrency=1,
        extra_engine_args=json.dumps(engine_args),
    )

    # The fit misses the measured prefill points of this profile by up to 11.7%.
    assert report.to_dict()["summary"]["mean_ttft_ms"] == pytest.approx(
        MEASURED_TTFT_MS[input_tokens], rel=0.12
    )


@pytest.mark.unit
def test_directory_with_an_old_ais_input_is_refused():
    with pytest.warns(FutureWarning, match=r"use --ais-perf-model VALUE\.$"):
        args = parse_args(
            ["--model-path", MODEL, "--ais-system", "h200_sxm", *DIRECTORY]
        )

    with pytest.raises(ValueError, match="^choose one timing provider"):
        use_planner_profile_data(args)


@pytest.mark.unit
def test_engine_args_file_keeps_its_own_timing(tmp_path):
    timing = {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 2.0}
    file = tmp_path / "engine.json"
    file.write_text(json.dumps({"engine": {"timing_model": timing}}))

    engine = started([*DIRECTORY, "--extra-engine-args", str(file)])[0]["engine"]

    assert engine["timing_model"] == timing


@pytest.mark.unit
def test_lookup_table_file_stays_the_timing_source(tmp_path):
    tables = tmp_path / "tables.npz"
    np.savez(
        tables,
        prefill_isl=np.array([128.0, 256.0]),
        prefill_ttft_ms=np.array([4.0, 8.0]),
        decode_active_kv_tokens=np.array([1024.0, 2048.0]),
        decode_context_length=np.array([128.0, 256.0]),
        decode_itl=np.array([[1.0, 1.5], [2.0, 2.5]]),
    )

    engine = started(["--planner-profile-data", str(tables)])[0]["engine"]
    timing = engine["timing_model"]

    assert timing["provider"] == "dynamo_profile"
    assert timing["config"] == {"path": str(tables)}


@pytest.mark.unit
def test_mocker_refuses_a_directory_with_too_few_points(too_few_points):
    args = parse_args(["--planner-profile-data", str(too_few_points)])

    with pytest.raises(ValueError, match="^ais_perf_model: the model is not ready"):
        use_planner_profile_data(args)


@pytest.mark.unit
def test_mocker_refuses_a_path_without_profile_data(tmp_path):
    args = parse_args(["--planner-profile-data", str(tmp_path)])

    with pytest.raises(FileNotFoundError, match="neither a mocker-format NPZ file"):
        use_planner_profile_data(args)


@pytest.mark.unit
def test_written_file_names_one_saved_model_for_each_worker_type(tmp_path):
    perf_model = write_ais_perf_model(PROFILE, tmp_path / "out", model=MODEL)

    assert perf_model == tmp_path / "out" / "ais_perf_model.json"
    assert [entry["worker_type"] for entry in json.loads(perf_model.read_text())] == [
        "prefill",
        "decode",
        "aggregated",
    ]
    for _, worker_type in WORKER_TYPES:
        config = engine_perf_config(str(perf_model), worker_type)
        assert config["deployment_limits"]["max_kv_tokens"] == MAX_KV_TOKENS
        snapshot = config["estimator_config"]["state_snapshot"]
        assert Path(snapshot["path"]) == perf_model.parent / worker_type


@pytest.mark.integration
def test_replay_takes_the_written_file(tmp_path):
    perf_model = write_ais_perf_model(PROFILE, tmp_path / "out", model=MODEL)

    report = run_synthetic_trace_replay(
        4168,
        50,
        3,
        replay_mode="offline",
        replay_concurrency=1,
        ais_perf_model=str(perf_model),
    )

    assert report.to_dict()["summary"]["mean_ttft_ms"] == pytest.approx(
        MEASURED_TTFT_MS[4168], rel=0.12
    )


@pytest.mark.unit
def test_too_few_points_write_nothing(too_few_points, tmp_path):
    with pytest.raises(ValueError, match="^ais_perf_model: the model is not ready"):
        write_ais_perf_model(too_few_points, tmp_path / "out", model=MODEL)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["profile"]


@pytest.mark.unit
def test_existing_output_directory_is_refused(tmp_path):
    (tmp_path / "out").mkdir()

    with pytest.raises(FileExistsError):
        write_ais_perf_model(PROFILE, tmp_path / "out", model=MODEL)

    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.timeout(180)
@pytest.mark.integration
def test_command_prints_the_file_it_wrote(tmp_path):
    output = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "dynamo.mocker.utils.planner_profiler_perf_data_converter",
    ]
    command += [str(PROFILE), str(output), "--model", MODEL, "--backend", "sglang"]

    done = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=120
    )

    assert done.stdout.splitlines()[-1] == str(output / "ais_perf_model.json")
    assert engine_perf_config(done.stdout.splitlines()[-1], "decode")["backend"] == (
        "sglang"
    )
