# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A profiler results directory as mocker timing.

The directory becomes an AIS perf model, or, with the deprecated directory
form of ``--planner-profile-data``, lookup tables.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dynamo.common.ais_perf_model import engine_perf_config
from dynamo.mocker.args import parse_args, resolve_planner_profile_data
from dynamo.mocker.config import load_mocker_engine_args
from dynamo.mocker.utils.planner_profiler_perf_data_converter import (
    convert_profile_results_to_npz,
)
from dynamo.mocker.utils.profile_results_ais_perf_model import write_ais_perf_model
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
DIRECTORY_WARNING = (
    r"^--planner-profile-data with a profiler results directory is deprecated and "
    r"will be removed in v1\.9; use --ais-perf-model with the file that "
    r"`python -m dynamo\.mocker\.utils\.profile_results_ais_perf_model` writes for "
    r"the directory\.$"
)


@pytest.fixture(scope="module")
def perf_model(tmp_path_factory) -> Path:
    """The AIS perf model file written for ``PROFILE``."""
    output = tmp_path_factory.mktemp("converted") / "perf_model"
    return write_ais_perf_model(PROFILE, output, model=MODEL)


@pytest.fixture(scope="module")
def lookup_tables(tmp_path_factory) -> Path:
    """The lookup tables that the deprecated converter writes for ``PROFILE``."""
    return convert_profile_results_to_npz(
        PROFILE, tmp_path_factory.mktemp("tables") / "perf_data.npz"
    )


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


def one_request_at_a_time(input_tokens: int, engine_timing: dict) -> dict:
    """Replay three requests, one at a time, and return the summary."""
    report = run_synthetic_trace_replay(
        input_tokens,
        50,
        3,
        replay_mode="offline",
        replay_concurrency=1,
        **engine_timing,
    )
    return report.to_dict()["summary"]


@pytest.mark.unit
def test_file_names_one_saved_model_for_each_worker_type(perf_model):
    assert perf_model.name == "ais_perf_model.json"
    assert [entry["worker_type"] for entry in json.loads(perf_model.read_text())] == [
        "prefill",
        "decode",
        "aggregated",
    ]
    for _, worker_type in WORKER_TYPES:
        config = engine_perf_config(str(perf_model), worker_type)
        written = {
            "model": MODEL,
            "backend": "vllm",
            "system": "unknown",
            "worker_type": worker_type,
            "estimation_mode": "fpm_regression",
        }
        assert {key: config[key] for key in written} == written
        assert config["deployment_limits"]["max_kv_tokens"] == MAX_KV_TOKENS
        # The saved state is next to the file.
        snapshot = config["estimator_config"]["state_snapshot"]
        assert Path(snapshot["path"]) == perf_model.parent / worker_type


@pytest.mark.unit
@pytest.mark.parametrize("mode, worker_type", WORKER_TYPES)
def test_mocker_takes_timing_and_kv_capacity_from_the_file(
    perf_model, mode, worker_type
):
    argv = ["--model-path", MODEL, "--disaggregation-mode", mode]
    argv += ["--ais-perf-model", str(perf_model)]

    engine = load_mocker_engine_args(parse_args(argv))["engine"]

    assert engine["timing_model"]["provider"] == "ais"
    assert engine["timing_model"]["config"]["worker_type"] == worker_type
    # The KV capacity of the profiled engine, in blocks of 64 tokens.
    assert engine["num_gpu_blocks"] == MAX_KV_TOKENS // 64

    engine = load_mocker_engine_args(
        parse_args(argv + ["--num-gpu-blocks-override", "4096"])
    )["engine"]
    assert engine["num_gpu_blocks"] == 4096


@pytest.mark.integration
@pytest.mark.parametrize("input_tokens", sorted(MEASURED_TTFT_MS))
def test_replay_through_the_file_keeps_measured_prefill_time(perf_model, input_tokens):
    summary = one_request_at_a_time(input_tokens, {"ais_perf_model": str(perf_model)})

    # The fit is within 12% of every such prefill point of this profile.
    assert summary["mean_ttft_ms"] == pytest.approx(
        MEASURED_TTFT_MS[input_tokens], rel=0.12
    )


@pytest.mark.integration
def test_replay_through_the_file_and_through_lookup_tables(perf_model, lookup_tables):
    """Decode time agrees inside the profiled range; short prefills differ."""
    table_timing = {
        "type": "external",
        "provider": "dynamo_profile",
        "config": {"path": str(lookup_tables)},
    }
    tables = {
        "extra_engine_args": json.dumps({"engine": {"timing_model": table_timing}})
    }
    model = {"ais_perf_model": str(perf_model)}

    # Contexts of about 3,200 tokens: both follow the same plane.
    assert one_request_at_a_time(3151, model)["mean_itl_ms"] == pytest.approx(
        one_request_at_a_time(3151, tables)["mean_itl_ms"], rel=0.015
    )
    # The line through the prefill points is far below the shortest one, which
    # the test above finds through the file.
    assert one_request_at_a_time(100, tables)["mean_ttft_ms"] < 1.0


@pytest.mark.unit
def test_too_few_points_write_nothing(tmp_path):
    profile = write_points(
        tmp_path / "profile",
        [(1000, 30.0), (4000, 90.0), (8000, 170.0)],
        [(0.1, 500.0, 7.0), (0.4, 2000.0, 15.0), (0.8, 4000.0, 30.0)],
    )

    with pytest.raises(ValueError, match="^ais_perf_model: the model is not ready"):
        write_ais_perf_model(profile, tmp_path / "out", model=MODEL)

    assert sorted(path.name for path in tmp_path.iterdir()) == ["profile"]


@pytest.mark.unit
def test_existing_output_directory_is_refused(tmp_path):
    (tmp_path / "out").mkdir()

    with pytest.raises(FileExistsError):
        write_ais_perf_model(PROFILE, tmp_path / "out", model=MODEL)

    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.integration
def test_command_prints_the_file_it_wrote(tmp_path):
    output = tmp_path / "out"
    command = [
        sys.executable,
        "-m",
        "dynamo.mocker.utils.profile_results_ais_perf_model",
    ]
    command += [str(PROFILE), str(output), "--model", MODEL, "--backend", "sglang"]

    done = subprocess.run(
        command, capture_output=True, text=True, check=True, timeout=120
    )

    assert done.stdout.splitlines()[-1] == str(output / "ais_perf_model.json")
    assert engine_perf_config(done.stdout.splitlines()[-1], "decode")["backend"] == (
        "sglang"
    )


@pytest.mark.unit
def test_lookup_tables_keep_their_values(lookup_tables):
    """The values the scikit-learn regression gave for this profile."""
    with np.load(lookup_tables) as tables:
        assert tables["prefill_isl"][[0, 99]] == pytest.approx([100.0, 16372.0])
        assert tables["prefill_ttft_ms"][[0, 49, 99]] == pytest.approx(
            [0.6065956666666105, 178.74199816419093, 360.5128170392157], rel=1e-9
        )
        assert tables["decode_active_kv_tokens"][99] == pytest.approx(910195.0)
        assert tables["decode_context_length"][[0, 99]] == pytest.approx([1.0, 13585.0])
        assert tables["decode_itl"][[0, 49, 99, 99], [0, 49, 0, 99]] == pytest.approx(
            [
                6.110762547385647,
                20.67015159320542,
                18534.580560736187,
                34.15789791294073,
            ],
            rel=1e-9,
        )


@pytest.mark.unit
def test_lookup_tables_fit_points_of_one_context_length(tmp_path):
    """Requests and KV tokens rise together, so the fit has one slope."""
    counts = [935, 931, 937, 934, 933, 934, 940, 935, 936, 934, 936, 937, 936, 937]
    profile = write_points(
        tmp_path / "profile",
        [(1000, 30.0), (6000, 120.0)],
        # Requests of 100 tokens: 6 ms plus 40 ns for each KV token.
        [(count / 1000, 100.0, 6 + 0.004 * count) for count in counts + [935]],
    )

    with np.load(
        convert_profile_results_to_npz(profile, tmp_path / "tables.npz")
    ) as tables:
        # The measured time of 940 requests.
        assert tables["decode_itl"][99, 99] == pytest.approx(9.76, rel=1e-6)


@pytest.mark.unit
def test_lookup_tables_without_kv_usage_are_finite(tmp_path):
    profile = write_points(
        tmp_path / "profile",
        [(1000, 30.0), (6000, 120.0)],
        [(0.0, 500.0, 7.0), (0.0, 4000.0, 8.0)],
    )

    with np.load(
        convert_profile_results_to_npz(profile, tmp_path / "tables.npz")
    ) as tables:
        assert np.isfinite(tables["decode_itl"]).all()


@pytest.mark.unit
def test_lookup_tables_refuse_a_time_that_is_not_a_number(tmp_path):
    profile = write_points(
        tmp_path / "profile",
        [(1000, 30.0), (6000, 120.0)],
        [(0.1, 500.0, float("nan")), (0.6, 4000.0, 20.0)],
    )

    with pytest.raises(ValueError, match="NaN or infinity"):
        convert_profile_results_to_npz(profile, tmp_path / "tables.npz")


@pytest.mark.unit
def test_lookup_tables_refuse_time_that_falls_with_tokens(tmp_path):
    decode = [(0.1, 500.0, 7.0), (0.6, 4000.0, 20.0)]
    falling_prefill = write_points(
        tmp_path / "prefill", [(1000, 90.0), (6000, 30.0)], decode
    )
    falling_decode = write_points(
        tmp_path / "decode",
        [(1000, 30.0), (6000, 120.0)],
        [(0.1, 500.0, 20.0), (0.3, 4000.0, 15.0), (0.6, 4000.0, 7.0)],
    )

    with pytest.raises(RuntimeError, match="^Failed to fit prefill regression"):
        convert_profile_results_to_npz(falling_prefill, tmp_path / "prefill.npz")
    with pytest.raises(RuntimeError, match="^Failed to fit decode regression"):
        convert_profile_results_to_npz(falling_decode, tmp_path / "decode.npz")


@pytest.mark.unit
def test_lookup_tables_ignore_time_that_falls_with_batch_size(tmp_path):
    """At one KV token count, more requests do not make a step faster."""
    profile = write_points(
        tmp_path / "profile",
        [(1000, 30.0), (6000, 120.0)],
        # 100 and 500 requests of 100 tokens, then 10 and 18 of 5,000 tokens.
        [
            (0.1, 100.0, 5.0),
            (0.5, 100.0, 4.0),
            (0.5, 5000.0, 30.0),
            (0.9, 5000.0, 55.0),
        ],
    )

    with np.load(
        convert_profile_results_to_npz(profile, tmp_path / "tables.npz")
    ) as tables:
        itl = tables["decode_itl"]

    # Along a row the KV token count is fixed and the batch size falls.
    assert np.all(np.diff(itl, axis=1) == 0)
    assert np.all(np.diff(itl[:, 0]) > 0)


@pytest.mark.unit
def test_directory_form_warns_once_the_tables_are_written():
    with pytest.warns(FutureWarning, match=DIRECTORY_WARNING) as warned:
        result = resolve_planner_profile_data(PROFILE)

    assert len(warned) == 1
    with np.load(result.npz_path) as tables:
        assert tables["decode_itl"].shape == (100, 100)


@pytest.mark.unit
def test_table_file_and_failed_conversion_do_not_warn(lookup_tables, tmp_path):
    # A warning is an error in this suite, so none is given here.
    assert resolve_planner_profile_data(lookup_tables).npz_path == lookup_tables
    with pytest.raises(FileNotFoundError, match="neither a mocker-format NPZ file"):
        resolve_planner_profile_data(tmp_path / "missing")
    falling = write_points(
        tmp_path / "falling",
        [(1000, 90.0), (6000, 30.0)],
        [(0.1, 500.0, 7.0), (0.6, 4000.0, 20.0)],
    )
    with pytest.raises(RuntimeError, match="^Failed to fit prefill regression"):
        resolve_planner_profile_data(falling)
