# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify disjoint shard coverage through real pytest collection."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_sharding import assign_groups, junit_identity


def test_balanced_assignment_is_independent_of_collection_order():
    groups = {"a": 11, "b": 10, "c": 8, "d": 7}
    assigned, loads = assign_groups(groups, 2)
    reverse, reverse_loads = assign_groups(dict(reversed(list(groups.items()))), 2)
    assert assigned == reverse
    assert loads == reverse_loads == [18, 18]


def test_junit_identity_preserves_parameter_values():
    assert junit_identity("tests/test_a.py::TestA::test_it[a.b/1]") == (
        "tests.test_a.TestA::test_it[a.b/1]"
    )


@pytest.fixture
def shard_suite(tmp_path):
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers =\n    gpu\n    profiled_vram_gib\n"
    )
    (tmp_path / "test_profiled.py").write_text(
        "import pytest\n"
        "@pytest.mark.gpu\n"
        "@pytest.mark.profiled_vram_gib(4)\n"
        "@pytest.mark.parametrize('value', range(5))\n"
        "def test_gpu(value): pass\n"
        "def test_cpu(): pass\n"
    )
    (tmp_path / "test_serial.py").write_text(
        "import pytest\npytestmark = pytest.mark.gpu\n"
        "@pytest.fixture(scope='module')\n"
        "def shared_state(): return {}\n"
        "def test_a(shared_state): shared_state['ready'] = True\n"
        "def test_b(shared_state): assert shared_state['ready']\n"
    )
    weights = {
        f"test_profiled::test_gpu[{i}]": duration
        for i, duration in enumerate([11, 10, 8, 7])
    }  # fifth test deliberately has no historical timing
    weights.update({"test_serial::test_a": 4, "test_serial::test_b": 3})
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"synthetic": weights}))
    env = dict(os.environ)
    env.update(
        PYTHONPATH=str(Path(__file__).parent),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        DYNAMO_GPU_SHARD_POOL="synthetic",
        DYNAMO_GPU_SHARD_DURATIONS=str(manifest),
    )

    def collect(index, count=2, execute=False, fail_gpu=False, **overrides):
        shard_env = dict(env)
        shard_env.update(
            DYNAMO_GPU_SHARD_INDEX=str(index), DYNAMO_GPU_SHARD_COUNT=str(count)
        )
        shard_env.update(overrides)
        if fail_gpu:
            path = tmp_path / "test_profiled.py"
            path.write_text(
                path.read_text().replace(
                    "def test_gpu(value): pass",
                    "def test_gpu(value): assert value != 2",
                )
            )
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "test_sharding",
                "-m",
                "gpu",
                *([] if execute else ["--collect-only"]),
                "-q",
            ],
            cwd=tmp_path,
            env=shard_env,
            capture_output=True,
            text=True,
            check=False,
        )

    return collect


def test_collection_has_no_gaps_duplicates_or_split_module_fixtures(shard_suite):
    results = [shard_suite(i) for i in range(2)]
    for result in results:
        assert result.returncode == 0, result.stdout + result.stderr
    sets = [
        {
            line
            for line in result.stdout.splitlines()
            if line.startswith("test_") and "::" in line
        }
        for result in results
    ]
    assert not (sets[0] & sets[1])
    assert sets[0] | sets[1] == {
        *(f"test_profiled.py::test_gpu[{i}]" for i in range(5)),
        "test_serial.py::test_a",
        "test_serial.py::test_b",
    }
    assert any(
        {"test_serial.py::test_a", "test_serial.py::test_b"} <= group for group in sets
    )
    assert all("1 tests use" in result.stdout for result in results)


def test_unsharded_collection_and_empty_shards_succeed(shard_suite):
    result = shard_suite(0, count=1)
    assert result.returncode == 0
    assert "7/8 tests collected" in result.stdout
    results = [shard_suite(i, count=9) for i in range(9)]
    assert all(result.returncode == 0 for result in results)
    assert any("0 tests;" in result.stdout for result in results)


@pytest.mark.parametrize(
    "overrides",
    [{"DYNAMO_GPU_SHARD_INDEX": "2"}, {"DYNAMO_GPU_SHARD_POOL": "missing"}],
)
def test_invalid_configuration_fails_closed(shard_suite, overrides):
    result = shard_suite(0, **overrides)
    assert result.returncode != 0


def test_execution_preserves_module_fixture_and_producer_consumer_order(shard_suite):
    # The consumer fails if the producer is split away or reordered. Every
    # shard starts a fresh pytest process, as independent CI job pods do.
    results = [shard_suite(index, execute=True) for index in range(2)]
    assert all(result.returncode == 0 for result in results), [
        result.stdout + result.stderr for result in results
    ]


def test_a_failure_in_either_shard_still_fails_its_job(shard_suite):
    results = [shard_suite(index, execute=True, fail_gpu=True) for index in range(2)]
    assert sorted(result.returncode for result in results) == [0, 1]
    assert sum("test_gpu[2]" in result.stdout for result in results) == 1
