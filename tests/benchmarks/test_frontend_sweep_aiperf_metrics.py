# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Characterization tests for AIPerf metric decoding in the frontend sweep executors.

These pin the observable behavior of the local and Kubernetes wrappers, including
where they intentionally differ (throughput normalization, non-dict latency
handling, artifact discovery, and incremental vs buffered assignment).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import pytest

# The sweep scripts are not an installed package; they import ``sweep_core``
# and ``sweep_k8s`` as top-level packages from this directory.
SCRIPTS_DIR = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "frontend" / "scripts"
)
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from sweep_core.models import (  # noqa: E402
    AiperfDimension,
    DeployDimension,
    RunResult,
    RunSpec,
)
from sweep_executors.k8s_dgd import _parse_k8s_aiperf_into_result  # noqa: E402
from sweep_executors.local import (  # noqa: E402
    _parse_aiperf_into_result,
    _parse_aiperf_json,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

METRIC_FIELDS = (
    "req_per_sec",
    "output_tok_per_sec",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "itl_p50_ms",
    "itl_p99_ms",
    "duration_sec",
)
LATENCY_FIELDS = ("ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms")

# Distinct nonzero values so an untouched field is easy to tell from an overwrite.
SENTINELS = MappingProxyType(
    {
        "req_per_sec": 101.0,
        "output_tok_per_sec": 102.0,
        "ttft_p50_ms": 103.0,
        "ttft_p99_ms": 104.0,
        "itl_p50_ms": 105.0,
        "itl_p99_ms": 106.0,
        "duration_sec": 107.0,
    }
)

FULL_EXPECTED = MappingProxyType(
    {
        "req_per_sec": 10.5,
        "output_tok_per_sec": 250.25,
        "ttft_p50_ms": 1.5,
        "ttft_p99_ms": 2.5,
        "itl_p50_ms": 3.5,
        "itl_p99_ms": 4.5,
        "duration_sec": 30.0,
    }
)


def _full_canonical(**overrides: Any) -> dict:
    """Return a fresh canonical AIPerf payload, so tests never share nested dicts."""
    data = {
        "request_throughput": {"avg": 10.5},
        "output_token_throughput": {"avg": 250.25},
        "time_to_first_token": {"p50": 1.5, "p99": 2.5},
        "inter_token_latency": {"p50": 3.5, "p99": 4.5},
        "benchmark_duration": 30.0,
    }
    data.update(overrides)
    return data


def _make_result(prepopulated: bool = False) -> RunResult:
    deploy = DeployDimension(backend="mocker", tokenizer="hf")
    spec = RunSpec(
        deploy=deploy,
        aiperf=AiperfDimension(concurrency=1, isl=8),
        deploy_key=deploy.deploy_key,
        run_id="run-0",
    )
    result = RunResult(run_spec=spec)
    if prepopulated:
        for name, value in SENTINELS.items():
            setattr(result, name, value)
    return result


def _metrics(result: RunResult) -> dict:
    return {name: getattr(result, name) for name in METRIC_FIELDS}


def _write_json(
    run_dir: Path, data: Any = None, raw: Optional[str] = None, model: str = ""
) -> Path:
    """Write ``aiperf/[<model>/]profile_export_aiperf.json`` under ``run_dir``."""
    target_dir = run_dir / "aiperf"
    if model:
        target_dir = target_dir / model
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "profile_export_aiperf.json"
    path.write_text(raw if raw is not None else json.dumps(data))
    return path


def _assert_same_values(actual: dict, expected: Mapping[str, Any]) -> None:
    """Compare values and their types, so 0 vs 0.0 vs None vs False is pinned."""
    assert actual == expected
    for key, value in expected.items():
        assert type(actual[key]) is type(value), key


# ── Canonical and legacy keys ───────────────────────────────────────────────


def test_canonical_keys_both_wrappers(tmp_path: Path) -> None:
    path = _write_json(tmp_path, _full_canonical())

    _assert_same_values(_parse_aiperf_json(path), FULL_EXPECTED)

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), FULL_EXPECTED)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), FULL_EXPECTED)


def test_legacy_latency_keys(tmp_path: Path) -> None:
    data = _full_canonical()
    del data["time_to_first_token"]
    del data["inter_token_latency"]
    data["ttft"] = {"p50": 1.5, "p99": 2.5}
    data["itl"] = {"p50": 3.5, "p99": 4.5}
    path = _write_json(tmp_path, data)

    _assert_same_values(_parse_aiperf_json(path), FULL_EXPECTED)

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), FULL_EXPECTED)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), FULL_EXPECTED)


def test_canonical_latency_key_wins_over_legacy(tmp_path: Path) -> None:
    data = _full_canonical()
    data["ttft"] = {"p50": 901.0, "p99": 902.0}
    data["itl"] = {"p50": 903.0, "p99": 904.0}
    path = _write_json(tmp_path, data)

    _assert_same_values(_parse_aiperf_json(path), FULL_EXPECTED)

    local = _make_result()
    _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), FULL_EXPECTED)

    k8s = _make_result()
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), FULL_EXPECTED)


def test_non_dict_canonical_latency_does_not_fall_back_to_legacy(
    tmp_path: Path,
) -> None:
    # The canonical key is present (null), so the legacy key is never consulted
    # and the percentile fields are skipped entirely.
    data = _full_canonical()
    data["time_to_first_token"] = None
    data["inter_token_latency"] = None
    data["ttft"] = {"p50": 901.0, "p99": 902.0}
    data["itl"] = {"p50": 903.0, "p99": 904.0}
    path = _write_json(tmp_path, data)

    parsed = _parse_aiperf_json(path)
    _assert_same_values(
        parsed,
        {"req_per_sec": 10.5, "output_tok_per_sec": 250.25, "duration_sec": 30.0},
    )

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    assert all(getattr(local, name) == 0 for name in LATENCY_FIELDS)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    assert all(getattr(k8s, name) == SENTINELS[name] for name in LATENCY_FIELDS)


# ── Missing fields ──────────────────────────────────────────────────────────


def test_missing_fields_default_to_zero(tmp_path: Path) -> None:
    # Absent latency keys default to {}, which is a dict, so the percentiles
    # are still set (to 0) in both wrappers.
    path = _write_json(tmp_path, {})
    zeros = {name: 0 for name in METRIC_FIELDS}

    _assert_same_values(_parse_aiperf_json(path), zeros)

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), zeros)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), zeros)


# ── Falsey / null throughput averages ───────────────────────────────────────


@pytest.mark.parametrize("raw", [None, 0.0, False])
def test_falsey_throughput_local_kept_k8s_normalized(tmp_path: Path, raw: Any) -> None:
    data = _full_canonical()
    data["request_throughput"] = {"avg": raw}
    data["output_token_throughput"] = {"avg": raw}
    path = _write_json(tmp_path, data)

    parsed = _parse_aiperf_json(path)
    for name in ("req_per_sec", "output_tok_per_sec"):
        assert parsed[name] == raw and type(parsed[name]) is type(raw)

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    for name in ("req_per_sec", "output_tok_per_sec"):
        value = getattr(local, name)
        assert value == raw and type(value) is type(raw)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    for name in ("req_per_sec", "output_tok_per_sec"):
        value = getattr(k8s, name)
        assert value == 0 and type(value) is int


# ── Latency percentiles ─────────────────────────────────────────────────────


@pytest.mark.parametrize("raw", [None, 0.0])
def test_falsey_latency_percentiles_become_zero(tmp_path: Path, raw: Any) -> None:
    data = _full_canonical()
    data["time_to_first_token"] = {"p50": raw, "p99": raw}
    data["inter_token_latency"] = {"p50": raw}  # p99 absent
    path = _write_json(tmp_path, data)

    parsed = _parse_aiperf_json(path)
    for name in LATENCY_FIELDS:
        assert parsed[name] == 0 and type(parsed[name]) is int

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    for name in LATENCY_FIELDS:
        assert getattr(local, name) == 0 and type(getattr(local, name)) is int
        assert getattr(k8s, name) == 0 and type(getattr(k8s, name)) is int


# ── Duration ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        (42.5, 42.5),
        (17, 17),
        (None, 0),
        ({"avg": 7.5, "unit": "sec"}, 7.5),
        ({"unit": "sec"}, 0),
        ({"avg": None}, None),
    ],
    ids=["float", "int", "null", "obj-avg", "obj-no-avg", "obj-null-avg"],
)
def test_duration_shapes(tmp_path: Path, raw: Any, expected: Any) -> None:
    data = _full_canonical()
    data["benchmark_duration"] = raw
    path = _write_json(tmp_path, data)

    parsed = _parse_aiperf_json(path)["duration_sec"]
    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)

    for value in (parsed, local.duration_sec, k8s.duration_sec):
        if expected is None:
            assert value is None
        else:
            assert value == expected and type(value) is type(expected)


# ── Non-dict latency with a prepopulated result ─────────────────────────────


@pytest.mark.parametrize("raw", [None, 5])
def test_non_dict_latency_local_zeroes_k8s_preserves(tmp_path: Path, raw: Any) -> None:
    data = _full_canonical()
    data["time_to_first_token"] = raw
    data["inter_token_latency"] = raw
    path = _write_json(tmp_path, data)

    parsed = _parse_aiperf_json(path)
    assert not any(name in parsed for name in LATENCY_FIELDS)

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    expected_local = dict(FULL_EXPECTED, **{name: 0 for name in LATENCY_FIELDS})
    _assert_same_values(_metrics(local), expected_local)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    expected_k8s = dict(
        FULL_EXPECTED, **{name: SENTINELS[name] for name in LATENCY_FIELDS}
    )
    _assert_same_values(_metrics(k8s), expected_k8s)


# ── Artifact discovery ──────────────────────────────────────────────────────


def test_local_model_subdirectory_fallback_uses_sorted_first(tmp_path: Path) -> None:
    second = _full_canonical(request_throughput={"avg": 999.0})
    _write_json(tmp_path, second, model="b-model")
    _write_json(tmp_path, _full_canonical(), model="a-model")

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), FULL_EXPECTED)

    # Kubernetes only checks the conventional path.
    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), SENTINELS)


def test_local_top_level_file_wins_over_subdirectory(tmp_path: Path) -> None:
    _write_json(tmp_path, _full_canonical())
    _write_json(tmp_path, _full_canonical(request_throughput={"avg": 999.0}), model="a")

    local = _make_result()
    _parse_aiperf_into_result(local, tmp_path)
    assert local.req_per_sec == 10.5


# ── Missing file / invalid JSON ─────────────────────────────────────────────


def test_missing_file(tmp_path: Path) -> None:
    assert _parse_aiperf_json(tmp_path / "nope.json") == {}

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), SENTINELS)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), SENTINELS)


def test_invalid_json_is_swallowed(tmp_path: Path) -> None:
    path = _write_json(tmp_path, raw="{not json")
    assert _parse_aiperf_json(path) == {}

    local = _make_result(prepopulated=True)
    _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), SENTINELS)

    k8s = _make_result(prepopulated=True)
    _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), SENTINELS)


# ── Malformed throughput is not caught ──────────────────────────────────────


@pytest.mark.parametrize("field", ["request_throughput", "output_token_throughput"])
def test_malformed_throughput_raises_attribute_error(
    tmp_path: Path, field: str
) -> None:
    data = _full_canonical()
    data[field] = 5
    path = _write_json(tmp_path, data)

    with pytest.raises(AttributeError):
        _parse_aiperf_json(path)

    # Local buffers before assigning, so nothing is written.
    local = _make_result(prepopulated=True)
    with pytest.raises(AttributeError):
        _parse_aiperf_into_result(local, tmp_path)
    _assert_same_values(_metrics(local), SENTINELS)

    k8s = _make_result(prepopulated=True)
    with pytest.raises(AttributeError):
        _parse_k8s_aiperf_into_result(k8s, tmp_path)


# ── Kubernetes incremental (partial) update ─────────────────────────────────


def test_k8s_partial_update_before_uncaught_error(tmp_path: Path) -> None:
    # Only AttributeError (``.get`` on a non-dict) can fail mid-decode, and it escapes;
    # Kubernetes keeps fields assigned before it and leaves later ones unchanged.
    data = _full_canonical()
    data["request_throughput"] = {"avg": 7}
    data["output_token_throughput"] = 5
    _write_json(tmp_path, data)

    k8s = _make_result(prepopulated=True)
    with pytest.raises(AttributeError):
        _parse_k8s_aiperf_into_result(k8s, tmp_path)
    _assert_same_values(_metrics(k8s), dict(SENTINELS, req_per_sec=7))
