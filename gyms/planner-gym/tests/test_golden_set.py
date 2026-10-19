# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for trace-agnostic Golden Set construction."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from autoscaling_arena.datasets import (
    TRACE_SCHEMA,
    build_golden_set,
    load_golden_set_recipe,
)
from autoscaling_arena.match_config import load_match_config
from autoscaling_arena.workloads import validate_mooncake_trace


def _raw_event(
    request_id: str,
    received_ms: int,
    *,
    input_length: int,
    output_length: int,
    block_size: int = 4,
    session_id: str | None = None,
) -> dict:
    hashes = [
        received_ms * 100 + index
        for index in range((input_length + block_size - 1) // block_size)
    ]
    event = {
        "schema": TRACE_SCHEMA,
        "event_type": "request_end",
        "event_time_unix_ms": received_ms + 10,
        "request": {
            "request_id": request_id,
            "request_received_ms": received_ms,
            "output_tokens": output_length,
            "total_time_ms": 10,
            "replay": {
                "input_length": input_length,
                "trace_block_size": block_size,
                "input_sequence_hashes": hashes,
            },
        },
    }
    if session_id is not None:
        event["agent_context"] = {"session_id": session_id}
    return {"event": event}


def _base_recipe() -> dict:
    return {
        "schema_version": 1,
        "selection": {
            "replacement": False,
            "session_policy": "preserve",
            "missing_session": "singleton",
            "max_count_error_fraction": 0.0,
            "filters": {"min_input_length": 1, "min_output_length": 1},
        },
        "arrival_process": {"type": "deterministic"},
        "pools": {
            "all": {"type": "all"},
            "input-heavy": {
                "type": "quantile",
                "field": "input_length",
                "lower_quantile": 0.5,
            },
            "output-heavy": {
                "type": "quantile",
                "field": "output_length",
                "lower_quantile": 0.5,
            },
        },
        "workloads": [
            {
                "name": "steady",
                "description": "test steady schedule",
                "phases": [{"duration_s": 10, "load_factor": 1.0, "pool": "all"}],
            },
            {
                "name": "composition",
                "description": "test composition shift",
                "phases": [
                    {"duration_s": 5, "load_factor": 1.0, "pool": "input-heavy"},
                    {"duration_s": 5, "load_factor": 1.0, "pool": "output-heavy"},
                ],
            },
        ],
    }


def _write_recipe(path: Path, recipe: dict | None = None):
    path.write_text(yaml.safe_dump(recipe or _base_recipe(), sort_keys=False))
    return load_golden_set_recipe(path)


def _write_source(root: Path) -> tuple[Path, list[str]]:
    source = root / "private-model-partition"
    source.mkdir()
    rows = []
    identifiers = []
    # Alternating tails make the two quantile pools visibly different. Rows are
    # deliberately reversed so output ordering cannot follow shard order.
    for index in range(60):
        request_id = f"sensitive-request-{index}"
        identifiers.append(request_id)
        if index < 30:
            input_length, output_length = 40, 2
        else:
            input_length, output_length = 4, 20
        rows.append(
            _raw_event(
                request_id,
                10_000 + index,
                input_length=input_length,
                output_length=output_length,
            )
        )

    plain = source / "trace-a.jsonl"
    plain.write_text("\n".join(json.dumps(row) for row in reversed(rows[:30])) + "\n")
    compressed = source / "trace-b.jsonl.gz"
    with gzip.open(compressed, "wt") as handle:
        for row in reversed(rows[30:]):
            handle.write(json.dumps(row) + "\n")
        # An identical overlapping request must be removed by request_id.
        handle.write(json.dumps(rows[0]) + "\n")
        # One invalid turn invalidates the complete visible tagged session.
        handle.write(
            json.dumps(
                _raw_event(
                    "bad-session-valid-turn",
                    20_000,
                    input_length=8,
                    output_length=2,
                    session_id="private-session-id",
                )
            )
            + "\n"
        )
        handle.write(json.dumps({"verification": {"status": "ok"}}) + "\n")
        handle.write(
            json.dumps(
                {
                    "event": {
                        "schema": TRACE_SCHEMA,
                        "event_type": "engine_start",
                        "event_time_unix_ms": 20_002,
                    }
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                _raw_event(
                    "bad-session-zero-turn",
                    20_001,
                    input_length=8,
                    output_length=0,
                    session_id="private-session-id",
                )
            )
            + "\n"
        )
    identifiers.extend(
        ["bad-session-valid-turn", "bad-session-zero-turn", "private-session-id"]
    )
    return source, identifiers


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_public_example_recipe_is_valid():
    recipe_path = (
        Path(__file__).resolve().parents[1] / "configs" / "golden-set.example.yaml"
    )

    recipe = load_golden_set_recipe(recipe_path)

    assert len(recipe.data["workloads"]) == 6
    assert recipe.data["pools"]["input-heavy"]["field"] == "input_to_output_ratio"


def test_recipe_rejects_names_reserved_by_match_config(tmp_path):
    recipe = _base_recipe()
    recipe["workloads"] = [
        {
            "name": "flat",
            "phases": [{"duration_s": 1, "load_factor": 1, "pool": "all"}],
        }
    ]
    path = tmp_path / "reserved.yaml"
    path.write_text(yaml.safe_dump(recipe))

    with pytest.raises(ValueError, match="conflicts with a built-in workload"):
        load_golden_set_recipe(path)


def test_build_is_deterministic_private_and_replayable(tmp_path):
    source, private_identifiers = _write_source(tmp_path)
    recipe = _write_recipe(tmp_path / "recipe.yaml")

    first = build_golden_set(
        recipe, [source], tmp_path / "out-a", reference_rps=1.0, seed=7
    )
    second = build_golden_set(
        recipe, [source], tmp_path / "out-b", reference_rps=1.0, seed=7
    )

    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert [path.read_bytes() for path in first.trace_paths] == [
        path.read_bytes() for path in second.trace_paths
    ]
    assert first.manifest["source"]["request_rows"] == 63
    assert first.manifest["source"]["non_request_rows"] == 2
    assert first.manifest["source"]["duplicate_requests"] == 1
    assert first.manifest["source"]["filtered_requests"] == 1
    assert first.manifest["source"]["filtered_session_requests"] == 1
    assert first.manifest["source"]["replayable_requests"] == 60
    assert first.manifest["source"]["loaded_hash_references"] == 334
    assert first.manifest["source"]["block_size"] == 4

    serialized = first.manifest_path.read_text()
    assert str(source) not in serialized
    assert "private-model-partition" not in serialized
    assert all(identifier not in serialized for identifier in private_identifiers)
    for path in first.trace_paths:
        assert validate_mooncake_trace(path, block_size=4, presorted=True) > 0
        assert all(
            set(record) == {"timestamp", "input_length", "output_length", "hash_ids"}
            for record in _records(path)
        )
        workload = next(
            item for item in first.manifest["workloads"] if item["file"] == path.name
        )
        assert workload["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    composition = next(
        item for item in first.manifest["workloads"] if item["name"] == "composition"
    )
    first_phase, second_phase = composition["phases"]
    assert first_phase["median_input_length"] > second_phase["median_input_length"]
    assert first_phase["median_output_length"] < second_phase["median_output_length"]
    assert (
        first_phase["median_input_to_output_ratio"]
        > second_phase["median_input_to_output_ratio"]
    )
    assert first_phase["start_ms"] == 0
    assert first_phase["end_ms"] == 5_000
    assert second_phase["start_ms"] == 5_000
    assert second_phase["end_ms"] == 10_000


def test_adding_workload_does_not_perturb_existing_trace(tmp_path):
    source, _ = _write_source(tmp_path)
    base = _base_recipe()
    steady_only = deepcopy(base)
    steady_only["workloads"] = [base["workloads"][0]]
    with_extra = deepcopy(base)
    with_extra["workloads"].reverse()

    one = build_golden_set(
        _write_recipe(tmp_path / "one.yaml", steady_only),
        [source],
        tmp_path / "one",
        reference_rps=1.0,
        seed=9,
    )
    two = build_golden_set(
        _write_recipe(tmp_path / "two.yaml", with_extra),
        [source],
        tmp_path / "two",
        reference_rps=1.0,
        seed=9,
    )

    assert (one.output_dir / "steady.jsonl").read_bytes() == (
        two.output_dir / "steady.jsonl"
    ).read_bytes()


def test_seeded_burst_overlay_materializes_inside_duration(tmp_path):
    source, _ = _write_source(tmp_path)
    recipe_data = _base_recipe()
    recipe_data["workloads"] = [
        {
            "name": "bursts",
            "duration_s": 20,
            "baseline_load_factor": 0.25,
            "pool": "all",
            "burst_overlay": {
                "count": 2,
                "width_s": [1, 2],
                "load_factor": [0.75, 1.0],
                "minimum_gap_s": 2,
            },
        }
    ]
    result = build_golden_set(
        _write_recipe(tmp_path / "bursts.yaml", recipe_data),
        [source],
        tmp_path / "bursts-out",
        reference_rps=1.0,
        seed=3,
    )

    workload = result.manifest["workloads"][0]
    schedule = workload["phases"][0]["schedule"]
    assert workload["duration_ms"] == 20_000
    assert len(schedule["bursts"]) == 2
    assert schedule["bursts"][0]["end_ms"] <= schedule["bursts"][1]["start_ms"]
    assert all(
        record["timestamp"] < 20_000 for record in _records(result.trace_paths[0])
    )


def test_whole_session_is_never_split_to_hit_target(tmp_path):
    source = tmp_path / "session.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                _raw_event(
                    f"turn-{index}",
                    index,
                    input_length=4,
                    output_length=2,
                    session_id="one-session",
                )
            )
            for index in range(2)
        )
        + "\n"
    )
    recipe_data = _base_recipe()
    recipe_data["workloads"] = [
        {
            "name": "one-request-target",
            "phases": [{"duration_s": 1, "load_factor": 1.0, "pool": "all"}],
        }
    ]
    recipe = _write_recipe(tmp_path / "session-recipe.yaml", recipe_data)

    with pytest.raises(ValueError, match="without splitting sessions"):
        build_golden_set(recipe, [source], tmp_path / "session-out", reference_rps=1.0)


def test_whole_session_selection_finds_feasible_atomic_combination(tmp_path):
    source = tmp_path / "sessions.jsonl"
    rows = []
    timestamp = 0
    for session_id, size in (("eight", 8), ("six", 6), ("four", 4)):
        for turn in range(size):
            rows.append(
                _raw_event(
                    f"{session_id}-{turn}",
                    timestamp,
                    input_length=4,
                    output_length=2,
                    session_id=session_id,
                )
            )
            timestamp += 1
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    recipe_data = _base_recipe()
    recipe_data["workloads"] = [
        {
            "name": "ten-request-target",
            "phases": [{"duration_s": 10, "load_factor": 1.0, "pool": "all"}],
        }
    ]
    recipe = _write_recipe(tmp_path / "sessions-recipe.yaml", recipe_data)

    result = build_golden_set(
        recipe, [source], tmp_path / "sessions-out", reference_rps=1.0
    )

    assert len(_records(result.trace_paths[0])) == 10


def test_mixed_formats_are_rejected(tmp_path):
    source = tmp_path / "mixed"
    source.mkdir()
    (source / "raw.jsonl").write_text(
        json.dumps(_raw_event("request-1", 1, input_length=4, output_length=2)) + "\n"
    )
    (source / "replay.jsonl").write_text(
        '{"timestamp":0,"input_length":4,"output_length":2,"hash_ids":[1]}\n'
    )
    recipe = _write_recipe(tmp_path / "recipe.yaml")

    with pytest.raises(ValueError, match="mixed trace formats"):
        build_golden_set(
            recipe,
            [source],
            tmp_path / "out",
            source_block_size=4,
            reference_rps=1.0,
        )


def test_mixed_raw_block_sizes_are_rejected(tmp_path):
    source = tmp_path / "mixed-blocks.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    _raw_event(
                        "request-1", 1, input_length=4, output_length=2, block_size=4
                    )
                ),
                json.dumps(
                    _raw_event(
                        "request-2", 2, input_length=8, output_length=2, block_size=8
                    )
                ),
            ]
        )
        + "\n"
    )
    recipe = _write_recipe(tmp_path / "recipe.yaml")

    with pytest.raises(ValueError, match="mixed trace block sizes"):
        build_golden_set(recipe, [source], tmp_path / "out", reference_rps=1.0)


def test_replay_input_requires_and_uses_explicit_block_size(tmp_path):
    source = tmp_path / "base.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    # Converted datasets may retain raw-source provenance; the
                    # complete root-level replay shape remains authoritative.
                    "schema": TRACE_SCHEMA,
                    "event_type": "request_end",
                    "timestamp": index,
                    "input_length": 4,
                    "output_length": 2,
                    "hash_ids": [100 + index],
                }
            )
            for index in range(20)
        )
        + "\n"
    )
    recipe_data = _base_recipe()
    recipe_data["workloads"] = [recipe_data["workloads"][0]]
    recipe = _write_recipe(tmp_path / "recipe.yaml", recipe_data)

    with pytest.raises(ValueError, match="source-block-size is required"):
        build_golden_set(recipe, [source], tmp_path / "missing", reference_rps=1.0)

    result = build_golden_set(
        recipe,
        [source],
        tmp_path / "valid",
        source_block_size=4,
        reference_rps=1.0,
    )
    assert result.manifest["source"]["format"] == "replay_jsonl"
    assert (
        validate_mooncake_trace(result.trace_paths[0], block_size=4, presorted=True)
        == 10
    )

    example_path = (
        Path(__file__).resolve().parents[1] / "configs" / "match.sim.example.yaml"
    )
    match_data = yaml.safe_load(example_path.read_text())
    match_data["backend"]["planner_config"] = str(
        example_path.parent / "planner.sim.example.yaml"
    )
    match_data["evaluations"] = yaml.safe_load(
        result.match_config_fragment_path.read_text()
    )["evaluations"]
    match_path = result.output_dir / "match.yaml"
    match_path.write_text(yaml.safe_dump(match_data, sort_keys=False))

    loaded = load_match_config(match_path)
    assert [evaluation.workload for evaluation in loaded.evaluations] == ["steady"]


def test_source_hash_guard_aborts_instead_of_silently_sampling(tmp_path):
    source, _ = _write_source(tmp_path)
    recipe = _write_recipe(tmp_path / "recipe.yaml")
    output = tmp_path / "guarded-out"

    with pytest.raises(ValueError, match="max_source_hashes=10"):
        build_golden_set(
            recipe,
            [source],
            output,
            reference_rps=1.0,
            max_source_hashes=10,
        )

    assert not output.exists()


def test_composition_workload_rejects_indistinguishable_pools(tmp_path):
    source = tmp_path / "homogeneous.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                _raw_event(
                    f"request-{index}",
                    index,
                    input_length=4,
                    output_length=2,
                )
            )
            for index in range(20)
        )
        + "\n"
    )
    recipe_data = _base_recipe()
    recipe_data["workloads"] = [recipe_data["workloads"][1]]
    recipe = _write_recipe(tmp_path / "recipe.yaml", recipe_data)
    output = tmp_path / "composition-out"

    with pytest.raises(ValueError, match="identical request groups"):
        build_golden_set(recipe, [source], output, reference_rps=1.0)

    assert not output.exists()


def test_failed_build_leaves_no_partial_output_or_temporary_directory(tmp_path):
    source, _ = _write_source(tmp_path)
    recipe_data = _base_recipe()
    recipe_data["workloads"] = [
        recipe_data["workloads"][0],
        {
            "name": "too-large",
            "phases": [{"duration_s": 1_000, "load_factor": 1.0, "pool": "all"}],
        },
    ]
    recipe = _write_recipe(tmp_path / "recipe.yaml", recipe_data)
    output = tmp_path / "atomic-out"

    with pytest.raises(ValueError, match="supply a larger source"):
        build_golden_set(recipe, [source], output, reference_rps=1.0)

    assert not output.exists()
    assert not list(tmp_path.glob(".atomic-out.tmp-*"))


def test_published_directory_honors_process_umask(tmp_path):
    source, _ = _write_source(tmp_path)
    recipe_data = _base_recipe()
    recipe_data["workloads"] = [recipe_data["workloads"][0]]
    recipe = _write_recipe(tmp_path / "recipe.yaml", recipe_data)

    previous_umask = os.umask(0o027)
    try:
        result = build_golden_set(
            recipe,
            [source],
            tmp_path / "published-out",
            reference_rps=1.0,
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(result.output_dir.stat().st_mode) == 0o750
