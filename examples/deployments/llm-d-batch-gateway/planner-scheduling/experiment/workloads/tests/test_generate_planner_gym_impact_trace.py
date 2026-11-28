# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import generate_planner_gym_impact_trace as trace_generator
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.timeout(30),
]


def _load_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_default_trace_has_exact_schedule_bounds_and_unique_hashes(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "impact.jsonl"

    metadata = trace_generator.write_trace(
        trace_path, trace_generator.TraceParameters()
    )
    records = _load_records(trace_path)
    sidecar = json.loads(
        trace_generator.metadata_path_for(trace_path).read_text(encoding="utf-8")
    )

    assert len(records) == 1080
    assert [
        sum(start <= record["timestamp"] < end for record in records)
        for start, end in ((0, 60_000), (60_000, 120_000), (120_000, 180_000))
    ] == [300, 480, 300]
    assert [phase["rate_rps"] for phase in metadata["parameters"]["phases"]] == [
        5.0,
        8.0,
        5.0,
    ]
    assert records == sorted(records, key=lambda record: record["timestamp"])
    assert all(
        448 <= record["input_length"] <= 576
        and 56 <= record["output_length"] <= 72
        and record["input_length"] + record["output_length"] < 4096
        for record in records
    )
    assert all(
        len(record["hash_ids"]) == math.ceil(record["input_length"] / 512)
        for record in records
    )
    all_hashes = [hash_id for record in records for hash_id in record["hash_ids"]]
    assert len(all_hashes) == len(set(all_hashes))
    assert metadata == sidecar
    assert metadata["validation"]["all_records_valid"] is True
    assert metadata["validation"]["phase_counts"] == [300, 480, 300]
    assert (
        metadata["trace"]["sha256"]
        == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    )


def test_same_seed_produces_identical_trace_and_new_seed_changes_it(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    third = tmp_path / "third.jsonl"

    first_metadata = trace_generator.write_trace(
        first, trace_generator.TraceParameters(seed=17)
    )
    second_metadata = trace_generator.write_trace(
        second, trace_generator.TraceParameters(seed=17)
    )
    third_metadata = trace_generator.write_trace(
        third, trace_generator.TraceParameters(seed=18)
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_metadata["trace"]["sha256"] == second_metadata["trace"]["sha256"]
    assert first.read_bytes() != third.read_bytes()
    assert first_metadata["trace"]["sha256"] != third_metadata["trace"]["sha256"]


def test_shared_prefix_requires_explicit_mode_and_keeps_tails_unique(
    tmp_path: Path,
) -> None:
    parameters = trace_generator.TraceParameters(
        phases=(trace_generator.Phase("only", 1.0, 4.0),),
        seed=3,
        input_tokens=1024,
        output_tokens=32,
        input_jitter_tokens=0,
        output_jitter_tokens=0,
        prefix_mode="shared",
        shared_prefix_blocks=1,
    )
    trace_path = tmp_path / "shared.jsonl"

    trace_generator.write_trace(trace_path, parameters)
    records = _load_records(trace_path)

    assert [record["hash_ids"][0] for record in records] == [0, 0, 0, 0]
    tails = [record["hash_ids"][1] for record in records]
    assert len(tails) == len(set(tails))
    assert 0 not in tails


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        (
            trace_generator.TraceParameters(
                input_tokens=4000,
                output_tokens=96,
                input_jitter_tokens=0,
                output_jitter_tokens=0,
            ),
            "strictly less",
        ),
        (
            trace_generator.TraceParameters(
                prefix_mode="shared", shared_prefix_blocks=0
            ),
            "must be positive",
        ),
        (
            trace_generator.TraceParameters(
                phases=(
                    trace_generator.Phase("same", 1, 1),
                    trace_generator.Phase("same", 1, 1),
                )
            ),
            "duplicate phase",
        ),
    ],
)
def test_invalid_contract_publishes_nothing(
    tmp_path: Path,
    parameters: trace_generator.TraceParameters,
    message: str,
) -> None:
    trace_path = tmp_path / "invalid.jsonl"

    with pytest.raises(trace_generator.TraceGenerationError, match=message):
        trace_generator.write_trace(trace_path, parameters)

    assert not trace_path.exists()
    assert not trace_generator.metadata_path_for(trace_path).exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_existing_artifacts_are_not_replaced_without_force(tmp_path: Path) -> None:
    trace_path = tmp_path / "existing.jsonl"
    metadata_path = trace_generator.metadata_path_for(trace_path)
    trace_path.write_text("original trace\n", encoding="utf-8")
    metadata_path.write_text("original metadata\n", encoding="utf-8")

    with pytest.raises(trace_generator.TraceGenerationError, match="already exists"):
        trace_generator.write_trace(trace_path, trace_generator.TraceParameters())

    assert trace_path.read_text(encoding="utf-8") == "original trace\n"
    assert metadata_path.read_text(encoding="utf-8") == "original metadata\n"


def test_cli_phase_parser_accepts_custom_schedule() -> None:
    parsed = trace_generator.parse_phase("pilot:2.5:3.2")

    assert parsed == trace_generator.Phase("pilot", 2.5, 3.2)
    assert parsed.duration_ms == 2500
    assert parsed.request_count == 8

    with pytest.raises(
        argparse.ArgumentTypeError, match="NAME:DURATION_SECONDS:RATE_RPS"
    ):
        trace_generator.parse_phase("missing-fields")
