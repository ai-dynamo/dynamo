#!/usr/bin/env python3
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

"""Generate the bounded Mooncake trace used by the Batch-impact experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GENERATOR_NAME = "generate_planner_gym_impact_trace"
GENERATOR_VERSION = 1
PHASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
RECORD_KEYS = {"timestamp", "input_length", "output_length", "hash_ids"}


class TraceGenerationError(ValueError):
    """A trace cannot be generated without violating its evidence contract."""


@dataclass(frozen=True)
class Phase:
    """One constant-rate phase in the open-loop request schedule."""

    name: str
    duration_seconds: float
    rate_rps: float

    @property
    def duration_ms(self) -> int:
        """Return the exactly representable phase duration in milliseconds."""
        return round(self.duration_seconds * 1000)

    @property
    def request_count(self) -> int:
        """Return the nearest whole request count, with halves rounded upward."""
        return math.floor(self.duration_seconds * self.rate_rps + 0.5)


DEFAULT_PHASES = (
    Phase("low-1", 60.0, 5.0),
    Phase("high", 60.0, 8.0),
    Phase("low-2", 60.0, 5.0),
)


@dataclass(frozen=True)
class TraceParameters:
    """Every parameter that determines the generated trace bytes."""

    phases: tuple[Phase, ...] = DEFAULT_PHASES
    seed: int = 20260918
    input_tokens: int = 512
    output_tokens: int = 64
    input_jitter_tokens: int = 64
    output_jitter_tokens: int = 8
    max_model_length: int = 4096
    block_size: int = 512
    prefix_mode: str = "none"
    shared_prefix_blocks: int = 0


def parse_phase(value: str) -> Phase:
    """Parse ``NAME:DURATION_SECONDS:RATE_RPS`` from one CLI value."""
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "phase must use NAME:DURATION_SECONDS:RATE_RPS"
        )
    name, duration_text, rate_text = parts
    try:
        duration_seconds = float(duration_text)
        rate_rps = float(rate_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "phase duration and rate must be numbers"
        ) from error
    phase = Phase(name=name, duration_seconds=duration_seconds, rate_rps=rate_rps)
    try:
        _validate_phase(phase)
    except TraceGenerationError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return phase


def _validate_phase(phase: Phase) -> None:
    if PHASE_NAME_RE.fullmatch(phase.name) is None:
        raise TraceGenerationError(
            f"invalid phase name {phase.name!r}; use letters, digits, '.', '_', or '-'"
        )
    if not math.isfinite(phase.duration_seconds) or phase.duration_seconds <= 0:
        raise TraceGenerationError(
            f"phase {phase.name!r} duration must be finite and positive"
        )
    duration_ms = phase.duration_seconds * 1000
    if not math.isfinite(duration_ms) or not math.isclose(
        duration_ms, round(duration_ms), abs_tol=1e-9
    ):
        raise TraceGenerationError(
            f"phase {phase.name!r} duration must have millisecond precision"
        )
    if not math.isfinite(phase.rate_rps) or phase.rate_rps <= 0:
        raise TraceGenerationError(
            f"phase {phase.name!r} rate must be finite and positive"
        )
    request_count = phase.duration_seconds * phase.rate_rps
    if not math.isfinite(request_count) or math.floor(request_count + 0.5) <= 0:
        raise TraceGenerationError(
            f"phase {phase.name!r} rate and duration produce no requests"
        )


def validate_parameters(parameters: TraceParameters) -> None:
    """Validate the complete generation contract before allocating records."""
    if isinstance(parameters.seed, bool) or not isinstance(parameters.seed, int):
        raise TraceGenerationError("seed must be an integer")
    if not parameters.phases:
        raise TraceGenerationError("at least one phase is required")
    phase_names: set[str] = set()
    for phase in parameters.phases:
        _validate_phase(phase)
        if phase.name in phase_names:
            raise TraceGenerationError(f"duplicate phase name: {phase.name!r}")
        phase_names.add(phase.name)

    positive_values = {
        "input_tokens": parameters.input_tokens,
        "output_tokens": parameters.output_tokens,
        "max_model_length": parameters.max_model_length,
        "block_size": parameters.block_size,
    }
    for name, value in positive_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TraceGenerationError(f"{name} must be a positive integer")

    jitter_values = {
        "input_jitter_tokens": parameters.input_jitter_tokens,
        "output_jitter_tokens": parameters.output_jitter_tokens,
        "shared_prefix_blocks": parameters.shared_prefix_blocks,
    }
    for name, value in jitter_values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TraceGenerationError(f"{name} must be a nonnegative integer")

    if parameters.input_tokens - parameters.input_jitter_tokens <= 0:
        raise TraceGenerationError("input jitter can produce a nonpositive length")
    if parameters.output_tokens - parameters.output_jitter_tokens <= 0:
        raise TraceGenerationError("output jitter can produce a nonpositive length")
    maximum_total = (
        parameters.input_tokens
        + parameters.input_jitter_tokens
        + parameters.output_tokens
        + parameters.output_jitter_tokens
    )
    if maximum_total >= parameters.max_model_length:
        raise TraceGenerationError(
            "maximum input plus output length must be strictly less than "
            f"max_model_length ({maximum_total} >= {parameters.max_model_length})"
        )
    if parameters.prefix_mode not in {"none", "shared"}:
        raise TraceGenerationError("prefix_mode must be 'none' or 'shared'")
    if parameters.prefix_mode == "none" and parameters.shared_prefix_blocks != 0:
        raise TraceGenerationError(
            "shared_prefix_blocks must be zero when prefix_mode is 'none'"
        )
    if parameters.prefix_mode == "shared" and parameters.shared_prefix_blocks <= 0:
        raise TraceGenerationError(
            "shared_prefix_blocks must be positive when prefix_mode is 'shared'"
        )


def _sample_length(rng: random.Random, center: int, jitter: int) -> int:
    return center + rng.randint(-jitter, jitter)


def generate_records(parameters: TraceParameters) -> list[dict[str, Any]]:
    """Generate deterministic, phase-stratified Mooncake records."""
    validate_parameters(parameters)
    rng = random.Random(parameters.seed)
    records: list[dict[str, Any]] = []
    phase_start_ms = 0
    next_unique_hash = parameters.shared_prefix_blocks

    for phase in parameters.phases:
        for request_index in range(phase.request_count):
            # One request lands in every equal-width stratum. This retains the
            # configured phase rate without introducing bursty count variance;
            # the seed controls the position within each stratum and the shape.
            relative_ms = int(
                (request_index + rng.random()) * phase.duration_ms / phase.request_count
            )
            relative_ms = min(relative_ms, phase.duration_ms - 1)
            input_length = _sample_length(
                rng, parameters.input_tokens, parameters.input_jitter_tokens
            )
            output_length = _sample_length(
                rng, parameters.output_tokens, parameters.output_jitter_tokens
            )
            block_count = math.ceil(input_length / parameters.block_size)
            if parameters.prefix_mode == "none":
                hash_ids = list(range(next_unique_hash, next_unique_hash + block_count))
                next_unique_hash += block_count
            else:
                shared_count = min(parameters.shared_prefix_blocks, block_count)
                tail_count = block_count - shared_count
                hash_ids = list(range(shared_count)) + list(
                    range(next_unique_hash, next_unique_hash + tail_count)
                )
                next_unique_hash += tail_count

            records.append(
                {
                    "timestamp": phase_start_ms + relative_ms,
                    "input_length": input_length,
                    "output_length": output_length,
                    "hash_ids": hash_ids,
                }
            )
        phase_start_ms += phase.duration_ms

    validate_records(records, parameters)
    return records


def validate_records(
    records: Sequence[dict[str, Any]], parameters: TraceParameters
) -> dict[str, Any]:
    """Validate every record and return observed evidence metadata."""
    validate_parameters(parameters)
    expected_count = sum(phase.request_count for phase in parameters.phases)
    if len(records) != expected_count:
        raise TraceGenerationError(
            f"trace has {len(records)} records; expected {expected_count}"
        )

    previous_timestamp = -1
    phase_index = 0
    phase_start_ms = 0
    phase_end_ms = parameters.phases[0].duration_ms
    phase_counts = [0 for _ in parameters.phases]
    seen_unique_hashes: set[int] = set()
    shared_hashes = set(range(parameters.shared_prefix_blocks))
    input_lengths: list[int] = []
    output_lengths: list[int] = []

    for record_index, record in enumerate(records):
        location = f"record {record_index}"
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise TraceGenerationError(
                f"{location} must contain exactly {sorted(RECORD_KEYS)}"
            )
        timestamp = record["timestamp"]
        input_length = record["input_length"]
        output_length = record["output_length"]
        hash_ids = record["hash_ids"]
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise TraceGenerationError(f"{location} timestamp must be an integer")
        if timestamp < previous_timestamp:
            raise TraceGenerationError(f"{location} timestamps are not sorted")
        previous_timestamp = timestamp
        while phase_index < len(parameters.phases) and timestamp >= phase_end_ms:
            phase_start_ms = phase_end_ms
            phase_index += 1
            if phase_index < len(parameters.phases):
                phase_end_ms += parameters.phases[phase_index].duration_ms
        if phase_index >= len(parameters.phases) or timestamp < phase_start_ms:
            raise TraceGenerationError(
                f"{location} timestamp {timestamp} is outside the phase schedule"
            )
        phase_counts[phase_index] += 1

        if (
            isinstance(input_length, bool)
            or not isinstance(input_length, int)
            or input_length <= 0
        ):
            raise TraceGenerationError(
                f"{location} input_length must be a positive integer"
            )
        if (
            isinstance(output_length, bool)
            or not isinstance(output_length, int)
            or output_length <= 0
        ):
            raise TraceGenerationError(
                f"{location} output_length must be a positive integer"
            )
        if input_length + output_length >= parameters.max_model_length:
            raise TraceGenerationError(
                f"{location} total length is not below max_model_length"
            )
        if not (
            parameters.input_tokens - parameters.input_jitter_tokens
            <= input_length
            <= parameters.input_tokens + parameters.input_jitter_tokens
        ):
            raise TraceGenerationError(f"{location} input_length is out of bounds")
        if not (
            parameters.output_tokens - parameters.output_jitter_tokens
            <= output_length
            <= parameters.output_tokens + parameters.output_jitter_tokens
        ):
            raise TraceGenerationError(f"{location} output_length is out of bounds")

        expected_hash_count = math.ceil(input_length / parameters.block_size)
        if not isinstance(hash_ids, list) or len(hash_ids) != expected_hash_count:
            raise TraceGenerationError(
                f"{location} must have {expected_hash_count} hash_ids"
            )
        if not all(
            isinstance(hash_id, int) and not isinstance(hash_id, bool) and hash_id >= 0
            for hash_id in hash_ids
        ):
            raise TraceGenerationError(
                f"{location} hash_ids must be nonnegative integers"
            )
        if len(set(hash_ids)) != len(hash_ids):
            raise TraceGenerationError(f"{location} repeats a hash_id")

        if parameters.prefix_mode == "none":
            reused = seen_unique_hashes.intersection(hash_ids)
            if reused:
                raise TraceGenerationError(
                    f"{location} reuses hash_ids in no-sharing mode: {sorted(reused)}"
                )
            seen_unique_hashes.update(hash_ids)
        else:
            shared_count = min(parameters.shared_prefix_blocks, expected_hash_count)
            if hash_ids[:shared_count] != list(range(shared_count)):
                raise TraceGenerationError(
                    f"{location} does not use the configured shared prefix"
                )
            tail = hash_ids[shared_count:]
            if shared_hashes.intersection(tail) or seen_unique_hashes.intersection(
                tail
            ):
                raise TraceGenerationError(f"{location} reuses a non-prefix hash_id")
            seen_unique_hashes.update(tail)

        input_lengths.append(input_length)
        output_lengths.append(output_length)

    expected_phase_counts = [phase.request_count for phase in parameters.phases]
    if phase_counts != expected_phase_counts:
        raise TraceGenerationError(
            f"observed phase counts {phase_counts}; expected {expected_phase_counts}"
        )

    return {
        "all_records_valid": True,
        "timestamps_sorted": True,
        "strict_total_below_max_model_length": True,
        "phase_counts": phase_counts,
        "input_length": {
            "min": min(input_lengths),
            "max": max(input_lengths),
            "mean": sum(input_lengths) / len(input_lengths),
        },
        "output_length": {
            "min": min(output_lengths),
            "max": max(output_lengths),
            "mean": sum(output_lengths) / len(output_lengths),
        },
        "unique_nonshared_hashes": len(seen_unique_hashes),
    }


def _serialize_records(records: Sequence[dict[str, Any]]) -> bytes:
    lines = [json.dumps(record, separators=(",", ":")) for record in records]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_temporary(target: Path, payload: bytes) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
        return Path(output.name)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metadata_path_for(output_path: Path) -> Path:
    """Return the mandatory metadata sidecar path for a trace path."""
    return output_path.with_name(f"{output_path.name}.metadata.json")


def _parameter_metadata(parameters: TraceParameters) -> dict[str, Any]:
    phase_start_ms = 0
    phases: list[dict[str, Any]] = []
    for phase in parameters.phases:
        phase_end_ms = phase_start_ms + phase.duration_ms
        phases.append(
            {
                "name": phase.name,
                "duration_seconds": phase.duration_seconds,
                "rate_rps": phase.rate_rps,
                "request_count": phase.request_count,
                "effective_rate_rps": phase.request_count / phase.duration_seconds,
                "start_ms": phase_start_ms,
                "end_ms_exclusive": phase_end_ms,
            }
        )
        phase_start_ms = phase_end_ms
    return {
        "seed": parameters.seed,
        "phases": phases,
        "input_tokens": parameters.input_tokens,
        "output_tokens": parameters.output_tokens,
        "input_jitter_tokens": parameters.input_jitter_tokens,
        "output_jitter_tokens": parameters.output_jitter_tokens,
        "max_model_length": parameters.max_model_length,
        "block_size": parameters.block_size,
        "prefix_mode": parameters.prefix_mode,
        "shared_prefix_blocks": parameters.shared_prefix_blocks,
    }


def _publish_temporary(temporary: Path, target: Path, *, force: bool) -> None:
    if force:
        os.replace(temporary, target)
        return
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise TraceGenerationError(f"output already exists: {target}") from error
    temporary.unlink()


def write_trace(
    output_path: Path,
    parameters: TraceParameters,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Validate and atomically publish a trace and checksum-bearing sidecar."""
    if output_path.suffix != ".jsonl":
        raise TraceGenerationError("output path must end in .jsonl")
    metadata_path = metadata_path_for(output_path)
    if not force:
        for target in (output_path, metadata_path):
            if target.exists():
                raise TraceGenerationError(f"output already exists: {target}")

    records = generate_records(parameters)
    validation = validate_records(records, parameters)
    trace_payload = _serialize_records(records)
    trace_sha256 = _sha256_bytes(trace_payload)
    metadata = {
        "schema_version": 1,
        "format": "mooncake_trace_jsonl",
        "generator": {
            "name": GENERATOR_NAME,
            "version": GENERATOR_VERSION,
        },
        "parameters": _parameter_metadata(parameters),
        "record_count": len(records),
        "duration_seconds": sum(phase.duration_seconds for phase in parameters.phases),
        "trace": {
            "filename": output_path.name,
            "bytes": len(trace_payload),
            "sha256": trace_sha256,
        },
        "validation": validation,
    }
    metadata_payload = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    trace_temporary: Path | None = None
    metadata_temporary: Path | None = None
    try:
        trace_temporary = _write_temporary(output_path, trace_payload)
        metadata_temporary = _write_temporary(metadata_path, metadata_payload)
        # Publish the self-contained trace first, then the sidecar which attests
        # to its final bytes. Each target appears only after a complete fsync.
        _publish_temporary(trace_temporary, output_path, force=force)
        trace_temporary = None
        _publish_temporary(metadata_temporary, metadata_path, force=force)
        metadata_temporary = None
    finally:
        for temporary in (trace_temporary, metadata_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return metadata


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--phase",
        action="append",
        type=parse_phase,
        metavar="NAME:DURATION_SECONDS:RATE_RPS",
        help=(
            "repeat for each ordered phase; defaults to "
            "low-1:60:5, high:60:8, low-2:60:5"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--input-tokens", type=_positive_int, default=512)
    parser.add_argument("--output-tokens", type=_positive_int, default=64)
    parser.add_argument("--input-jitter-tokens", type=_nonnegative_int, default=64)
    parser.add_argument("--output-jitter-tokens", type=_nonnegative_int, default=8)
    parser.add_argument("--max-model-length", type=_positive_int, default=4096)
    parser.add_argument("--block-size", type=_positive_int, default=512)
    parser.add_argument("--prefix-mode", choices=("none", "shared"), default="none")
    parser.add_argument("--shared-prefix-blocks", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="atomically replace existing outputs"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one trace from command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    parameters = TraceParameters(
        phases=tuple(args.phase) if args.phase else DEFAULT_PHASES,
        seed=args.seed,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        input_jitter_tokens=args.input_jitter_tokens,
        output_jitter_tokens=args.output_jitter_tokens,
        max_model_length=args.max_model_length,
        block_size=args.block_size,
        prefix_mode=args.prefix_mode,
        shared_prefix_blocks=args.shared_prefix_blocks,
    )
    try:
        metadata = write_trace(args.output, parameters, force=args.force)
    except (OSError, TraceGenerationError) as error:
        parser.error(str(error))
    summary = {
        "metadata": str(metadata_path_for(args.output)),
        "record_count": metadata["record_count"],
        "sha256": metadata["trace"]["sha256"],
        "trace": str(args.output),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
