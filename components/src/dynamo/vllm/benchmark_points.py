# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema for explicit vLLM self-benchmark points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BenchmarkMode = Literal["prefill", "decode", "agg"]
BENCHMARK_MODES: tuple[BenchmarkMode, ...] = ("prefill", "decode", "agg")


class _PointCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    batch_size: int = Field(gt=0)


class PartitionSpec(BaseModel):
    """How a point's totals are spread across its requests.

    The default -- omitting the field entirely -- is the historical equal
    split. A rung instead raises ``high_count`` requests by ``fraction`` and
    lets the rest absorb the deficit, holding the totals exactly fixed.

    Which axis is perturbed matters: ``axis="new"`` varies the freshly
    computed tokens at a fixed KV read, ``axis="kv"`` does the reverse. Work
    models whose per-request cost is linear along the untouched axis see that
    axis's contribution cancel, which is what makes a single rung solve a
    single coefficient.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # "both" perturbs the new-token and KV axes together, which is what
    # real traffic does; the single-axis values exist to isolate one work
    # term at a time during calibration.
    axis: Literal["new", "kv", "both"]
    high_count: int = Field(gt=0)
    fraction: float = Field(gt=0.0, lt=1.0)


class PrefillPointCandidate(_PointCandidate):
    total_prefill_tokens: int = Field(gt=0)
    total_kv_read_tokens: int = Field(ge=0)
    # None means the equal split; schema_version 1 manifests never set it.
    partition: PartitionSpec | None = None
    # Explicit ``(new_tokens, kv_read_tokens)`` per request, in order.
    #
    # A PartitionSpec describes a spread by its shape, which is enough when the
    # spread is the point. It is not enough for regime calibration: those rows
    # are solved from an inequality on every request -- one group pinned exactly
    # at ``topk + 1`` or at a launch-bound floor, the integer remainder placed on
    # whichever side the regime leaves headroom on -- and a fraction cannot
    # express any of that. Rounding it into a fraction moves a row across the
    # bound and silently changes which kernel the batch measures.
    # ``list[list[int]]`` rather than tuples: the model is strict and JSON has
    # no tuple, so a tuple annotation would reject every real manifest.
    rows: list[list[int]] | None = None

    @model_validator(mode="after")
    def validate_totals(self) -> PrefillPointCandidate:
        if self.total_prefill_tokens < self.batch_size:
            raise ValueError("total_prefill_tokens must be at least batch_size")
        if 0 < self.total_kv_read_tokens < self.batch_size:
            raise ValueError("total_kv_read_tokens must be zero or at least batch_size")
        if self.rows is not None:
            if self.partition is not None:
                raise ValueError("rows and partition are mutually exclusive")
            if any(len(row) != 2 for row in self.rows):
                raise ValueError(
                    "every row must be [new_tokens, kv_read_tokens], two entries"
                )
            if len(self.rows) != self.batch_size:
                raise ValueError(
                    f"rows has {len(self.rows)} entries, batch_size is {self.batch_size}"
                )
            if any(new_tokens < 1 for new_tokens, _ in self.rows):
                raise ValueError("every row must carry at least one new token")
            if any(kv_read < 0 for _, kv_read in self.rows):
                raise ValueError("kv read tokens cannot be negative")
            # The label is a difference against the equal-length batch with the
            # SAME totals, so a manifest whose rows drift from its own totals
            # would be measuring the extra tokens as well as the spread.
            new_total = sum(new_tokens for new_tokens, _ in self.rows)
            kv_total = sum(kv_read for _, kv_read in self.rows)
            if new_total != self.total_prefill_tokens:
                raise ValueError(
                    f"rows sum to {new_total} new tokens, "
                    f"total_prefill_tokens is {self.total_prefill_tokens}"
                )
            if kv_total != self.total_kv_read_tokens:
                raise ValueError(
                    f"rows sum to {kv_total} kv read tokens, "
                    f"total_kv_read_tokens is {self.total_kv_read_tokens}"
                )
        if self.partition is not None:
            if self.batch_size < 2:
                raise ValueError("partition requires batch_size >= 2")
            if self.partition.high_count >= self.batch_size:
                raise ValueError("partition.high_count must be less than batch_size")
            if self.partition.axis in ("kv", "both") and self.total_kv_read_tokens == 0:
                raise ValueError(
                    f'partition axis "{self.partition.axis}" requires '
                    "total_kv_read_tokens > 0"
                )
        return self


class DecodePointCandidate(_PointCandidate):
    total_kv_read_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_totals(self) -> DecodePointCandidate:
        if self.total_kv_read_tokens < self.batch_size:
            raise ValueError("total_kv_read_tokens must be at least batch_size")
        return self


class BenchmarkPoints(BaseModel):
    """Versioned, ordered benchmark-point manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    # v1: totals only, always split evenly.
    # v2: prefill points may carry an explicit PartitionSpec.
    # v3: prefill points may carry explicit per-request rows.
    schema_version: int = Field(strict=True, ge=1, le=3)
    prefill: list[PrefillPointCandidate]
    decode: list[DecodePointCandidate]

    @model_validator(mode="after")
    def validate_schema_version(self) -> BenchmarkPoints:
        if self.schema_version < 2 and any(
            point.partition is not None for point in self.prefill
        ):
            raise ValueError("partition requires schema_version >= 2")
        if self.schema_version < 3 and any(
            point.rows is not None for point in self.prefill
        ):
            raise ValueError("rows requires schema_version >= 3")
        return self


def load_benchmark_points_file(path: str) -> BenchmarkPoints:
    """Load and validate a benchmark manifest before workers start."""

    try:
        return BenchmarkPoints.model_validate_json(Path(path).read_bytes())
    except Exception as error:
        raise ValueError(f"--benchmark-points-file {path!r}: {error}") from error
