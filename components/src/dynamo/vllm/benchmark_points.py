# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema for explicit vLLM self-benchmark points."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BenchmarkMode = Literal["prefill", "decode", "agg"]
BENCHMARK_MODES: tuple[BenchmarkMode, ...] = ("prefill", "decode", "agg")


class _BenchmarkPoint(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    batch_size: int = Field(gt=0)


class PrefillBenchmarkPoint(_BenchmarkPoint):
    total_prefill_tokens: int = Field(gt=0)
    total_kv_read_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> PrefillBenchmarkPoint:
        if self.total_prefill_tokens < self.batch_size:
            raise ValueError("total_prefill_tokens must be at least batch_size")
        if 0 < self.total_kv_read_tokens < self.batch_size:
            raise ValueError("total_kv_read_tokens must be zero or at least batch_size")
        return self


class DecodeBenchmarkPoint(_BenchmarkPoint):
    total_kv_read_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_totals(self) -> DecodeBenchmarkPoint:
        if self.total_kv_read_tokens < self.batch_size:
            raise ValueError("total_kv_read_tokens must be at least batch_size")
        return self


class BenchmarkPoints(BaseModel):
    """Versioned, ordered benchmark-point manifest."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: int = Field(strict=True, ge=1, le=1)
    prefill: list[PrefillBenchmarkPoint]
    decode: list[DecodeBenchmarkPoint]

    def require_points_for(self, mode: BenchmarkMode) -> None:
        if mode in ("prefill", "agg") and not self.prefill:
            raise ValueError(
                f"prefill must contain at least one point for benchmark mode {mode!r}"
            )
        if mode in ("decode", "agg") and not self.decode:
            raise ValueError(
                f"decode must contain at least one point for benchmark mode {mode!r}"
            )


def load_benchmark_points_file(path: str, mode: BenchmarkMode) -> BenchmarkPoints:
    """Load and validate a benchmark manifest before workers start."""

    try:
        points = BenchmarkPoints.model_validate_json(Path(path).read_bytes())
        points.require_points_for(mode)
        return points
    except Exception as error:
        raise ValueError(f"--benchmark-points-file {path!r}: {error}") from error
