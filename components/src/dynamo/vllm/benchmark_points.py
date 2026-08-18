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


class PrefillPointCandidate(_PointCandidate):
    total_prefill_tokens: int = Field(gt=0)
    total_kv_read_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_totals(self) -> PrefillPointCandidate:
        if self.total_prefill_tokens < self.batch_size:
            raise ValueError("total_prefill_tokens must be at least batch_size")
        if 0 < self.total_kv_read_tokens < self.batch_size:
            raise ValueError("total_kv_read_tokens must be zero or at least batch_size")
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

    schema_version: int = Field(strict=True, ge=1, le=1)
    prefill: list[PrefillPointCandidate]
    decode: list[DecodePointCandidate]


def load_benchmark_points_file(path: str) -> BenchmarkPoints:
    """Load and validate a benchmark manifest before workers start."""

    try:
        return BenchmarkPoints.model_validate_json(Path(path).read_bytes())
    except Exception as error:
        raise ValueError(f"--benchmark-points-file {path!r}: {error}") from error
