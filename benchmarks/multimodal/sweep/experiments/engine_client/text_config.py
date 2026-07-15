# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration contract for the text engine-client benchmark."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

EXPECTED_RUNTIMES = {"vllm-serve", "dynamo-async", "dynamo-sync"}


@dataclass(frozen=True)
class RuntimeConfig:
    workflow: str
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class TextSweepConfig:
    model: str
    concurrency: int
    request_count: int
    warmup_count: int
    target_isl: int
    osl: int
    repeats: int
    port: int
    timeout: int
    max_model_len: int
    max_num_seqs: int
    kv_cache_memory_bytes: int
    prefix_caching: bool
    runtimes: dict[str, RuntimeConfig]
    trial_order: tuple[tuple[str, ...], ...]
    source_path: Path
    source_sha256: str

    @classmethod
    def load(cls, path: Path) -> "TextSweepConfig":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        runtimes = {
            str(label): RuntimeConfig(
                workflow=str(value["workflow"]),
                extra_args=tuple(str(arg) for arg in value.get("extra_args", [])),
            )
            for label, value in raw["runtimes"].items()
        }
        config = cls(
            model=str(raw["model"]),
            concurrency=int(raw["concurrency"]),
            request_count=int(raw["request_count"]),
            warmup_count=int(raw["warmup_count"]),
            target_isl=int(raw["target_isl"]),
            osl=int(raw["osl"]),
            repeats=int(raw["repeats"]),
            port=int(raw["port"]),
            timeout=int(raw["timeout"]),
            max_model_len=int(raw["max_model_len"]),
            max_num_seqs=int(raw["max_num_seqs"]),
            kv_cache_memory_bytes=int(raw["kv_cache_memory_bytes"]),
            prefix_caching=bool(raw["prefix_caching"]),
            runtimes=runtimes,
            trial_order=tuple(
                tuple(str(label) for label in order) for order in raw["trial_order"]
            ),
            source_path=path.resolve(),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.model != "Qwen/Qwen2.5-1.5B-Instruct":
            raise ValueError("text engine-client benchmark requires Qwen2.5-1.5B")
        if self.concurrency != 1:
            raise ValueError("text engine-client benchmark requires concurrency 1")
        fixed_controls = {
            "request_count": (self.request_count, 1000),
            "warmup_count": (self.warmup_count, 20),
            "target_isl": (self.target_isl, 740),
            "osl": (self.osl, 70),
        }
        changed = [
            f"{name}={actual} (expected {expected})"
            for name, (actual, expected) in fixed_controls.items()
            if actual != expected
        ]
        if changed:
            raise ValueError(
                "text engine-client benchmark fixed controls changed: "
                + ", ".join(changed)
            )
        if self.repeats != 5 or len(self.trial_order) != self.repeats:
            raise ValueError("text engine-client benchmark requires five trial orders")
        if set(self.runtimes) != EXPECTED_RUNTIMES:
            raise ValueError(f"runtimes must be exactly {sorted(EXPECTED_RUNTIMES)}")
        for order in self.trial_order:
            if len(order) != len(EXPECTED_RUNTIMES) or set(order) != EXPECTED_RUNTIMES:
                raise ValueError(
                    "each trial order must contain every runtime exactly once"
                )
        positive_fields: dict[str, Any] = {
            "request_count": self.request_count,
            "warmup_count": self.warmup_count,
            "target_isl": self.target_isl,
            "osl": self.osl,
            "port": self.port,
            "timeout": self.timeout,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "kv_cache_memory_bytes": self.kv_cache_memory_bytes,
        }
        invalid = [name for name, value in positive_fields.items() if int(value) <= 0]
        if invalid:
            raise ValueError(f"benchmark values must be positive: {invalid}")
