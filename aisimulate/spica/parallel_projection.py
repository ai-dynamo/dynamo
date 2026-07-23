# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured features and deterministic projection for parallel configs.

Vizier searches a compact, regular feature space.  The existing parallel
enumerator remains the source of truth: every suggestion is projected onto one
of the branch's backend-compatible, KV-feasible configs before replay.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from .search_space import BranchSpace

ParallelConfig = ReplicaParallelConfig | DisaggParallelConfig

USED_GPU_RATIO = "used_gpu_ratio"
PREFILL_GPU_SHARE = "prefill_gpu_share"
AGG_GPUS_PER_ENGINE = "agg_num_gpus_per_engine_target"
PREFILL_GPUS_PER_ENGINE = "prefill_num_gpus_per_engine_target"
DECODE_GPUS_PER_ENGINE = "decode_num_gpus_per_engine_target"
AGG_ATTENTION_MODE = "agg_attention_mode"
PREFILL_ATTENTION_MODE = "prefill_attention_mode"
DECODE_ATTENTION_MODE = "decode_attention_mode"
AGG_FFN_MODE = "agg_ffn_mode"
PREFILL_FFN_MODE = "prefill_ffn_mode"
DECODE_FFN_MODE = "decode_ffn_mode"

_ATTENTION_MODE_ORDER = ("tp", "dp")
_FFN_MODE_ORDER = ("ep", "tp")


@dataclass(frozen=True)
class ParallelParameter:
    """One Vizier-facing latent dimension."""

    name: str
    kind: Literal["float", "discrete", "categorical"]
    default: float | str
    values: tuple[float | str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    log_scale: bool = False

    @property
    def is_constant(self) -> bool:
        if self.kind == "float":
            return self.minimum == self.maximum
        return len(self.values) == 1


@dataclass(frozen=True)
class ParallelProjection:
    """A latent request and the valid config selected for replay."""

    config: ParallelConfig
    requested_features: dict[str, float | str]
    actual_features: dict[str, float | str]
    distance: float
    mode_projected: bool

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_features": self.requested_features,
            "actual_features": self.actual_features,
            "projection_distance": self.distance,
            "mode_projected": self.mode_projected,
            "actual_parallel_config": asdict(self.config),
        }


def _all_shapes(config: ParallelConfig) -> tuple[ParallelShape, ...]:
    if isinstance(config, ReplicaParallelConfig):
        return (config.shape,)
    return (config.prefill.shape, config.decode.shape)


def _attention_mode(shape: ParallelShape) -> str:
    # The enumerator emits pure attention TP or DP.  G=1 is canonicalized as TP.
    return "dp" if shape.dp > 1 else "tp"


def _ffn_mode(shape: ParallelShape) -> str:
    # The enumerator emits pure MoE TP or EP.  G=1 is canonicalized as TP.
    return "ep" if shape.moe_ep > 1 else "tp"


def _config_key(config: ParallelConfig) -> tuple[int, ...]:
    def role_key(role: ReplicaParallelConfig) -> tuple[int, ...]:
        shape = role.shape
        return (shape.tp, shape.pp, shape.dp, shape.moe_tp, shape.moe_ep, role.replicas)

    if isinstance(config, ReplicaParallelConfig):
        return role_key(config)
    return (*role_key(config.prefill), *role_key(config.decode))


def _ordered_values(values: set[str], preferred: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value for value in preferred if value in values)


def _geometric_default(values: tuple[float, ...]) -> float:
    target = math.sqrt(min(values) * max(values))
    return min(
        values, key=lambda value: (abs(math.log(value) - math.log(target)), value)
    )


class ParallelConfigProjector:
    """Encode and project one deployment-mode branch's parallel config pool."""

    def __init__(self, branch: BranchSpace):
        if not branch.parallel_configs:
            raise ValueError("parallel projection requires at least one valid config")

        self.branch = branch
        self.gpu_budget = branch.gpu_budget or max(
            config.total_gpus for config in branch.parallel_configs
        )
        self.is_moe = any(
            shape.moe_tp > 1 or shape.moe_ep > 1
            for config in branch.parallel_configs
            for shape in _all_shapes(config)
        )
        self._features = {
            config: self._encode(config) for config in branch.parallel_configs
        }
        self.parameters = self._build_parameters()
        self.constants = {
            parameter.name: parameter.default
            for parameter in self.parameters
            if parameter.is_constant
        }

    def _role_features(
        self, prefix: str, role: ReplicaParallelConfig
    ) -> dict[str, float | str]:
        features: dict[str, float | str] = {
            f"{prefix}_num_gpus_per_engine_target": float(role.shape.gpus_per_worker),
            f"{prefix}_attention_mode": _attention_mode(role.shape),
        }
        if self.is_moe:
            features[f"{prefix}_ffn_mode"] = _ffn_mode(role.shape)
        return features

    def _encode(self, config: ParallelConfig) -> dict[str, float | str]:
        features: dict[str, float | str] = {
            USED_GPU_RATIO: config.total_gpus / self.gpu_budget
        }
        if isinstance(config, ReplicaParallelConfig):
            features.update(self._role_features("agg", config))
            return features

        features[PREFILL_GPU_SHARE] = config.prefill.total_gpus / config.total_gpus
        features.update(self._role_features("prefill", config.prefill))
        features.update(self._role_features("decode", config.decode))
        return features

    def _float_parameter(self, name: str, *, default: float) -> ParallelParameter:
        values = [float(features[name]) for features in self._features.values()]
        minimum, maximum = min(values), max(values)
        return ParallelParameter(
            name=name,
            kind="float",
            minimum=minimum,
            maximum=maximum,
            default=min(max(default, minimum), maximum),
        )

    def _discrete_parameter(self, name: str) -> ParallelParameter:
        values = tuple(
            sorted({float(features[name]) for features in self._features.values()})
        )
        return ParallelParameter(
            name=name,
            kind="discrete",
            values=values,
            default=_geometric_default(values),
            log_scale=True,
        )

    def _categorical_parameter(
        self, name: str, preferred: tuple[str, ...]
    ) -> ParallelParameter:
        present = {str(features[name]) for features in self._features.values()}
        values = _ordered_values(present, preferred)
        return ParallelParameter(
            name=name, kind="categorical", values=values, default=values[0]
        )

    def _build_parameters(self) -> tuple[ParallelParameter, ...]:
        parameters = [self._float_parameter(USED_GPU_RATIO, default=1.0)]
        if self.branch.deployment_mode == "agg":
            parameters.extend(
                [
                    self._discrete_parameter(AGG_GPUS_PER_ENGINE),
                    self._categorical_parameter(
                        AGG_ATTENTION_MODE, _ATTENTION_MODE_ORDER
                    ),
                ]
            )
            if self.is_moe:
                parameters.append(
                    self._categorical_parameter(AGG_FFN_MODE, _FFN_MODE_ORDER)
                )
            return tuple(parameters)

        parameters.extend(
            [
                self._float_parameter(PREFILL_GPU_SHARE, default=0.5),
                self._discrete_parameter(PREFILL_GPUS_PER_ENGINE),
                self._discrete_parameter(DECODE_GPUS_PER_ENGINE),
                self._categorical_parameter(
                    PREFILL_ATTENTION_MODE, _ATTENTION_MODE_ORDER
                ),
                self._categorical_parameter(
                    DECODE_ATTENTION_MODE, _ATTENTION_MODE_ORDER
                ),
            ]
        )
        if self.is_moe:
            parameters.extend(
                [
                    self._categorical_parameter(PREFILL_FFN_MODE, _FFN_MODE_ORDER),
                    self._categorical_parameter(DECODE_FFN_MODE, _FFN_MODE_ORDER),
                ]
            )
        return tuple(parameters)

    def requested_features(self, params: dict[str, Any]) -> dict[str, float | str]:
        requested: dict[str, float | str] = {}
        for parameter in self.parameters:
            value = params.get(parameter.name, parameter.default)
            requested[parameter.name] = (
                str(value) if parameter.kind == "categorical" else float(value)
            )
        return requested

    def project(self, params: dict[str, Any], backend: str) -> ParallelProjection:
        requested = self.requested_features(params)
        candidates = [
            config
            for config in self.branch.parallel_configs
            if backend in self.branch.supported_backends.get(config, frozenset())
        ]
        if not candidates:
            raise ValueError(
                f"backend {backend!r} has no valid parallel config in this branch"
            )

        categorical_names = [
            parameter.name
            for parameter in self.parameters
            if parameter.kind == "categorical"
        ]

        def mismatches(config: ParallelConfig) -> int:
            actual = self._features[config]
            return sum(actual[name] != requested[name] for name in categorical_names)

        min_mismatches = min(mismatches(config) for config in candidates)
        candidates = [
            config for config in candidates if mismatches(config) == min_mismatches
        ]

        numeric_parameters = [
            parameter
            for parameter in self.parameters
            if parameter.kind != "categorical"
        ]
        backend_features = [
            self._features[config]
            for config in self.branch.parallel_configs
            if backend in self.branch.supported_backends.get(config, frozenset())
        ]

        def transformed(name: str, value: float) -> float:
            return (
                math.log2(value)
                if name.endswith("num_gpus_per_engine_target")
                else value
            )

        ranges: dict[str, tuple[float, float]] = {}
        for parameter in numeric_parameters:
            values = [
                transformed(parameter.name, float(features[parameter.name]))
                for features in backend_features
            ]
            ranges[parameter.name] = (min(values), max(values))

        def numeric_distance(config: ParallelConfig) -> float:
            actual = self._features[config]
            distance = 0.0
            for parameter in numeric_parameters:
                name = parameter.name
                lower, upper = ranges[name]
                if upper == lower:
                    continue
                delta = (
                    transformed(name, float(actual[name]))
                    - transformed(name, float(requested[name]))
                ) / (upper - lower)
                distance += delta * delta
            return distance

        selected = min(
            candidates,
            key=lambda config: (numeric_distance(config), _config_key(config)),
        )
        return ParallelProjection(
            config=selected,
            requested_features=requested,
            actual_features=dict(self._features[selected]),
            distance=numeric_distance(selected),
            mode_projected=min_mismatches > 0,
        )
