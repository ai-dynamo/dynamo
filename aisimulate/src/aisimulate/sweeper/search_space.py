# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the per-branch candidate space the sampler searches over.

A *branch* is one **deployment_mode** (agg / disagg) — one Vizier study each, since
agg and disagg have structurally different parallel configs. ``backend`` is NOT a
branch: it is a searched categorical knob within the study. For each mode we take the
**union** of every configured backend's KV-feasible parallel configs
(:func:`aisimulate.sweeper.model_hw.parallel_configs_for`) as the valid projection pool, recording per
config which backends support it. The sampler projects structured latent features onto this
pool. Backends with no perf DB, no viable config, or no support from the injected replay
runner are dropped from the backend knob.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from .config import SmartSearchConfig
from .kv_estimate import NoPerfDatabase
from .model_hw import NoViableParallelConfig, parallel_configs_for
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from .replay import RunnerCapabilities

_ParallelConfig = ReplicaParallelConfig | DisaggParallelConfig

_AGG_ENGINE = ("agg_max_num_batched_tokens", "agg_max_num_seqs")
_DISAGG_ENGINE = (
    "prefill_max_num_batched_tokens",
    "prefill_max_num_seqs",
    "decode_max_num_batched_tokens",
    "decode_max_num_seqs",
)


@dataclass(frozen=True)
class BranchSpace:
    """One ``deployment_mode`` branch of the search (backend is a searched knob)."""

    deployment_mode: str
    # Union of every searched backend's KV-feasible parallel configs.
    parallel_configs: tuple[_ParallelConfig, ...]
    # parallel config -> the backends for which it is legal+KV-feasible. Projection
    # hard-filters on this map; the search loop keeps a defensive gate.
    supported_backends: dict[_ParallelConfig, frozenset[str]]
    # Searchable atomic knob -> its configured choice list (incl. "backend").
    knob_choices: dict[str, list[Any]]
    # Pinned branch budget; optional only for lightweight unit-test fixtures.
    gpu_budget: int | None = None
    # Continuous workload dimensions. Currently only Pareto ``kv_load_ratio`` uses
    # this; list-valued component knobs remain discrete choices above.
    float_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


def _engine_knobs(deployment_mode: str) -> tuple[str, ...]:
    return _AGG_ENGINE if deployment_mode == "agg" else _DISAGG_ENGINE


def _shape_from_dict(d: dict[str, Any]) -> ParallelShape:
    """A per-worker :class:`ParallelShape` from a pinned shape dict. Omitted dims
    default to 1 (so dense models can write just ``{tp: N}``); ``pp`` defaults to 1."""
    if "tp" not in d:
        raise ValueError(f"a parallel_configs shape needs a 'tp' field, got {d}")
    return ParallelShape(
        tp=int(d["tp"]),
        dp=int(d.get("attention_dp", 1)),
        moe_tp=int(d.get("moe_tp", 1)),
        moe_ep=int(d.get("moe_ep", 1)),
        pp=int(d.get("pp", 1)),
    )


def _replica_from_dict(d: dict[str, Any]) -> ReplicaParallelConfig:
    return ReplicaParallelConfig(
        shape=_shape_from_dict(d), replicas=int(d.get("replicas", 1))
    )


def _parse_parallel_entry(entry: dict[str, Any], deployment_mode: str):
    """Parse one pinned ``parallel_configs`` entry into the config object: a flat
    shape dict for agg, or a ``{prefill, decode}`` pair for disagg."""
    if deployment_mode == "agg":
        return _replica_from_dict(entry)
    return DisaggParallelConfig(
        prefill=_replica_from_dict(entry["prefill"]),
        decode=_replica_from_dict(entry["decode"]),
    )


def branch_knob_choices(search_space, deployment_mode: str) -> dict[str, list[Any]]:
    """Backend-owned atomic knobs for one deployment branch."""
    names = _engine_knobs(deployment_mode)
    return {name: list(getattr(search_space, name)) for name in names}


def enumerate_branches(
    config: SmartSearchConfig,
    *,
    max_seq_len: int | None = None,
    runner_capabilities: RunnerCapabilities | None = None,
) -> list[BranchSpace]:
    """One :class:`BranchSpace` per ``deployment_mode``. Within each, ``backend`` is a
    searched knob: the parallel-config domain is the **union** of every configured
    backend's KV-feasible configs, tagged with which backends support each.

    A backend with no perf DB / no viable config for a mode is dropped (skipped). A mode
    for which *no* backend is viable is skipped with a warning (so a viable mode still
    runs); only if **no** mode is viable does it raise :class:`NoViableParallelConfig`. A
    *pinned* config that is legal for no backend is a hard error (fail fast — the pin is
    wrong). ``max_seq_len`` is forwarded to :func:`parallel_configs_for` (``None`` -> the
    model's max context length).
    """
    ss = config.search_space
    branches: list[BranchSpace] = []
    skipped: list[str] = []  # modes dropped because no backend was viable
    # Dedupe modes (preserving order): a repeated deployment_mode would yield duplicate
    # branches and hence colliding Vizier study_ids (one study per mode).
    for deployment_mode in dict.fromkeys(ss.deployment_mode):
        # Pinned configs (if any) are parsed once, then validated per backend; otherwise
        # each backend contributes its full enumerated menu.
        pinned = (
            [_parse_parallel_entry(e, deployment_mode) for e in ss.parallel_configs]
            if ss.parallel_configs
            else None
        )
        support: dict[_ParallelConfig, set[str]] = {}
        runner_incompatible = [
            backend
            for backend in ss.backend
            if runner_capabilities is not None
            and not runner_capabilities.supports_backend_topology(
                backend, deployment_mode
            )
        ]
        for backend in ss.backend:
            if (
                runner_capabilities is not None
                and not runner_capabilities.supports_backend_topology(
                    backend, deployment_mode
                )
            ):
                continue
            try:
                legal = parallel_configs_for(
                    ss.model_name,
                    ss.hardware_sku,
                    gpu_budget=ss.gpu_budget,
                    deployment_mode=deployment_mode,
                    backend=backend,
                    min_gpu_budget=ss.min_gpu_budget,
                    max_seq_len=max_seq_len,
                )
            except (NoPerfDatabase, NoViableParallelConfig):
                continue  # backend unusable for this mode -> drop it from the search
            legal_set = set(legal)
            for cfg in pinned if pinned is not None else legal:
                if cfg in legal_set:
                    support.setdefault(cfg, set()).add(backend)

        if not support:
            if pinned is not None:
                # an explicit pin that no backend can run is a user error -> fail fast
                raise NoViableParallelConfig(
                    f"deployment_mode={deployment_mode!r}: no configured backend can run the pinned "
                    f"parallel_configs (illegal shape, replay-incompatible backend, or no perf DB)"
                )
            # natural infeasibility for this mode -> skip it, keep any viable modes
            warnings.warn(
                f"smart-sweep: deployment_mode={deployment_mode!r} skipped — no configured backend "
                f"has a viable parallel config within gpu_budget={ss.gpu_budget}"
                + (
                    f"; runner-incompatible backends={runner_incompatible}"
                    if runner_incompatible
                    else ""
                ),
                stacklevel=2,
            )
            skipped.append(deployment_mode)
            continue
        if pinned is not None:
            illegal = [c for c in pinned if c not in support]
            if illegal:
                raise NoViableParallelConfig(
                    f"pinned parallel_configs are legal/KV-feasible for no configured backend: {illegal}"
                )

        knob_choices = branch_knob_choices(ss, deployment_mode)
        viable_backends = set().union(*support.values())
        knob_choices["backend"] = [
            backend
            for backend in dict.fromkeys(ss.backend)
            if backend in viable_backends
        ]
        float_ranges: dict[str, tuple[float, float]] = {}
        kv_load_range = config.workload.kv_load_ratio_range
        if kv_load_range is not None:
            float_ranges["kv_load_ratio"] = kv_load_range
        elif config.workload.kv_load_ratio is not None:
            # A scalar KV load is pinned for both scalar and Pareto goals. Keep it in
            # the constant path so every decoded selection carries the requested ratio.
            knob_choices["kv_load_ratio"] = [float(config.workload.kv_load_ratio)]
        branches.append(
            BranchSpace(
                deployment_mode=deployment_mode,
                parallel_configs=tuple(support),
                supported_backends={cfg: frozenset(bs) for cfg, bs in support.items()},
                knob_choices=knob_choices,
                gpu_budget=ss.gpu_budget,
                float_ranges=float_ranges,
            )
        )

    if not branches:
        raise NoViableParallelConfig(
            f"no deployment_mode has a viable parallel config (skipped {skipped}); check "
            f"backends / model / hardware / gpu_budget={ss.gpu_budget}"
        )
    return branches
