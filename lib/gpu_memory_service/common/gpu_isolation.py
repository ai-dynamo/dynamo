# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One switch for how a failover isolates the predecessor's GPU work.

``DYN_GMS_GPU_ISOLATION`` selects a coherent set of defaults:

``mps``
    Engines run as MPS clients. GMS terminates a dead client's MPS context
    (``gms-mps`` quiescence provider), the crash interlock is on, and frozen
    predecessor pages are reclaimed only with that proof (``gpu-proof``).

``process``
    Engines are plain CUDA processes, without MPS. The driver tears down a
    process's CUDA context when it exits, so the predecessor is fenced by its
    writer-cohort lifetime guards (``process-lifetime`` provider), the crash
    interlock is off, and reclaim uses ``process-death-timeout``.

Unset keeps the individual settings and their historical defaults. Every
individual variable still overrides the mode.
"""

from __future__ import annotations

import os

ISOLATION_ENV = "DYN_GMS_GPU_ISOLATION"
_MODES = ("mps", "process")

_QUIESCENCE_PROVIDER = {"mps": "gms-mps", "process": "process-lifetime"}
_CRASH_INTERLOCK = {"mps": "1", "process": "0"}
_RECLAIM_POLICY = {"mps": "gpu-proof", "process": "process-death-timeout"}


def gpu_isolation_mode() -> str | None:
    """Return ``mps``, ``process``, or None when the switch is unset."""
    raw = os.environ.get(ISOLATION_ENV, "").strip().lower()
    if not raw:
        return None
    if raw not in _MODES:
        raise ValueError(f"{ISOLATION_ENV}={raw!r} must be one of {_MODES}")
    return raw


def default_quiescence_provider() -> str | None:
    mode = gpu_isolation_mode()
    return None if mode is None else _QUIESCENCE_PROVIDER[mode]


def default_crash_interlock() -> str:
    mode = gpu_isolation_mode()
    return "0" if mode is None else _CRASH_INTERLOCK[mode]


def default_reclaim_policy() -> str:
    mode = gpu_isolation_mode()
    return "gpu-proof" if mode is None else _RECLAIM_POLICY[mode]
