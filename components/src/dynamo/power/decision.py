# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EPD's decision file: where it lives, what it contains, how to expand it.

EPD's ``file`` actuator (``epd.actuation.handoff.file``) publishes every
decision atomically (tmp + ``os.replace``) to
``{LOG_DIR}/epd/actuation/power_decision.json`` with the schema::

    {
      "version": 1,
      "sequence": 7,                      # 1-based publisher counter
      "timestamp": 1760000000.123,        # time.time() on the controller
      "per_node_gpu_power_limit_w": {"node01": 1350, "node02": 1420}
    }

The watts are an absolute per-GPU cap applied to every GPU of that node (in
this package: every GPU of that node's GPU group). The applier reads it;
nothing in this module imports EPD.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

DECISION_FILE_RELATIVE_PATH = os.path.join("epd", "actuation", "power_decision.json")
DECISION_SCHEMA_VERSION = 1


class DecisionFormatError(ValueError):
    """The decision file does not follow EPD's schema."""


@dataclass(frozen=True)
class PowerDecisionFile:
    """One parsed ``power_decision.json``."""

    version: int
    sequence: int
    timestamp: float
    per_node_gpu_power_limit_w: Mapping[str, int]

    def nodes(self) -> tuple[str, ...]:
        return tuple(self.per_node_gpu_power_limit_w)


def decision_file_path(log_dir: str) -> str:
    """The decision file EPD writes for the controller whose ``LOG_DIR`` is ``log_dir``.

    Mirrors ``epd.actuation.handoff.file.decision_path``: EPD roots all of its
    output at ``{abspath(LOG_DIR)}/epd``. The launcher passes this path to the
    applier as ``--decision-file``.
    """
    return os.path.join(os.path.abspath(log_dir), DECISION_FILE_RELATIVE_PATH)


def parse_decision(payload: object) -> PowerDecisionFile:
    """Validate a decoded JSON document against EPD's schema."""
    if not isinstance(payload, Mapping):
        raise DecisionFormatError("decision must be a JSON object")
    try:
        version = int(payload["version"])
        sequence = int(payload["sequence"])
        timestamp = float(payload["timestamp"])
        caps_raw = payload["per_node_gpu_power_limit_w"]
    except KeyError as exc:
        raise DecisionFormatError(f"decision is missing key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise DecisionFormatError(f"decision has a malformed field: {exc}") from exc
    if version != DECISION_SCHEMA_VERSION:
        raise DecisionFormatError(
            f"decision schema version {version} is not the supported "
            f"{DECISION_SCHEMA_VERSION}"
        )
    if not isinstance(caps_raw, Mapping):
        raise DecisionFormatError("per_node_gpu_power_limit_w must be an object")
    caps = {}
    for node, watts in caps_raw.items():
        if isinstance(watts, bool) or not isinstance(watts, (int, float)):
            raise DecisionFormatError(
                f"cap for node {node!r} must be a number of watts, got {watts!r}"
            )
        if watts <= 0:
            raise DecisionFormatError(
                f"cap for node {node!r} must be > 0 W, got {watts}"
            )
        caps[str(node)] = int(watts)
    return PowerDecisionFile(
        version=version,
        sequence=sequence,
        timestamp=timestamp,
        per_node_gpu_power_limit_w=caps,
    )


def read_decision_file(path: str) -> PowerDecisionFile:
    """Read and validate ``path``.

    Raises :class:`FileNotFoundError` when the file is absent,
    :class:`json.JSONDecodeError` when it is not JSON and
    :class:`DecisionFormatError` when it is JSON of the wrong shape.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return parse_decision(payload)


def node_watts(decision: PowerDecisionFile, node: str) -> int | None:
    """This node's per-GPU cap in watts, or ``None`` when the decision has no entry."""
    return decision.per_node_gpu_power_limit_w.get(node)


def per_gpu_watts(watts: int, gpu_indices: Sequence[int]) -> dict[int, int]:
    """Expand one node cap to the GPUs of its group (EPD caps every GPU alike)."""
    return {int(index): int(watts) for index in gpu_indices}


__all__ = [
    "DECISION_FILE_RELATIVE_PATH",
    "DECISION_SCHEMA_VERSION",
    "DecisionFormatError",
    "PowerDecisionFile",
    "decision_file_path",
    "node_watts",
    "parse_decision",
    "per_gpu_watts",
    "read_decision_file",
]
