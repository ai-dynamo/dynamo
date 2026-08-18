# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for the Power Agent's mutable managed-GPU state.

Why this lives in its own module instead of in ``power_agent.py``:

The daemon entrypoint runs ``power_agent.py`` as the top-level ``__main__``
module — every launch path does this:

  * the image ``ENTRYPOINT ["python", "/app/power_agent.py"]`` (Dockerfile),
  * the Helm DaemonSet ``command: [python, /app/power_agent.py]``
    (templates/daemonset.yaml), and
  * the dev-pod ``exec python3 /scripts/power_agent.py`` (templates/dev-pod.yaml).

Meanwhile ``actuator.py`` reaches back into the agent via ``import
power_agent`` (e.g. ``NvmlActuator.apply_cap`` delegates to
``power_agent._apply_cap`` and ``DcgmActuator._record_managed_state``
records the cap). Because the running module is ``__main__`` but the
actuator imports it under the name ``power_agent``, Python materialises
**two distinct module objects** in ``sys.modules`` — each with its own
module-level globals.

If the managed-GPU sets lived in ``power_agent.py`` they would therefore
exist as two independent copies: the actuator would record freshly-capped
GPUs into the ``power_agent`` copy while shutdown cleanup (running in
``__main__``) restored from the ``__main__`` copy — which would always be
empty. The failure is silent and total: every cap leaks past graceful
shutdown because the restore loop never sees a single managed GPU.

Hosting the state here — imported under its canonical name ``managed_state``
by ``power_agent.py`` — guarantees exactly one copy regardless of how the
agent was launched: the ``__main__`` instance and the canonical
``power_agent`` instance both ``import managed_state``, which resolves to the
*same* cached module object, so their ``_managed_gpu_indices`` /
``_previously_managed`` aliases converge on one set. (``actuator.py`` reaches
that set through ``power_agent`` — e.g. ``power_agent._managed_gpu_indices`` —
rather than importing ``managed_state`` itself.) This module deliberately
imports nothing from ``power_agent`` or ``actuator`` so it can never
participate in an import cycle.

NOTE FOR PACKAGING: any new launch surface must ship this file alongside
``power_agent.py`` / ``actuator.py`` (the image ``COPY`` and the dev-pod
script ConfigMap both include it). A missing ``managed_state.py`` fails
loudly at startup with ``ModuleNotFoundError: No module named
'managed_state'``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("power_agent")

# Absolute path of the persisted managed-GPU state file (UUID-gated orphan
# recovery). Kept here so every module agrees on one location.
MANAGED_STATE_PATH = "/var/lib/dynamo-power-agent/managed_gpus.json"

STATE_VERSION = 2
STATIC_CONTROL_MODE = "static"
TRANSACTIONAL_CONTROL_MODE = "transactional-replica-fence"
_CONTROL_MODES = {STATIC_CONTROL_MODE, TRANSACTIONAL_CONTROL_MODE}
_OWNERSHIP_FIELDS = (
    "controlMode",
    "dgdUID",
    "component",
    "podUID",
    "allocationID",
    "targetWatts",
)


class ManagedStateError(ValueError):
    """Raised when a nonempty durable ownership document is unsafe to use."""


def empty_managed_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "managed": {}}


def _legacy_static_record() -> dict[str, Any]:
    """Conservative v2 representation of one Phase 1 UUID-only record."""
    return {
        "controlMode": STATIC_CONTROL_MODE,
        "dgdUID": "",
        "component": "",
        "podUID": "",
        "allocationID": "",
        "targetWatts": 0,
    }


def static_ownership_record() -> dict[str, Any]:
    """Return a new v2 record for one legacy/static managed GPU."""
    return _legacy_static_record()


def _validate_record(gpu_uuid: str, record: Any) -> dict[str, Any]:
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ManagedStateError("managed GPU UUID keys must be nonempty strings")
    if not isinstance(record, dict):
        raise ManagedStateError(f"ownership record for {gpu_uuid} must be an object")
    missing = [field for field in _OWNERSHIP_FIELDS if field not in record]
    if missing:
        raise ManagedStateError(
            f"ownership record for {gpu_uuid} is missing {', '.join(missing)}"
        )
    control_mode = record["controlMode"]
    if control_mode not in _CONTROL_MODES:
        raise ManagedStateError(
            f"ownership record for {gpu_uuid} has invalid controlMode"
        )
    for field in ("dgdUID", "component", "podUID", "allocationID"):
        if not isinstance(record[field], str):
            raise ManagedStateError(
                f"ownership record for {gpu_uuid} field {field} must be a string"
            )
        if control_mode == TRANSACTIONAL_CONTROL_MODE and not record[field]:
            raise ManagedStateError(
                f"transactional ownership record for {gpu_uuid} field {field} "
                "must be nonempty"
            )
    target_watts = record["targetWatts"]
    if isinstance(target_watts, bool) or not isinstance(target_watts, int):
        raise ManagedStateError(
            f"ownership record for {gpu_uuid} targetWatts must be an integer"
        )
    minimum_target = 1 if control_mode == TRANSACTIONAL_CONTROL_MODE else 0
    if target_watts < minimum_target:
        raise ManagedStateError(
            f"ownership record for {gpu_uuid} targetWatts is out of range"
        )
    return dict(record)


def _validate_v2(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ManagedStateError(
            f"managed state has unexpected root type {type(raw).__name__}"
        )
    if raw.get("version") != STATE_VERSION:
        raise ManagedStateError(
            f"unsupported managed state version {raw.get('version')!r}"
        )
    managed = raw.get("managed")
    if not isinstance(managed, dict):
        raise ManagedStateError("managed state field 'managed' must be an object")
    return {
        "version": STATE_VERSION,
        "managed": {
            gpu_uuid: _validate_record(gpu_uuid, record)
            for gpu_uuid, record in managed.items()
        },
    }


def _migrate_v1(raw: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {"version", "managed_uuids"}
    unexpected_fields = set(raw) - allowed_fields
    if "managed_uuids" not in raw or unexpected_fields:
        details = []
        if "managed_uuids" not in raw:
            details.append("missing required field 'managed_uuids'")
        if unexpected_fields:
            details.append(f"unexpected fields {sorted(unexpected_fields)!r}")
        raise ManagedStateError("invalid legacy managed state: " + "; ".join(details))
    managed_uuids = raw["managed_uuids"]
    if not isinstance(managed_uuids, list):
        raise ManagedStateError(
            "legacy managed_uuids type "
            f"{type(managed_uuids).__name__} (expected list)"
        )
    invalid_count = sum(1 for value in managed_uuids if not isinstance(value, str))
    if invalid_count:
        logger.warning(
            "Managed-GPU state contained %d non-string entries; dropping them.",
            invalid_count,
        )
    managed = {
        gpu_uuid: _legacy_static_record()
        for gpu_uuid in managed_uuids
        if isinstance(gpu_uuid, str) and gpu_uuid
    }
    return {"version": STATE_VERSION, "managed": managed}


def load_managed_state(
    path: str | os.PathLike[str] = MANAGED_STATE_PATH,
    *,
    control_mode: str = STATIC_CONTROL_MODE,
) -> dict[str, Any]:
    """Load v2 ownership state, migrating the Phase 1 UUID-only form.

    A missing or empty file is a valid first boot. Every nonempty malformed
    document raises ``ManagedStateError``; transactional callers therefore fail
    closed, while the static compatibility wrapper can retain its existing
    inconclusive-read behavior by catching the exception and skipping recovery.
    """
    if control_mode not in _CONTROL_MODES:
        raise ValueError(f"unknown control mode {control_mode!r}")
    state_path = Path(path)
    try:
        with open(state_path, encoding="utf-8") as input_file:
            encoded = input_file.read()
    except FileNotFoundError:
        return empty_managed_state()
    if not encoded.strip():
        return empty_managed_state()
    try:
        raw = json.loads(encoded)
    except json.JSONDecodeError as e:
        raise ManagedStateError(f"managed state JSONDecodeError: {e}") from e
    if isinstance(raw, dict) and ("version" not in raw or raw.get("version") == 1):
        return _migrate_v1(raw)
    return _validate_v2(raw)


def save_managed_state(
    state: dict[str, Any],
    path: str | os.PathLike[str] = MANAGED_STATE_PATH,
) -> None:
    """Validate and atomically replace the durable v2 state document."""
    validated = _validate_v2(state)
    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = state_path.with_name(f"{state_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(validated, output, sort_keys=True, separators=(",", ":"))
    os.replace(temporary_path, state_path)


def authorize_enrollment(state: dict[str, Any], gpu_uuid: str, dgd_uid: str) -> bool:
    """Return whether ``dgd_uid`` may create or refresh this GPU record."""
    validated = _validate_v2(state)
    existing = validated["managed"].get(gpu_uuid)
    if existing is None:
        return True
    return (
        existing["controlMode"] == TRANSACTIONAL_CONTROL_MODE
        and existing["dgdUID"] == dgd_uid
    )


def enroll_managed_gpu(
    gpu_uuid: str,
    ownership: dict[str, Any],
    path: str | os.PathLike[str] = MANAGED_STATE_PATH,
) -> dict[str, Any]:
    """Atomically enroll one UUID without permitting cross-DGD overwrite."""
    record = _validate_record(gpu_uuid, ownership)
    state = load_managed_state(path, control_mode=record["controlMode"])
    if record["controlMode"] == TRANSACTIONAL_CONTROL_MODE and not authorize_enrollment(
        state, gpu_uuid, record["dgdUID"]
    ):
        raise ManagedStateError(
            f"GPU {gpu_uuid} is already enrolled by another DGD UID"
        )
    if state["managed"].get(gpu_uuid) == record:
        return state
    state["managed"][gpu_uuid] = record
    save_managed_state(state, path)
    return state


# In-process set of physical GPU indices this running agent has capped.
# Populated by every successful cap write (NVML via ``power_agent._apply_cap``,
# DCGM via ``DcgmActuator._record_managed_state``) and classified by shutdown
# cleanup (``power_agent._shutdown_cleanup``, invoked from the reconcile loop
# after SIGTERM). Static ownership restores default TGP; transactional
# ownership keeps the cap live for identity-safe adoption by the replacement.
managed_gpu_indices: set[int] = set()

# Identity paired with each process-local managed index at the successful cap
# write. Shared across the entrypoint/canonical module copies just like the
# index set, so a UUID-addressed release can retire an NVML index even when an
# independent discovery snapshot transiently omits that GPU.
managed_gpu_uuid_by_index: dict[int, str] = {}

# Persisted across restarts (``MANAGED_STATE_PATH``): the UUIDs this agent
# currently OWNS a below-default cap on — added on a successful cap write and
# PRUNED again once that cap is released/restored to default (runtime release,
# static SIGTERM cleanup, or cold-start orphan recovery). Active transactional
# ownership survives SIGTERM. It is NOT an append-only "ever capped" ledger
# (that is ``DcgmActuator._capped_uuids``); it is the live,
# cross-incarnation ownership set used for UUID-gated cold-start orphan recovery
# so a restart only touches GPUs it still owns. Always mutated in place — never
# rebind this name, or the alias that ``power_agent.py`` and ``actuator.py``
# hold would split and re-introduce the dual-copy bug described above.
previously_managed: set[str] = set()

# UUIDs whose cap acquisition completed (hardware cap is live and in-memory
# ownership was recorded) but whose durable ADD to MANAGED_STATE_PATH failed.
# This must be shared for the same reason as ``previously_managed``: cap writes
# run through the actuator's canonical ``import power_agent`` module, while the
# reconcile loop that flushes the retry queue runs in the entrypoint module.
pending_acquisition: set[str] = set()

# Transactional acquisitions must retain their full ownership record across the
# ``power_agent``/``__main__`` module boundary. A UUID-only retry would degrade
# the record to static and make graceful Agent replacement restore a cap that
# must remain live.
pending_transactional_acquisition: dict[str, dict[str, Any]] = {}
