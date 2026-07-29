# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actuator abstraction for the Power Agent.

Defines the `Actuator` Protocol and two implementations:

  - `NvmlActuator` — the path that shipped in PR #9682. Used on
    clusters where the GPU Operator's `dcgm.enabled=false` (default).
    Delegates to the existing module-level NVML helpers in
    `power_agent.py` so the existing NvmlActuator tests pass unchanged.
  - `DcgmActuator` — used on clusters where the GPU Operator's
    `dcgm.enabled=true`. Connects standalone-TCP to the
    `nvidia-dcgm` hostengine and routes per-GPU `dcgmConfigSet`
    writes through it. PID
    enumeration intentionally stays on NVML because DCGM does not
    expose a snapshot-of-running-PIDs API
    (`DCGM_FI_DEV_COMPUTE_PIDS` is a time-series field).

The two are mutually exclusive at chart-install time — a Power
Agent process binds to exactly one actuator at startup and holds
it for its lifetime.

Lazy imports
------------
Every method that touches `pynvml`, `pydcgm`, `dcgm_structs`, or
`dcgmvalue` imports it inside the method body rather than at module
top. Three reasons:

  1. The NVML-only deployment (the chart default) must not pay the
     `libdcgm.so` dlopen cost (~3 s on slow base images).
  2. Tests patch `power_agent.pynvml`; the `NvmlActuator` methods
     access pynvml via that re-exported reference so the patches
     stay effective without test changes.
  3. Avoids a circular import (`power_agent` imports `actuator`,
     `NvmlActuator` calls into `power_agent`).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Optional, Protocol, TypeVar, runtime_checkable

logger = logging.getLogger("power_agent.actuator")

T = TypeVar("T")

# Smallest numeric DCGM blank sentinel. All numeric DCGM blank / not-found /
# not-supported sentinels are far above any real GPU power limit, so reject the
# whole range instead of depending on `dcgmvalue` being importable on this path.
_DCGM_NUMERIC_BLANK_MIN = float(0x7FFFFFF0)


class _GpuIdentityMismatch(Exception):
    """Raised inside the DCGM cap-write path when the physical GPU at the
    target index no longer matches the UUID captured at `apply_cap` entry.

    A DCGM hostengine reconnect inside `apply_cap` (the `_with_reconnect`
    retry) can re-enumerate GPU indices, so the index the cap was derived
    for may now host a different physical GPU. Writing then would apply a
    cap computed from GPU-A's workload onto GPU-B. This is deliberately NOT
    a `DCGMError`, so `_with_reconnect` does not try to recover from it; it
    propagates to `apply_cap`, which skips the write (no clobber), records
    `apply_failures_total`, and lets the next reconcile cycle re-attribute
    and retry against the fresh enumeration.
    """


@runtime_checkable
class Actuator(Protocol):
    """Protocol implemented by every power-cap actuator.

    Two implementations exist:
      - NvmlActuator (default; PR #9682 behaviour unchanged)
      - DcgmActuator (opt-in via --actuator=dcgm)

    The two are mutually exclusive at chart-install time — a given
    Power Agent process binds to exactly one actuator at startup and
    holds it for its lifetime.
    """

    name: str  # "nvml" | "dcgm"

    def init(self) -> None:
        """Initialize the underlying library. Called once at agent startup."""
        ...

    def shutdown(self) -> None:
        """Release library resources. Called on agent shutdown."""
        ...

    def device_count(self) -> int:
        """Return the number of GPUs visible to this actuator."""
        ...

    def get_uuid(self, gpu_idx: int) -> str:
        """Return the GPU UUID string for the given index."""
        ...

    def list_running_pids(
        self, gpu_idx: int, expected_uuid: Optional[str] = None
    ) -> list[int]:
        """Snapshot the compute PIDs currently running on the GPU.

        `expected_uuid` binds the PID snapshot to the physical GPU the caller
        anchored its policy decision to. When supplied, the implementation MUST
        attribute PIDs to that UUID — not to whatever GPU a re-enumeration may
        have parked on `gpu_idx` — and raise `_GpuIdentityMismatch` if that GPU
        is no longer resolvable. Without this, a DCGM A->B->A re-enumeration
        could feed GPU-B's workload into a cap that `apply_cap` (which only
        guards the write DESTINATION) then writes onto GPU-A. The NVML path binds by index for the process lifetime and
        does not re-enumerate, so it accepts the argument for Protocol
        uniformity without needing it (mirrors `apply_cap`).
        """
        ...

    def constraints_w(self, gpu_idx: int) -> tuple[int, int]:
        """Return the (min_w, max_w) SKU power-cap settable range."""
        ...

    def current_w(self, gpu_idx: int) -> int:
        """Return the power cap currently applied to the GPU (watts).

        Distinct from `default_w` and from `constraints_w` — this is
        whatever value the driver has live on the GPU right now, set
        by whichever process last wrote it (us, nvidia-smi, vendor
        firmware default). Used by the orphan-recovery guard to skip
        no-op writes when the cap is already at default.
        """
        ...

    def default_w(self, gpu_idx: int) -> int:
        """Return the factory-default TGP for the GPU (watts).

        NVML side: `nvmlDeviceGetPowerManagementDefaultLimit`. DCGM
        side: `DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF` (field 163, distinct
        from `_MAX` field 162; see DcgmActuator.restore_default).
        """
        ...

    def apply_cap(
        self, gpu_idx: int, watts: int, expected_uuid: Optional[str] = None
    ) -> int:
        """Write a per-GPU power cap. Returns the effective post-clamp value.

        `expected_uuid` is the GPU identity the caller anchored the policy
        decision to (the reconcile loop captures it BEFORE the PID snapshot
        that produced `watts`, so the whole attribution → cap → write sequence
        refers to one physical GPU). When supplied, the
        implementation must verify the index still hosts that UUID at write
        time and refuse (counting an apply failure) on a mismatch, rather than
        applying one GPU's workload-derived cap to whatever GPU a DCGM
        re-enumeration moved onto the index. When omitted (None), the
        implementation captures its own entry-time identity as a best-effort
        anchor. The NVML path binds by index for the process lifetime and does
        not re-enumerate, so it accepts the argument for Protocol uniformity
        without needing it.

        Return-value contract:

            The returned int is the **effective post-clamp value** —
            `max(min_w, min(watts, max_w))` against the SKU constraints —
            regardless of whether the underlying write to NVML / DCGM
            succeeded. This matches both implementations' actual behaviour
            (NvmlActuator returns effective_w even on NVMLError; DcgmActuator
            returns effective_w even on DCGMError). The reason: callers
            (`PowerAgent._reconcile_gpu`, Prometheus exporters, log
            aggregators) need a non-Optional value to record what the agent
            *intended* to apply; success/failure is reported separately via
            `metrics.apply_failures_total`. Earlier doc wording said "actually
            applied" which suggested a contract this method does not hold —
            corrected here.

        A failed write does NOT raise from `apply_cap` itself; the actuator
        logs the failure and increments `apply_failures_total`. Callers that
        need to detect failure should observe the metric, not the return
        value.
        """
        ...

    def restore_default(self, gpu_idx: int) -> Optional[bool]:
        """Restore the factory-default TGP for the GPU.

        Return False when the restore was intentionally SKIPPED and the cap may
        still be live, so the caller must retain ownership. On the DCGM path
        that covers three cases: the managed physical GPU is no longer locatable
        (re-enumerated away and not found), the resolved index's identity could
        not be confirmed at write time (a proven mid-write re-enumeration OR an
        unverifiable identity read — the guarded write fails closed), or the
        identity-bound power read found the index re-enumerated onto a different
        GPU. None/True mean the restore path completed (True restored a live
        below-default cap; None means nothing of ours remained).
        """
        ...

    def restore_default_by_uuid(self, uuid: str) -> Optional[bool]:
        """Restore the factory-default TGP for the GPU carrying `uuid`.

        Identity-stable peer of `restore_default`: the target index is
        resolved from `uuid` at write time, so a driver/hostengine
        re-enumeration between the caller's UUID probe and the write
        cannot land the restore (and the caller's subsequent prune) on a
        different physical GPU. Cold-start orphan recovery and the SIGTERM
        sweep use this instead of the index-keyed `restore_default`, which
        is only safe once `apply_cap` has recorded the index→UUID mapping.

        Returns:
          * ``True``  - a live below-default cap was restored.
          * ``None``  - the UUID resolved but the GPU is already at/above
            default (nothing of ours to restore), or a clean scan proved
            the GPU is no longer present.
          * ``False`` - the UUID could not be located conclusively (a probe
            raised, e.g. a transient outage), so the GPU may still carry our
            cap; the caller must keep the UUID and retry later.
        """
        ...

    def scan_uuid_index_map(self) -> tuple[dict[str, int], bool]:
        """Snapshot every currently-visible GPU as ``{uuid: index}`` plus a
        ``conclusive`` flag.

        ``conclusive`` is True only when the ENTIRE topology was enumerated
        without error in a single consistent view, so a caller may treat a UUID
        absent from the map as proof the GPU is gone. On any partial / failed
        enumeration it is False and the map must be treated as non-authoritative
        (a missing UUID is NOT proof of absence).

        Cold-start orphan recovery uses this instead of a
        ``range(device_count())`` + per-index ``get_uuid`` loop. On the DCGM path
        ``get_uuid`` reconnects on its OWN, and a reconnect can GROW discovery (a
        hostengine restart re-enumerating more GPUs); with the loop bound fixed
        at the pre-growth length the newly-enumerated indices are never visited,
        yet no probe raises — so a persisted UUID that merely MOVED to a new
        index looks absent and gets pruned. Implementations MUST build the map
        from ONE consistent enumeration (DCGM: inside a single ``_with_reconnect``
        that re-materializes the topology length, mirroring
        ``_resolve_idx_for_uuid``) so a mid-scan growth is fully rescanned.
        """
        ...


class NvmlActuator:
    """NVML-backed actuator — the default and only path in PR #9682.

    All methods lazy-import `power_agent` (which owns the module-level
    NVML helpers, `_managed_gpu_indices`, and the persistent state) and
    `pynvml` to keep the test patches that target `power_agent.pynvml`
    effective. This intentionally re-uses the exact code paths the
    existing 43 tests cover.

    Construction
    ------------
    `apply_cap` and `restore_default` need a metrics object to record
    clamping events, apply failures, and the applied-limit gauge.
    `PowerAgent` constructs the actuator with its own metrics
    (`NvmlActuator(self.metrics)`); tests that only exercise queries
    (`device_count`, `get_uuid`, etc.) can construct with no args.
    """

    name: str = "nvml"

    def __init__(self, metrics: Optional[Any] = None) -> None:
        # `metrics` is `power_agent.PowerAgentMetrics` in production. Typed
        # as Any here so this module doesn't need to import power_agent at
        # module load time (would create a cycle: power_agent imports
        # actuator → actuator imports power_agent at top level → ImportError).
        self._metrics = metrics

    def init(self) -> None:
        """Initialize NVML — this actuator OWNS the NVML lifecycle.

        Moved here from `PowerAgent.__init__` so each
        actuator fully owns its library lifecycle and the init/shutdown
        counts balance per mode:

          * NVML mode  — this actuator does the single `nvmlInit()` here
            and the single `nvmlShutdown()` in `shutdown()`.
          * DCGM mode  — `DcgmActuator` does its own (guarded) `nvmlInit()`
            for its `list_running_pids` reads, and its own `nvmlShutdown()`.

        Previously `PowerAgent.__init__` called `nvmlInit()` unconditionally
        AND `DcgmActuator.init()` called it again, so DCGM mode had two inits
        against one shutdown — leaking one NVML reference for the process
        lifetime. Owning the lifecycle per-actuator removes that imbalance.
        """
        import pynvml

        pynvml.nvmlInit()

    def shutdown(self) -> None:
        """Shut NVML down (pairs with the `nvmlInit()` in `init()`).

        Best-effort like `DcgmActuator.shutdown()`: shutdown cleanup
        (`power_agent._shutdown_cleanup`) must complete regardless, so a
        teardown fault is logged (with traceback) rather than raised.
        """
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            logger.exception(
                "pynvml.nvmlShutdown() failed during NvmlActuator shutdown; "
                "continuing shutdown.",
            )

    def device_count(self) -> int:
        import pynvml

        return pynvml.nvmlDeviceGetCount()

    def get_uuid(self, gpu_idx: int) -> str:
        import power_agent
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        return power_agent._nvml_uuid(handle)

    def scan_uuid_index_map(self) -> tuple[dict[str, int], bool]:
        """NVML identity snapshot. NVML indices are process-stable and NVML does
        not reconnect / re-enumerate mid-scan, so a clean pass over
        ``range(nvmlDeviceGetCount())`` is conclusive; a device-count read
        failure or any per-index UUID read failure marks it inconclusive."""
        import pynvml

        try:
            count = pynvml.nvmlDeviceGetCount()
        except Exception as e:
            logger.warning(
                "NVML device-count read failed during identity scan; "
                "treating as inconclusive: %s",
                e,
            )
            return {}, False

        mapping: dict[str, int] = {}
        conclusive = True
        for idx in range(count):
            try:
                mapping[self.get_uuid(idx)] = idx
            except Exception as e:
                conclusive = False
                logger.warning(
                    "NVML UUID read failed for index %d during identity scan; "
                    "marking scan inconclusive: %s",
                    idx,
                    e,
                )
        return mapping, conclusive

    def list_running_pids(
        self, gpu_idx: int, expected_uuid: Optional[str] = None
    ) -> list[int]:
        # NVML per-process index ordering is stable for the process lifetime
        # (no re-enumeration), so `gpu_idx` already identifies the anchored
        # GPU and `expected_uuid` needs no rebinding here — accepted for
        # Protocol uniformity, mirroring NvmlActuator.apply_cap.
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        return [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]

    def constraints_w(self, gpu_idx: int) -> tuple[int, int]:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        return min_mw // 1000, max_mw // 1000

    def current_w(self, gpu_idx: int) -> int:
        """Return the cap currently applied (mW from NVML, returned in W).

        Mirrors the inline `pynvml.nvmlDeviceGetPowerManagementLimit`
        call that was in `_restore_orphaned_gpus_on_startup` before
        the Protocol extension lifted it onto the actuator
        surface. Lazy-imports pynvml so test patches that target
        `power_agent.pynvml` (the legacy NVML test surface) stay
        effective.
        """
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        return pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000

    def default_w(self, gpu_idx: int) -> int:
        """Return the factory-default TGP (mW from NVML, returned in W).

        Mirrors the inline `pynvml.nvmlDeviceGetPowerManagementDefaultLimit`
        call that lived in `_restore_orphaned_gpus_on_startup` and the
        shutdown restore path (`_shutdown_cleanup`). Read once, then compared
        against `current_w` by the orphan-recovery guard.
        """
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        return pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) // 1000

    def apply_cap(
        self, gpu_idx: int, watts: int, expected_uuid: Optional[str] = None
    ) -> int:
        """Apply a power cap to GPU `gpu_idx`, returning the effective watts.

        Delegates to `power_agent._apply_cap`, which contains the
        clamping, managed-state tracking, metrics updates, and
        NVMLError handling exercised by `test_apply_cap.py`.
        `_apply_cap` returns `None` (predates the Protocol), so we
        re-derive the post-clamp effective_w from constraints here
        WITHOUT calling `_clamp_to_constraints` again — a previous
        version did, which double-logged the clamp warning and
        double-incremented `cap_clamped_total` for any out-of-range
        request. Constraint reads
        are pure (no side effects), so reading min/max here and
        clamping locally is equivalent to peeking at what `_apply_cap`
        will produce.

        `expected_uuid` is accepted for Protocol uniformity but not used:
        NVML binds the device handle by index for the process lifetime and
        has no reconnect/re-enumeration model like DCGM's hostengine, so there
        is no in-process window in which the index could move onto a different
        physical GPU between the caller's identity capture and this write. The
        DCGM re-enumeration guard lives in `DcgmActuator.apply_cap`.
        """
        if self._metrics is None:
            raise RuntimeError(
                "NvmlActuator.apply_cap requires a metrics object; "
                "construct as NvmlActuator(power_agent_metrics)."
            )

        import power_agent
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        # _apply_cap runs the SINGLE clamp (with its log + metric side-
        # effects). We only need the post-clamp number to return.
        power_agent._apply_cap(handle, gpu_idx, watts, self._metrics)
        try:
            min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            min_w, max_w = min_mw // 1000, max_mw // 1000
            return max(min_w, min(watts, max_w))
        except pynvml.NVMLError:
            # If constraints read fails, fall back to the requested
            # value — _apply_cap's own _clamp_to_constraints has the
            # same fallback (`return requested_w`), so the two paths
            # remain consistent.
            return watts

    def restore_default(self, gpu_idx: int) -> Optional[bool]:
        """Restore the factory-default TGP via NVML.

        Mirrors the inline NVML calls in `power_agent._shutdown_cleanup`
        and `power_agent._restore_orphaned_gpus_on_startup` so
        `DcgmActuator.restore_default` (which reads
        `DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF`) is a peer of this method,
        not a special case.
        """
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
        # Keep the applied-limit gauge in sync with what is now LIVE on the GPU
        # (the factory default) — apply ticks it via `power_agent._apply_cap`, so
        # a restore must too or Prometheus keeps reporting the released cap
        # .
        if self._metrics is not None:
            self._metrics.applied_limit_watts.labels(gpu=str(gpu_idx)).set(
                default_mw // 1000
            )
        return True

    def restore_default_by_uuid(self, uuid: str) -> Optional[bool]:
        """Identity-stable restore for the NVML path.

        NVML's per-process index ordering is stable (unlike DCGM, which can
        re-enumerate after a hostengine reconnect), but the orphan-recovery
        and SIGTERM callers are actuator-agnostic and rely on the Protocol
        contract, so we resolve `uuid` to its current index here and apply
        the same `current_w < default_w` guard `DcgmActuator` does. Returns
        ``True`` / ``None`` / ``False`` per the Protocol.
        """
        scan_complete = True
        match_idx: Optional[int] = None
        for idx in range(self.device_count()):
            try:
                if self.get_uuid(idx) == uuid:
                    match_idx = idx
                    break
            except Exception as e:
                scan_complete = False
                logger.warning(
                    "NVML: failed to inspect GPU index %d while resolving "
                    "managed UUID %s: %s",
                    idx,
                    uuid,
                    e,
                )
        if match_idx is None:
            # Clean scan with no match -> GPU gone (safe to prune). Incomplete
            # scan -> indeterminate; keep the UUID for the next attempt.
            return None if scan_complete else False
        current = self.current_w(match_idx)
        if current < self.default_w(match_idx):
            self.restore_default(match_idx)  # updates the gauge to the default
            return True
        # Visible and already at/above default: nothing of ours to restore, but
        # sync the gauge to the LIVE value so a GPU restored to default
        # externally stops reporting our old cap.
        if self._metrics is not None:
            self._metrics.applied_limit_watts.labels(gpu=str(match_idx)).set(current)
        return None


class DcgmActuator:
    """DCGM-backed actuator — routes per-GPU power caps through `nvidia-dcgm`.

    Used on clusters where the GPU Operator's `dcgm.enabled=true`.
    Connects standalone-TCP to the operator-managed `nvidia-dcgm`
    hostengine and calls `dcgmConfigSet(mPowerLimit.val=W)` for the cap
    write. Per NVIDIA's DCGM Configuration API, `dcgmConfigSet` records
    the value as the GPU's "target configuration", and DCGM itself
    "maintains [the target configuration] and automatically enforces
    [it] after a GPU reset or reinitialization is completed"
    (https://docs.nvidia.com/datacenter/dcgm/latest/dcgm-api/dcgm-api-config.html).
    So that reset/reinit auto-reapply — the single resilience property
    DCGM buys over NVML — is in effect after every successful Set; there
    is no tick-driven re-enforce loop in DCGM. It does **not** make the
    cap survive Power Agent restart (the agent restores default at shutdown
    regardless).

    The Power Agent deliberately issues ONLY `dcgmConfigSet`, never the
    `dcgmConfigEnforce` manual re-assert. Per the API docs `dcgmConfigEnforce`
    "manually enforce[s] the configuration" only if it was "already
    configured using the API dcgmConfigSet" — it registers nothing new, so
    right after a successful Set it is a redundant no-op (the current state
    already equals the target). An earlier revision exposed it behind an
    optional `enforce` flag; that surface was removed as
    semantically dead config.

    Asymmetric reads (intentional)
    ------------------------------
    `list_running_pids` calls `pynvml.nvmlDeviceGetComputeRunningProcesses`
    even on the DCGM path. DCGM has no public snapshot-of-running-PIDs
    API: `DCGM_FI_DEV_COMPUTE_PIDS` is a time-series field where each
    value decodes to one `c_dcgmRunningProcess_t` (per
    `dcgm_field_helpers.py:67-71`), and there is no
    `dcgmGetDeviceProcesses` exported in `libdcgm.so`. This is a
    property of the upstream API shapes, not a design preference —
    DCGM is built for time-series GPU monitoring, NVML for snapshot
    device queries.

    Stale-handle recovery
    ---------------------
    When the operator restarts `nvidia-dcgm` (pod eviction, upgrade,
    chart rollback), the cached `DcgmHandle` and per-GPU `DcgmGroup`
    objects become invalid; the next API call raises
    `dcgmExceptionClass(DCGM_ST_CONNECTION_NOT_VALID)` per the
    upstream test (`DCGM/testing/python3/tests/test_connection.py:48-87`).
    Every hostengine-touching method routes through `_with_reconnect`,
    which catches that specific error code exactly once, flushes the
    group cache, rebuilds the handle, and retries. Persistent failure
    propagates so the agent's reconcile loop logs it and the next
    reconcile tick tries again.

    `restore_default` reads field 163
    --------------------------------
    DCGM exposes four distinct power-limit fields
    (`dcgm_fields.py:232-238`): MIN, MAX, DEF, current. NVML's
    `nvmlDeviceGetPowerManagementDefaultLimit` returns the
    factory default (`DEF`), which on every shipped data-center SKU
    equals MAX but is conceptually distinct. We read DEF to match
    NVML's restore semantics byte-for-byte on hypothetical future
    SKUs where default < max.
    """

    name: str = "dcgm"

    DEFAULT_HOST = "nvidia-dcgm.gpu-operator.svc.cluster.local"
    DEFAULT_PORT = 5555

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        metrics: Optional[Any] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._metrics = metrics

        # All hostengine state — populated in init(), cleared in
        # shutdown() and _with_reconnect's recovery path.
        self._handle: Optional[Any] = None
        self._system: Optional[Any] = None
        self._discovered_gpu_ids: list[int] = []
        self._groups: dict[int, Any] = {}  # gpu_idx -> DcgmGroup

        # NVML init/shutdown are refcounted by the driver. `init()` runs on
        # EVERY `_with_reconnect` recovery (a stale-handle rebuild re-invokes
        # it), but `shutdown()` runs once — so calling `nvmlInit()` from each
        # init() would leak one refcount per reconnect and leave NVML
        # initialized after our single `nvmlShutdown()`. Track whether WE hold
        # the init so we call `nvmlInit()` at most once per actuator lifetime
        # and pair it with the one `nvmlShutdown()`.
        self._nvml_initialized = False

        # Cross-library identity map — built lazily on first
        # `list_running_pids` call (see _ensure_identity_map). DCGM
        # gpuIds and NVML indices live in separate identity spaces:
        # DCGM preserves gpuId across detach/attach by UUID matching
        # (DcgmCacheManager.cpp:1230-1296), while NVML re-enumerates
        # per-process; MIG can split the surfaces further. Routing
        # the NVML PID read by `gpu_idx` would therefore read the
        # wrong GPU on any node where the two libraries disagree on
        # ordering. We translate via UUID instead — the only hardware
        # identifier both libraries agree on. Cleared on reconnect
        # because DCGM may re-enumerate after the hostengine restart.
        self._dcgm_uuid_by_idx: Optional[list[str]] = None
        self._nvml_index_by_uuid: Optional[dict[str, int]] = None

        # In-process restore identity snapshot. `_managed_gpu_indices` stores
        # the reconcile-loop integer index, but DCGM may re-enumerate after a
        # hostengine restart. Keep the UUID we actually capped so SIGTERM
        # restore can relocate the physical GPU before writing default TGP.
        self._managed_uuid_by_idx: dict[int, str] = {}

        # Append-only set of every UUID THIS process capped. Unlike
        # `_managed_uuid_by_idx` (keyed by the unstable integer index, so a
        # re-cap of a re-enumerated index overwrites the prior UUID) and
        # `_managed_gpu_indices` (a set of indices that can't represent two
        # GPUs that occupied one index at different times), this never loses
        # an entry on re-enumeration. It is the OWNERSHIP filter for the
        # SIGTERM UUID sweep: the persisted `_previously_managed` set is
        # cross-incarnation (it can hold UUIDs orphan recovery kept but this
        # process never capped, e.g. GPUs with a running workload), so the
        # sweep must restrict to UUIDs we actually capped to avoid resetting
        # a cap owned by another workflow. See `_shutdown_cleanup`.
        self._capped_uuids: set[str] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Open the hostengine connection. Fails loud if unreachable.

        An operator who set `agent.actuator: dcgm` in the chart values
        declared that DCGM is mandatory; silently falling back to NVML
        would mask a misconfiguration. The chart catches typos at
        template time (via `validateActuator`); this method catches
        runtime unreachability with a clear stack trace.

        Fails fast on DCGM 3.x bindings: `_apply_cap_inner` uses
        `dcgm_structs.c_dcgmDeviceConfig_v2`, which only exists in 4.x.
        Without the check below, the agent would connect cleanly, run
        one reconcile tick worth of cap-writes (raising `AttributeError`
        on the first GPU), and exit mid-tick — leaving some GPUs capped
        and SIGTERM-restore unable to run because the actuator never
        finished registering. Failing here surfaces the version skew
        as an actionable startup error instead.
        """
        import dcgm_structs
        import pydcgm

        if not hasattr(dcgm_structs, "c_dcgmDeviceConfig_v2"):
            raise RuntimeError(
                "DcgmActuator requires DCGM >= 4.0 bindings "
                "(dcgm_structs.c_dcgmDeviceConfig_v2 is missing — "
                "DCGM 3.x bindings detected). Rebuild the Power Agent "
                "image against a DCGM 4.x base via "
                "`--build-arg DCGM_IMAGE=nvcr.io/nvidia/cloud-native/"
                "dcgm:<4.x-tag>` — pin the tag your GPU Operator's "
                "nvidia-dcgm hostengine deploys (client >= hostengine); see "
                "the DCGM_IMAGE default in deploy/power-agent/Dockerfile."
            )

        self._handle = pydcgm.DcgmHandle(
            ipAddress=f"{self._host}:{self._port}",
            opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO,
            persistAfterDisconnect=False,
            timeoutMs=5000,
        )
        self._system = self._handle.GetSystem()
        # GPU enumeration uses the hostengine's view (same as
        # dcgm-exporter, dcgmi, the customer's existing DCGM stack).
        self._discovered_gpu_ids = sorted(self._system.discovery.GetAllGpuIds())

        # NVML is also initialized so list_running_pids() can call
        # nvmlDeviceGetComputeRunningProcesses (see class docstring).
        # pynvml is already a dependency of PR #9682 — no new image
        # cost. Guard with `_nvml_initialized` so a reconnect-driven init()
        # doesn't re-`nvmlInit()` and leak a refcount past our single
        # `nvmlShutdown()`.
        import pynvml

        if not self._nvml_initialized:
            pynvml.nvmlInit()
            self._nvml_initialized = True

        logger.info(
            "DcgmActuator connected to %s:%d (%d GPUs)",
            self._host,
            self._port,
            len(self._discovered_gpu_ids),
        )

    def shutdown(self) -> None:
        """Release hostengine + NVML resources.

        Defensive against the hostengine already being gone — by
        SIGTERM time it may already be evicted, so the Shutdown call
        itself can raise `DCGM_ST_CONNECTION_NOT_VALID`.

        Per PR #9682 CodeRabbit review, individual cleanup failures are
        LOGGED (with traceback) rather than silently dropped — silent
        catches made hostengine / NVML shutdown faults invisible in
        pod logs. We still don't re-raise from this method: cleanup is
        best-effort and the caller (`power_agent._shutdown_cleanup`) needs
        to finish the rest of the teardown regardless so the container
        exits promptly.
        """
        if self._handle is not None:
            for gpu_id, grp in self._groups.items():
                try:
                    grp.Delete()
                except Exception:
                    logger.exception(
                        "DCGM group Delete() failed for gpu_id=%s; "
                        "continuing shutdown.",
                        gpu_id,
                    )
            self._groups.clear()
            try:
                self._handle.Shutdown()
            except Exception:
                logger.exception(
                    "DCGM hostengine Shutdown() failed (host=%s port=%d); "
                    "continuing shutdown.",
                    self._host,
                    self._port,
                )
            self._handle = None
            self._system = None
        # Drop the identity-map cache so a subsequent init()/use
        # rebuilds it cleanly.
        self._dcgm_uuid_by_idx = None
        self._nvml_index_by_uuid = None
        # Only shut NVML down if WE initialized it (pairs with the guarded
        # nvmlInit in init()); clear the flag so a later init() on the same
        # instance re-initializes cleanly.
        if self._nvml_initialized:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                logger.exception(
                    "pynvml.nvmlShutdown() failed during DcgmActuator shutdown; "
                    "continuing shutdown.",
                )
            finally:
                self._nvml_initialized = False

    # ------------------------------------------------------------------
    # Stale-handle recovery
    # ------------------------------------------------------------------

    def _with_reconnect(self, op: Callable[[], T]) -> T:
        """Run `op`; on `DCGM_ST_CONNECTION_NOT_VALID`, rebuild + retry once.

        See the class docstring for the reasoning behind
        single-retry semantics, dropping (not Deleting) the group cache,
        and inline (not watchdog-thread) reconnection.
        """
        import dcgm_structs

        try:
            return op()
        except dcgm_structs.DCGMError as e:
            # Only recover from CONNECTION_NOT_VALID — other DCGM errors
            # (NOT_SUPPORTED, GENERIC_ERROR, etc.) indicate a different
            # class of failure that recovery would only mask.
            if e.value != dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID:
                raise

            logger.warning(
                "nvidia-dcgm connection lost (DCGM_ST_CONNECTION_NOT_VALID); "
                "rebuilding handle and flushing %d cached DcgmGroup(s).",
                len(self._groups),
            )

            # Drop the stale handle + group cache. Do NOT try to
            # `.Delete()` cached groups: their backing hostengine state
            # is already gone, and the Delete call would itself raise
            # CONNECTION_NOT_VALID.
            self._groups.clear()
            # Invalidate the cross-library identity map. DCGM may
            # re-enumerate after the hostengine restart (different
            # gpuId ordering if the operator changed visibility), so
            # we must rebuild on next `list_running_pids` call.
            self._dcgm_uuid_by_idx = None
            self._nvml_index_by_uuid = None
            try:
                if self._handle is not None:
                    self._handle.Shutdown()
            except Exception:
                # Intentional: this is best-effort cleanup of an already
                # dead hostengine connection. Aborting reconnect on
                # cleanup failure would be strictly worse — the stale
                # handle is local-only, we're about to drop it. Log so
                # the failure is still visible in outage diagnosis (silent
                # swallowing would hide hostengine faults), then
                # continue to re-init.
                logger.exception(
                    "DcgmActuator reconnect cleanup: handle.Shutdown() "
                    "failed on the stale handle; dropping it and "
                    "continuing with re-init."
                )
            self._handle = None
            self._system = None

            # Re-establish handle + GPU discovery. If init() itself
            # raises, the exception propagates: a sustained outage
            # should be visible to the reconcile loop (and operators),
            # not silently retried indefinitely.
            self.init()

            # One retry. Any further error propagates so the operator
            # sees a real outage in their logs / Prometheus
            # `apply_failures_total`.
            return op()

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def device_count(self) -> int:
        # `_discovered_gpu_ids` is populated by `init()` and refreshed ONLY by a
        # `_with_reconnect` re-init, which fires only on a
        # `DCGM_ST_CONNECTION_NOT_VALID` raised by an actual hostengine call. An
        # EMPTY cached topology is therefore a trap: the reconcile loop iterates
        # `range(0)`, issues no hostengine call, so nothing can raise
        # CONNECTION_NOT_VALID, so discovery is never rebuilt — an agent that
        # connected before the hostengine finished enumerating GPUs (agent /
        # nvidia-dcgm startup-ordering race) or hit a transient empty discovery
        # would loop forever over zero GPUs, silently enforcing NO caps. Re-probe
        # discovery when the cache is empty so GPUs that appear after startup are
        # picked up on the next reconcile cycle without a restart. NVML has no
        # analogue: its `device_count()` reads `nvmlDeviceGetCount()` live.
        if not self._discovered_gpu_ids:
            self._rediscover_gpu_ids()
        return len(self._discovered_gpu_ids)

    def _rediscover_gpu_ids(self) -> None:
        """Best-effort refresh of `_discovered_gpu_ids` from the hostengine.

        Recovery for a transiently-empty topology (see `device_count`). Runs the
        discovery read inside `_with_reconnect` so a stale handle reconnects and
        re-enumerates first. Fully best-effort: a still-empty result or ANY read
        failure keeps the last-known set and never raises (a rediscovery fault
        must not abort the reconcile cycle). A non-empty refresh invalidates the
        cross-library identity map so it rebuilds against the new ordering on the
        next `list_running_pids`.
        """
        if self._system is None:
            return
        try:
            discovered = self._with_reconnect(
                lambda: sorted(self._system.discovery.GetAllGpuIds())
            )
        except Exception as e:
            logger.warning(
                "DCGM GPU re-discovery failed; keeping last-known %d GPU(s): %s",
                len(self._discovered_gpu_ids),
                e,
            )
            return
        if discovered and discovered != self._discovered_gpu_ids:
            logger.info(
                "DCGM GPU discovery refreshed: %d GPU(s) now visible (was %d); "
                "recovered a transiently-empty topology without a restart.",
                len(discovered),
                len(self._discovered_gpu_ids),
            )
            self._discovered_gpu_ids = discovered
            # Ordering may have changed; force the identity map to rebuild.
            self._dcgm_uuid_by_idx = None
            self._nvml_index_by_uuid = None

    def get_uuid(self, gpu_idx: int) -> str:
        """Read the GPU UUID via the synchronous device-info API.

        Uses `DcgmSystem.discovery.GetGpuAttributes` (which wraps
        `dcgmGetDeviceAttributes`) rather than `dcgmEntityGetLatestValues`
        with `DCGM_FI_DEV_UUID`. The latter pulls from DCGM's field cache,
        which only populates a field once *some* consumer subscribes to
        it via `dcgmWatchFields`. On a freshly started hostengine with no
        companion watcher (e.g. a standalone `nv-hostengine` in dev/CI,
        or any production cluster whose `nvidia-dcgm-exporter`
        configuration doesn't include UUID), the field cache returns the
            string-blank sentinel `<<<NULL>>>`, which silently broke
            cross-library identity mapping when UUID was sourced from the
            field cache. `GetGpuAttributes` is
        the documented synchronous device-info call for static
        descriptors (UUID, brand, name, serial, PCI BDF, MIG mode) and
        does not depend on the field cache, so it works on any
        hostengine the moment GPU discovery completes.
        """

        return self._with_reconnect(lambda: self._read_uuid_raw(gpu_idx))

    def _read_uuid_raw(self, gpu_idx: int) -> str:
        """UUID read WITHOUT the `_with_reconnect` wrapper.

        `get_uuid` wraps this for external callers. The cap-write identity
        guard (`_apply_cap_inner._write_set`) is itself already running
        inside `_with_reconnect`, so it calls this directly to avoid
        nesting reconnect logic; a `CONNECTION_NOT_VALID` here propagates to
        the enclosing `_with_reconnect`, which handles the single retry.
        """
        gpu_id = self._discovered_gpu_ids[gpu_idx]
        attrs = self._system.discovery.GetGpuAttributes(gpu_id)
        raw = attrs.identifiers.uuid
        return self._normalize_uuid(raw, source=f"DCGM gpu_id={gpu_id}")

    def list_running_pids(
        self, gpu_idx: int, expected_uuid: Optional[str] = None
    ) -> list[int]:
        """Snapshot of compute PIDs on the GPU — via NVML, even on the DCGM path.

        DCGM has no public snapshot-of-running-PIDs API. See the class
        docstring for the full reasoning.
        Lazy-imports pynvml so this method is callable from tests that
        mock the import.

        Identity mapping
        ----------------
        `gpu_idx` is the DCGM-ordered index used throughout the
        reconcile loop, NOT an NVML index. We translate via UUID
        (UUID -> NVML index via _nvml_index_by_uuid) because the two
        libraries live in separate identity spaces — see
        DcgmCacheManager.cpp:1230-1296.

        Identity binding
        --------------------------------------
        When `expected_uuid` is supplied, the NVML index is resolved
        straight from THAT UUID rather than from `gpu_idx`. A DCGM
        hostengine reconnect can re-enumerate `gpu_idx` onto a DIFFERENT
        physical GPU between the caller's identity capture and this read;
        resolving by index would then attribute the other GPU's PIDs to
        `expected_uuid`, and the cap derived from them would be written onto
        the anchored GPU by the identity-guarded `apply_cap` (which only
        verifies the write DESTINATION). Resolving by UUID guarantees the PID
        snapshot belongs to the anchored GPU; if that GPU is no longer
        resolvable we fail closed with `_GpuIdentityMismatch` so the caller
        skips this cycle instead of mis-routing.
        """
        import pynvml

        self._ensure_identity_map()
        if expected_uuid is not None:
            uuid = expected_uuid
        else:
            uuid = self._dcgm_uuid_by_idx[gpu_idx]
        try:
            nvml_idx = self._nvml_index_by_uuid[uuid]
        except KeyError as err:
            if expected_uuid is not None:
                # The anchored GPU is no longer resolvable to an NVML index
                # (re-enumeration or hot-unplug). Fail closed — the caller
                # skips PID attribution this cycle rather than reading a
                # different GPU's workload.
                raise _GpuIdentityMismatch(
                    f"DcgmActuator.list_running_pids: anchored UUID {uuid!r} "
                    "is no longer resolvable to an NVML index (re-enumeration "
                    "or hot-unplug); skipping PID attribution this cycle."
                ) from err
            # Unanchored path: _ensure_identity_map raises loudly on missing
            # UUIDs at build time, so reaching here means a GPU disappeared
            # from NVML's view between map build and now. Surface it instead
            # of silently mis-routing.
            raise RuntimeError(
                f"DcgmActuator.list_running_pids: GPU UUID {uuid!r} "
                f"(DCGM gpu_idx={gpu_idx}) is no longer visible to "
                "NVML. Suspect mid-reconcile device hot-unplug."
            ) from err
        handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
        return [p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)]

    def _ensure_identity_map(self) -> None:
        """Build the DCGM-idx <-> NVML-idx UUID map if not already built.

        Lazy (cap-only paths don't pay the cost). Invalidated by
        `_with_reconnect` (DCGM may re-enumerate post-restart) and
        `shutdown`. Raises RuntimeError on cross-library UUID
        mismatch — silent mis-routing of PID reads to the wrong GPU
        is the failure mode worth failing loud on. DCGM reads use
        `_with_reconnect` so the first call survives a hostengine
        restart between agent startup and first reconcile.
        """
        if self._dcgm_uuid_by_idx is not None and self._nvml_index_by_uuid is not None:
            return

        import pynvml

        def _read_dcgm_uuids() -> list[str]:
            # Indexed by gpu_idx -> _discovered_gpu_ids[gpu_idx].
            # Safe to re-iterate on _with_reconnect retry because
            # _discovered_gpu_ids is repopulated by init() on recovery.
            # Uses GetGpuAttributes (synchronous device-info API) — see
            # `get_uuid` docstring for why `dcgmEntityGetLatestValues +
            # DCGM_FI_DEV_UUID` returns `<<<NULL>>>` on a fresh hostengine
            # and would silently break the cross-library identity map.
            uuids: list[str] = []
            for gpu_id in self._discovered_gpu_ids:
                attrs = self._system.discovery.GetGpuAttributes(gpu_id)
                raw = attrs.identifiers.uuid
                uuids.append(self._normalize_uuid(raw, source=f"DCGM gpu_id={gpu_id}"))
            return uuids

        dcgm_uuids = self._with_reconnect(_read_dcgm_uuids)

        # NVML side: enumerate this process's NVML indices -> UUIDs.
        # No DCGM dependency here — bare pynvml.
        nvml_index_by_uuid: dict[str, int] = {}
        nvml_count = pynvml.nvmlDeviceGetCount()
        for nvml_idx in range(nvml_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_idx)
            raw_uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuid = self._normalize_uuid(raw_uuid, source=f"NVML index={nvml_idx}")
            nvml_index_by_uuid[uuid] = nvml_idx

        missing = [u for u in dcgm_uuids if u not in nvml_index_by_uuid]
        if missing:
            raise RuntimeError(
                f"DcgmActuator: {len(missing)} GPU UUID(s) visible to "
                f"DCGM are not visible to NVML in this process: "
                f"{missing!r}. The two libraries must agree on the GPU "
                "set before the DCGM actuator can route per-GPU PID "
                "reads. Likely causes: NVIDIA_VISIBLE_DEVICES differs "
                "between the Power Agent pod and the nvidia-dcgm pod, "
                "MIG-mode mismatch, or concurrent device hot-plug."
            )

        # Publish both halves atomically so a partial write can't be
        # observed by a concurrent reader (the reconcile loop is
        # single-threaded today, but a future caller might not be).
        self._dcgm_uuid_by_idx = dcgm_uuids
        self._nvml_index_by_uuid = nvml_index_by_uuid

        logger.info(
            "DcgmActuator identity map built: %d GPUs reconciled "
            "between DCGM and NVML via UUID.",
            len(dcgm_uuids),
        )

    def _power_limits(self, gpu_idx: int) -> "Any":
        """Read all four DCGM power-limit fields in one RPC, via GetGpuAttributes.

        `DcgmSystem.discovery.GetGpuAttributes` returns a
        `c_dcgmDeviceAttributes_v3` struct whose `.powerLimits` member
        carries `curPowerLimit`, `defaultPowerLimit`,
        `enforcedPowerLimit`, `minPowerLimit`, `maxPowerLimit` — all
        in watts. This consolidates `constraints_w` / `current_w` /
        `default_w` onto the synchronous device-info API (same
        rationale as `get_uuid`, see its docstring): the field-cache
        API `dcgmEntityGetLatestValues` returns `DCGM_FP64_BLANK`
        (= 140737488355328.0) for any field no DCGM consumer has
        watched, which silently broke an earlier field-cache-based
        implementation on every fresh hostengine. `GetGpuAttributes` works on any
        hostengine the moment GPU discovery completes.
        """

        def _op() -> "Any":
            gpu_id = self._discovered_gpu_ids[gpu_idx]
            return self._system.discovery.GetGpuAttributes(gpu_id).powerLimits

        return self._with_reconnect(_op)

    def _power_limits_with_uuid(self, gpu_idx: int) -> tuple["Any", str]:
        """Read `powerLimits` AND the UUID from ONE `GetGpuAttributes` RPC.

        Returns ``(powerLimits, uuid)`` sourced from the *same* attributes
        snapshot. `apply_cap` and the restore paths use this to bind the
        wattage they are about to write (clamped against min/max, or the
        factory default) to the identity that wattage was derived from —
        closing the ABA provenance gap: with
        min/max/default read through one `_with_reconnect` op and the
        identity-guarded write through another, a double re-enumeration
        (A→B→A) could clamp GPU-A's request against GPU-B's SKU range (or
        write GPU-B's default onto GPU-A) while the final destination guard
        still sees A. Reading the limits and their owning UUID atomically
        lets the caller reject a snapshot whose UUID != the anchored
        identity before the write. `.identifiers.uuid` and `.powerLimits`
        are members of the one `c_dcgmDeviceAttributes_v3` struct, so this
        costs no extra RPC over the separate reads it replaces.
        """

        def _op() -> tuple["Any", str]:
            gpu_id = self._discovered_gpu_ids[gpu_idx]
            attrs = self._system.discovery.GetGpuAttributes(gpu_id)
            uuid = self._normalize_uuid(
                attrs.identifiers.uuid, source=f"DCGM gpu_id={gpu_id}"
            )
            return attrs.powerLimits, uuid

        return self._with_reconnect(_op)

    def constraints_w(self, gpu_idx: int) -> tuple[int, int]:
        """Return (min_w, max_w) — the SKU's settable power-cap range.

        Reads `powerLimits.minPowerLimit` + `maxPowerLimit` from the
        device-info API. Distinct from `defaultPowerLimit` (which
        `restore_default` reads).
        """
        pl = self._power_limits(gpu_idx)
        return (
            self._coerce_power_limit_watts(pl.minPowerLimit, "minPowerLimit", gpu_idx),
            self._coerce_power_limit_watts(pl.maxPowerLimit, "maxPowerLimit", gpu_idx),
        )

    def current_w(self, gpu_idx: int) -> int:
        """Return the GPU's current power cap in watts.

        Reads `powerLimits.curPowerLimit`. Used by orphan-recovery
        to skip writes that would no-op.
        """
        return self._coerce_power_limit_watts(
            self._power_limits(gpu_idx).curPowerLimit, "curPowerLimit", gpu_idx
        )

    def default_w(self, gpu_idx: int) -> int:
        """Return the GPU's factory-default power limit in watts.

        Reads `powerLimits.defaultPowerLimit` — the byte-for-byte DCGM
        equivalent of NVML's `nvmlDeviceGetPowerManagementDefaultLimit`.
        Distinct from `maxPowerLimit` (max settable); on every shipped
        data-center SKU the two are numerically equal, but reading the
        right field keeps the DCGM path semantically aligned with NVML
        on hypothetical future SKUs where default < max.
        """
        return self._coerce_power_limit_watts(
            self._power_limits(gpu_idx).defaultPowerLimit,
            "defaultPowerLimit",
            gpu_idx,
        )

    @staticmethod
    def _normalize_uuid(raw: Any, source: str) -> str:
        """Normalize a DCGM/NVML UUID and reject blank cache sentinels.

        DCGM/NVML bindings vary between ``bytes`` and ``str``. Both are
        accepted, but blank field-cache sentinels such as ``<<<NULL>>>`` are
        not valid hardware identities and must not enter the cross-library
        map.
        """
        try:
            uuid = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
        except UnicodeDecodeError as e:
            raise RuntimeError(f"DcgmActuator: non-ASCII UUID from {source}") from e

        uuid = uuid.strip()
        if not uuid or uuid == "<<<NULL>>>" or uuid.lower() == "none":
            raise RuntimeError(
                f"DcgmActuator: invalid blank UUID from {source}: {uuid!r}"
            )
        return uuid

    @staticmethod
    def _coerce_power_limit_watts(value: Any, field_name: str, gpu_idx: int) -> int:
        """Coerce a GetGpuAttributes power-limit field and reject blanks."""
        try:
            numeric = float(value)
        except (TypeError, ValueError) as e:
            raise RuntimeError(
                f"DcgmActuator: invalid DCGM power limit {field_name} for "
                f"GPU {gpu_idx}: {value!r}"
            ) from e

        if not math.isfinite(numeric):
            raise RuntimeError(
                f"DcgmActuator: non-finite DCGM power limit {field_name} for "
                f"GPU {gpu_idx}: {value!r}"
            )

        if numeric >= _DCGM_NUMERIC_BLANK_MIN:
            raise RuntimeError(
                f"DcgmActuator: blank DCGM power limit {field_name} for "
                f"GPU {gpu_idx}: {value!r}"
            )

        watts = int(numeric)
        if watts <= 0:
            raise RuntimeError(
                f"DcgmActuator: non-positive DCGM power limit {field_name} for "
                f"GPU {gpu_idx}: {value!r}"
            )
        return watts

    # ------------------------------------------------------------------
    # Write methods
    # ------------------------------------------------------------------

    def apply_cap(
        self, gpu_idx: int, watts: int, expected_uuid: Optional[str] = None
    ) -> int:
        """Write a per-GPU cap via `dcgmConfigSet`; return the effective watts.

        Mirrors `power_agent._apply_cap`'s clamp + metrics + state-track
        contract on the DCGM side. DCGMError other than
        CONNECTION_NOT_VALID is absorbed into `apply_failures_total`
        and the call returns the effective post-clamp watts regardless
        (per the Actuator Protocol). SIGTERM / orphan-recovery callers want write
        failures surfaced as exceptions instead — they call
        `restore_default`, which uses `_apply_cap_inner`.

        `expected_uuid` anchors the write to a physical GPU. The reconcile
        loop captures it BEFORE the PID snapshot that produced `watts` and
        passes it here, so the entire attribution → policy → write sequence
        refers to one GPU: if a DCGM reconnect re-enumerates the index
        anywhere in that window, the in-transaction re-verification in
        `_write_set` detects the mismatch and the write is skipped (retried
        next reconcile) rather than applying one GPU's workload-derived cap to
        another. When the caller supplies it, we do NOT re-capture here —
        re-capturing would reopen the very attribution-to-write window the
        anchor exists to close. When omitted (direct/legacy callers, tests),
        we capture our own entry-time identity as a best-effort anchor,
        failing closed if even that is unreadable.
        """
        if self._metrics is None:
            raise RuntimeError(
                "DcgmActuator.apply_cap requires a metrics object; "
                "construct as DcgmActuator(..., metrics=power_agent_metrics)."
            )

        if expected_uuid is None:
            # No caller-supplied anchor: capture the identity of the GPU we are
            # about to cap FIRST — before ANY other reconnect-capable DCGM call
            # (including `constraints_w` below) — and REQUIRE it. `watts` was
            # derived from the workload running on THIS index; if we cannot
            # establish which physical GPU that is, we cannot guarantee the cap
            # lands on the GPU whose workload produced it. Capturing before
            # `constraints_w` matters because a reconnect *inside* the
            # constraints read could re-enumerate the index onto a different
            # GPU; taking the identity first (and re-verifying it
            # in-transaction in `_write_set`) turns that into a detected
            # mismatch instead of silently adopting the new occupant as
            # "expected". Fail closed (apply failure, retried next reconcile
            # once identity is readable) rather than risk applying one GPU's
            # workload cap to another. Mirrors `_write_set`'s in-transaction
            # read, which also fails closed.
            try:
                expected_uuid = self.get_uuid(gpu_idx)
            except Exception as e:
                expected_uuid = None
                logger.error(
                    "Skipping DCGM cap write for GPU %d: could not read its "
                    "identity before the write, so the cap cannot be safely "
                    "attributed to the GPU whose workload produced it "
                    "(retrying next reconcile): %s",
                    gpu_idx,
                    e,
                )
                self._metrics.apply_failures_total.inc()

        # Clamp against the SKU range. Read the min/max AND the identity they
        # belong to from ONE GetGpuAttributes snapshot, so the constraints we
        # clamp against are provably GPU `expected_uuid`'s — not some other
        # GPU a mid-read re-enumeration parked on the index. Done AFTER the identity capture so a reconnect here can't
        # pre-empt/replace the expected identity; also honors the Actuator
        # Protocol return contract (the effective post-clamp watts) on the
        # fail-closed early-outs below. The read handles its own stale-handle
        # recovery.
        #
        # A DCGMError that SURVIVES that recovery (a non-connection error, or a
        # connection loss still failing after the single reconnect) is absorbed
        # into `apply_failures_total` and the requested watts returned unclamped
        # — the same fallback `NvmlActuator.apply_cap` uses when its constraints
        # read fails. No Set happens, so no cap is made live, which is exactly
        # what `apply_failures_total` counts; absorbing it here (rather than
        # letting it escape to `reconcile_once`'s generic per-GPU guard) keeps
        # this method consistent with its documented "DCGMError absorbed" return
        # contract. A blank/garbage power-limit VALUE is a `RuntimeError` from
        # `_coerce_power_limit_watts`, NOT a DCGMError, and is deliberately left
        # to surface (DCGM returning nonsense must be loud) — mirroring the
        # narrow `except dcgm_structs.DCGMError` on the write below.
        import dcgm_structs

        try:
            pl, constraints_uuid = self._power_limits_with_uuid(gpu_idx)
        except dcgm_structs.DCGMError as e:
            logger.error(
                "DCGM constraints read for GPU %d failed; skipping cap write "
                "and counting an apply failure: %s",
                gpu_idx,
                e,
            )
            self._metrics.apply_failures_total.inc()
            return watts
        min_w = self._coerce_power_limit_watts(
            pl.minPowerLimit, "minPowerLimit", gpu_idx
        )
        max_w = self._coerce_power_limit_watts(
            pl.maxPowerLimit, "maxPowerLimit", gpu_idx
        )
        effective_w = self._clamp_with_metrics(watts, min_w, max_w, gpu_idx)

        if expected_uuid is None:
            # Entry identity was unreadable (logged + counted above). Return
            # the clamped effective watts per the Actuator Protocol; NO write
            # happened.
            return effective_w

        if constraints_uuid != expected_uuid:
            # The SKU range we just clamped against came from a DIFFERENT
            # physical GPU than the one this cap is anchored to (a re-enumeration
            # during the constraints read). Writing `effective_w` would apply a
            # value clamped to another GPU's limits; fail closed and retry next
            # reconcile once the enumeration settles. The
            # write-path guard in `_write_set` catches destination mismatches;
            # this catches wattage-provenance mismatches.
            logger.error(
                "Skipping DCGM cap write for GPU %d: SKU constraints were read "
                "from UUID %s but the cap is anchored to %s (DCGM re-enumeration "
                "during the constraints read); retrying next reconcile.",
                gpu_idx,
                constraints_uuid,
                expected_uuid,
            )
            self._metrics.apply_failures_total.inc()
            return effective_w

        # Lazy import (deferred until we're actually about to call into
        # DCGM): keeps the actuator constructible and the "no metrics"
        # guard above runnable on hosts without the DCGM Python
        # bindings (tests, NVML-only nodes). Surrounding methods
        # (`_with_reconnect`, `_apply_cap_inner`, etc.) follow the
        # same pattern. Scoped to before the try so the narrowed
        # `except dcgm_structs.DCGMError` can resolve.
        import dcgm_structs

        try:
            return self._apply_cap_inner(
                gpu_idx, effective_w, expected_uuid=expected_uuid
            )
        except _GpuIdentityMismatch as e:
            # The target index re-enumerated onto a different GPU during the
            # write. Skip rather than clobber the unrelated GPU; the next
            # reconcile cycle re-attributes pods against the fresh
            # enumeration and retries. Counts as an apply failure so the
            # skipped cap surfaces on `apply_failures_total`.
            logger.error(
                "Skipping DCGM cap write for GPU %d → %d W: %s",
                gpu_idx,
                effective_w,
                e,
            )
            self._metrics.apply_failures_total.inc()
            return effective_w
        except dcgm_structs.DCGMError as e:
            # Narrow on purpose: only DCGM write errors
            # are part of the "cap-write failed, log + bump metric +
            # return effective_w" contract. AttributeError on the
            # pydcgm bindings (we hit one such bug around
            # dcgmvalue.DCGM_INT32_BLANK — see _apply_cap_inner
            # docstring), RuntimeError from sustained
            # CONNECTION_NOT_VALID after _with_reconnect re-init, or
            # any other programming defect MUST surface so it gets
            # fixed instead of being silently absorbed as a normal
            # apply failure.
            logger.error(
                "dcgmConfigSet GPU %d → %d W failed: %s",
                gpu_idx,
                effective_w,
                e,
            )
            self._metrics.apply_failures_total.inc()
            return effective_w

    def _apply_cap_inner(
        self,
        gpu_idx: int,
        effective_w: int,
        expected_uuid: Optional[str] = None,
        record_ownership: bool = True,
    ) -> int:
        """Inner cap-write path that propagates Set failures as exceptions.

        Precondition: `effective_w` is already clamped to SKU range.

        `_write_set` reads the GPU identity inside the same `_with_reconnect`
        transaction, immediately before the Set, and fails closed if it cannot
        read it: a cap that cannot be attributed to a stable UUID cannot be
        persisted, relocated, or restored. When `expected_uuid` is supplied it
        is compared against that transaction-local read and a proven mismatch
        also fails closed, so a reconnect between identity capture/resolution
        and the write can never silently relocate the write onto — or report
        success for — a different GPU. The verified identity is returned
        alongside the wattage and threaded into `_record_managed_state` (no
        post-Set re-read), so bookkeeping records the GPU the cap landed on.

        `record_ownership` controls whether a successful write CLAIMS ownership
        (`_managed_gpu_indices` + `_capped_uuids` + persisted UUID). `apply_cap`
        passes True. The restore/release paths pass False: they RELINQUISH a
        cap, so re-recording ownership would leave the UUID in `_capped_uuids`
        and let a later SIGTERM sweep reset a cap another workflow installed on
        that GPU after our release.
        """

        def _write_set() -> tuple[int, str]:
            import dcgm_structs
            import pydcgm

            gpu_id = self._discovered_gpu_ids[gpu_idx]

            # Identity read INSIDE the write transaction, immediately before
            # Set. Runs on the FIRST attempt and every `_with_reconnect` retry
            # (which rebuilds `_discovered_gpu_ids` via `init()`), so the
            # identity we verify AND the identity we hand to bookkeeping is the
            # one the cap actually lands on — no window between the read and
            # the Set, and no post-Set reconnect-capable re-read that a
            # re-enumeration could move onto a different GPU.
            # FAIL CLOSED on any identity we cannot READ. A cap we cannot
            # attribute to a stable UUID cannot be safely owned: it can't be
            # persisted for orphan recovery, relocated on re-enumeration, or
            # restored at SIGTERM. If we wrote it anyway and tracked only the
            # index, a later reconnect that re-enumerated this index onto a
            # different GPU would make the SIGTERM restore un-cap the WRONG GPU
            # while the real one leaks. So refuse the write
            # instead — `apply_cap` turns `_GpuIdentityMismatch` into an
            # `apply_failures_total` tick, and the next reconcile cycle retries
            # once the identity is readable. CONNECTION_NOT_VALID still
            # surfaces to `_with_reconnect` for a reconnect + retry first.
            try:
                observed_uuid = self._read_uuid_raw(gpu_idx)
            except dcgm_structs.DCGMError:
                raise
            except Exception as e:
                raise _GpuIdentityMismatch(
                    f"could not read GPU index {gpu_idx} identity before cap "
                    "write; refusing to apply a cap we cannot attribute to a "
                    "stable GPU (would risk a wrong-GPU restore and a permanent "
                    f"cap leak): {e}"
                ) from e
            # When an identity was captured/resolved before the write, a
            # PROVEN mismatch means a reconnect re-enumerated `gpu_idx` onto a
            # different GPU between capture and this write. Refuse for BOTH
            # callers — an apply must not clobber a re-enumerated GPU with a
            # workload-derived cap, and a restore must not write "default" onto
            # (and then falsely report success for) the wrong GPU, which would
            # make its caller prune the still-capped UUID. When no identity was available (`expected_uuid is None`),
            # `observed_uuid` above IS the identity — read transaction-locally,
            # so bookkeeping still records the GPU the cap actually lands on.
            if expected_uuid is not None and observed_uuid != expected_uuid:
                raise _GpuIdentityMismatch(
                    f"GPU index {gpu_idx} now hosts UUID {observed_uuid} "
                    f"but the cap was computed for UUID {expected_uuid} "
                    "(DCGM re-enumeration during the cap write)"
                )

            grp = self._groups.get(gpu_idx)
            if grp is None:
                grp = pydcgm.DcgmGroup(
                    self._handle,
                    groupName=f"dynamo-power-agent-gpu-{gpu_idx}",
                    groupType=dcgm_structs.DCGM_GROUP_EMPTY,
                )
                grp.AddGpu(gpu_id)
                self._groups[gpu_idx] = grp

            # DCGM_INT32_BLANK lives in `dcgmvalue` (NOT `dcgm_structs`)
            # in the upstream pydcgm bindings — confirmed against DCGM
            # 4.5.3 (`/shared/pydcgm/dcgmvalue.py:17`). An earlier revision
            # reached for `dcgm_structs.DCGM_INT32_BLANK`, which doesn't
            # exist there, and apply_cap raised AttributeError on the
            # first cap write against a real hostengine.
            import dcgmvalue

            cfg = dcgm_structs.c_dcgmDeviceConfig_v2()
            cfg.version = dcgm_structs.dcgmDeviceConfig_version2
            # Blank every field DCGM would otherwise reset out from
            # under us — we only intend to write the power limit.
            cfg.mEccMode = dcgmvalue.DCGM_INT32_BLANK
            cfg.mComputeMode = dcgmvalue.DCGM_INT32_BLANK
            cfg.mPerfState.syncBoost = dcgmvalue.DCGM_INT32_BLANK
            cfg.mPerfState.targetClocks.smClock = dcgmvalue.DCGM_INT32_BLANK
            cfg.mPerfState.targetClocks.memClock = dcgmvalue.DCGM_INT32_BLANK
            # mWorkloadPowerProfiles MUST be explicitly blanked. The
            # ctypes constructor zero-initializes the array, and DCGM's
            # config manager treats an all-zero workload-profile array
            # as ACTION_CLEAR (DcgmConfigManagerTests.cpp:207-231).
            # Without this loop, every cap write would silently wipe
            # whatever workload power profiles the customer or another
            # tool had configured.
            for bitmap_index in range(
                dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE
            ):
                cfg.mWorkloadPowerProfiles[bitmap_index] = dcgmvalue.DCGM_INT32_BLANK
            cfg.mPowerLimit.type = dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL
            cfg.mPowerLimit.val = effective_w

            grp.config.Set(cfg)
            return effective_w, observed_uuid

        # Set is the load-bearing call; failure here means the cap
        # was NOT applied. Propagate so apply_cap can absorb (metric
        # tick) or restore_default can surface to SIGTERM.
        result, observed_uuid = self._with_reconnect(_write_set)

        # Set succeeded → cap is LIVE on the GPU. `observed_uuid` is the
        # identity `_write_set` verified in-transaction (guaranteed non-empty;
        # an unreadable identity fails the write before here), so bookkeeping
        # records the GPU the cap landed on with no post-Set re-read.
        #
        # No follow-up dcgmConfigEnforce: `dcgmConfigSet` already records the
        # cap as DCGM's target configuration, which DCGM auto-reapplies after a
        # GPU reset/reinit; dcgmConfigEnforce would only manually re-assert that
        # already-set target, so it is redundant here.
        #
        # The applied-limit gauge reflects what is LIVE on the GPU right now,
        # so it updates on EVERY successful write — independent of ownership
        # recording. A restore/release writes the factory default with
        # `record_ownership=False`; without this the gauge would keep reporting
        # the released cap forever.
        if self._metrics is not None:
            self._metrics.applied_limit_watts.labels(gpu=str(gpu_idx)).set(result)

        # Restore/release callers pass `record_ownership=False`: they are
        # relinquishing the cap, so claiming ownership here would leave the UUID
        # in `_capped_uuids` for a later sweep to act on.
        if record_ownership:
            self._record_managed_state(gpu_idx, verified_uuid=observed_uuid)

        return result

    def _record_managed_state(self, gpu_idx: int, verified_uuid: str) -> None:
        """Record post-Set OWNERSHIP bookkeeping. Runs on the Set-success path
        only when the caller CLAIMS ownership; the cap is LIVE on the GPU and
        SIGTERM / orphan-recovery must find it. The applied-limit gauge is NOT
        updated here — `_apply_cap_inner` ticks it on every successful write
        (including ownership-less restores) so the metric never goes stale.

        `verified_uuid` is the identity `_write_set` read for `gpu_idx`
        immediately before the Set, inside the same `_with_reconnect`
        transaction. It is persisted as-is — there is NO `get_uuid` lookup
        here, because a post-Set reconnect-capable re-read could re-enumerate
        onto a different GPU and persist the wrong UUID, leaking the cap we
        just applied. It is always a real UUID: `_write_set`
        fails closed (never reaching this bookkeeping) if it cannot read the
        identity, so a cap is never tracked by index alone."""
        import power_agent

        power_agent._managed_gpu_indices.add(gpu_idx)
        try:
            self._managed_uuid_by_idx[gpu_idx] = verified_uuid
            # Ownership record for the SIGTERM sweep (never overwritten on
            # re-enumeration, unlike `_managed_uuid_by_idx[gpu_idx]`).
            self._capped_uuids.add(verified_uuid)
            power_agent._record_managed_gpu_by_uuid(verified_uuid)
        except Exception as e:
            # UUID persistence failure is non-fatal: the cap was applied,
            # only the persistent orphan-recovery record is missed.
            logger.warning(
                "DCGM apply succeeded on GPU %d but UUID persistence failed: %s",
                gpu_idx,
                e,
            )

    def restore_default(self, gpu_idx: int) -> Optional[bool]:
        """Restore factory-default TGP by reading field 163 then re-capping.

        Goes through `_apply_cap_inner` (not `apply_cap`) so DCGM
        write failures propagate to the SIGTERM/orphan-recovery
        callers as exceptions instead of being absorbed. Default is
        in-constraints by definition, so skipping `apply_cap`'s
        re-clamp is safe.
        """
        restore_idx = self._resolve_managed_idx(gpu_idx)
        if restore_idx is None:
            return False
        # Guard the write with the UUID we intend to restore so a reconnect
        # BETWEEN `_resolve_managed_idx` and the Set can't relocate the write
        # onto a different GPU (`_resolve_managed_idx` only proves identity at
        # resolution time). `_apply_cap_inner` FAILS CLOSED — a proven
        # mid-write mismatch OR an unverifiable identity raises
        # `_GpuIdentityMismatch`; treat that like the "not conclusively
        # located" case — return False so the SIGTERM/release caller keeps the
        # UUID (never prunes a still-live cap, never writes "default" onto a
        # GPU we cannot confirm) and cold-start orphan recovery retries
        # . `want_uuid` is None only for the (now
        # near-unreachable) case of a tracked index with no recorded UUID —
        # apply fails closed on an unreadable identity, so every tracked index
        # carries one; even then `_apply_cap_inner` still fails closed if the
        # in-transaction identity read fails, it just skips the mismatch check.
        want_uuid = self._managed_uuid_by_idx.get(gpu_idx)
        try:
            if want_uuid is not None:
                # Bind the default we write to the identity it came from: read
                # both the default limit AND the UUID from ONE snapshot and
                # reject an A->B->A re-enumeration that could otherwise source
                # GPU-B's default and write it onto GPU-A on a heterogeneous
                # node — the trailing `_apply_cap_inner` destination guard alone
                # would still pass. This subsumes the
                # separate `default_w(restore_idx)` read.
                pl, pl_uuid = self._power_limits_with_uuid(restore_idx)
                if pl_uuid != want_uuid:
                    logger.warning(
                        "Skipping default restore for managed GPU %d: the "
                        "resolved index re-enumerated between resolution and "
                        "the power read (%s -> %s); leaving the cap tracked for "
                        "orphan recovery rather than writing another GPU's "
                        "default.",
                        gpu_idx,
                        want_uuid,
                        pl_uuid,
                    )
                    return False
                default_val = self._coerce_power_limit_watts(
                    pl.defaultPowerLimit, "defaultPowerLimit", restore_idx
                )
            else:
                # No recorded UUID for this index (near-unreachable — apply
                # fails closed on an unreadable identity). No anchor to bind
                # provenance to; fall back to the separate read and let
                # `_apply_cap_inner`'s in-transaction identity read guard the
                # write.
                default_val = self.default_w(restore_idx)
            self._apply_cap_inner(
                restore_idx,
                default_val,
                expected_uuid=want_uuid,
                record_ownership=False,
            )
        except _GpuIdentityMismatch as e:
            logger.warning(
                "Skipping default restore for managed GPU %d: could not "
                "confirm the target GPU's identity at write time "
                "(re-enumeration or unreadable), so the cap is left tracked "
                "for orphan recovery rather than pruned: %s",
                gpu_idx,
                e,
            )
            return False
        return True

    def managed_uuids(self) -> set[str]:
        """UUIDs THIS process capped — the ownership scope for the SIGTERM
        UUID sweep. A copy so callers can iterate while pruning the
        persisted set. See `_capped_uuids` and `_shutdown_cleanup`."""
        return set(self._capped_uuids)

    def retire_managed_uuid(self, uuid: str) -> None:
        """Drop `uuid` from this actuator's ownership records after its cap has
        been released and the GPU restored to default.

        `_capped_uuids` is the set the shutdown sweep treats as "capped by this
        process". A runtime release relinquishes ownership, so the UUID must
        leave that set — otherwise a later sweep could reset a cap another
        workflow installed on the GPU after our release.

        Also drop every index that projected to this UUID from BOTH the
        index→UUID map and the shared `_managed_gpu_indices` set. A UUID can be
        projected at more than one index (re-enumeration re-capped its old index
        onto a different GPU); dropping only the caller's index would leave the
        other projection index falsely managed, and its later unrelated occupant
        could then be released on stale integer membership alone.
        """
        import power_agent

        self._capped_uuids.discard(uuid)
        for idx in [i for i, u in self._managed_uuid_by_idx.items() if u == uuid]:
            del self._managed_uuid_by_idx[idx]
            power_agent._managed_gpu_indices.discard(idx)

    def restore_default_by_uuid(self, uuid: str) -> Optional[bool]:
        """Restore the originally-capped `uuid` at its CURRENT index.

        SIGTERM safety net for the re-enumeration re-cap collision.
        The in-memory managed maps (`_managed_gpu_indices`,
        `_managed_uuid_by_idx`) are keyed by integer index, so re-capping an
        index that DCGM re-enumerated onto a *different* physical GPU
        overwrites the displaced GPU's record:

            1. cap GPU-A at idx 0  -> indices={0}, uuid_by_idx={0: A}
            2. DCGM reconnect re-enumerates -> idx 0 is now GPU-B, A moved
            3. re-cap idx 0 (now GPU-B) -> uuid_by_idx[0] overwritten to B;
               the index set still just holds {0}, so A is no longer
               reachable from the index-keyed SIGTERM loop and its cap
               leaks if the agent is removed without a restart.

        The displaced UUID survives in the append-only `_capped_uuids`
        ownership set, so `power_agent._shutdown_cleanup` sweeps
        `managed_uuids()` through here after the index loop to catch any cap
        the index-keyed pass missed. Identity is resolved at restore time,
        so the write always lands on the GPU that actually carries the cap.
        Scoping to `_capped_uuids` (rather than the cross-incarnation
        `managed_gpus.json` set) keeps the sweep from resetting a cap this
        process never applied.

        Returns a three-valued result that both callers
        (`power_agent._shutdown_cleanup` and `_restore_orphaned_gpus_on_startup`)
        interpret identically: retire ownership (prune `uuid` from the
        persisted set) on a CONCLUSIVE ``True`` or ``None``, and RETAIN it on
        an inconclusive ``False``:
          * ``True``  - a live below-default cap was restored. Retire: the cap
            is now gone, so nothing of ours remains on that GPU.
          * ``None``  - the UUID resolved, its identity was confirmed in the
            SAME snapshot as the power reads, and the GPU is already at/above
            default (no live cap of ours to restore); or a clean scan proved
            the GPU is no longer present. Either way it is CONCLUSIVE that no
            cap of ours remains, so retire — keeping a UUID whose cap is proven
            gone would let a future startup clobber a later unrelated cap on the
            same physical GPU.
          * ``False`` - the UUID could not be located conclusively (a
            relocation scan probe raised, e.g. a transient DCGM outage); or
            the resolved index's identity could not be confirmed at write time
            (a proven mid-write re-enumeration OR an unverifiable read — the
            guarded write fails closed); or the identity-bound power read found
            the resolved index re-enumerated onto a different GPU (its snapshot
            UUID no longer matches) or could not be read — so the GPU may still
            carry our cap. Retain the UUID; cold-start orphan recovery retries
            on the next boot.
        """
        idx, scan_complete = self._resolve_idx_for_uuid(uuid)
        if idx is None:
            # Clean scan with no match -> the GPU is gone and its cap left
            # with it (safe to prune). Incomplete scan -> indeterminate, so
            # keep the UUID for the next orphan-recovery attempt.
            return None if scan_complete else False
        # Read current + default limits AND the identity they belong to from
        # ONE snapshot, so the value we may write (the factory default) and the
        # at/above-default decision are both provably GPU `uuid`'s — not a
        # different GPU a re-enumeration parked on `idx` between
        # `_resolve_idx_for_uuid` and this read. This also
        # subsumes the old separate post-read identity recheck: a snapshot whose
        # UUID no longer matches means the index moved, so we fail closed here.
        try:
            pl, pl_uuid = self._power_limits_with_uuid(idx)
        except Exception as e:
            logger.warning(
                "SIGTERM UUID sweep: could not read power limits/identity for "
                "index %d (UUID %s); keeping the UUID for cold-start orphan "
                "recovery: %s",
                idx,
                uuid,
                e,
            )
            return False
        if pl_uuid != uuid:
            logger.warning(
                "SIGTERM UUID sweep: index %d re-enumerated between UUID "
                "resolution and the power reads (%s -> %s); the original GPU may "
                "still carry our cap at its new index. Keeping the UUID for "
                "cold-start orphan recovery.",
                idx,
                uuid,
                pl_uuid,
            )
            return False
        current_w = self._coerce_power_limit_watts(
            pl.curPowerLimit, "curPowerLimit", idx
        )
        default_w = self._coerce_power_limit_watts(
            pl.defaultPowerLimit, "defaultPowerLimit", idx
        )
        if current_w < default_w:
            # Guard the write with `uuid` so a reconnect BETWEEN this read and
            # the Set can't relocate the write onto — and falsely report success
            # for — a different GPU. The write value (`default_w`) came from the
            # same snapshot whose UUID we just confirmed is `uuid`, so it is GPU
            # `uuid`'s own default, not a value a mid-read re-enumeration sourced
            # from another SKU. The guarded write FAILS
            # CLOSED: a proven mid-write mismatch OR an unverifiable identity
            # raises `_GpuIdentityMismatch`; return False so the caller KEEPS
            # `uuid` (a True return would prune it while its cap is still live)
            # and orphan recovery retries on the next boot.
            try:
                self._apply_cap_inner(
                    idx, default_w, expected_uuid=uuid, record_ownership=False
                )
            except _GpuIdentityMismatch as e:
                logger.warning(
                    "SIGTERM UUID sweep: could not confirm index %d still "
                    "hosts UUID %s at write time (re-enumeration or "
                    "unreadable); skipping and keeping the UUID for cold-start "
                    "orphan recovery: %s",
                    idx,
                    uuid,
                    e,
                )
                return False
            logger.warning(
                "SIGTERM UUID sweep restored a cap the index-keyed pass "
                "missed: UUID %s now at index %d (DCGM re-enumeration "
                "re-cap collision).",
                uuid,
                idx,
            )
            return True
        # Same GPU (its UUID was confirmed in the same snapshot as the power
        # reads) and genuinely at/above default: nothing of ours to restore, but
        # sync the gauge to the LIVE value so a GPU restored to default
        # externally stops reporting our old cap.
        if self._metrics is not None:
            self._metrics.applied_limit_watts.labels(gpu=str(idx)).set(current_w)
        return None

    def managed_uuid_for_idx(self, gpu_idx: int) -> str:
        """Return the UUID originally capped for `gpu_idx`, if known.

        `power_agent._shutdown_cleanup` uses this after a successful restore to
        prune `managed_gpus.json`. If DCGM re-enumerated and `restore_default`
        relocated the write, pruning by `get_uuid(gpu_idx)` would remove the
        wrong UUID.
        """
        return self._managed_uuid_by_idx.get(gpu_idx) or self.get_uuid(gpu_idx)

    def _resolve_managed_idx(self, gpu_idx: int) -> Optional[int]:
        """Resolve the *current* index to restore for a managed `gpu_idx`.

        DCGM may re-enumerate after a hostengine restart, so the integer
        index recorded at cap time can point at a different physical GPU by
        SIGTERM. We relocate by the UUID we actually capped.

        The fallback rule turns on whether the index is PROVEN wrong:

        * Identity unreadable (the `get_uuid(gpu_idx)` probe itself raised) —
          mismatch is NOT proven. DCGM may be briefly down; we best-effort
          restore at the original index and let `_apply_cap_inner`'s own
          reconnect-and-retry attempt the write. We have not shown the index
          is wrong, so this cannot target a known-wrong GPU.
        * Proven mismatch (`get_uuid(gpu_idx) != want_uuid`) — the original
          index now hosts a DIFFERENT physical GPU, so writing default TGP
          there would clobber an unrelated GPU AND (because the SIGTERM caller
          prunes `want_uuid` after a "successful" restore) drop the actually-
          capped GPU from `managed_gpus.json`, leaking its cap permanently.
          We therefore NEVER fall back to the original index after a proven
          mismatch: we relocate to the index that currently hosts `want_uuid`,
          or, if it can't be located (gone, or the scan was inconclusive),
          return ``None`` to skip without writing or pruning so cold-start
          orphan recovery retries on the next boot.
        """
        want_uuid = self._managed_uuid_by_idx.get(gpu_idx)
        if want_uuid is None:
            # No recorded identity (UUID read failed at cap time). Nothing to
            # relocate against; restore at the original index.
            return gpu_idx

        try:
            current_uuid = self.get_uuid(gpu_idx)
        except Exception as e:
            # Can't verify identity -> mismatch NOT proven. Do NOT treat this
            # as "GPU gone" (transient hostengine blip vs hardware removal),
            # and the index isn't shown to be wrong, so best-effort restore
            # at the original index; the cap-write retry path may reconnect.
            logger.warning(
                "Could not verify managed GPU index %d (capped UUID %s) "
                "before restore; attempting best-effort restore at the "
                "original index: %s",
                gpu_idx,
                want_uuid,
                e,
            )
            return gpu_idx

        if current_uuid == want_uuid:
            return gpu_idx

        # PROVEN mismatch: gpu_idx now hosts a DIFFERENT physical GPU
        # (current_uuid), so writing default TGP there is wrong. Relocate to
        # the index that currently hosts the capped UUID, or skip entirely —
        # never write to the known-wrong original index.
        relocated_idx, scan_complete = self._resolve_idx_for_uuid(want_uuid)
        if relocated_idx is not None:
            logger.warning(
                "Managed GPU index %d now resolves to UUID %s; restoring "
                "capped UUID %s at its current index %d.",
                gpu_idx,
                current_uuid,
                want_uuid,
                relocated_idx,
            )
            return relocated_idx

        # Could not locate the capped UUID. Skip without writing to the
        # known-wrong original index and without pruning (the SIGTERM caller
        # leaves want_uuid in managed_gpus.json so orphan recovery retries).
        if scan_complete:
            logger.warning(
                "Managed GPU UUID %s is no longer visible after DCGM "
                "re-enumeration (index %d now hosts UUID %s); skipping "
                "default restore. Cold-start orphan recovery will retry.",
                want_uuid,
                gpu_idx,
                current_uuid,
            )
        else:
            logger.warning(
                "Managed GPU UUID %s could not be located after a proven "
                "index mismatch (index %d now hosts UUID %s; relocation scan "
                "incomplete); skipping default restore rather than risk "
                "writing to the wrong GPU. Cold-start orphan recovery will "
                "retry.",
                want_uuid,
                gpu_idx,
                current_uuid,
            )
        return None

    def _resolve_idx_for_uuid(self, uuid: str) -> tuple[Optional[int], bool]:
        """Find the current index whose UUID matches `uuid`.

        Returns ``(idx, scan_complete)``. ``scan_complete`` is True only when
        every GPU was inspected without error, so callers can distinguish a
        positive "not present" (clean scan, no match) from an indeterminate
        result (a probe raised, e.g. a transient DCGM outage). Callers must
        not treat an incomplete scan as proof the GPU is gone.

        The whole scan runs inside ONE ``_with_reconnect`` and reads each index
        with ``_read_uuid_raw`` (no per-index reconnect). This closes a
        false-"gone" hole that the old per-index ``get_uuid`` had: ``get_uuid``
        reconnects on its OWN, and a reconnect can GROW discovery (a hostengine
        restart re-enumerating more GPUs). With the loop bound already fixed at
        the pre-growth length, the target UUID's new index was never visited,
        yet no probe raised — so it returned ``(None, True)`` and callers pruned
        a still-live cap. Here a ``CONNECTION_NOT_VALID`` instead propagates to
        the single outer handler, which rebuilds discovery and re-invokes the
        scan against the NEW topology (length re-materialized on every entry),
        so a growth is fully rescanned. A shrink was already safe (out-of-range
        gpu_ids would raise and mark the scan incomplete).
        """
        import dcgm_structs

        def _scan() -> tuple[Optional[int], bool]:
            scan_complete = True
            for idx in range(len(self._discovered_gpu_ids)):
                try:
                    if self._read_uuid_raw(idx) == uuid:
                        return idx, True
                except dcgm_structs.DCGMError as e:
                    if e.value == dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID:
                        # Let the outer _with_reconnect rebuild the topology and
                        # restart the COMPLETE scan; a partial pre-reconnect scan
                        # cannot be trusted (indices may have moved).
                        raise
                    scan_complete = False
                    logger.warning(
                        "Failed to inspect GPU index %d while resolving managed "
                        "UUID %s: %s",
                        idx,
                        uuid,
                        e,
                    )
                except Exception as e:
                    scan_complete = False
                    logger.warning(
                        "Failed to inspect GPU index %d while resolving managed "
                        "UUID %s: %s",
                        idx,
                        uuid,
                        e,
                    )
            return None, scan_complete

        return self._with_reconnect(_scan)

    def scan_uuid_index_map(self) -> tuple[dict[str, int], bool]:
        """DCGM identity snapshot built inside ONE ``_with_reconnect``.

        Mirrors ``_resolve_idx_for_uuid``'s growth-safety: the scan re-reads
        ``len(self._discovered_gpu_ids)`` on every entry, and a
        ``CONNECTION_NOT_VALID`` propagates to the single outer handler, which
        rebuilds discovery (new topology length) and re-runs the WHOLE scan — so
        a hostengine reconnect that grew the GPU set mid-scan is fully rescanned
        rather than silently leaving newly-enumerated indices unvisited. Any
        other per-index error marks the scan inconclusive; a sustained
        outage / reconnect failure that escapes ``_with_reconnect`` is reported
        as ``({}, False)`` (weak evidence — the caller must not prune on it),
        which also covers the ``device_count() == 0`` outage case the naive
        cached-count path could not distinguish from a genuinely empty node.
        """
        import dcgm_structs

        # An empty cached topology is NOT trustworthy evidence of a GPU-less
        # node on the DCGM path. Power Agent runs on GPU nodes, so zero
        # discovered GPUs far more likely means discovery never populated, or a
        # dropped hostengine connection that the scan below could not surface:
        # `range(0)` issues no hostengine call, so `_with_reconnect` never fires
        # and the loop would return ``({}, True)`` — causing recovery to prune
        # EVERY persisted UUID as absent (a cap-leak / ownership-loss fail-open).
        # Report inconclusive instead so recovery retains persisted UUIDs and
        # retries next boot. (NVML, by contrast, has no reconnect/stale-topology
        # hazard: a clean `nvmlDeviceGetCount()==0` there is genuinely empty.)
        if not self._discovered_gpu_ids:
            logger.warning(
                "DCGM identity scan found an empty topology (0 discovered "
                "GPUs); treating as INCONCLUSIVE so orphan recovery retains "
                "persisted UUIDs this boot rather than pruning them all as "
                "absent."
            )
            return {}, False

        def _scan() -> tuple[dict[str, int], bool]:
            mapping: dict[str, int] = {}
            conclusive = True
            for idx in range(len(self._discovered_gpu_ids)):
                try:
                    mapping[self._read_uuid_raw(idx)] = idx
                except dcgm_structs.DCGMError as e:
                    if e.value == dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID:
                        # Restart the COMPLETE scan against rebuilt topology; a
                        # partial pre-reconnect map cannot be trusted (indices
                        # may have moved AND the set may have grown).
                        raise
                    conclusive = False
                    logger.warning(
                        "Failed to inspect GPU index %d during DCGM identity "
                        "scan; marking scan inconclusive: %s",
                        idx,
                        e,
                    )
                except Exception as e:
                    conclusive = False
                    logger.warning(
                        "Failed to inspect GPU index %d during DCGM identity "
                        "scan; marking scan inconclusive: %s",
                        idx,
                        e,
                    )
            return mapping, conclusive

        try:
            return self._with_reconnect(_scan)
        except Exception as e:
            logger.warning(
                "DCGM identity scan failed (%s: %s); treating as inconclusive "
                "so orphan recovery neither prunes nor restores this boot.",
                type(e).__name__,
                e,
            )
            return {}, False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clamp_with_metrics(
        self, requested_w: int, min_w: int, max_w: int, gpu_idx: int
    ) -> int:
        """SKU clamp matching `power_agent._clamp_to_constraints` semantics.

        Re-implemented (rather than delegated) because the NVML helper
        takes a pynvml handle, which we don't have on this path. The
        metric labels and log messages are identical so dashboards see
        the same shape regardless of actuator.
        """
        if requested_w < min_w:
            logger.warning(
                "Requested cap %d W below SKU min %d W on GPU %d; clamping up.",
                requested_w,
                min_w,
                gpu_idx,
            )
            self._metrics.cap_clamped_total.labels(direction="min").inc()
            return min_w
        if requested_w > max_w:
            logger.warning(
                "Requested cap %d W above SKU max %d W on GPU %d; clamping down.",
                requested_w,
                max_w,
                gpu_idx,
            )
            self._metrics.cap_clamped_total.labels(direction="max").inc()
            return max_w
        return requested_w
