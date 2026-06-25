# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actuator abstraction for the Power Agent.

Defines the `Actuator` Protocol and two implementations:

  - `NvmlActuator` — the path that shipped in PR #9682. Used on
    clusters where the GPU Operator's `dcgm.enabled=false` (default).
    Delegates to the existing module-level NVML helpers in
    `power_agent.py` so the 43 PR A tests pass unchanged.
  - `DcgmActuator` — used on clusters where the GPU Operator's
    `dcgm.enabled=true`. Connects standalone-TCP to the
    `nvidia-dcgm` hostengine and routes per-GPU `dcgmConfigSet`
    + optional `dcgmConfigEnforce` writes through it. PID
    enumeration intentionally stays on NVML because DCGM does not
    expose a snapshot-of-running-PIDs API
    (`DCGM_FI_DEV_COMPUTE_PIDS` is a time-series field).

The two are mutually exclusive at chart-install time — a Power
Agent process binds to exactly one actuator at startup and holds
it for its lifetime.

Lazy imports
------------
Every method that touches `pynvml`, `pydcgm`, `dcgm_structs`,
`dcgm_agent`, or `dcgm_fields` imports it inside the method body
rather than at module top. Three reasons:

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
      - DcgmActuator (opt-in via --actuator=dcgm; added in PR B)

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

    def list_running_pids(self, gpu_idx: int) -> list[int]:
        """Snapshot the compute PIDs currently running on the GPU."""
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

    def apply_cap(self, gpu_idx: int, watts: int) -> int:
        """Write a per-GPU power cap. Returns the effective post-clamp value.

        Return-value contract (v1.6 clarification per review comment #9):

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

        Return False only when the restore was intentionally skipped because
        the managed physical GPU is no longer visible; None/True mean the
        restore path completed.
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
        # power_agent.PowerAgent.__init__ already calls pynvml.nvmlInit()
        # before constructing the actuator in PR A; we accept that as our
        # init contract. PR B's DcgmActuator does its own connection
        # setup here, so the Protocol still earns its keep — PR B will
        # also move NVML init into this method for symmetry.
        return None

    def shutdown(self) -> None:
        # power_agent._handle_sigterm calls pynvml.nvmlShutdown() at
        # SIGTERM time; the actuator's shutdown is a no-op in PR A to
        # avoid a double-shutdown. PR B will broaden this for DCGM and
        # consolidate the NVML side in a follow-up.
        return None

    def device_count(self) -> int:
        import pynvml

        return pynvml.nvmlDeviceGetCount()

    def get_uuid(self, gpu_idx: int) -> str:
        import power_agent
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        return power_agent._nvml_uuid(handle)

    def list_running_pids(self, gpu_idx: int) -> list[int]:
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
        the v1.5 Protocol extension lifted it onto the actuator
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
        call that lived in `_restore_orphaned_gpus_on_startup` and in
        `_handle_sigterm`. Read once, then compared against
        `current_w` by the orphan-recovery guard.
        """
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        return pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) // 1000

    def apply_cap(self, gpu_idx: int, watts: int) -> int:
        """Apply a power cap to GPU `gpu_idx`, returning the effective watts.

        Delegates to `power_agent._apply_cap`, which contains the
        clamping, managed-state tracking, metrics updates, and
        NVMLError handling exercised by `test_apply_cap.py`.
        `_apply_cap` returns `None` (predates the Protocol), so we
        re-derive the post-clamp effective_w from constraints here
        WITHOUT calling `_clamp_to_constraints` again — a previous
        version did, which double-logged the clamp warning and
        double-incremented `cap_clamped_total` for any out-of-range
        request (v1.6 fix per review comment #8). Constraint reads
        are pure (no side effects), so reading min/max here and
        clamping locally is equivalent to peeking at what `_apply_cap`
        will produce.
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

        Mirrors the inline NVML calls in `power_agent._handle_sigterm`
        and `power_agent._restore_orphaned_gpus_on_startup` so
        `DcgmActuator.restore_default` (which reads
        `DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF`) is a peer of this method,
        not a special case.
        """
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
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
        if self.current_w(match_idx) < self.default_w(match_idx):
            self.restore_default(match_idx)
            return True
        # Visible and already at/above default: nothing of ours to restore.
        return None


class DcgmActuator:
    """DCGM-backed actuator — routes per-GPU power caps through `nvidia-dcgm`.

    Used on clusters where the GPU Operator's `dcgm.enabled=true`.
    Connects standalone-TCP to the operator-managed `nvidia-dcgm`
    hostengine, calls `dcgmConfigSet(mPowerLimit.val=W)` for the cap
    write, and optionally calls `dcgmConfigEnforce` to register the
    cap as DCGM's "target configuration" so it is automatically
    re-applied after a GPU reset or reinit
    (`DcgmConfigManager.h:113-117`, `dcgm_test_apis.h:180-183`).
    That reset/reinit auto-reapply is the single resilience property
    DCGM genuinely buys over NVML — there is **no** tick-driven
    re-enforce loop in DCGM source, and `dcgmConfigEnforce` does
    **not** make the cap survive Power Agent restart (the agent's
    SIGTERM handler restores default regardless).

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
        enforce: bool = False,
        metrics: Optional[Any] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._enforce = enforce
        self._metrics = metrics

        # All hostengine state — populated in init(), cleared in
        # shutdown() and _with_reconnect's recovery path.
        self._handle: Optional[Any] = None
        self._system: Optional[Any] = None
        self._discovered_gpu_ids: list[int] = []
        self._groups: dict[int, Any] = {}  # gpu_idx -> DcgmGroup

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
        # a cap owned by another workflow. See `_handle_sigterm`.
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
                "dcgm:4.5.1-1-ubuntu22.04` (or newer 4.x tag); see "
                "deploy/power-agent/Dockerfile."
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
        # cost.
        import pynvml

        pynvml.nvmlInit()

        logger.info(
            "DcgmActuator connected to %s:%d (enforce=%s, %d GPUs)",
            self._host,
            self._port,
            self._enforce,
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
        best-effort and callers (`power_agent._handle_sigterm`) need
        to proceed to `_shutdown.set()` regardless so the container
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
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            logger.exception(
                "pynvml.nvmlShutdown() failed during DcgmActuator shutdown; "
                "continuing shutdown.",
            )

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
                # the failure is still visible in outage diagnosis (per
                # PR9790 review on silent exception swallowing), then
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
        return len(self._discovered_gpu_ids)

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
        cross-library identity mapping pre-v1.10. `GetGpuAttributes` is
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

    def list_running_pids(self, gpu_idx: int) -> list[int]:
        """Snapshot of compute PIDs on the GPU — via NVML, even on the DCGM path.

        DCGM has no public snapshot-of-running-PIDs API. See the class
        docstring for the full reasoning.
        Lazy-imports pynvml so this method is callable from tests that
        mock the import.

        Identity mapping
        ----------------
        `gpu_idx` is the DCGM-ordered index used throughout the
        reconcile loop, NOT an NVML index. We translate via UUID
        (DCGM's gpuId -> UUID via _dcgm_uuid_by_idx, then UUID ->
        NVML index via _nvml_index_by_uuid) because the two libraries
        live in separate identity spaces — see DcgmCacheManager.cpp:
        1230-1296.
        """
        import pynvml

        self._ensure_identity_map()
        uuid = self._dcgm_uuid_by_idx[gpu_idx]
        try:
            nvml_idx = self._nvml_index_by_uuid[uuid]
        except KeyError as err:
            # Should not happen — _ensure_identity_map raises loudly
            # on missing UUIDs at build time. If we get here, a GPU
            # disappeared from NVML's view between identity-map build
            # and now (driver crash, hot-unplug). Surface it instead
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
        watched, which silently broke the v1.8 implementation on
        every fresh hostengine. `GetGpuAttributes` works on any
        hostengine the moment GPU discovery completes.
        """

        def _op() -> "Any":
            gpu_id = self._discovered_gpu_ids[gpu_idx]
            return self._system.discovery.GetGpuAttributes(gpu_id).powerLimits

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

        blank_values = set()
        try:
            import dcgmvalue

            blank_values.update(
                {
                    float(dcgmvalue.DCGM_INT32_BLANK),
                    float(dcgmvalue.DCGM_INT64_BLANK),
                    float(dcgmvalue.DCGM_FP64_BLANK),
                }
            )
        except Exception:
            # Unit tests and production DCGM paths patch/import dcgmvalue.
            # If that import is unavailable here, the finite/positive checks
            # below still catch the dangerous non-numeric cases.
            pass

        if numeric in blank_values:
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

    def apply_cap(self, gpu_idx: int, watts: int) -> int:
        """Write a per-GPU cap via `dcgmConfigSet`; return the effective watts.

        Mirrors `power_agent._apply_cap`'s clamp + metrics + state-track
        contract on the DCGM side. DCGMError other than
        CONNECTION_NOT_VALID is absorbed into `apply_failures_total`
        and the call returns the effective post-clamp watts regardless
        (per the Actuator Protocol). SIGTERM / orphan-recovery callers want write
        failures surfaced as exceptions instead — they call
        `restore_default`, which uses `_apply_cap_inner`.
        """
        if self._metrics is None:
            raise RuntimeError(
                "DcgmActuator.apply_cap requires a metrics object; "
                "construct as DcgmActuator(..., metrics=power_agent_metrics)."
            )

        # Capture the identity of the GPU we are about to cap, BEFORE the
        # write. A DCGM hostengine reconnect inside `_apply_cap_inner`
        # (the `_with_reconnect` retry after CONNECTION_NOT_VALID) can
        # re-enumerate indices, so the write must verify that `gpu_idx`
        # still hosts this same UUID — otherwise a cap derived from this
        # GPU's workload could land on whatever GPU took over the index.
        # Best-effort: if the identity is unreadable here the mismatch is
        # NOT proven, so we proceed and let the write attempt happen
        # (mirrors `_resolve_managed_idx`'s "unproven mismatch" stance).
        try:
            expected_uuid: Optional[str] = self.get_uuid(gpu_idx)
        except Exception as e:
            logger.warning(
                "Could not read GPU %d identity before cap write; "
                "proceeding without the re-enumeration guard: %s",
                gpu_idx,
                e,
            )
            expected_uuid = None

        # Clamp against the SKU range. constraints_w handles its own
        # stale-handle recovery, so we don't re-wrap it here.
        min_w, max_w = self.constraints_w(gpu_idx)
        effective_w = self._clamp_with_metrics(watts, min_w, max_w, gpu_idx)

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
            # Narrow on purpose (PR9790 review): only DCGM write errors
            # are part of the "cap-write failed, log + bump metric +
            # return effective_w" contract. AttributeError on the
            # pydcgm bindings (we hit one such bug at v1.10 around
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
        self, gpu_idx: int, effective_w: int, expected_uuid: Optional[str] = None
    ) -> int:
        """Inner cap-write path that propagates Set failures as exceptions.

        Precondition: `effective_w` is already clamped to SKU range.

        `expected_uuid` (set only by `apply_cap`, never by the restore
        paths which do their own index relocation) is the GPU identity
        captured before the write. It is re-checked inside `_write_set`
        AFTER any `_with_reconnect` re-init, so a re-enumeration that moved
        `gpu_idx` onto a different GPU raises `_GpuIdentityMismatch` instead
        of writing this GPU's cap onto the wrong one.
        Bookkeeping (`_managed_gpu_indices`, UUID persistence,
        `applied_limit_watts`) runs immediately after a successful
        Set — even if the optional Enforce that follows fails. The
        cap is LIVE on the GPU as soon as Set returns (DCGM invokes
        `nvmlDeviceSetPowerManagementLimit` synchronously inside
        `dcgmConfigSet`), so we MUST track it for SIGTERM and
        orphan-recovery regardless of whether the target-config
        registration via Enforce succeeded. Extracted from `apply_cap`
        so `restore_default` can surface Set failures to SIGTERM, and
        split from Enforce so partial success (Set OK, Enforce failed)
        doesn't drop managed-state tracking.
        """

        def _write_set() -> int:
            import dcgm_structs
            import pydcgm

            gpu_id = self._discovered_gpu_ids[gpu_idx]

            # Re-enumeration guard. This runs on the FIRST attempt and on
            # any post-reconnect retry (both go through `_with_reconnect`).
            # On the retry, `_discovered_gpu_ids` was rebuilt by `init()`,
            # so reading the UUID raw (no nested reconnect) tells us whether
            # `gpu_idx` still hosts the GPU this cap was computed for. Once an
            # `expected_uuid` was captured at apply entry, BOTH a proven
            # mismatch AND an unverifiable recheck abort the write: an apply
            # writes a workload-derived cap, so we refuse rather than risk
            # clobbering a re-enumerated GPU, and let the next reconcile cycle
            # retry. (Entry-time unreadability yields `expected_uuid is None`
            # — nothing to protect — and proceeds best-effort below.)
            if expected_uuid is not None:
                try:
                    current_uuid = self._read_uuid_raw(gpu_idx)
                except dcgm_structs.DCGMError:
                    # Surface to `_with_reconnect` (CONNECTION_NOT_VALID ->
                    # reconnect + retry; anything else -> apply_cap's
                    # DCGMError handler). Identity not proven wrong.
                    raise
                except Exception as e:
                    # We captured an identity at apply entry but now cannot
                    # confirm `gpu_idx` still hosts it. Unlike the RESTORE
                    # path (which writes the harmless factory default and so
                    # best-efforts an unprovable mismatch), an apply writes a
                    # workload-derived cap, so a write here risks clobbering a
                    # re-enumerated GPU we cannot rule out. Refuse the write
                    # and let the next reconcile cycle re-attribute and retry
                    # (PR9790 review follow-up). Surfaced as an apply failure
                    # by apply_cap's `_GpuIdentityMismatch` handler.
                    raise _GpuIdentityMismatch(
                        f"could not verify GPU index {gpu_idx} identity before "
                        f"cap write (expected UUID {expected_uuid}); refusing "
                        f"to write rather than risk a wrong-GPU clobber: {e}"
                    ) from e
                if current_uuid != expected_uuid:
                    raise _GpuIdentityMismatch(
                        f"GPU index {gpu_idx} now hosts UUID {current_uuid} "
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
            # 4.5.3 (`/shared/pydcgm/dcgmvalue.py:17`). Pre-v1.10 we
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
            return effective_w

        # Set is the load-bearing call; failure here means the cap
        # was NOT applied. Propagate so apply_cap can absorb (metric
        # tick) or restore_default can surface to SIGTERM.
        result = self._with_reconnect(_write_set)

        # Set succeeded → cap is LIVE on the GPU. Record managed state
        # before attempting Enforce so that if Enforce fails we still
        # know we're tracking this GPU.
        self._record_managed_state(gpu_idx, result)

        # Optional: register the cap as DCGM's "target configuration"
        # so the hostengine auto-reapplies after GPU reset/reinit. A
        # failure here does NOT mean the cap is gone — Set already
        # made it live. We log + tick a dedicated metric so dashboards
        # can distinguish "Enforce-only failure (cap is live)" from
        # "apply_failures_total (cap is NOT live)".
        if self._enforce:

            def _write_enforce() -> None:
                # Re-look up the group: _with_reconnect recovery would
                # have cleared self._groups, so we need to tolerate
                # absence here (the group is re-created lazily by
                # the next _write_set or this lookup raises clearly).
                grp = self._groups.get(gpu_idx)
                if grp is None:
                    raise RuntimeError(
                        f"DcgmActuator: group cache for GPU {gpu_idx} was "
                        "cleared between Set and Enforce; recovery in flight."
                    )
                grp.config.Enforce()

            try:
                self._with_reconnect(_write_enforce)
            except Exception as e:
                logger.warning(
                    "dcgmConfigEnforce failed for GPU %d after successful Set "
                    "(cap is live and tracked; only auto-reapply-after-reset "
                    "target-config registration is missing): %s",
                    gpu_idx,
                    e,
                )
                if self._metrics is not None:
                    self._metrics.dcgm_enforce_failures_total.inc()

        return result

    def _record_managed_state(self, gpu_idx: int, watts: int) -> None:
        """Record post-Set bookkeeping. Shared by Set-success path and
        the Set-OK-Enforce-failed path — both leave the cap LIVE on
        the GPU, both need SIGTERM / orphan-recovery to find it."""
        import power_agent

        power_agent._managed_gpu_indices.add(gpu_idx)
        try:
            uuid = self.get_uuid(gpu_idx)
            self._managed_uuid_by_idx[gpu_idx] = uuid
            # Ownership record for the SIGTERM sweep (never overwritten on
            # re-enumeration, unlike `_managed_uuid_by_idx[gpu_idx]`).
            self._capped_uuids.add(uuid)
            power_agent._record_managed_gpu_by_uuid(uuid)
        except Exception as e:
            # UUID lookup failure is non-fatal: the cap was applied,
            # only the persistent orphan-recovery record is missed.
            logger.warning(
                "DCGM apply succeeded on GPU %d but UUID persistence failed: %s",
                gpu_idx,
                e,
            )
        if self._metrics is not None:
            self._metrics.applied_limit_watts.labels(gpu=str(gpu_idx)).set(watts)

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
        self._apply_cap_inner(restore_idx, self.default_w(restore_idx))
        return True

    def managed_uuids(self) -> set[str]:
        """UUIDs THIS process capped — the ownership scope for the SIGTERM
        UUID sweep. A copy so callers can iterate while pruning the
        persisted set. See `_capped_uuids` and `_handle_sigterm`."""
        return set(self._capped_uuids)

    def restore_default_by_uuid(self, uuid: str) -> Optional[bool]:
        """Restore the originally-capped `uuid` at its CURRENT index.

        SIGTERM safety net for the re-enumeration re-cap collision raised
        in PR9790 review. The in-memory managed maps (`_managed_gpu_indices`,
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
        ownership set, so `power_agent._handle_sigterm` sweeps
        `managed_uuids()` through here after the index loop to catch any cap
        the index-keyed pass missed. Identity is resolved at restore time,
        so the write always lands on the GPU that actually carries the cap.
        Scoping to `_capped_uuids` (rather than the cross-incarnation
        `managed_gpus.json` set) keeps the sweep from resetting a cap this
        process never applied.

        Returns (the caller prunes `uuid` from the persisted set ONLY on
        ``True`` — see `power_agent._handle_sigterm` for why ``None`` is
        intentionally kept rather than pruned):
          * ``True``  - a live below-default cap was restored.
          * ``None``  - the UUID resolved but the GPU is already at/above
            default (no live cap of ours to restore), or a clean scan
            proved the GPU is no longer present. Nothing to restore.
          * ``False`` - the UUID could not be located conclusively (a
            relocation scan probe raised, e.g. a transient DCGM outage), so
            the GPU may still carry our cap; cold-start orphan recovery
            retries on the next boot.
        """
        idx, scan_complete = self._resolve_idx_for_uuid(uuid)
        if idx is None:
            # Clean scan with no match -> the GPU is gone and its cap left
            # with it (safe to prune). Incomplete scan -> indeterminate, so
            # keep the UUID for the next orphan-recovery attempt.
            return None if scan_complete else False
        if self.current_w(idx) < self.default_w(idx):
            self._apply_cap_inner(idx, self.default_w(idx))
            logger.warning(
                "SIGTERM UUID sweep restored a cap the index-keyed pass "
                "missed: UUID %s now at index %d (DCGM re-enumeration "
                "re-cap collision).",
                uuid,
                idx,
            )
            return True
        # Visible and already at/above default: nothing of ours to restore.
        return None

    def managed_uuid_for_idx(self, gpu_idx: int) -> str:
        """Return the UUID originally capped for `gpu_idx`, if known.

        `power_agent._handle_sigterm` uses this after a successful restore to
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
        """
        scan_complete = True
        for idx in range(len(self._discovered_gpu_ids)):
            try:
                if self.get_uuid(idx) == uuid:
                    return idx, True
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
