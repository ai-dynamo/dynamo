# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DcgmActuator tests.

Asserts the DCGM path's hostengine interactions, deliberately
asymmetric reads (NVML for PIDs, DCGM for caps), and stale-handle
recovery against a fully-mocked pydcgm surface. Test scope:

  * Protocol satisfaction (DcgmActuator implements Actuator).
  * Lifecycle: init() opens the hostengine + runs nvmlInit; shutdown()
    tears down both, defensively against an already-gone hostengine.
  * Read surface: get_uuid (decodes bytes→str), constraints_w
    (reads MIN=161 + MAX=162), list_running_pids (uses NVML, NOT DCGM
    — the most load-bearing test in this file).
  * Write surface: apply_cap clamps the request, writes the
    `c_dcgmDeviceConfig_v2.mPowerLimit.val` via dcgmConfigSet ONLY (the
    agent never issues the redundant dcgmConfigEnforce re-assert), updates
    module-level managed-state, and bumps apply_failures_total on
    hostengine errors.
  * restore_default reads DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF (field 163,
    not MAX=162) — guards against reading the wrong power-limit field.
  * Stale-handle recovery: CONNECTION_NOT_VALID triggers single retry
    after rebuilding handle + flushing group cache. Other DCGMErrors
    propagate without retry.

Mocked at the import boundary: pydcgm/dcgm_structs/dcgm_agent/dcgm_fields
are not real Python packages in the dev/CI environment (they ship
inside the vendored DCGM container image). DcgmActuator's lazy
`import pydcgm` inside method bodies makes patch.dict('sys.modules')
clean.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from actuator import Actuator, DcgmActuator, _GpuIdentityMismatch

# ---------------------------------------------------------------------------
# Mock factories. Centralised so individual tests stay short and the shape of
# the mocked DCGM surface is documented in one place.
# ---------------------------------------------------------------------------


def _make_dcgm_modules():
    """Build a fresh set of mocked pydcgm/dcgm_structs/dcgm_agent/dcgm_fields.

    Returns a dict suitable for `patch.dict("sys.modules", ...)`. Each
    call returns brand-new MagicMocks so cross-test contamination is
    impossible. Constants get real numeric values so any code path that
    compares them (e.g. `e.value != DCGM_ST_CONNECTION_NOT_VALID`)
    works correctly.
    """

    class DCGMError(Exception):
        """Real exception subclass — needed so try/except in production
        code catches actual instances, not MagicMock surrogates."""

        def __init__(self, value):
            super().__init__(f"DCGMError({value})")
            self.value = value

    dcgm_structs = MagicMock()
    dcgm_structs.DCGMError = DCGMError
    # Sentinel values — actual upstream constants differ but the only
    # property we rely on is that they're distinct and DCGMError.value
    # comparisons work.
    dcgm_structs.DCGM_ST_OK = 0
    dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID = -42
    dcgm_structs.DCGM_ST_GENERIC_ERROR = -1
    dcgm_structs.DCGM_OPERATION_MODE_AUTO = 1
    dcgm_structs.DCGM_GROUP_EMPTY = 0
    dcgm_structs.DCGM_CONFIG_POWER_CAP_INDIVIDUAL = 0
    dcgm_structs.dcgmDeviceConfig_version2 = 0x02000000
    # Matches upstream dcgm_structs.py:1475. Production code uses this
    # to bound the mWorkloadPowerProfiles blanking loop; if it's wrong
    # in the mock, the loop runs the wrong number of times and the
    # workload-profile blanking test catches that.
    dcgm_structs.DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE = 8

    # Each call returns a *new* config object so apply_cap's struct
    # initialization writes don't leak across test runs.
    def _new_config():
        cfg = MagicMock()
        cfg.mPerfState = MagicMock()
        cfg.mPerfState.targetClocks = MagicMock()
        cfg.mPowerLimit = MagicMock()
        return cfg

    dcgm_structs.c_dcgmDeviceConfig_v2 = MagicMock(side_effect=_new_config)

    dcgm_fields = MagicMock()
    # These constants match upstream `dcgm_fields.py` field IDs exactly
    # — load-bearing because production code passes them as field IDs
    # and the tests assert they're the *right* IDs (esp. DEF=163 vs
    # MAX=162).
    dcgm_fields.DCGM_FE_GPU = 1
    dcgm_fields.DCGM_FI_DEV_UUID = 54
    dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT = 160
    dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN = 161
    dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX = 162
    dcgm_fields.DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF = 163

    # DCGM_INT32_BLANK lives in `dcgmvalue` (NOT `dcgm_structs`) per
    # the upstream pydcgm bindings — see
    # /shared/pydcgm/dcgmvalue.py:17 in the DCGM 4.5.3 release.
    # apply_cap reaches for `dcgmvalue.DCGM_INT32_BLANK` exactly the way
    # the production code does.
    dcgmvalue = MagicMock()
    dcgmvalue.DCGM_INT32_BLANK = 0x7FFFFFF0
    dcgmvalue.DCGM_INT64_BLANK = 0x7FFFFFFFFFFFFFF0
    dcgmvalue.DCGM_FP64_BLANK = 140737488355328.0
    dcgmvalue.DCGM_STR_BLANK = "<<<NULL>>>"

    return {
        "pydcgm": MagicMock(),
        "dcgm_structs": dcgm_structs,
        "dcgm_agent": MagicMock(),
        "dcgm_fields": dcgm_fields,
        "dcgmvalue": dcgmvalue,
    }


def _make_gpu_attrs(
    uuid, *, min_w=100, max_w=700, default_w=700, current_w=700, enforced_w=700
):
    """Build a `c_dcgmDeviceAttributes_v3` shape mock.

    Mirrors the upstream struct DcgmActuator reads via
    `DcgmSystem.discovery.GetGpuAttributes(gpu_id)`. Two members are
    actually accessed by production code:

      - `.identifiers.uuid` — used by `get_uuid` and
        `_ensure_identity_map`.
      - `.powerLimits.{cur,default,enforced,min,max}PowerLimit` —
        used by `constraints_w`, `current_w`, `default_w`
        (all three read from this one struct).

    The fields are integers (watts) in the real struct
    (`c_dcgmDevicePowerLimits_v1`). Mock defaults match the values
    observed on the A100 e2e parity rig so apply_cap clamp math
    works without per-test overrides.
    """
    attrs = MagicMock()
    attrs.identifiers = MagicMock()
    attrs.identifiers.uuid = uuid
    attrs.powerLimits = MagicMock()
    attrs.powerLimits.minPowerLimit = min_w
    attrs.powerLimits.maxPowerLimit = max_w
    attrs.powerLimits.defaultPowerLimit = default_w
    attrs.powerLimits.curPowerLimit = current_w
    attrs.powerLimits.enforcedPowerLimit = enforced_w
    return attrs


def _wire_handle(pydcgm_mock, gpu_ids=(0, 1), uuid_by_gpu_id=None):
    """Make pydcgm.DcgmHandle(...) return a usable handle mock.

    `handle.handle` is what dcgm_agent calls receive as their first
    arg. `handle.GetSystem().discovery.GetAllGpuIds()` is what
    `DcgmActuator.init` calls to populate `_discovered_gpu_ids`.
    `handle.GetSystem().discovery.GetGpuAttributes(gpu_id)` is the
    single device-info API used by `get_uuid`, `constraints_w`,
    `current_w`, `default_w`, and `_ensure_identity_map` — the
    static-read consolidation (see `actuator.DcgmActuator._power_limits` and
    `get_uuid` docstrings for why all static reads go through this
    one struct, not through `dcgmEntityGetLatestValues`).

    `uuid_by_gpu_id` (dict) lets a test customize what UUID each
    GPU ID returns. Default: `gpu_id -> f"GPU-uuid-{gpu_id}"`.
    """
    handle = MagicMock()
    handle.handle = 0xDEADBEEF
    discovery = handle.GetSystem.return_value.discovery
    discovery.GetAllGpuIds.return_value = list(gpu_ids)
    if uuid_by_gpu_id is None:
        uuid_by_gpu_id = {gid: f"GPU-uuid-{gid}".encode() for gid in gpu_ids}
    discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
        uuid_by_gpu_id[gid]
    )
    pydcgm_mock.DcgmHandle.return_value = handle
    return handle


def _make_value(*, str_val=None, dbl_val=None):
    """Build a vals[i] mock with .value.str / .value.dbl set.

    `dcgmEntityGetLatestValues` returns a list of structs; production
    code accesses `.value.str` for UUID and `.value.dbl` for numeric
    fields. Mocking both keeps the helper reusable across read tests.
    """
    val = MagicMock()
    val.value = MagicMock()
    if str_val is not None:
        val.value.str = str_val
    if dbl_val is not None:
        val.value.dbl = dbl_val
    return val


def _make_initialized_actuator(
    modules=None, gpu_ids=(0, 1), metrics=None, uuid_by_gpu_id=None
):
    """Build + init() a DcgmActuator with a fully-wired set of mocks.

    Returns (actuator, modules, handle, nvml) so tests can assert on
    calls against any of the mocks. Saves ~10 lines of setup per test.

    `uuid_by_gpu_id` is forwarded to `_wire_handle` so tests that
    exercise `get_uuid` can pin the value GetGpuAttributes returns per
    gpu_id without re-wiring the discovery mock by hand.
    """
    modules = modules or _make_dcgm_modules()
    metrics = metrics or MagicMock()
    actuator = DcgmActuator(host="test", port=5555, metrics=metrics)
    handle = _wire_handle(
        modules["pydcgm"], gpu_ids=gpu_ids, uuid_by_gpu_id=uuid_by_gpu_id
    )
    nvml = MagicMock()
    with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
        actuator.init()
    return actuator, modules, handle, nvml


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


class TestProtocolSatisfaction(unittest.TestCase):
    """DcgmActuator must structurally satisfy Actuator (@runtime_checkable)."""

    def test_dcgm_actuator_is_instance_of_actuator(self):
        # Construction must NOT touch DCGM (it's lazy until init()).
        self.assertIsInstance(DcgmActuator(), Actuator)

    def test_name_attribute_is_dcgm(self):
        self.assertEqual(DcgmActuator.name, "dcgm")
        self.assertEqual(DcgmActuator().name, "dcgm")

    def test_all_protocol_methods_present(self):
        actuator = DcgmActuator()
        for method in (
            "init",
            "shutdown",
            "device_count",
            "get_uuid",
            "list_running_pids",
            "constraints_w",
            "current_w",
            "default_w",
            "apply_cap",
            "restore_default",
            "restore_default_by_uuid",
            "scan_uuid_index_map",
        ):
            self.assertTrue(
                callable(getattr(actuator, method, None)),
                f"DcgmActuator missing or non-callable method: {method}",
            )

    def test_default_host_and_port_match_gpu_operator(self):
        """Defaults must match upstream nvidia-dcgm Service."""
        self.assertEqual(
            DcgmActuator.DEFAULT_HOST,
            "nvidia-dcgm.gpu-operator.svc.cluster.local",
        )
        self.assertEqual(DcgmActuator.DEFAULT_PORT, 5555)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestInit(unittest.TestCase):
    def test_init_opens_handle_and_runs_nvml_init(self):
        modules = _make_dcgm_modules()
        handle = _wire_handle(modules["pydcgm"], gpu_ids=(0, 1, 2, 3))
        nvml = MagicMock()
        actuator = DcgmActuator(
            host="hostengine.example", port=5555, metrics=MagicMock()
        )
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.init()

        modules["pydcgm"].DcgmHandle.assert_called_once()
        args, kwargs = modules["pydcgm"].DcgmHandle.call_args
        self.assertEqual(kwargs["ipAddress"], "hostengine.example:5555")
        self.assertEqual(kwargs["timeoutMs"], 5000)
        self.assertEqual(actuator._discovered_gpu_ids, [0, 1, 2, 3])
        # Both NVML and DCGM are initialized on the DCGM path because
        # list_running_pids uses NVML even when --actuator=dcgm.
        nvml.nvmlInit.assert_called_once_with()
        # Discovery is sorted so device_count→gpu_id mapping is stable.
        handle.GetSystem.return_value.discovery.GetAllGpuIds.assert_called_once_with()

    def test_init_sorts_discovered_gpu_ids(self):
        """If DCGM returns IDs out of order, we must sort them — the
        gpu_idx → gpu_id mapping is positional and unsorted IDs would
        scramble the mapping seen by callers (annotation lookup, etc.)."""
        modules = _make_dcgm_modules()
        _wire_handle(modules["pydcgm"], gpu_ids=(3, 0, 2, 1))
        actuator = DcgmActuator(metrics=MagicMock())
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            actuator.init()
        self.assertEqual(actuator._discovered_gpu_ids, [0, 1, 2, 3])

    def test_init_rejects_dcgm_3x_bindings_with_actionable_error(self):
        """`_apply_cap_inner` uses `dcgm_structs.c_dcgmDeviceConfig_v2`,
        which only exists in DCGM 4.x. With 3.x bindings the agent
        would connect cleanly, run one reconcile tick worth of cap
        writes (raising AttributeError on the first GPU), and exit
        mid-tick — leaving some GPUs capped and the SIGTERM-restore
        path unable to run because the actuator never finished
        registering. init() must fail BEFORE opening the hostengine
        connection with a message that names the missing struct and
        points at the Dockerfile fix.
        """
        modules = _make_dcgm_modules()
        # Simulate 3.x bindings by deleting the 4.x-only struct from the
        # mocked module surface — MagicMock's `del` semantics flip
        # subsequent `hasattr(...)` to False, exactly what the guard
        # in `DcgmActuator.init` checks.
        del modules["dcgm_structs"].c_dcgmDeviceConfig_v2
        _wire_handle(modules["pydcgm"], gpu_ids=(0,))
        actuator = DcgmActuator(host="hostengine.example", port=5555)
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(RuntimeError) as ctx:
                actuator.init()
        msg = str(ctx.exception)
        self.assertIn("DCGM >= 4.0", msg)
        self.assertIn("c_dcgmDeviceConfig_v2", msg)
        self.assertIn("DCGM_IMAGE", msg)
        # Guard must fail BEFORE any pydcgm calls so misconfigured
        # deployments don't half-init and leave hostengine sockets
        # dangling. DcgmHandle is the first pydcgm-touching call.
        modules["pydcgm"].DcgmHandle.assert_not_called()

    def test_nvml_init_runs_once_across_reconnects(self):
        """init() runs on EVERY `_with_reconnect` recovery, but nvmlInit is
        refcounted and shutdown() runs once — so nvmlInit must fire at most
        once per actuator lifetime, else each reconnect leaks an NVML refcount
        past our single nvmlShutdown. Calling init()
        repeatedly stands in for the reconnect-driven rebuilds."""
        modules = _make_dcgm_modules()
        _wire_handle(modules["pydcgm"], gpu_ids=(0, 1))
        nvml = MagicMock()
        actuator = DcgmActuator(host="h", port=5555, metrics=MagicMock())
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.init()
            actuator.init()  # simulate a _with_reconnect rebuild
            actuator.init()
        nvml.nvmlInit.assert_called_once_with()

    def test_nvml_reinitializes_after_full_shutdown(self):
        """A paired shutdown() calls nvmlShutdown and clears the guard, so a
        subsequent init() (a fresh lifetime) must nvmlInit again — the guard
        prevents leaks, it must not prevent legitimate re-initialization."""
        modules = _make_dcgm_modules()
        _wire_handle(modules["pydcgm"], gpu_ids=(0,))
        nvml = MagicMock()
        actuator = DcgmActuator(host="h", port=5555, metrics=MagicMock())
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.init()
            actuator.shutdown()
            actuator.init()
        self.assertEqual(nvml.nvmlInit.call_count, 2)
        nvml.nvmlShutdown.assert_called_once()


class TestShutdown(unittest.TestCase):
    def test_shutdown_releases_groups_handle_and_nvml(self):
        actuator, modules, handle, nvml = _make_initialized_actuator()
        # Seed a cached group so we can assert .Delete() is called.
        group_a = MagicMock()
        actuator._groups[0] = group_a

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.shutdown()

        group_a.Delete.assert_called_once()
        handle.Shutdown.assert_called_once()
        nvml.nvmlShutdown.assert_called_once()
        # State should be cleared so a subsequent init() rebuilds cleanly.
        self.assertEqual(actuator._groups, {})
        self.assertIsNone(actuator._handle)

    def test_shutdown_is_idempotent_against_dead_hostengine(self):
        """If the hostengine pod is already gone, shutdown() must not
        raise — SIGTERM ordering can leave us reaching for a dead
        socket. group.Delete and handle.Shutdown should be swallowed."""
        actuator, modules, handle, nvml = _make_initialized_actuator()
        group = MagicMock()
        group.Delete.side_effect = Exception("hostengine gone")
        actuator._groups[0] = group
        handle.Shutdown.side_effect = Exception("hostengine gone")

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.shutdown()  # must not raise

        self.assertIsNone(actuator._handle)

    def test_shutdown_logs_group_delete_failure(self):
        """Per PR #9682 CodeRabbit review on actuator.py shutdown.

        Pre-fix the group.Delete() failure was silently swallowed
        (`except Exception: pass`). We still don't re-raise (cleanup
        is best-effort) but the traceback must appear in pod logs so
        operators can diagnose DCGM resource leaks.
        """
        actuator, modules, handle, nvml = _make_initialized_actuator()
        group = MagicMock()
        group.Delete.side_effect = Exception("group delete failed")
        actuator._groups[7] = group  # specific gpu_id we'll look for in log

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertLogs("power_agent.actuator", level="ERROR") as cm:
                actuator.shutdown()

        joined = "\n".join(cm.output)
        self.assertIn("group delete failed", joined)
        # The gpu_id context must be in the log message so operators
        # can spot WHICH GPU's group leaked.
        self.assertIn("gpu_id=7", joined)

    def test_shutdown_logs_hostengine_shutdown_failure(self):
        """`handle.Shutdown()` failure (e.g. DCGM_ST_CONNECTION_NOT_VALID)
        must be logged with host/port context, not silently dropped."""
        actuator, modules, handle, nvml = _make_initialized_actuator()
        handle.Shutdown.side_effect = Exception("connection not valid")

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertLogs("power_agent.actuator", level="ERROR") as cm:
                actuator.shutdown()

        joined = "\n".join(cm.output)
        self.assertIn("connection not valid", joined)
        # host:port context should let operators correlate with
        # specific hostengine pod/service.
        self.assertIn(actuator._host, joined)

    def test_shutdown_logs_nvml_shutdown_failure(self):
        """`pynvml.nvmlShutdown()` failure during DcgmActuator.shutdown()
        must surface — pre-fix it was silently dropped."""
        actuator, modules, handle, nvml = _make_initialized_actuator()
        nvml.nvmlShutdown.side_effect = Exception("nvml teardown failed")

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertLogs("power_agent.actuator", level="ERROR") as cm:
                actuator.shutdown()

        joined = "\n".join(cm.output)
        self.assertIn("nvml teardown failed", joined)


# ---------------------------------------------------------------------------
# Read surface
# ---------------------------------------------------------------------------


class TestDeviceCount(unittest.TestCase):
    def test_device_count_returns_len_of_discovered_ids(self):
        actuator, *_ = _make_initialized_actuator(gpu_ids=(0, 1, 2, 3))
        self.assertEqual(actuator.device_count(), 4)

    def test_device_count_zero_when_no_gpus(self):
        actuator, *_ = _make_initialized_actuator(gpu_ids=())
        self.assertEqual(actuator.device_count(), 0)


class TestGetUuid(unittest.TestCase):
    """get_uuid MUST use the synchronous device-info API.

    An earlier implementation read UUID via `dcgmEntityGetLatestValues`
    with field 54 (DCGM_FI_DEV_UUID). That pulls from DCGM's field cache,
    which only populates when *some* DCGM consumer has previously
    subscribed via `dcgmWatchFields`. On a fresh hostengine with no
    companion watcher (standalone `nv-hostengine`, or a production
    cluster whose `nvidia-dcgm-exporter` doesn't watch UUID), the
    cache returns the string-blank sentinel `<<<NULL>>>` and
    cross-library identity mapping silently breaks. The current
    implementation routes UUID reads through `DcgmSystem.discovery.GetGpuAttributes`
    (which wraps `dcgmGetDeviceAttributes`) — the documented
    synchronous device-info API for static descriptors.
    """

    def test_get_uuid_calls_get_gpu_attributes(self):
        actuator, modules, handle, nvml = _make_initialized_actuator(
            uuid_by_gpu_id={0: b"GPU-deadbeef-1234", 1: b"GPU-other"}
        )
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            uuid = actuator.get_uuid(0)
        self.assertEqual(uuid, "GPU-deadbeef-1234")
        # GetGpuAttributes is the source of truth, NOT the field cache.
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.assert_called_with(0)

    def test_get_uuid_does_not_use_field_cache(self):
        """Regression guard for the field-cache-avoidance contract.

        Even if a future refactor adds `dcgmEntityGetLatestValues`
        calls to other DCGM methods, `get_uuid` must never request
        DCGM_FI_DEV_UUID (54) through the field cache — that returned
        `<<<NULL>>>` in our 2026-05-20 e2e parity run on a fresh
        nv-hostengine. The fail-loud guarantee is: ZERO calls to
        `dcgmEntityGetLatestValues` triggered solely by a `get_uuid`.
        """
        actuator, modules, *_ = _make_initialized_actuator()
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            actuator.get_uuid(0)
        modules["dcgm_agent"].dcgmEntityGetLatestValues.assert_not_called()

    def test_get_uuid_handles_str_return(self):
        """Some DCGM bindings return str, others bytes. Both work."""
        actuator, modules, *_ = _make_initialized_actuator(
            uuid_by_gpu_id={0: "GPU-already-str", 1: "GPU-other"}
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            self.assertEqual(actuator.get_uuid(0), "GPU-already-str")

    def test_get_uuid_rejects_blank_string_sentinel(self):
        """The DCGM string-blank sentinel is not a usable hardware identity.

        If it enters the UUID map, DCGM/NVML PID routing can silently
        collapse multiple GPUs onto the same fake identity. Fail before
        the map is built instead.
        """
        modules = _make_dcgm_modules()
        actuator, modules, *_ = _make_initialized_actuator(
            modules=modules,
            uuid_by_gpu_id={
                0: modules["dcgmvalue"].DCGM_STR_BLANK,
                1: "GPU-other",
            },
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(RuntimeError) as ctx:
                actuator.get_uuid(0)
        self.assertIn("invalid blank UUID", str(ctx.exception))
        self.assertIn("DCGM gpu_id=0", str(ctx.exception))

    def test_get_uuid_rejects_non_ascii_bytes(self):
        """Malformed bytes should fail loudly instead of becoming mojibake."""
        actuator, modules, *_ = _make_initialized_actuator(
            uuid_by_gpu_id={0: b"\xff\xfe", 1: b"GPU-other"}
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(RuntimeError) as ctx:
                actuator.get_uuid(0)
        self.assertIn("non-ASCII UUID", str(ctx.exception))

    def test_get_uuid_uses_discovered_gpu_id_not_idx(self):
        """If DCGM enumerates GPU IDs [10, 20], gpu_idx=1 must query
        gpu_id=20, NOT gpu_id=1. Catches the off-by-mapping bug."""
        actuator, modules, handle, _ = _make_initialized_actuator(
            gpu_ids=(10, 20),
            uuid_by_gpu_id={10: b"GPU-ten", 20: b"GPU-twenty"},
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            uuid = actuator.get_uuid(1)
        self.assertEqual(uuid, "GPU-twenty")
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.assert_called_with(20)


def _wire_identity_map(modules, nvml, handle, dcgm_uuids, nvml_uuids):
    """Seed mocks for the lazy UUID-keyed identity map.

    `dcgm_uuids` is the list of UUIDs DCGM returns for its discovered
    GPUs, in the order `_discovered_gpu_ids` iterates them.
    `nvml_uuids` is the list of UUIDs NVML returns by index 0..N-1.
    The two lists can be in different orders (or contain different
    sets) to exercise the cross-library mapping. UUIDs may be `bytes`
    or `str`; we mirror upstream where some bindings return each.

    `handle` is the DcgmHandle mock returned by `_wire_handle` — we
    overwrite its `GetSystem().discovery.GetGpuAttributes.side_effect`
    so the device-info API returns the right UUID per gpu_id.
    """
    # DCGM side: GetGpuAttributes(gpu_id).identifiers.uuid per discovered GPU.
    gpu_ids = handle.GetSystem.return_value.discovery.GetAllGpuIds.return_value
    uuid_by_gpu_id = dict(zip(gpu_ids, dcgm_uuids))
    discovery = handle.GetSystem.return_value.discovery
    discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
        uuid_by_gpu_id[gid]
    )

    # NVML side: count + per-index handle + per-handle UUID.
    nvml.nvmlDeviceGetCount.return_value = len(nvml_uuids)
    handles = [f"nvml_handle_{i}" for i in range(len(nvml_uuids))]
    nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: handles[i]
    uuid_by_handle = dict(zip(handles, nvml_uuids))
    nvml.nvmlDeviceGetUUID.side_effect = lambda h: uuid_by_handle[h]


class TestListRunningPidsUsesNvml(unittest.TestCase):
    """The crown jewel test — DCGM path uses NVML for PIDs.

    If a future refactor accidentally routes the PID read through DCGM
    (which has no snapshot API), the agent would lose PID enumeration
    on the DCGM path. These tests guard that contract.

    Identity-map note: `list_running_pids` first builds (lazily) a
    UUID-keyed map between DCGM gpuIds and NVML indices. UUID reads on
    the DCGM side go through
    `DcgmSystem.discovery.GetGpuAttributes` (synchronous device-info
    API) rather than `dcgmEntityGetLatestValues` (field cache). The
    "no DCGM time-series read" guarantee is therefore: zero
    `dcgmEntityGetLatestValues` calls triggered solely by
    `list_running_pids`.
    """

    def test_list_running_pids_uses_pynvml_get_compute_running_processes(self):
        actuator, modules, handle, _ = _make_initialized_actuator()
        nvml = MagicMock()
        # Same UUIDs DCGM and NVML see, in the same order → identity map
        # is the identity function (gpu_idx 0 → nvml index 0).
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
            nvml_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
        )
        proc_a = MagicMock(pid=1234)
        proc_b = MagicMock(pid=5678)
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc_a, proc_b]

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            pids = actuator.list_running_pids(0)

        self.assertEqual(pids, [1234, 5678])
        # The compute-PID enumeration itself goes through NVML.
        nvml.nvmlDeviceGetComputeRunningProcesses.assert_called_once_with(
            "nvml_handle_0"
        )
        # The DCGM-side UUID read goes through GetGpuAttributes (device-
        # info API), NOT dcgmEntityGetLatestValues (field cache). If any
        # call lands on the field-cache API, we've regressed either onto
        # the field-cache UUID-read bug or — worse — onto the
        # time-series PID-read path this actuator deliberately avoids.
        modules["dcgm_agent"].dcgmEntityGetLatestValues.assert_not_called()

    def test_list_running_pids_returns_empty_when_no_processes(self):
        actuator, modules, handle, _ = _make_initialized_actuator()
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
            nvml_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
        )
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            self.assertEqual(actuator.list_running_pids(0), [])

    def test_list_running_pids_routes_via_uuid_not_gpu_idx(self):
        """Misaligned DCGM/NVML orderings: same UUIDs, different index spaces.

        DCGM enumerates [10, 20] (i.e. gpu_idx 0 → DCGM gpu_id 10,
        UUID 'GPU-a'); NVML enumerates them in the opposite order
        (index 0 → 'GPU-b', index 1 → 'GPU-a'). list_running_pids(0)
        must route the NVML handle lookup by UUID, hitting NVML
        index 1, not index 0. This is the load-bearing correctness
        test for the cross-library identity mapping.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(gpu_ids=(10, 20))
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-a", b"GPU-b"],
            nvml_uuids=[b"GPU-b", b"GPU-a"],  # reversed!
        )
        proc = MagicMock(pid=999)
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc]

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            pids = actuator.list_running_pids(0)

        self.assertEqual(pids, [999])
        # The handle returned for "GPU-a" lives at NVML index 1, NOT 0.
        # If routing fell through on raw gpu_idx, we'd see index 0.
        nvml.nvmlDeviceGetHandleByIndex.assert_any_call(1)
        nvml.nvmlDeviceGetComputeRunningProcesses.assert_called_once_with(
            "nvml_handle_1"
        )

    def test_list_running_pids_binds_to_expected_uuid(self):
        """With `expected_uuid`, the PID snapshot is
        attributed to THAT GPU's NVML index, not to whatever `gpu_idx`
        currently maps to. Here gpu_idx 0 is 'GPU-a', but the caller anchored
        'GPU-b', so the read must route to GPU-b's NVML index — otherwise a
        re-enumeration could feed GPU-a's workload into a cap the identity
        guard then writes onto GPU-b.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(gpu_ids=(10, 20))
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-a", b"GPU-b"],
            nvml_uuids=[b"GPU-a", b"GPU-b"],  # aligned
        )
        proc = MagicMock(pid=777)
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc]

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            pids = actuator.list_running_pids(0, expected_uuid="GPU-b")

        self.assertEqual(pids, [777])
        # Bound to GPU-b (NVML index 1), NOT gpu_idx 0 (GPU-a, NVML index 0).
        nvml.nvmlDeviceGetComputeRunningProcesses.assert_called_once_with(
            "nvml_handle_1"
        )

    def test_list_running_pids_raises_on_unresolvable_expected_uuid(self):
        """If the anchored UUID is no longer visible to NVML (re-enumeration or
        hot-unplug), fail closed with `_GpuIdentityMismatch` so the caller skips
        this cycle instead of attributing a different GPU's PIDs."""
        actuator, modules, handle, _ = _make_initialized_actuator(gpu_ids=(10, 20))
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-a", b"GPU-b"],
            nvml_uuids=[b"GPU-a", b"GPU-b"],
        )

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertRaises(_GpuIdentityMismatch):
                actuator.list_running_pids(0, expected_uuid="GPU-ghost")

    def test_identity_map_is_lazy_and_cached(self):
        """Build once, reuse on subsequent calls.

        Per-call rebuilds would spam DCGM with N UUID reads every
        reconcile (and N NVML enumerations). The map only invalidates
        on _with_reconnect; until then, both halves are reused.
        """
        actuator, modules, handle, _ = _make_initialized_actuator()
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
            nvml_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
        )
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.list_running_pids(0)
            actuator.list_running_pids(1)
            actuator.list_running_pids(0)

        # DCGM: one GetGpuAttributes call per discovered GPU, regardless
        # of how many times list_running_pids was called.
        discovery = handle.GetSystem.return_value.discovery
        self.assertEqual(discovery.GetGpuAttributes.call_count, 2)
        # NVML enumeration: exactly nvml.nvmlDeviceGetCount() handle
        # lookups for the map build, plus one per list_running_pids
        # call. Count() itself called only once (during map build).
        nvml.nvmlDeviceGetCount.assert_called_once()

    def test_identity_map_raises_on_uuid_missing_from_nvml(self):
        """If DCGM sees a UUID NVML doesn't, surface it loud at first use.

        Silent mis-routing would land cap reads/writes on the wrong
        physical GPU. The operator needs to fix the visibility
        mismatch (NVIDIA_VISIBLE_DEVICES, MIG mode, etc.) — not have
        the agent quietly limp along.
        """
        actuator, modules, handle, _ = _make_initialized_actuator()
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-aaaa", b"GPU-missing"],
            nvml_uuids=[b"GPU-aaaa"],
        )

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertRaises(RuntimeError) as ctx:
                actuator.list_running_pids(0)

        self.assertIn("GPU-missing", str(ctx.exception))
        self.assertIn("DCGM", str(ctx.exception))
        self.assertIn("NVML", str(ctx.exception))

    def test_identity_map_invalidated_on_reconnect(self):
        """After _with_reconnect rebuilds the handle, the map must be rebuilt.

        DCGM may re-enumerate post-restart (different gpuId ordering
        if visibility changed). Stale UUID mappings would silently
        route PID reads to the wrong NVML index.
        """
        actuator, modules, handle, _ = _make_initialized_actuator()
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
            nvml_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
        )
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.list_running_pids(0)  # builds the map
            self.assertIsNotNone(actuator._dcgm_uuid_by_idx)

            DCGMError = modules["dcgm_structs"].DCGMError
            CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

            # Trigger recovery via _with_reconnect; the second op() call
            # is irrelevant — we just want the cache clear to fire.
            calls = {"n": 0}

            def op():
                calls["n"] += 1
                if calls["n"] == 1:
                    raise DCGMError(CONNECTION_NOT_VALID)
                return None

            actuator._with_reconnect(op)

        self.assertIsNone(actuator._dcgm_uuid_by_idx)
        self.assertIsNone(actuator._nvml_index_by_uuid)

    def test_identity_map_build_recovers_from_connection_not_valid(self):
        """First list_running_pids survives a hostengine restart.

        Injects CONNECTION_NOT_VALID on the first UUID read, healthy
        responses on retry. Asserts list_running_pids returns the PIDs
        (no propagated exception), DcgmHandle was rebuilt, identity
        map is populated from the retry data.
        """
        actuator, modules, handle, _ = _make_initialized_actuator()
        nvml = MagicMock()

        DCGMError = modules["dcgm_structs"].DCGMError
        CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

        # _with_reconnect replays the whole _read_dcgm_uuids closure on
        # retry. The retry-side path calls init() which builds a *new*
        # DcgmHandle (and therefore a new discovery mock chain via
        # pydcgm.DcgmHandle.return_value). Two-phase wire-up:
        #
        #   Phase 1 (pre-recovery, current handle): GetGpuAttributes
        #     raises CONNECTION_NOT_VALID on first call. _with_reconnect
        #     catches, calls init(), gets a new handle.
        #   Phase 2 (post-recovery, new handle): re-wire on the same
        #     pydcgm.DcgmHandle.return_value (which init() uses), this
        #     time returning healthy UUIDs.
        #
        # In practice the test mocks DcgmHandle.return_value (one mock,
        # shared across both init() calls), so we just configure the
        # GetGpuAttributes side_effect to fail-then-succeed per call.
        gpu_ids_iter = iter(["GPU-aaaa", "GPU-bbbb"] * 4)
        call_state = {"raised": False}

        def get_gpu_attributes_side_effect(gid):
            if not call_state["raised"]:
                call_state["raised"] = True
                raise DCGMError(CONNECTION_NOT_VALID)
            uuid = next(gpu_ids_iter)
            return _make_gpu_attrs(uuid.encode())

        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = get_gpu_attributes_side_effect

        nvml.nvmlDeviceGetCount.return_value = 2
        handles = ["nvml_handle_0", "nvml_handle_1"]
        nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: handles[i]
        nvml_uuids = {"nvml_handle_0": b"GPU-aaaa", "nvml_handle_1": b"GPU-bbbb"}
        nvml.nvmlDeviceGetUUID.side_effect = lambda h: nvml_uuids[h]
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [MagicMock(pid=42)]

        handle_calls_before = modules["pydcgm"].DcgmHandle.call_count

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            pids = actuator.list_running_pids(0)

        self.assertEqual(pids, [42])
        self.assertEqual(
            modules["pydcgm"].DcgmHandle.call_count,
            handle_calls_before + 1,
            "_with_reconnect should have rebuilt DcgmHandle once.",
        )
        self.assertEqual(actuator._dcgm_uuid_by_idx, ["GPU-aaaa", "GPU-bbbb"])
        self.assertEqual(actuator._nvml_index_by_uuid, {"GPU-aaaa": 0, "GPU-bbbb": 1})

    def test_cached_identity_map_fails_loud_on_mid_run_nvml_disappearance(self):
        """If a GPU disappears from NVML after the map is built, do not
        fall back to raw gpu_idx routing."""
        actuator, modules, handle, _ = _make_initialized_actuator()
        nvml = MagicMock()
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
            nvml_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
        )
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.list_running_pids(0)  # builds the cache
            actuator._nvml_index_by_uuid.pop("GPU-aaaa")
            with self.assertRaises(RuntimeError) as ctx:
                actuator.list_running_pids(0)

        self.assertIn("no longer visible to NVML", str(ctx.exception))
        self.assertIn("GPU-aaaa", str(ctx.exception))


class TestConstraintsW(unittest.TestCase):
    """constraints_w reads `powerLimits.{min,max}PowerLimit`
    via the synchronous device-info API (GetGpuAttributes), NOT
    `dcgmEntityGetLatestValues + DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN/MAX`.

    An earlier implementation read field 161/162 from the field cache,
    which returned `DCGM_FP64_BLANK = 140737488355328.0` on a fresh
    hostengine with no companion watcher. The cap-clamp math then
    clamped every request up to 2^47 W and apply_cap exploded.
    `GetGpuAttributes.powerLimits` carries the same values
    (independently confirmed against the A100 e2e probe matching NVML
    byte-for-byte) but doesn't depend on the field cache.
    """

    def test_constraints_w_returns_min_and_max_from_power_limits(self):
        actuator, modules, handle, _ = _make_initialized_actuator()
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", min_w=100, max_w=700
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            min_w, max_w = actuator.constraints_w(0)
        self.assertEqual((min_w, max_w), (100, 700))
        # The settable range came from the device-info API, not the
        # field cache. Regression guard for the field-cache-avoidance contract.
        discovery.GetGpuAttributes.assert_called_with(0)
        modules["dcgm_agent"].dcgmEntityGetLatestValues.assert_not_called()

    def test_constraints_w_rejects_blank_power_limits(self):
        """GetGpuAttributes returning a DCGM blank sentinel must not feed
        clamp math with a gigantic fake watt value."""
        modules = _make_dcgm_modules()
        actuator, modules, handle, _ = _make_initialized_actuator(modules=modules)
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", min_w=modules["dcgmvalue"].DCGM_FP64_BLANK, max_w=700
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(RuntimeError) as ctx:
                actuator.constraints_w(0)

        self.assertIn("blank DCGM power limit minPowerLimit", str(ctx.exception))


class TestCurrentWAndDefaultW(unittest.TestCase):
    """current_w / default_w read `powerLimits.curPowerLimit` /
    `defaultPowerLimit` via GetGpuAttributes — the consolidation
    onto the synchronous device-info API. The orphan-recovery guard
    depends on these returning the real values (not the field-cache
    blank sentinel), otherwise every cap-restore would no-op or write
    nonsense.
    """

    def test_current_w_returns_powerlimits_cur(self):
        actuator, modules, handle, _ = _make_initialized_actuator()
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", current_w=423
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            self.assertEqual(actuator.current_w(0), 423)
        modules["dcgm_agent"].dcgmEntityGetLatestValues.assert_not_called()

    def test_current_w_returns_cur_not_enforced(self):
        """Current and enforced power limits can diverge; current_w must
        report the live current value."""
        actuator, modules, handle, _ = _make_initialized_actuator()
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", current_w=275, enforced_w=333
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            self.assertEqual(actuator.current_w(0), 275)

    def test_default_w_returns_powerlimits_default(self):
        """defaultPowerLimit is NOT maxPowerLimit. Even on SKUs where
        they're numerically equal today, reading the wrong attribute
        would silently regress on future SKUs where default < max."""
        actuator, modules, handle, _ = _make_initialized_actuator()
        discovery = handle.GetSystem.return_value.discovery
        # Distinct values so a future refactor that read maxPowerLimit
        # instead would fail this test loudly.
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", default_w=400, max_w=500
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            self.assertEqual(actuator.default_w(0), 400)
        modules["dcgm_agent"].dcgmEntityGetLatestValues.assert_not_called()

    def test_current_w_uses_with_reconnect_on_stale_handle(self):
        """current_w wraps its read in _with_reconnect. A
        CONNECTION_NOT_VALID mid-orphan-recovery must trigger a
        rebuild + retry, not abort the agent."""
        actuator, modules, handle, _ = _make_initialized_actuator()
        DCGMError = modules["dcgm_structs"].DCGMError
        CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

        discovery = handle.GetSystem.return_value.discovery
        call_state = {"raised": False}

        def side_effect(gid):
            if not call_state["raised"]:
                call_state["raised"] = True
                raise DCGMError(CONNECTION_NOT_VALID)
            return _make_gpu_attrs(b"GPU-x", current_w=400)

        discovery.GetGpuAttributes.side_effect = side_effect

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            self.assertEqual(actuator.current_w(0), 400)

        # Reconnect happened (init() rebuilt DcgmHandle once).
        self.assertEqual(modules["pydcgm"].DcgmHandle.call_count, 2)

    def test_default_w_uses_with_reconnect_on_stale_handle(self):
        actuator, modules, handle, _ = _make_initialized_actuator()
        DCGMError = modules["dcgm_structs"].DCGMError
        CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

        discovery = handle.GetSystem.return_value.discovery
        call_state = {"raised": False}

        def side_effect(gid):
            if not call_state["raised"]:
                call_state["raised"] = True
                raise DCGMError(CONNECTION_NOT_VALID)
            return _make_gpu_attrs(b"GPU-x", default_w=700)

        discovery.GetGpuAttributes.side_effect = side_effect

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            self.assertEqual(actuator.default_w(0), 700)

        self.assertEqual(modules["pydcgm"].DcgmHandle.call_count, 2)

    def test_default_w_rejects_nan_power_limit(self):
        actuator, modules, handle, _ = _make_initialized_actuator()
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", default_w=float("nan")
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(RuntimeError) as ctx:
                actuator.default_w(0)

        self.assertIn(
            "non-finite DCGM power limit defaultPowerLimit", str(ctx.exception)
        )


# ---------------------------------------------------------------------------
# Write surface — apply_cap
# ---------------------------------------------------------------------------


class TestApplyCap(unittest.TestCase):
    def setUp(self):
        # Reset module-level managed state so each test starts clean.
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def _seed_constraints_and_uuid(
        self, modules, handle, min_w=100, max_w=700, uuid=b"GPU-x"
    ):
        """Wire constraints + UUID via GetGpuAttributes (single API).

        An earlier implementation split these two reads across two DCGM
        APIs (field cache for constraints, GetGpuAttributes for UUID).
        Now both come from the same
        `GetGpuAttributes(gid)` call — constraints from
        `.powerLimits.{min,max}PowerLimit`, UUID from
        `.identifiers.uuid`. No `dcgmEntityGetLatestValues` calls are
        triggered by apply_cap on the read side any more.
        """
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            uuid, min_w=min_w, max_w=max_w
        )

    def test_apply_cap_requires_metrics(self):
        actuator = DcgmActuator()  # no metrics
        with self.assertRaises(RuntimeError) as ctx:
            actuator.apply_cap(0, 300)
        self.assertIn("metrics", str(ctx.exception).lower())

    def test_apply_cap_happy_path_within_constraints(self):
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(modules, handle)

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            result = actuator.apply_cap(0, 300)

        self.assertEqual(result, 300)
        # A DcgmGroup was created and the GPU added to it.
        modules["pydcgm"].DcgmGroup.assert_called_once()
        group = modules["pydcgm"].DcgmGroup.return_value
        group.AddGpu.assert_called_once_with(0)
        # The cap write went through grp.config.Set(cfg) with mPowerLimit.val=300.
        group.config.Set.assert_called_once()
        cfg = group.config.Set.call_args.args[0]
        self.assertEqual(cfg.mPowerLimit.val, 300)
        # State tracking parity with the NVML path.
        self.assertIn(0, power_agent._managed_gpu_indices)
        metrics.applied_limit_watts.labels.assert_called_with(gpu="0")
        metrics.applied_limit_watts.labels.return_value.set.assert_called_with(300)
        metrics.apply_failures_total.inc.assert_not_called()
        # The capped UUID is recorded in the ownership set the SIGTERM sweep
        # restricts to.
        self.assertEqual(actuator.managed_uuids(), {"GPU-x"})

    def test_recap_after_reenumeration_keeps_displaced_uuid_owned(self):
        """The core of sttts's leak: the index-keyed `_managed_uuid_by_idx`
        is overwritten when a re-enumerated index is re-capped, but the
        append-only ownership set (`managed_uuids`) must retain the displaced
        UUID so the SIGTERM sweep can still find and restore it."""
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        discovery = handle.GetSystem.return_value.discovery

        # First cap: idx 0 resolves to GPU-A.
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", min_w=100, max_w=700
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.apply_cap(0, 300)

        # DCGM re-enumerates: idx 0 now resolves to GPU-B; re-cap idx 0.
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-B", min_w=100, max_w=700
        )
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.apply_cap(0, 300)

        # The index-keyed projection lost GPU-A (the original bug shape)...
        self.assertEqual(actuator._managed_uuid_by_idx[0], "GPU-B")
        # ...but the ownership set retains BOTH, so the sweep can relocate
        # and restore the displaced GPU-A on shutdown.
        self.assertEqual(actuator.managed_uuids(), {"GPU-A", "GPU-B"})

    def test_apply_cap_skips_write_on_reenumeration_during_write(self):
        """Identity-stable cap write.

        If `gpu_idx` re-enumerates onto a DIFFERENT physical GPU between the
        identity captured at `apply_cap` entry and the pre-`Set` re-check
        (e.g. a hostengine reconnect inside `_with_reconnect` re-orders
        indices), the cap — derived from the ORIGINAL GPU's workload — must
        NOT be written to the new GPU. `apply_cap` skips the `Set`, ticks
        `apply_failures_total`, records nothing as managed (no clobber), and
        returns the effective watts; the next reconcile cycle re-attributes
        and retries against the fresh enumeration.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(modules, handle, min_w=100, max_w=700)

        # Both the entry identity capture (via get_uuid) and the pre-Set
        # re-verification call `_read_uuid_raw`. Return GPU-A first (the
        # identity the cap is computed for) then GPU-B (the re-enumerated
        # occupant) so the guard fires.
        with patch.object(actuator, "_read_uuid_raw", side_effect=["GPU-A", "GPU-B"]):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                result = actuator.apply_cap(0, 300)

        # Effective watts still returned (per the Actuator Protocol), but NO cap write.
        self.assertEqual(result, 300)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        metrics.apply_failures_total.inc.assert_called_once()
        # Nothing tracked as managed — we refused to write, so there is no
        # cap of ours on either GPU to track.
        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_apply_cap_refuses_when_entry_identity_unreadable(self):
        """Entry-identity policy: if the entry-time identity
        lookup fails, the apply must FAIL CLOSED even if the identity would be
        readable in-transaction.

        `watts` is derived from the workload on THIS index. Transaction-local
        identity fixes ownership bookkeeping, but without an entry identity to
        compare against it cannot establish that the current GPU is the one
        whose workload produced the cap — a reconnect could have re-enumerated
        the index onto a different GPU, and we would apply GPU-A's cap to GPU-B
        (recorded, wrongly for policy, as GPU-B's). So refuse at entry: no Set,
        an apply-failure tick, nothing tracked; the next reconcile retries.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        # Identity IS readable in-transaction (GetGpuAttributes → GPU-A); the
        # refusal is driven purely by the entry-time failure, proving the
        # entry identity is required regardless of transaction readability.
        self._seed_constraints_and_uuid(modules, handle, uuid="GPU-A")

        with patch.object(
            actuator, "get_uuid", side_effect=RuntimeError("transient identity blip")
        ):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                result = actuator.apply_cap(0, 300)

        # Effective watts still returned (Actuator Protocol), but NO write and
        # NO tracking of any kind.
        self.assertEqual(result, 300)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        metrics.apply_failures_total.inc.assert_called_once()
        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, actuator._managed_uuid_by_idx)
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_apply_cap_rejects_constraints_read_from_reenumerated_gpu(self):
        """Wattage provenance: the SKU constraints the
        cap is clamped against must come from the SAME physical GPU the cap is
        anchored to.

        The expected UUID is captured (GPU-A) BEFORE the constraints read;
        `_power_limits_with_uuid` reads the min/max AND the identity they
        belong to from one snapshot. If a reconnect re-enumerates the index
        onto GPU-B during that read, the snapshot's UUID (GPU-B) no longer
        matches the anchor (GPU-A), so apply_cap fails closed rather than
        writing a value clamped to another GPU's SKU range. This closes the
        ABA hole a plain in-transaction destination check would miss.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(
            modules, handle, uuid="GPU-A", min_w=100, max_w=700
        )

        # `state` models which physical GPU currently occupies index 0. The
        # identity-bound constraints read flips it GPU-A -> GPU-B, simulating a
        # reconnect / re-enumeration DURING the constraints read, and reports
        # the new occupant's identity alongside its limits.
        state = {"uuid": "GPU-A"}

        def flip_during_power_read(_idx):
            state["uuid"] = "GPU-B"
            pl = MagicMock()
            pl.minPowerLimit = 100
            pl.maxPowerLimit = 700
            return pl, "GPU-B"

        with patch.object(
            actuator, "get_uuid", side_effect=lambda _idx: state["uuid"]
        ), patch.object(
            actuator, "_power_limits_with_uuid", side_effect=flip_during_power_read
        ):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                result = actuator.apply_cap(0, 300)

        # Entry captured GPU-A before the constraints read flipped the index to
        # GPU-B; the constraints-provenance check sees GPU-B -> mismatch ->
        # refuse. Effective watts still returned per the Protocol; no Set, no
        # tracking.
        self.assertEqual(result, 300)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        metrics.apply_failures_total.inc.assert_called_once()
        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, actuator._managed_uuid_by_idx)
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_apply_cap_uses_caller_supplied_identity_without_recapture(self):
        """PID-snapshot anchor: when the caller supplies
        `expected_uuid` (the reconcile loop captures it BEFORE the PID snapshot
        that produced `watts`), apply_cap must use it verbatim and must NOT
        re-capture via `get_uuid` — re-capturing here would reopen the
        attribution-to-write window the anchor exists to close. The
        in-transaction recheck verifies the supplied UUID; on a match the Set
        fires and the GPU is recorded under that UUID.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(
            modules, handle, uuid="GPU-A", min_w=100, max_w=700
        )

        with patch.object(
            actuator,
            "get_uuid",
            side_effect=AssertionError(
                "get_uuid must not be called when expected_uuid is supplied"
            ),
        ), patch.object(actuator, "_read_uuid_raw", return_value="GPU-A"):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                result = actuator.apply_cap(0, 300, expected_uuid="GPU-A")

        self.assertEqual(result, 300)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_called_once()
        metrics.apply_failures_total.inc.assert_not_called()
        self.assertEqual(actuator.managed_uuids(), {"GPU-A"})
        self.assertEqual(actuator._managed_uuid_by_idx.get(0), "GPU-A")

    def test_apply_cap_caller_supplied_identity_mismatch_fails_closed(self):
        """PID-snapshot anchor: if the caller-supplied
        `expected_uuid` no longer matches the GPU on the index at write time —
        a re-enumeration ANYWHERE between the pre-snapshot capture and the Set,
        including the window between PID attribution and this call — refuse:
        no Set, an apply-failure tick, nothing tracked. The next reconcile
        re-captures against the fresh enumeration and retries.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(modules, handle, min_w=100, max_w=700)

        with patch.object(
            actuator,
            "get_uuid",
            side_effect=AssertionError(
                "get_uuid must not be called when expected_uuid is supplied"
            ),
        ), patch.object(actuator, "_read_uuid_raw", return_value="GPU-B"):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                result = actuator.apply_cap(0, 300, expected_uuid="GPU-A")

        self.assertEqual(result, 300)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        metrics.apply_failures_total.inc.assert_called_once()
        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, actuator._managed_uuid_by_idx)
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_apply_cap_skips_write_when_recheck_identity_unverifiable(self):
        """Once an identity IS captured
        at entry, a pre-`Set` recheck that cannot read the UUID must NOT
        fall through and write. Enumeration may have changed and we can't
        prove otherwise, so refuse (apply failure + retry next cycle) rather
        than risk writing this GPU's cap onto a re-enumerated occupant.

        Contrast with `test_apply_cap_refuses_when_entry_identity_unreadable`,
        which refuses at ENTRY; this one refuses at the in-transaction recheck
        after a successful entry capture.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(modules, handle, min_w=100, max_w=700)

        # Entry capture succeeds (GPU-A); the pre-Set recheck raises a
        # non-DCGM error → unverifiable identity with an expected UUID in hand.
        with patch.object(
            actuator,
            "_read_uuid_raw",
            side_effect=["GPU-A", RuntimeError("identity read failed at recheck")],
        ):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                result = actuator.apply_cap(0, 300)

        self.assertEqual(result, 300)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        metrics.apply_failures_total.inc.assert_called_once()
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_apply_cap_records_transaction_verified_uuid_not_reconnect_reread(self):
        """Bookkeeping persists the identity verified INSIDE
        the write transaction, not a fresh reconnect-capable re-read.

        A hostengine restart in the window between the successful `Set` and
        `_record_managed_state` can re-enumerate `get_uuid(gpu_idx)` onto a
        DIFFERENT physical GPU. Trusting that re-read would persist the wrong
        UUID and leak the cap we just applied. The recorded UUID must be the
        one `_write_set` confirmed at Set time (GPU-A), so `_record_managed_state`
        must NOT call `get_uuid` again on the guarded path.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        # Constraints + the in-transaction identity recheck (`_read_uuid_raw`,
        # via GetGpuAttributes) both resolve to GPU-A, so the guarded Set
        # succeeds against GPU-A.
        self._seed_constraints_and_uuid(modules, handle, uuid="GPU-A")

        # get_uuid yields GPU-A at apply entry (→ expected_uuid) then GPU-B on
        # any later call (the simulated post-Set re-enumeration). The fix must
        # NOT consume that second value for bookkeeping.
        with patch.object(actuator, "get_uuid", side_effect=["GPU-A", "GPU-B"]):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                result = actuator.apply_cap(0, 300)

        self.assertEqual(result, 300)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_called_once()
        # Bookkeeping recorded the transaction-verified GPU-A, not the
        # re-enumerated GPU-B a post-Set get_uuid re-read would have returned.
        self.assertEqual(actuator.managed_uuids(), {"GPU-A"})
        self.assertEqual(actuator._managed_uuid_by_idx[0], "GPU-A")

    def test_apply_cap_clamps_above_max(self):
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(modules, handle, min_w=100, max_w=700)

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            result = actuator.apply_cap(0, 900)

        self.assertEqual(result, 700)
        cfg = modules["pydcgm"].DcgmGroup.return_value.config.Set.call_args.args[0]
        self.assertEqual(cfg.mPowerLimit.val, 700)
        metrics.cap_clamped_total.labels.assert_called_with(direction="max")

    def test_apply_cap_clamps_below_min(self):
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(modules, handle, min_w=100, max_w=700)

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            result = actuator.apply_cap(0, 50)

        self.assertEqual(result, 100)
        metrics.cap_clamped_total.labels.assert_called_with(direction="min")

    def test_apply_cap_never_calls_enforce(self):
        """The agent issues ONLY dcgmConfigSet, never dcgmConfigEnforce.

        dcgmConfigSet already records the cap as DCGM's target config
        (auto-reapplied after reset), so the dcgmConfigEnforce manual
        re-assert is redundant. It was removed as dead config; this locks that in so a refactor can't reintroduce
        the extra RPC.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        self._seed_constraints_and_uuid(modules, handle)

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.apply_cap(0, 300)

        modules["pydcgm"].DcgmGroup.return_value.config.Enforce.assert_not_called()

    def test_apply_cap_dcgm_generic_error_bumps_failure_metric_and_returns(self):
        """Non-CONNECTION_NOT_VALID DCGMErrors are non-fatal — the
        reconcile loop logs + continues on to the next GPU."""
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        # Constraints come from GetGpuAttributes.powerLimits —
        # default handle wiring (min=100, max=700) is fine for this test.
        self._seed_constraints_and_uuid(modules, handle, min_w=100, max_w=700)
        group = modules["pydcgm"].DcgmGroup.return_value
        group.config.Set.side_effect = modules["dcgm_structs"].DCGMError(
            modules["dcgm_structs"].DCGM_ST_GENERIC_ERROR
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            result = actuator.apply_cap(0, 300)

        # Returns the *requested* (effective) watts even on failure
        # because downstream callers / Prometheus need a number.
        self.assertEqual(result, 300)
        metrics.apply_failures_total.inc.assert_called_once()
        # GPU is NOT recorded as managed — we don't track an unsuccessful
        # write, otherwise restore-on-shutdown would touch GPUs we never
        # actually capped.
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_apply_cap_propagates_non_dcgm_exception(self):
        """Non-DCGMError exceptions MUST propagate out of apply_cap.

        Regression guard: the original
        ``except Exception`` masked programming/binding defects
        (e.g. the ``dcgmvalue.DCGM_INT32_BLANK`` AttributeError,
        documented in ``_apply_cap_inner``'s docstring) as normal
        apply failures, which both silenced the bug AND incorrectly
        ticked ``apply_failures_total`` — a metric reserved for
        "the cap write itself failed". After narrowing to
        ``dcgm_structs.DCGMError``, an AttributeError surfaces to
        the reconcile loop instead of being absorbed.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        self._seed_constraints_and_uuid(modules, handle, min_w=100, max_w=700)
        group = modules["pydcgm"].DcgmGroup.return_value
        # Simulate a non-DCGM defect inside the write path — e.g. a
        # binding-attribute mismatch on a future pydcgm version.
        group.config.Set.side_effect = AttributeError(
            "simulated pydcgm binding regression"
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(AttributeError) as ctx:
                actuator.apply_cap(0, 300)

        self.assertIn("simulated pydcgm binding regression", str(ctx.exception))
        # The "cap write failed" failure metric MUST NOT tick — that
        # metric is reserved for DCGMError outcomes. A binding bug is
        # a different failure class and using the same metric would
        # hide the regression behind normal cap-failure alerting.
        metrics.apply_failures_total.inc.assert_not_called()
        # No managed-state bookkeeping ran (we never reached the
        # success path), and no applied_limit_watts gauge tick.
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        metrics.applied_limit_watts.labels.assert_not_called()

    def test_apply_cap_blanks_workload_power_profiles(self):
        """Every workload-profile slot MUST be set to DCGM_INT32_BLANK.

        ctypes zero-initializes c_dcgmDeviceConfig_v2, and DCGM's
        config manager treats an all-zero workload-profile array as
        ACTION_CLEAR (see DcgmConfigManagerTests.cpp:207-231 —
        "Initialized target config is cleared by zeroed new config").
        Without the blanking loop in apply_cap, every cap write would
        silently clear whatever workload power profiles were on the
        GPU. This test guards that contract: all 8 slots must be
        BLANK (sentinel 0x7FFFFFF0), never zero.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        self._seed_constraints_and_uuid(modules, handle)

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.apply_cap(0, 300)

        cfg = modules["pydcgm"].DcgmGroup.return_value.config.Set.call_args.args[0]
        # DCGM_INT32_BLANK lives in dcgmvalue (NOT dcgm_structs) per the
        # upstream pydcgm bindings — the production import matches this.
        # See _make_dcgm_modules for the mock setup.
        blank = modules["dcgmvalue"].DCGM_INT32_BLANK
        size = modules["dcgm_structs"].DCGM_WORKLOAD_POWER_PROFILE_ARRAY_SIZE

        # __setitem__ must have been called once per slot, in index
        # order, each time with the BLANK sentinel. Order matters less
        # than completeness, so we collect the set of (index, value)
        # pairs and compare against the expected coverage.
        setitem_calls = cfg.mWorkloadPowerProfiles.__setitem__.call_args_list
        observed = {call.args[0]: call.args[1] for call in setitem_calls}
        self.assertEqual(
            observed,
            {i: blank for i in range(size)},
            f"Expected every slot in [0, {size}) blanked to {blank:#x}; "
            f"got {observed!r}. An all-zero array would trigger "
            "DCGM ACTION_CLEAR and wipe workload power profiles.",
        )

    def test_apply_cap_caches_group_across_calls(self):
        """Per-GPU DcgmGroup is created once, reused on subsequent
        applies. Saves the DcgmGroup create + AddGpu RPCs per reconcile.

        Constraints + UUID both come from GetGpuAttributes — a single
        lambda satisfies any number of apply_cap calls.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        self._seed_constraints_and_uuid(modules, handle)

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.apply_cap(0, 300)
            actuator.apply_cap(0, 400)

        # DcgmGroup must be constructed exactly once — second call
        # reuses the cached entry.
        self.assertEqual(modules["pydcgm"].DcgmGroup.call_count, 1)


# ---------------------------------------------------------------------------
# Write surface — restore_default
# ---------------------------------------------------------------------------


class TestRestoreDefault(unittest.TestCase):
    """restore_default MUST read field 163 (DEF), not 162 (MAX).

    DCGM lacks a "reset to default" verb, so we read the factory
    default explicitly. Field 162 is the
    *maximum settable* power limit; on shipped data-center SKUs they're
    numerically equal, but reading 163 ensures correctness on
    hypothetical future SKUs where default < max.
    """

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def test_restore_default_reads_defaultPowerLimit_not_max(self):
        """restore_default reads `powerLimits.defaultPowerLimit`, NOT
        `maxPowerLimit`. Even on shipped data-center SKUs where the
        two are numerically equal, future SKUs may diverge — and the
        whole point of distinct fields is to track the difference.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        # Distinct values so a future refactor that read maxPowerLimit
        # would fail this test loudly. min=100/max=500/default=400.
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", min_w=100, max_w=500, default_w=400
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.restore_default(0)

        # The cap write was issued at the FACTORY DEFAULT (400 W), not
        # the SKU max (500 W). All power-limit reads go through
        # GetGpuAttributes, so this also guards that no field-cache read
        # leaked into the path.
        cfg = modules["pydcgm"].DcgmGroup.return_value.config.Set.call_args.args[0]
        self.assertEqual(cfg.mPowerLimit.val, 400)
        modules["dcgm_agent"].dcgmEntityGetLatestValues.assert_not_called()

    def test_restore_default_relocates_managed_uuid_after_reenumeration(self):
        """A hostengine restart can change gpu_idx -> gpu_id ordering.

        If GPU index 0 was capped while it resolved to UUID A, but later
        resolves to UUID B, restore_default(0) must restore UUID A at its
        current index rather than writing default TGP to UUID B.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        actuator._discovered_gpu_ids = [20, 10]
        discovery.GetGpuAttributes.side_effect = lambda gid: {
            10: _make_gpu_attrs("GPU-A", default_w=410),
            20: _make_gpu_attrs("GPU-B", default_w=620),
        }[gid]

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                actuator.restore_default(0)

        cfg = modules["pydcgm"].DcgmGroup.return_value.config.Set.call_args.args[0]
        self.assertEqual(cfg.mPowerLimit.val, 410)
        group = modules["pydcgm"].DcgmGroup.return_value
        group.AddGpu.assert_called_with(10)
        self.assertIn("restoring capped UUID GPU-A", "\n".join(cm.output))
        self.assertEqual(actuator.managed_uuid_for_idx(0), "GPU-A")

    def test_restore_default_skips_when_managed_uuid_left_node(self):
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        actuator._discovered_gpu_ids = [20]
        discovery.GetGpuAttributes.side_effect = lambda gid: {
            20: _make_gpu_attrs("GPU-B", default_w=620),
        }[gid]

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                actuator.restore_default(0)

        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        self.assertIn("GPU-A is no longer visible", "\n".join(cm.output))

    def test_restore_default_best_effort_when_identity_unreadable(self):
        """A transient DCGM read failure at restore must NOT be treated as
        'GPU gone'. We fall back to the original index so `_apply_cap_inner`'s
        own reconnect-and-retry can still attempt the restore — abandoning a
        capped GPU on an unverifiable read is worse than a best-effort write.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: {
            10: _make_gpu_attrs("GPU-A", default_w=410),
        }[gid]

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            with patch.object(
                actuator, "get_uuid", side_effect=RuntimeError("hostengine blip")
            ):
                with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                    result = actuator.restore_default(0)

        self.assertTrue(result)
        cfg = modules["pydcgm"].DcgmGroup.return_value.config.Set.call_args.args[0]
        self.assertEqual(cfg.mPowerLimit.val, 410)
        self.assertIn("best-effort restore at the", "\n".join(cm.output))

    def test_restore_default_skips_when_scan_incomplete_after_proven_mismatch(self):
        """After a PROVEN index mismatch, an inconclusive relocation scan must
        NOT fall back to the original index. That index now hosts a different
        physical GPU, so writing default TGP there would clobber an unrelated
        GPU and — because the SIGTERM caller prunes the capped UUID after a
        "successful" restore — drop the real capped GPU from managed_gpus.json,
        leaking its cap. We skip without writing instead; cold-start orphan
        recovery retries on the next boot.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        actuator._discovered_gpu_ids = [20, 30]

        def _attrs(gid):
            if gid == 20:
                return _make_gpu_attrs("GPU-B", default_w=620)
            raise RuntimeError("hostengine blip on gpu_id 30")

        discovery.GetGpuAttributes.side_effect = _attrs

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                result = actuator.restore_default(0)

        self.assertFalse(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        self.assertIn("relocation scan incomplete", "\n".join(cm.output))

    def test_restore_default_raises_on_write_failure(self):
        """restore_default propagates DCGM write failure to the caller.

        Without this contract, a refactor that re-routed restore_default
        through apply_cap (which absorbs failures into
        apply_failures_total) would let shutdown cleanup log a false-
        positive "restored" message when dcgmConfigSet(default) failed.
        Asserts: DCGMError raised, no bookkeeping side-effects.
        """
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        # Default value via GetGpuAttributes.
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(b"GPU-x")
        group = modules["pydcgm"].DcgmGroup.return_value
        DCGMError = modules["dcgm_structs"].DCGMError
        group.config.Set.side_effect = DCGMError(
            modules["dcgm_structs"].DCGM_ST_GENERIC_ERROR
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(DCGMError):
                actuator.restore_default(0)

        # The exception path must NOT update managed-state — an earlier
        # restore_default would have left _managed_gpu_indices intact
        # via apply_cap's success-path bookkeeping; the current path skips
        # bookkeeping because _apply_cap_inner raises before it runs.
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        metrics.applied_limit_watts.labels.assert_not_called()

    def test_restore_default_returns_false_on_midwrite_reenumeration(self):
        """A reconnect BETWEEN index resolution and the Set re-enumerates the
        resolved index onto a different physical GPU.

        `_resolve_managed_idx` only proves identity at resolution time, so the
        write itself must re-verify. On a proven mid-write mismatch
        `restore_default` must NOT write the default to the wrong GPU and must
        return False — a True return makes `_shutdown_cleanup` prune the still-
        live cap, leaking it.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410
        )

        # Resolution sees idx 0 still hosting GPU-A; the in-transaction recheck
        # then sees GPU-B (re-enumerated mid-write) → proven mismatch.
        with patch.object(actuator, "get_uuid", return_value="GPU-A"), patch.object(
            actuator, "_read_uuid_raw", return_value="GPU-B"
        ):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                    result = actuator.restore_default(0)

        self.assertFalse(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        self.assertIn("could not confirm the target GPU", "\n".join(cm.output))

    def test_restore_default_fails_closed_when_recheck_identity_unreadable(self):
        """A restore that cannot READ the target identity
        at write time must FAIL CLOSED, not best-effort write.

        Writing the factory "default" is NOT harmless when the index may have
        re-enumerated onto a GPU-B that another workflow capped below default:
        the write would clobber GPU-B, and returning success would make the
        caller prune the still-capped GPU-A (leak). An unverifiable identity is
        therefore treated like a proven mismatch — no write, return False, keep
        the UUID for orphan recovery. (DCGM connection errors still surface to
        `_with_reconnect` for a retry; this covers the non-DCGM unreadable
        case, which the earlier best-effort behaviour silently swallowed.)
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410
        )

        with patch.object(actuator, "get_uuid", return_value="GPU-A"), patch.object(
            actuator,
            "_read_uuid_raw",
            side_effect=RuntimeError("identity unreadable at recheck"),
        ):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                    result = actuator.restore_default(0)

        self.assertFalse(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        self.assertIn("could not confirm the target GPU", "\n".join(cm.output))


# ---------------------------------------------------------------------------
# Write surface — restore_default_by_uuid (SIGTERM UUID sweep)
# ---------------------------------------------------------------------------


class TestRestoreDefaultByUuid(unittest.TestCase):
    """`restore_default_by_uuid` is the SIGTERM safety net for the
    re-enumeration re-cap collision: it resolves a UUID to its CURRENT
    index and restores default TGP there, independent of the lossy
    index-keyed in-memory maps."""

    def test_restores_displaced_uuid_at_current_index(self):
        """The leaked GPU from sttts's trace: UUID A was capped at idx 0,
        DCGM re-enumerated A to idx 1, and idx 0 was re-capped onto B
        (overwriting `_managed_uuid_by_idx[0]`). Sweeping UUID A must
        restore it at its current index 1, below default."""
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [20, 10]
        discovery.GetGpuAttributes.side_effect = lambda gid: {
            10: _make_gpu_attrs("GPU-A", default_w=410, current_w=300),
            20: _make_gpu_attrs("GPU-B", default_w=620, current_w=620),
        }[gid]

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            result = actuator.restore_default_by_uuid("GPU-A")

        self.assertTrue(result)
        cfg = modules["pydcgm"].DcgmGroup.return_value.config.Set.call_args.args[0]
        self.assertEqual(cfg.mPowerLimit.val, 410)
        modules["pydcgm"].DcgmGroup.return_value.AddGpu.assert_called_with(10)

    def test_returns_none_when_already_at_default(self):
        """UUID resolves but the GPU is already at/above default — we hold
        no live cap, so return None (caller prunes) and issue no write."""
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410, current_w=410
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            result = actuator.restore_default_by_uuid("GPU-A")

        self.assertIsNone(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()

    def test_returns_none_when_uuid_gone_after_clean_scan(self):
        """A clean scan with no match proves the GPU left the node; its cap
        left with it, so return None (safe to prune) with no write."""
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [20]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-B", default_w=620, current_w=620
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            result = actuator.restore_default_by_uuid("GPU-A")

        self.assertIsNone(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()

    def test_returns_false_when_scan_incomplete(self):
        """A relocation-scan probe raising (transient DCGM outage) is NOT
        proof the GPU is gone. Return False so the caller KEEPS the UUID for
        cold-start orphan recovery rather than pruning a possibly-live cap."""
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [20, 30]

        def _attrs(gid):
            if gid == 20:
                return _make_gpu_attrs("GPU-B", default_w=620, current_w=620)
            raise RuntimeError("hostengine blip on gpu_id 30")

        discovery.GetGpuAttributes.side_effect = _attrs

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertLogs("power_agent.actuator", level="WARNING"):
                result = actuator.restore_default_by_uuid("GPU-A")

        self.assertFalse(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()

    def test_returns_false_on_midwrite_reenumeration(self):
        """A reconnect BETWEEN `_resolve_idx_for_uuid` (plus the current/default
        reads) and the Set re-enumerates the resolved index onto a different
        GPU. The sweep must return False (NOT True) so `_shutdown_cleanup`
        keeps the UUID — a True return would prune a still-live cap — and must
        not write the default to the wrong GPU.
        """
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410, current_w=300
        )

        # `_resolve_idx_for_uuid` now reads via `_read_uuid_raw` too, so the
        # first read (resolution) must see GPU-A (match at idx 0) and the second
        # (`_write_set`'s in-transaction recheck) must see GPU-B — the index
        # re-enumerated mid-write. The identity-bound power read in between goes
        # through `_power_limits_with_uuid` (GetGpuAttributes -> GPU-A), so it is
        # unaffected by this patch.
        with patch.object(actuator, "_read_uuid_raw", side_effect=["GPU-A", "GPU-B"]):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                    result = actuator.restore_default_by_uuid("GPU-A")

        self.assertFalse(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        self.assertIn("could not confirm index", "\n".join(cm.output))

    def test_returns_false_on_reenumeration_before_power_read(self):
        """On the NO-write path a reconnect BETWEEN
        `_resolve_idx_for_uuid` and the identity-bound power read can park a
        DIFFERENT GPU (already at/above default) on the resolved index while
        the GPU we own moved elsewhere with its cap still live. Because the
        limits AND their owning UUID now come from ONE snapshot, the sweep sees
        the mismatch (GPU-B, not the resolved GPU-A) and returns False (retain
        ownership), issuing no write, rather than pruning on None and leaking
        the still-live cap that moved elsewhere."""
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [10]
        # Resolution sees GPU-A at idx 0 ...
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410, current_w=410
        )
        # ... but the identity-bound power read returns GPU-B (the index
        # re-enumerated onto a different GPU already at default).
        pl = MagicMock()
        pl.curPowerLimit = 410
        pl.defaultPowerLimit = 410
        with patch.object(
            actuator, "_power_limits_with_uuid", return_value=(pl, "GPU-B")
        ):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                    result = actuator.restore_default_by_uuid("GPU-A")

        self.assertFalse(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        self.assertIn("re-enumerated between UUID", "\n".join(cm.output))

    def test_returns_false_when_power_read_unreadable(self):
        """No-write path, the identity-bound power read raises (transient DCGM
        outage): the sweep cannot prove the resolved index still hosts the UUID,
        so it fails closed (False, retain ownership) rather than pruning on
        None."""
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410, current_w=410
        )

        with patch.object(
            actuator, "_power_limits_with_uuid", side_effect=RuntimeError("blip")
        ):
            with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
                "power_agent._persist_managed_gpus"
            ):
                with self.assertLogs("power_agent.actuator", level="WARNING") as cm:
                    result = actuator.restore_default_by_uuid("GPU-A")

        self.assertFalse(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        self.assertIn("could not read power limits/identity", "\n".join(cm.output))


# ---------------------------------------------------------------------------
# Ownership retirement on release
# ---------------------------------------------------------------------------


class TestOwnershipRetirement(unittest.TestCase):
    """A cap that is APPLIED claims ownership (`_capped_uuids`); a cap that is
    RELEASED/RESTORED must NOT re-claim it, and an explicit release must retire
    it — otherwise the SIGTERM sweep would reset a cap another workflow
    installed on that GPU after our release."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def test_retire_managed_uuid_drops_capped_uuid_and_index_projection(self):
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            b"GPU-x", min_w=100, max_w=700
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.apply_cap(0, 300)

        # Applying a cap claims ownership.
        self.assertEqual(actuator.managed_uuids(), {"GPU-x"})
        self.assertEqual(actuator._managed_uuid_by_idx.get(0), "GPU-x")

        # Retiring drops the UUID from the ownership set AND any stale index
        # projection pointing at it — from both the index→UUID map and the
        # shared managed-index set — so a later sweep no longer treats it as
        # ours.
        actuator.retire_managed_uuid("GPU-x")
        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, actuator._managed_uuid_by_idx)
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_retire_managed_uuid_drops_all_projection_indices(self):
        """Regression for the multi-projection leak: a UUID re-capped across a
        re-enumeration is projected at more than one index. Retiring it must
        clear EVERY projection index from both maps — leaving one behind keeps
        that index falsely managed, so its later unrelated occupant could be
        released on stale integer membership alone."""
        actuator, _, _, _ = _make_initialized_actuator(metrics=MagicMock())
        # GPU-A capped at idx 0, then re-enumerated onto idx 1 and re-capped:
        # both indices project to GPU-A and both are in the managed set.
        actuator._capped_uuids.add("GPU-A")
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        actuator._managed_uuid_by_idx[1] = "GPU-A"
        power_agent._managed_gpu_indices.update({0, 1})

        actuator.retire_managed_uuid("GPU-A")

        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, actuator._managed_uuid_by_idx)
        self.assertNotIn(1, actuator._managed_uuid_by_idx)
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn(1, power_agent._managed_gpu_indices)

    def test_restore_default_does_not_reclaim_ownership(self):
        """A runtime release restores default through `restore_default`. That
        write must NOT re-record ownership (record_ownership=False): if it did,
        the just-released UUID would sit back in `_capped_uuids` for the next
        sweep to act on."""
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        discovery = handle.GetSystem.return_value.discovery
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", min_w=100, max_w=700, default_w=410
        )
        actuator._managed_uuid_by_idx[0] = "GPU-A"

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            result = actuator.restore_default(0)

        self.assertTrue(result)
        # The default was written, but ownership was NOT (re-)claimed.
        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, power_agent._managed_gpu_indices)

    def test_restore_default_by_uuid_does_not_reclaim_ownership(self):
        """Cold-start orphan recovery restores by UUID on a fresh process whose
        ownership set is empty. Restoring must not populate `_capped_uuids` —
        the GPU is being handed BACK to default, not adopted."""
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=MagicMock())
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410, current_w=300
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            result = actuator.restore_default_by_uuid("GPU-A")

        self.assertTrue(result)
        cfg = modules["pydcgm"].DcgmGroup.return_value.config.Set.call_args.args[0]
        self.assertEqual(cfg.mPowerLimit.val, 410)
        # Fresh process: the restore must not adopt the GPU into ownership.
        self.assertEqual(actuator.managed_uuids(), set())
        self.assertNotIn(0, power_agent._managed_gpu_indices)


# ---------------------------------------------------------------------------
# Stale-handle recovery (_with_reconnect)
# ---------------------------------------------------------------------------


class TestStaleHandleRecovery(unittest.TestCase):
    def test_connection_not_valid_triggers_single_retry(self):
        """First op() raises CONNECTION_NOT_VALID; second op() succeeds.
        Production code must rebuild the handle, flush groups, retry."""
        modules = _make_dcgm_modules()
        _wire_handle(modules["pydcgm"], gpu_ids=(0, 1))
        nvml = MagicMock()
        actuator = DcgmActuator(metrics=MagicMock())

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.init()

            # Seed a cached group so we can assert it's flushed.
            actuator._groups[0] = MagicMock()

            DCGMError = modules["dcgm_structs"].DCGMError
            CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

            call_count = {"n": 0}

            def op():
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise DCGMError(CONNECTION_NOT_VALID)
                return "ok"

            result = actuator._with_reconnect(op)

        self.assertEqual(result, "ok")
        self.assertEqual(call_count["n"], 2)
        # Groups must have been cleared during recovery — caching a
        # stale-handle group would re-raise on the next reconcile.
        self.assertEqual(actuator._groups, {})
        # DcgmHandle constructor was called twice: once in init(), once
        # during recovery in _with_reconnect.
        self.assertEqual(modules["pydcgm"].DcgmHandle.call_count, 2)

    def test_other_dcgm_errors_propagate_without_retry(self):
        """A NOT_SUPPORTED / GENERIC_ERROR / etc. must propagate
        immediately — recovering would mask the wrong class of failure."""
        modules = _make_dcgm_modules()
        _wire_handle(modules["pydcgm"])
        nvml = MagicMock()
        actuator = DcgmActuator(metrics=MagicMock())

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.init()

            DCGMError = modules["dcgm_structs"].DCGMError
            GENERIC_ERROR = modules["dcgm_structs"].DCGM_ST_GENERIC_ERROR

            calls = {"n": 0}

            def op():
                calls["n"] += 1
                raise DCGMError(GENERIC_ERROR)

            with self.assertRaises(DCGMError):
                actuator._with_reconnect(op)

        # Exactly one attempt — no retry, no reconnect.
        self.assertEqual(calls["n"], 1)
        self.assertEqual(modules["pydcgm"].DcgmHandle.call_count, 1)

    def test_persistent_connection_failure_propagates_after_one_retry(self):
        """If reconnect succeeds but op() raises CONNECTION_NOT_VALID
        again, the second exception propagates — no infinite retry."""
        modules = _make_dcgm_modules()
        _wire_handle(modules["pydcgm"])
        nvml = MagicMock()
        actuator = DcgmActuator(metrics=MagicMock())

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.init()

            DCGMError = modules["dcgm_structs"].DCGMError
            CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

            calls = {"n": 0}

            def op():
                calls["n"] += 1
                raise DCGMError(CONNECTION_NOT_VALID)

            with self.assertRaises(DCGMError):
                actuator._with_reconnect(op)

        self.assertEqual(calls["n"], 2)  # one initial + one retry

    def test_reconnect_init_failure_propagates_and_clears_stale_state(self):
        """A sustained hostengine outage during reconnect propagates; stale
        groups and identity maps are still flushed before the failed init."""
        modules = _make_dcgm_modules()
        _wire_handle(modules["pydcgm"])
        nvml = MagicMock()
        actuator = DcgmActuator(metrics=MagicMock())

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            actuator.init()
            actuator._groups[0] = MagicMock()
            actuator._dcgm_uuid_by_idx = ["GPU-aaaa"]
            actuator._nvml_index_by_uuid = {"GPU-aaaa": 0}

            DCGMError = modules["dcgm_structs"].DCGMError
            CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID
            modules["pydcgm"].DcgmHandle.side_effect = RuntimeError("hostengine down")

            with self.assertRaises(RuntimeError) as ctx:
                actuator._with_reconnect(
                    lambda: (_ for _ in ()).throw(DCGMError(CONNECTION_NOT_VALID))
                )

        self.assertIn("hostengine down", str(ctx.exception))
        self.assertEqual(actuator._groups, {})
        self.assertIsNone(actuator._dcgm_uuid_by_idx)
        self.assertIsNone(actuator._nvml_index_by_uuid)
        self.assertIsNone(actuator._handle)

    def test_non_dcgm_exception_propagates_immediately(self):
        """Random Python exceptions (not DCGMError subclasses) must
        also pass through — _with_reconnect is DCGM-specific."""
        actuator, modules, *_ = _make_initialized_actuator(metrics=MagicMock())
        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with self.assertRaises(ValueError):
                actuator._with_reconnect(
                    lambda: (_ for _ in ()).throw(ValueError("nope"))
                )


class TestResolveIdxTopologyGrowth(unittest.TestCase):
    """`_resolve_idx_for_uuid` must not falsely report a UUID
    "gone" when a reconnect GROWS the topology mid-scan. The whole scan runs in
    ONE `_with_reconnect` reading via `_read_uuid_raw`; a CONNECTION_NOT_VALID
    rebuilds discovery and restarts the scan against the new (larger) GPU set,
    so a UUID that moved to a newly discovered index is found, not pruned."""

    def test_resolve_idx_rescans_grown_topology_after_reconnect(self):
        actuator, modules, _, _ = _make_initialized_actuator(metrics=MagicMock())
        DCGMError = modules["dcgm_structs"].DCGMError
        CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

        # Pre-reconnect the actuator sees ONE GPU that is not the target.
        actuator._discovered_gpu_ids = [10]
        state = {"reconnected": False}

        def fake_init():
            # A hostengine restart re-enumerated MORE GPUs; the target UUID now
            # lives at the newly discovered index 1.
            actuator._discovered_gpu_ids = [10, 20]
            state["reconnected"] = True

        def read_uuid(idx):
            gid = actuator._discovered_gpu_ids[idx]
            if not state["reconnected"] and gid == 10:
                # The stale handle raises on the first probe, before growth.
                raise DCGMError(CONNECTION_NOT_VALID)
            return {10: "GPU-B", 20: "GPU-A"}[gid]

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with patch.object(actuator, "init", side_effect=fake_init), patch.object(
                actuator, "_read_uuid_raw", side_effect=read_uuid
            ):
                idx, scan_complete = actuator._resolve_idx_for_uuid("GPU-A")

        self.assertTrue(state["reconnected"])
        # Found at the GROWN index (1), NOT falsely reported gone (None).
        self.assertEqual(idx, 1)
        self.assertTrue(scan_complete)


class TestScanUuidIndexMap(unittest.TestCase):
    """`scan_uuid_index_map` builds a conclusive {uuid: index} snapshot inside
    ONE `_with_reconnect`, so cold-start orphan recovery can prune absent UUIDs
    without the per-index-`get_uuid` false-gone hole (a reconnect that GROWS the
    topology mid-scan is fully rescanned)."""

    def test_clean_scan_returns_full_map_conclusive(self):
        actuator, modules, _, _ = _make_initialized_actuator(metrics=MagicMock())
        actuator._discovered_gpu_ids = [10, 20]

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with patch.object(
                actuator,
                "_read_uuid_raw",
                side_effect=lambda idx: {0: "GPU-A", 1: "GPU-B"}[idx],
            ):
                mapping, conclusive = actuator.scan_uuid_index_map()

        self.assertEqual(mapping, {"GPU-A": 0, "GPU-B": 1})
        self.assertTrue(conclusive)

    def test_rescans_grown_topology_after_reconnect(self):
        """The reconnect-growth case: a UUID that moved to a newly-enumerated
        index must appear in the map (not be missing → pruned as absent)."""
        actuator, modules, _, _ = _make_initialized_actuator(metrics=MagicMock())
        DCGMError = modules["dcgm_structs"].DCGMError
        CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID

        actuator._discovered_gpu_ids = [10]
        state = {"reconnected": False}

        def fake_init():
            actuator._discovered_gpu_ids = [10, 20]
            state["reconnected"] = True

        def read_uuid(idx):
            gid = actuator._discovered_gpu_ids[idx]
            if not state["reconnected"] and gid == 10:
                raise DCGMError(CONNECTION_NOT_VALID)
            return {10: "GPU-B", 20: "GPU-A"}[gid]

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with patch.object(actuator, "init", side_effect=fake_init), patch.object(
                actuator, "_read_uuid_raw", side_effect=read_uuid
            ):
                mapping, conclusive = actuator.scan_uuid_index_map()

        self.assertTrue(state["reconnected"])
        # GPU-A (the moved UUID) is present at its GROWN index 1.
        self.assertEqual(mapping, {"GPU-B": 0, "GPU-A": 1})
        self.assertTrue(conclusive)

    def test_non_connection_error_is_inconclusive_but_keeps_others(self):
        actuator, modules, _, _ = _make_initialized_actuator(metrics=MagicMock())
        DCGMError = modules["dcgm_structs"].DCGMError
        GENERIC_ERROR = modules["dcgm_structs"].DCGM_ST_GENERIC_ERROR
        actuator._discovered_gpu_ids = [10, 20]

        def read_uuid(idx):
            if idx == 0:
                return "GPU-A"
            raise DCGMError(GENERIC_ERROR)  # non-connection → mark inconclusive

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with patch.object(actuator, "_read_uuid_raw", side_effect=read_uuid):
                mapping, conclusive = actuator.scan_uuid_index_map()

        self.assertEqual(mapping, {"GPU-A": 0})
        self.assertFalse(conclusive)

    def test_empty_cached_topology_is_inconclusive(self):
        """0 discovered GPUs on the DCGM path is NOT trusted as a genuinely
        empty node (Power Agent runs on GPU nodes): `range(0)` issues no
        hostengine call, so `_with_reconnect` never fires and a naive scan would
        return ({}, True), pruning EVERY persisted UUID. It must be inconclusive
        so recovery retains them."""
        actuator, modules, _, _ = _make_initialized_actuator(metrics=MagicMock())
        actuator._discovered_gpu_ids = []

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            mapping, conclusive = actuator.scan_uuid_index_map()

        self.assertEqual(mapping, {})
        self.assertFalse(conclusive)

    def test_sustained_outage_returns_empty_inconclusive(self):
        """A CONNECTION_NOT_VALID that persists across the single reconnect
        retry escapes `_with_reconnect`; scan_uuid_index_map reports ({}, False)
        so recovery neither prunes nor restores (device_count==0-due-to-outage
        is NOT mistaken for a genuinely empty node)."""
        actuator, modules, _, _ = _make_initialized_actuator(metrics=MagicMock())
        DCGMError = modules["dcgm_structs"].DCGMError
        CONNECTION_NOT_VALID = modules["dcgm_structs"].DCGM_ST_CONNECTION_NOT_VALID
        actuator._discovered_gpu_ids = [10]

        def read_uuid(idx):
            raise DCGMError(CONNECTION_NOT_VALID)  # persists across the retry

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            with patch.object(actuator, "init", side_effect=lambda: None), patch.object(
                actuator, "_read_uuid_raw", side_effect=read_uuid
            ):
                mapping, conclusive = actuator.scan_uuid_index_map()

        self.assertEqual(mapping, {})
        self.assertFalse(conclusive)


class TestRestoreUpdatesAppliedGauge(unittest.TestCase):
    """Stale-gauge guard: a restore to factory default
    writes with record_ownership=False, but the applied-limit gauge tracks what
    is LIVE on the GPU, so it must still tick — otherwise Prometheus keeps
    reporting the released cap forever."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def test_restore_default_updates_applied_limit_gauge(self):
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        discovery = handle.GetSystem.return_value.discovery
        actuator._managed_uuid_by_idx[0] = "GPU-A"
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=410
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}), patch(
            "power_agent._persist_managed_gpus"
        ):
            actuator.restore_default(0)

        metrics.applied_limit_watts.labels.assert_called_with(gpu="0")
        metrics.applied_limit_watts.labels.return_value.set.assert_called_with(410)

    def test_restore_default_by_uuid_already_at_default_syncs_gauge(self):
        """When `restore_default_by_uuid` finds the GPU already at default it
        writes nothing (returns None) but must still sync the gauge to the LIVE
        value, so a GPU restored to default externally stops reporting our old
        cap."""
        metrics = MagicMock()
        actuator, modules, handle, _ = _make_initialized_actuator(metrics=metrics)
        discovery = handle.GetSystem.return_value.discovery
        actuator._discovered_gpu_ids = [10]
        discovery.GetGpuAttributes.side_effect = lambda gid: _make_gpu_attrs(
            "GPU-A", default_w=700, current_w=700
        )

        with patch.dict("sys.modules", {**modules, "pynvml": MagicMock()}):
            result = actuator.restore_default_by_uuid("GPU-A")

        self.assertIsNone(result)
        modules["pydcgm"].DcgmGroup.return_value.config.Set.assert_not_called()
        metrics.applied_limit_watts.labels.assert_called_with(gpu="0")
        metrics.applied_limit_watts.labels.return_value.set.assert_called_with(700)


if __name__ == "__main__":
    unittest.main()
