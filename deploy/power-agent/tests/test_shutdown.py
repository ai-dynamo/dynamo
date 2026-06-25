# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SIGTERM handler.

v1.6 wiring (per review comment #6): `_handle_sigterm` now dispatches
through the module-level `_active_actuator` (set by
`PowerAgent.__init__`) instead of going straight to `pynvml`. This is
required so that on `actuator: dcgm` the cap restore flows through the
hostengine and DCGM's target-config record stays consistent with the
driver-level cap (otherwise DCGM's reset/reinit auto-reapply would
silently re-push the *old* cap, contradicting the SIGTERM intent).

This file covers three scenarios:
  1. With an actuator registered — `restore_default` and `shutdown` are
     called on the actuator, never directly via pynvml.
  2. Without an actuator registered (defensive fallback) — raw NVML
     restore runs so the GPU isn't left at a custom cap.
  3. Edge cases: no managed GPUs, actuator.restore_default raising,
     _shutdown event still fires.
"""

import os
import runpy
import signal
import unittest
from unittest.mock import MagicMock, patch

import power_agent


class TestMainEntrypoint(unittest.TestCase):
    def test_fresh_module_copy_shares_managed_state(self):
        """A second, independently-executed copy of power_agent.py — exactly
        what `python /app/power_agent.py` (module `__main__`) plus the
        actuator's `import power_agent` produce — must alias the SAME
        managed-state objects. Otherwise caps recorded through one copy are
        invisible to the SIGTERM handler running in the other, and every cap
        leaks past graceful shutdown.
        """
        import managed_state

        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "power_agent.py"
        )

        # run_name != "__main__" so the entrypoint guard does not invoke main().
        ns = runpy.run_path(path, run_name="power_agent_fresh_copy")

        self.assertIs(ns["_managed_gpu_indices"], managed_state.managed_gpu_indices)
        self.assertIs(ns["_previously_managed"], managed_state.previously_managed)
        self.assertEqual(ns["_MANAGED_STATE_PATH"], managed_state.MANAGED_STATE_PATH)


class _SigtermTestBase(unittest.TestCase):
    def setUp(self):
        # Reset module-level state between tests so each starts from a
        # known clean slate (some prior test may have left _active_actuator
        # populated).
        power_agent._managed_gpu_indices.clear()
        power_agent._shutdown.clear()
        self._saved_active = power_agent._active_actuator
        power_agent._active_actuator = None

    def tearDown(self):
        power_agent._active_actuator = self._saved_active
        power_agent._managed_gpu_indices.clear()
        power_agent._shutdown.clear()


class TestSigtermViaActuator(_SigtermTestBase):
    """v1.6 happy path — actuator registered, dispatch via it."""

    def test_restores_default_via_actuator_for_each_managed_gpu(self):
        power_agent._managed_gpu_indices.update([0, 1, 2])
        actuator = MagicMock()
        actuator.name = "nvml"
        power_agent._active_actuator = actuator

        # pynvml should be untouched on this path.
        mock_nvml = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        # Restore was dispatched to the actuator for each managed GPU.
        self.assertEqual(actuator.restore_default.call_count, 3)
        actuator.restore_default.assert_any_call(0)
        actuator.restore_default.assert_any_call(1)
        actuator.restore_default.assert_any_call(2)
        # Shutdown was dispatched to the actuator (not raw nvmlShutdown).
        actuator.shutdown.assert_called_once()
        mock_nvml.nvmlShutdown.assert_not_called()
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        self.assertTrue(power_agent._shutdown.is_set())

    def test_dcgm_actuator_routes_restore_through_dcgm_not_nvml(self):
        """The load-bearing test for review comment #6.

        Pre-v1.6, SIGTERM on `actuator: dcgm` used raw NVML, leaving the
        DCGM hostengine's target-config record stale. This test verifies
        the dispatch now goes through the DCGM actuator surface — so the
        actual production code will issue `dcgmConfigSet(default_w)` and
        keep target-config in sync.
        """
        power_agent._managed_gpu_indices.add(7)
        dcgm_actuator = MagicMock()
        dcgm_actuator.name = "dcgm"
        power_agent._active_actuator = dcgm_actuator

        mock_nvml = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        dcgm_actuator.restore_default.assert_called_once_with(7)
        dcgm_actuator.shutdown.assert_called_once()
        # CRITICAL: raw NVML must NEVER be called when an actuator is
        # registered. If this fires, comment #6 has regressed.
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        mock_nvml.nvmlShutdown.assert_not_called()

    def test_actuator_restore_failure_does_not_prevent_shutdown(self):
        """A failing restore on one GPU must not stop the loop or
        prevent actuator.shutdown() from being called."""
        power_agent._managed_gpu_indices.update([0, 1])
        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.restore_default.side_effect = [
            RuntimeError("DCGM down"),
            None,  # GPU 1 succeeds
        ]
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        # Both GPUs were attempted.
        self.assertEqual(actuator.restore_default.call_count, 2)
        # Shutdown still ran despite the per-GPU failure.
        actuator.shutdown.assert_called_once()
        self.assertTrue(power_agent._shutdown.is_set())

    def test_actuator_shutdown_failure_does_not_prevent_shutdown_event(self):
        """Even if actuator.shutdown() raises (e.g. DCGM hostengine
        gone), the _shutdown event must still be set so the main loop
        can exit. Otherwise the agent would hang on SIGTERM."""
        power_agent._managed_gpu_indices.add(0)
        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.shutdown.side_effect = RuntimeError("hostengine gone")
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        self.assertTrue(power_agent._shutdown.is_set())

    def test_actuator_shutdown_failure_is_logged_not_silently_swallowed(self):
        """Per PR #9682 CodeRabbit review on power_agent.py:355.

        Pre-fix the shutdown try/except swallowed failures silently
        (`except Exception: pass`), so a DCGM hostengine fault or NVML
        teardown error at SIGTERM time was invisible in pod logs.
        We still don't re-raise (the agent MUST exit cleanly) but the
        failure now writes a traceback at ERROR level so operators
        can correlate with hostengine / driver events post-mortem.
        """
        power_agent._managed_gpu_indices.add(0)
        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.shutdown.side_effect = RuntimeError("hostengine gone")
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with self.assertLogs("power_agent", level="ERROR") as cm:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # The original RuntimeError message must appear in the log
        # output (logger.exception attaches the traceback).
        joined = "\n".join(cm.output)
        self.assertIn("hostengine gone", joined)
        # Sanity: shutdown event still fires.
        self.assertTrue(power_agent._shutdown.is_set())

    def test_raw_nvml_shutdown_failure_is_logged(self):
        """Defensive-fallback path: when no actuator is registered and
        `pynvml.nvmlShutdown()` itself raises, the failure must be
        logged (was silently swallowed pre-fix)."""
        # No managed GPUs so we skip the per-GPU restore loop and go
        # straight to the actuator-or-NVML shutdown branch.
        mock_nvml = MagicMock()
        mock_nvml.nvmlShutdown.side_effect = RuntimeError(
            "NVML library gone (driver unloaded)"
        )

        with patch.object(power_agent, "pynvml", mock_nvml):
            with self.assertLogs("power_agent", level="ERROR") as cm:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        joined = "\n".join(cm.output)
        self.assertIn("NVML library gone", joined)
        self.assertTrue(power_agent._shutdown.is_set())


class TestSigtermFallback(_SigtermTestBase):
    """Defensive fallback — actuator not yet registered.

    Should only happen if SIGTERM fires during PowerAgent.__init__ before
    `_active_actuator = self._actuator` runs. We still want a sane
    restore on those rare timings, so the handler falls through to raw
    NVML rather than abandoning a cap'd GPU.
    """

    def test_no_actuator_restores_via_raw_nvml(self):
        power_agent._managed_gpu_indices.update([0, 1, 2])
        # _active_actuator stays None per _SigtermTestBase.setUp.
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"handle_{i}"
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700_000
        mock_nvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # Pre-v1.6 behaviour preserved as the fallback path.
        self.assertEqual(mock_nvml.nvmlDeviceSetPowerManagementLimit.call_count, 3)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_any_call("handle_0", 700_000)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_any_call("handle_1", 700_000)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_any_call("handle_2", 700_000)
        mock_nvml.nvmlShutdown.assert_called_once()
        self.assertTrue(power_agent._shutdown.is_set())

    def test_no_managed_gpus_still_shuts_down(self):
        # Neither actuator nor managed GPUs.
        mock_nvml = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        mock_nvml.nvmlShutdown.assert_called_once()
        self.assertTrue(power_agent._shutdown.is_set())

    def test_nvml_error_does_not_prevent_shutdown(self):
        power_agent._managed_gpu_indices.add(0)
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("nvml error")
        mock_nvml.NVMLError = Exception

        with patch.object(power_agent, "pynvml", mock_nvml):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        mock_nvml.nvmlShutdown.assert_called_once()
        self.assertTrue(power_agent._shutdown.is_set())


class TestSigtermPrunesManagedGpusState(_SigtermTestBase):
    """Per PR9790 Codex adversarial review (finding #1).

    Pre-fix `_handle_sigterm` restored each managed GPU to default but
    never pruned that GPU's UUID from `managed_gpus.json`. On the next
    startup, orphan recovery would treat the stale UUID as agent-owned
    and could clobber a cap applied by another workflow on a now-idle
    GPU (different DGD, manual `nvidia-smi -pl`, vendor firmware).

    This file pins:
      1. Successful actuator restore → UUID discarded + state persisted.
      2. Failed restore → UUID NOT discarded (cap may still be live).
      3. UUID-lookup failure post-restore → warn, don't crash, don't prune.
      4. Raw-NVML fallback path → uses `_nvml_uuid(handle)` and prunes.
      5. Persist failure → log, still proceed to shutdown.
    """

    def setUp(self):
        super().setUp()
        self._saved_pm = power_agent._previously_managed.copy()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._previously_managed.clear()
        power_agent._previously_managed.update(self._saved_pm)
        super().tearDown()

    def test_successful_restore_prunes_uuid_and_persists(self):
        power_agent._managed_gpu_indices.update([0, 1])
        power_agent._previously_managed.update({"uuid-A", "uuid-B", "uuid-stale"})

        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.side_effect = lambda idx: {0: "uuid-A", 1: "uuid-B"}[idx]
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # Both restored UUIDs discarded; the unrelated "uuid-stale" entry
        # remains (we only own GPUs whose indices are in
        # _managed_gpu_indices).
        self.assertEqual(power_agent._previously_managed, {"uuid-stale"})
        # State persisted exactly once with the pruned set.
        persist.assert_called_once_with({"uuid-stale"})

    def test_successful_restore_prunes_actuator_managed_uuid_when_available(self):
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.update({"uuid-capped", "uuid-now-at-index"})

        class _DcgmLikeActuator:
            name = "dcgm"

            def __init__(self):
                self.restore_default = MagicMock()
                self.shutdown = MagicMock()
                self.get_uuid = MagicMock(return_value="uuid-now-at-index")
                self._managed_uuid_for_idx = MagicMock(return_value="uuid-capped")

            def managed_uuid_for_idx(self, gpu_idx):
                return self._managed_uuid_for_idx(gpu_idx)

        actuator = _DcgmLikeActuator()
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        actuator.restore_default.assert_called_once_with(0)
        actuator._managed_uuid_for_idx.assert_called_once_with(0)
        actuator.get_uuid.assert_not_called()
        self.assertEqual(power_agent._previously_managed, {"uuid-now-at-index"})
        persist.assert_called_once_with({"uuid-now-at-index"})

    def test_skipped_restore_does_not_prune(self):
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("uuid-capped")

        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.restore_default.return_value = False
        actuator.get_uuid.return_value = "uuid-capped"
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        actuator.restore_default.assert_called_once_with(0)
        actuator.get_uuid.assert_not_called()
        self.assertEqual(power_agent._previously_managed, {"uuid-capped"})
        persist.assert_called_once_with({"uuid-capped"})

    def test_failed_restore_does_not_prune(self):
        """If `restore_default` raises, the cap may still be live — we
        must NOT prune so the next startup's orphan recovery can reset it.
        """
        power_agent._managed_gpu_indices.update([0, 1])
        power_agent._previously_managed.update({"uuid-A", "uuid-B"})

        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.restore_default.side_effect = [
            RuntimeError("DCGM down on GPU 0"),
            None,  # GPU 1 succeeds
        ]
        actuator.get_uuid.side_effect = lambda idx: {0: "uuid-A", 1: "uuid-B"}[idx]
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # GPU 0's UUID is retained (restore failed); GPU 1's is pruned.
        self.assertEqual(power_agent._previously_managed, {"uuid-A"})
        persist.assert_called_once_with({"uuid-A"})

    def test_uuid_lookup_failure_is_logged_but_not_fatal(self):
        """`actuator.get_uuid` raising after a successful restore must not
        crash the SIGTERM handler. The state file retains the entry; the
        next startup's orphan recovery harmlessly sees current_w >=
        default_w and skips the redundant write."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("uuid-A")

        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.side_effect = RuntimeError("UUID query failed")
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with self.assertLogs("power_agent", level="WARNING") as cm:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        joined = "\n".join(cm.output)
        self.assertIn("UUID query failed", joined)
        # UUID NOT pruned because we couldn't identify which one to prune.
        self.assertEqual(power_agent._previously_managed, {"uuid-A"})
        self.assertTrue(power_agent._shutdown.is_set())

    def test_raw_nvml_fallback_prunes_via_nvml_uuid(self):
        """When actuator is not registered, the fallback path must still
        prune using `_nvml_uuid(handle)`."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("uuid-FALLBACK")

        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700_000

        with patch.object(power_agent, "pynvml", mock_nvml):
            with patch.object(power_agent, "_nvml_uuid", return_value="uuid-FALLBACK"):
                with patch.object(power_agent, "_persist_managed_gpus") as persist:
                    power_agent._handle_sigterm(signal.SIGTERM, None)

        self.assertEqual(power_agent._previously_managed, set())
        persist.assert_called_once_with(set())

    def test_persist_failure_does_not_prevent_shutdown(self):
        """Disk write failure at SIGTERM (read-only volume, full disk)
        must not hang the agent. The shutdown event still fires."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("uuid-A")

        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.return_value = "uuid-A"
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(
                power_agent,
                "_persist_managed_gpus",
                side_effect=OSError("read-only filesystem"),
            ):
                with self.assertLogs("power_agent", level="WARNING") as cm:
                    power_agent._handle_sigterm(signal.SIGTERM, None)

        joined = "\n".join(cm.output)
        self.assertIn("read-only filesystem", joined)
        self.assertTrue(power_agent._shutdown.is_set())


class TestSigtermDcgmUuidSweep(_SigtermTestBase):
    """PR9790 review follow-up (@sttts): the index-keyed restore loop can
    MISS a still-capped GPU when DCGM re-enumerated and a later reconcile
    re-capped its old index onto a different physical GPU. The displaced
    UUID survives in the per-process ownership set, so SIGTERM sweeps
    `actuator.managed_uuids()` through `restore_default_by_uuid` to catch
    the leak.

    Critically (per the independent review caveat), the sweep is scoped to
    UUIDs THIS process capped — not the cross-incarnation `_previously_managed`
    set — and prunes ONLY on a True (restored-a-live-cap) result, matching
    the index loop's and orphan recovery's invariants.
    """

    class _SweepActuator:
        """DCGM-like actuator exposing the UUID sweep surface.

        `owned` is the set of UUIDs THIS process capped (what
        `managed_uuids()` returns) — the ownership scope the sweep must
        respect. It is intentionally distinct from the persisted
        `_previously_managed` set so tests can pin that cross-incarnation
        entries are never swept.
        """

        name = "dcgm"

        def __init__(
            self, *, owned=None, sweep_results=None, idx_uuid=None, sweep_raises=()
        ):
            self.shutdown = MagicMock()
            self.restore_default = MagicMock(return_value=True)
            self._owned = set(owned or [])
            self._sweep_results = sweep_results or {}
            self._idx_uuid = idx_uuid or {}
            self._sweep_raises = set(sweep_raises)
            self.swept = []

        def get_uuid(self, gpu_idx):
            return self._idx_uuid.get(gpu_idx)

        def managed_uuid_for_idx(self, gpu_idx):
            return self._idx_uuid.get(gpu_idx)

        def managed_uuids(self):
            return set(self._owned)

        def restore_default_by_uuid(self, uuid):
            self.swept.append(uuid)
            if uuid in self._sweep_raises:
                raise RuntimeError(f"DCGM write failed for {uuid}")
            return self._sweep_results.get(uuid)

    def setUp(self):
        super().setUp()
        self._saved_pm = power_agent._previously_managed.copy()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._previously_managed.clear()
        power_agent._previously_managed.update(self._saved_pm)
        super().tearDown()

    def test_sweep_restores_leaked_uuid_not_in_index_set(self):
        """sttts's exact trace: idx 0 was re-capped onto GPU-B (so the index
        loop restores and prunes GPU-B), while the displaced GPU-A survives
        only in `_previously_managed`. The sweep must restore GPU-A by UUID
        and prune it."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.update({"GPU-A", "GPU-B"})

        actuator = self._SweepActuator(
            owned={"GPU-A", "GPU-B"},
            idx_uuid={0: "GPU-B"},
            # GPU-B was already restored to default by the index loop, so the
            # sweep sees it at default (None); GPU-A is the leaked one.
            sweep_results={"GPU-A": True, "GPU-B": None},
        )
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # Index loop pruned GPU-B; sweep restored + pruned the leaked GPU-A.
        self.assertEqual(set(actuator.swept), {"GPU-A", "GPU-B"})
        self.assertEqual(power_agent._previously_managed, set())
        persist.assert_called_once_with(set())
        self.assertTrue(power_agent._shutdown.is_set())

    def test_sweep_ignores_unowned_persisted_uuid(self):
        """The ownership guard: a cross-incarnation UUID that startup orphan
        recovery KEPT (e.g. a GPU with a running workload) but THIS process
        never capped must NEVER be swept — even if it would resolve to a
        restorable below-default cap. Resetting it on shutdown would clobber
        a cap owned by another workflow."""
        power_agent._previously_managed.add("uuid-foreign")

        actuator = self._SweepActuator(
            owned=set(),  # this process capped nothing
            sweep_results={"uuid-foreign": True},  # would restore IF swept
        )
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # Never resolved/restored, and left untouched in the persisted set.
        self.assertEqual(actuator.swept, [])
        self.assertEqual(power_agent._previously_managed, {"uuid-foreign"})
        persist.assert_called_once_with({"uuid-foreign"})

    def test_sweep_keeps_owned_uuid_already_at_default(self):
        """An owned UUID that resolves to an already-at-default GPU (None)
        must be LEFT in place, not pruned — matching orphan recovery and
        avoiding the over-aggressive drop the reviewer warned about."""
        power_agent._previously_managed.add("uuid-mine")

        actuator = self._SweepActuator(
            owned={"uuid-mine"}, sweep_results={"uuid-mine": None}
        )
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        self.assertEqual(actuator.swept, ["uuid-mine"])
        self.assertEqual(power_agent._previously_managed, {"uuid-mine"})
        persist.assert_called_once_with({"uuid-mine"})

    def test_sweep_keeps_uuid_on_inconclusive_scan(self):
        """A False result (transient DCGM outage / incomplete relocation
        scan) means the GPU may still carry our cap — keep the UUID so
        cold-start orphan recovery retries."""
        power_agent._previously_managed.add("uuid-indeterminate")

        actuator = self._SweepActuator(
            owned={"uuid-indeterminate"},
            sweep_results={"uuid-indeterminate": False},
        )
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        self.assertEqual(power_agent._previously_managed, {"uuid-indeterminate"})
        persist.assert_called_once_with({"uuid-indeterminate"})

    def test_sweep_exception_keeps_uuid_and_still_shuts_down(self):
        """If `restore_default_by_uuid` raises, the cap may still be live, so
        keep the UUID; the failure is logged and shutdown still completes."""
        power_agent._previously_managed.add("uuid-boom")

        actuator = self._SweepActuator(owned={"uuid-boom"}, sweep_raises={"uuid-boom"})
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with self.assertLogs("power_agent", level="ERROR") as cm:
                with patch.object(power_agent, "_persist_managed_gpus") as persist:
                    power_agent._handle_sigterm(signal.SIGTERM, None)

        self.assertIn("uuid-boom", "\n".join(cm.output))
        self.assertEqual(power_agent._previously_managed, {"uuid-boom"})
        persist.assert_called_once_with({"uuid-boom"})
        self.assertTrue(power_agent._shutdown.is_set())

    def test_sweep_does_not_run_on_nvml_actuator(self):
        """NVML indices are stable within a process, so the index loop is
        already complete and there is no UUID-relocation surface.

        The real `NvmlActuator` gained `restore_default_by_uuid` (for
        identity-stable orphan recovery) but deliberately does NOT track
        capped UUIDs, so it has NO `managed_uuids()`. The sweep gate must
        therefore require BOTH methods: gating on `restore_default_by_uuid`
        alone would enter the sweep on NVML and then `AttributeError` on the
        missing `managed_uuids()`, skipping persist/shutdown/_shutdown.set()
        (PR9790 review follow-up). This actuator mirrors that exact surface
        — `restore_default_by_uuid` present on the class, `managed_uuids`
        absent — and the method raises if the sweep wrongly invokes it.
        """

        class _NvmlLikeActuator:
            name = "nvml"

            # On the class (not the instance) so `hasattr(type(actuator),
            # ...)` matches it exactly as it does the real NvmlActuator.
            def restore_default_by_uuid(self, uuid):  # pragma: no cover
                raise AssertionError("UUID sweep must not run on the NVML actuator")

            def __init__(self):
                self.restore_default = MagicMock(return_value=True)
                self.shutdown = MagicMock()
                self.get_uuid = MagicMock(return_value="uuid-managed")

        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.update({"uuid-managed", "uuid-leftover"})
        actuator = _NvmlLikeActuator()
        # Pin the real NVML surface: relocating restore present, ownership-set
        # tracking absent — exactly the combination the sweep gate must reject.
        self.assertTrue(hasattr(type(actuator), "restore_default_by_uuid"))
        self.assertFalse(hasattr(type(actuator), "managed_uuids"))
        power_agent._active_actuator = actuator

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # Index loop pruned the managed UUID; the leftover is preserved
        # because no UUID sweep runs on NVML. Critically, shutdown completed
        # (persist + actuator.shutdown + _shutdown.set) — the regression
        # would have skipped all three via an AttributeError.
        self.assertEqual(power_agent._previously_managed, {"uuid-leftover"})
        persist.assert_called_once_with({"uuid-leftover"})
        actuator.shutdown.assert_called_once()
        self.assertTrue(power_agent._shutdown.is_set())


if __name__ == "__main__":
    unittest.main()
