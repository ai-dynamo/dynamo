# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for graceful shutdown.

Shutdown is split in two:

  * ``_handle_sigterm`` is the signal handler. It runs on the main thread
    between bytecodes and must do NOTHING but request shutdown (set the
    ``_shutdown`` event) — performing cap restores there races an in-flight
    reconcile that could record a still-live cap after the handler swept and
    shut down.
  * ``_shutdown_cleanup`` performs the actual restore + UUID sweep + persist +
    actuator shutdown. ``run()`` calls it exactly once from its ``finally``
    after the reconcile loop exits, so the in-flight write is fully recorded
    before cleanup runs. It always receives the live actuator (``run()`` passes
    ``self._actuator``), so there is no None / raw-NVML fallback.

Restore is dispatched through the actuator so ``actuator: dcgm`` restores via
``dcgmConfigSet`` and keeps DCGM's target-config record consistent with the
driver-level cap.
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
        invisible to the shutdown cleanup running in the other, and every cap
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


class _ShutdownTestBase(unittest.TestCase):
    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._shutdown.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._shutdown.clear()


class TestSigtermHandlerRequestsShutdownOnly(_ShutdownTestBase):
    """The signal handler must ONLY set the shutdown event — no restores, no
    actuator shutdown, no persistence. Doing heavy work in a signal handler is
    the race the P1#2 fix removes."""

    def test_handler_sets_shutdown_event(self):
        self.assertFalse(power_agent._shutdown.is_set())
        power_agent._handle_sigterm(signal.SIGTERM, None)
        self.assertTrue(power_agent._shutdown.is_set())

    def test_handler_does_no_cleanup_work(self):
        power_agent._managed_gpu_indices.update([0, 1])
        with patch.object(power_agent, "_shutdown_cleanup") as cleanup:
            power_agent._handle_sigterm(signal.SIGTERM, None)
        # The handler defers all cleanup to run()'s finally.
        cleanup.assert_not_called()


class TestRunLoopRunsCleanupOnce(_ShutdownTestBase):
    """`run()` performs cleanup exactly once from its `finally`, after the
    in-flight reconcile has returned — never from the signal handler."""

    def _bare_agent(self, reconcile):
        agent = power_agent.PowerAgent.__new__(power_agent.PowerAgent)
        agent._actuator = MagicMock()
        agent.node_name = "node-1"
        agent.safe_default_watts = 500
        agent.reconcile_once = reconcile
        return agent

    def test_cleanup_runs_once_on_normal_shutdown(self):
        def reconcile():
            power_agent._shutdown.set()  # SIGTERM equivalent

        agent = self._bare_agent(reconcile)
        with patch.object(power_agent.signal, "signal"), patch.object(
            power_agent, "_shutdown_cleanup"
        ) as cleanup:
            agent.run()

        cleanup.assert_called_once_with(agent._actuator)

    def test_cleanup_still_runs_when_reconcile_raises(self):
        """A crash in the reconcile loop must not skip cleanup — the cap
        restore is the whole point of graceful shutdown."""

        def reconcile():
            power_agent._shutdown.set()
            raise RuntimeError("reconcile blew up")

        agent = self._bare_agent(reconcile)
        with patch.object(power_agent.signal, "signal"), patch.object(
            power_agent, "_shutdown_cleanup"
        ) as cleanup:
            agent.run()

        cleanup.assert_called_once_with(agent._actuator)


class TestCleanupViaActuator(_ShutdownTestBase):
    """`_shutdown_cleanup` dispatches restore + shutdown through the actuator."""

    def test_restores_default_via_actuator_for_each_managed_gpu(self):
        power_agent._managed_gpu_indices.update([0, 1, 2])
        actuator = MagicMock()
        actuator.name = "nvml"

        power_agent._shutdown_cleanup(actuator)

        self.assertEqual(actuator.restore_default.call_count, 3)
        actuator.restore_default.assert_any_call(0)
        actuator.restore_default.assert_any_call(1)
        actuator.restore_default.assert_any_call(2)
        actuator.shutdown.assert_called_once()

    def test_dcgm_actuator_routes_restore_through_dcgm(self):
        """Load-bearing: on `actuator: dcgm`, cleanup must dispatch through
        the DCGM actuator surface (which issues `dcgmConfigSet(default_w)`),
        not raw NVML."""
        power_agent._managed_gpu_indices.add(7)
        dcgm_actuator = MagicMock()
        dcgm_actuator.name = "dcgm"

        power_agent._shutdown_cleanup(dcgm_actuator)

        dcgm_actuator.restore_default.assert_called_once_with(7)
        dcgm_actuator.shutdown.assert_called_once()

    def test_per_gpu_restore_failure_does_not_prevent_shutdown(self):
        """A failing restore on one GPU must not stop the loop or prevent
        actuator.shutdown() from being called."""
        power_agent._managed_gpu_indices.update([0, 1])
        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.restore_default.side_effect = [RuntimeError("DCGM down"), None]

        power_agent._shutdown_cleanup(actuator)

        self.assertEqual(actuator.restore_default.call_count, 2)
        actuator.shutdown.assert_called_once()

    def test_actuator_shutdown_failure_is_logged_not_raised(self):
        """Even if actuator.shutdown() raises, cleanup must not propagate — the
        failure is logged (with traceback) so run()'s finally completes and the
        container exits cleanly (PR #9682 CodeRabbit review)."""
        power_agent._managed_gpu_indices.add(0)
        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.shutdown.side_effect = RuntimeError("hostengine gone")

        with self.assertLogs("power_agent", level="ERROR") as cm:
            power_agent._shutdown_cleanup(actuator)  # must not raise

        self.assertIn("hostengine gone", "\n".join(cm.output))

    def test_no_managed_gpus_still_shuts_actuator_down(self):
        actuator = MagicMock()
        actuator.name = "nvml"

        power_agent._shutdown_cleanup(actuator)

        actuator.restore_default.assert_not_called()
        actuator.shutdown.assert_called_once()


class TestCleanupPrunesManagedGpusState(_ShutdownTestBase):
    """Cleanup restores + prunes managed-GPU state.

    Cleanup restores each managed GPU to default AND prunes that GPU's UUID
    from `managed_gpus.json`; otherwise the next startup's orphan recovery
    treats the stale UUID as agent-owned and could clobber a cap applied by
    another workflow on a now-idle GPU.
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

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        # Both restored UUIDs discarded; the unrelated "uuid-stale" entry
        # remains (we only own GPUs whose indices are in _managed_gpu_indices).
        self.assertEqual(power_agent._previously_managed, {"uuid-stale"})
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

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

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

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        actuator.restore_default.assert_called_once_with(0)
        actuator.get_uuid.assert_not_called()
        self.assertEqual(power_agent._previously_managed, {"uuid-capped"})
        persist.assert_called_once_with({"uuid-capped"})

    def test_failed_restore_does_not_prune(self):
        """If `restore_default` raises, the cap may still be live — we must NOT
        prune so the next startup's orphan recovery can reset it."""
        power_agent._managed_gpu_indices.update([0, 1])
        power_agent._previously_managed.update({"uuid-A", "uuid-B"})

        actuator = MagicMock()
        actuator.name = "dcgm"
        actuator.restore_default.side_effect = [
            RuntimeError("DCGM down on GPU 0"),
            None,  # GPU 1 succeeds
        ]
        actuator.get_uuid.side_effect = lambda idx: {0: "uuid-A", 1: "uuid-B"}[idx]

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        # GPU 0's UUID is retained (restore failed); GPU 1's is pruned.
        self.assertEqual(power_agent._previously_managed, {"uuid-A"})
        persist.assert_called_once_with({"uuid-A"})

    def test_uuid_lookup_failure_is_logged_but_not_fatal(self):
        """`actuator.get_uuid` raising after a successful restore must not crash
        cleanup. The state file retains the entry; the next startup harmlessly
        sees current_w >= default_w and skips."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("uuid-A")

        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.side_effect = RuntimeError("UUID query failed")

        with patch.object(power_agent, "_persist_managed_gpus"):
            with self.assertLogs("power_agent", level="WARNING") as cm:
                power_agent._shutdown_cleanup(actuator)

        self.assertIn("UUID query failed", "\n".join(cm.output))
        self.assertEqual(power_agent._previously_managed, {"uuid-A"})

    def test_persist_failure_does_not_prevent_shutdown(self):
        """Disk write failure at shutdown (read-only volume, full disk) must not
        stop the actuator shutdown."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("uuid-A")

        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.return_value = "uuid-A"

        with patch.object(
            power_agent,
            "_persist_managed_gpus",
            side_effect=OSError("read-only filesystem"),
        ):
            with self.assertLogs("power_agent", level="WARNING") as cm:
                power_agent._shutdown_cleanup(actuator)

        self.assertIn("read-only filesystem", "\n".join(cm.output))
        actuator.shutdown.assert_called_once()


class TestCleanupDcgmUuidSweep(_ShutdownTestBase):
    """The index-keyed restore loop can MISS a
    still-capped GPU when DCGM re-enumerated and a later reconcile re-capped its
    old index onto a different physical GPU. The displaced UUID survives in the
    per-process ownership set, so cleanup sweeps `actuator.managed_uuids()`
    through `restore_default_by_uuid` to catch the leak.

    The sweep is scoped to UUIDs THIS process capped — not the cross-incarnation
    `_previously_managed` set. It prunes on any CONCLUSIVE result — True or None
    — and retains only on the inconclusive False.
    """

    class _SweepActuator:
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
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.update({"GPU-A", "GPU-B"})

        actuator = self._SweepActuator(
            owned={"GPU-A", "GPU-B"},
            idx_uuid={0: "GPU-B"},
            sweep_results={"GPU-A": True, "GPU-B": None},
        )

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        self.assertEqual(set(actuator.swept), {"GPU-A", "GPU-B"})
        self.assertEqual(power_agent._previously_managed, set())
        persist.assert_called_once_with(set())

    def test_sweep_ignores_unowned_persisted_uuid(self):
        """The ownership guard: a cross-incarnation UUID startup recovery KEPT
        but THIS process never capped must NEVER be swept, even if it would
        resolve to a restorable below-default cap."""
        power_agent._previously_managed.add("uuid-foreign")

        actuator = self._SweepActuator(
            owned=set(), sweep_results={"uuid-foreign": True}
        )

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        self.assertEqual(actuator.swept, [])
        self.assertEqual(power_agent._previously_managed, {"uuid-foreign"})
        persist.assert_called_once_with({"uuid-foreign"})

    def test_sweep_retires_owned_uuid_already_at_default(self):
        power_agent._previously_managed.add("uuid-mine")

        actuator = self._SweepActuator(
            owned={"uuid-mine"}, sweep_results={"uuid-mine": None}
        )

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        self.assertEqual(actuator.swept, ["uuid-mine"])
        self.assertEqual(power_agent._previously_managed, set())
        persist.assert_called_once_with(set())

    def test_sweep_keeps_uuid_on_inconclusive_scan(self):
        power_agent._previously_managed.add("uuid-indeterminate")

        actuator = self._SweepActuator(
            owned={"uuid-indeterminate"},
            sweep_results={"uuid-indeterminate": False},
        )

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        self.assertEqual(power_agent._previously_managed, {"uuid-indeterminate"})
        persist.assert_called_once_with({"uuid-indeterminate"})

    def test_sweep_exception_keeps_uuid_and_still_shuts_down(self):
        power_agent._previously_managed.add("uuid-boom")

        actuator = self._SweepActuator(owned={"uuid-boom"}, sweep_raises={"uuid-boom"})

        with self.assertLogs("power_agent", level="ERROR") as cm:
            with patch.object(power_agent, "_persist_managed_gpus") as persist:
                power_agent._shutdown_cleanup(actuator)

        self.assertIn("uuid-boom", "\n".join(cm.output))
        self.assertEqual(power_agent._previously_managed, {"uuid-boom"})
        persist.assert_called_once_with({"uuid-boom"})
        actuator.shutdown.assert_called_once()

    def test_indexed_restore_retires_uuid_before_sweep(self):
        """A NORMAL index-keyed restore must retire the UUID
        from the actuator's ownership set, so the subsequent UUID sweep does not
        redundantly reprocess it — and cannot erase a cap an external writer
        installs on the just-restored GPU between the indexed restore and the
        sweep. The sweep should then see ONLY UUIDs the index loop could not
        cover (here: none)."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        class _RetiringSweepActuator:
            name = "dcgm"

            def __init__(self):
                self.shutdown = MagicMock()
                self.restore_default = MagicMock(return_value=True)
                self.swept = []
                self.retired = []
                self._owned = {"GPU-A"}

            def get_uuid(self, gpu_idx):  # pragma: no cover
                return "GPU-A"

            def managed_uuid_for_idx(self, gpu_idx):
                return "GPU-A"

            def managed_uuids(self):
                return set(self._owned)

            def retire_managed_uuid(self, uuid):
                self.retired.append(uuid)
                self._owned.discard(uuid)

            def restore_default_by_uuid(self, uuid):  # pragma: no cover
                self.swept.append(uuid)
                return None

        actuator = _RetiringSweepActuator()

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        actuator.restore_default.assert_called_once_with(0)
        # The index loop retired GPU-A, so the sweep had nothing left to do.
        self.assertEqual(actuator.retired, ["GPU-A"])
        self.assertEqual(actuator.swept, [])
        self.assertEqual(power_agent._previously_managed, set())
        persist.assert_called_once_with(set())

    def test_sweep_does_not_run_on_nvml_actuator(self):
        """NVML indices are stable within a process, so the index loop is
        already complete. The real `NvmlActuator` gained `restore_default_by_uuid`
        but deliberately has NO `managed_uuids()`, so the sweep gate must require
        BOTH methods — gating on `restore_default_by_uuid` alone would enter the
        sweep on NVML and `AttributeError` on the missing `managed_uuids()`,
        skipping persist + shutdown."""

        class _NvmlLikeActuator:
            name = "nvml"

            def restore_default_by_uuid(self, uuid):  # pragma: no cover
                raise AssertionError("UUID sweep must not run on the NVML actuator")

            def __init__(self):
                self.restore_default = MagicMock(return_value=True)
                self.shutdown = MagicMock()
                self.get_uuid = MagicMock(return_value="uuid-managed")

        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.update({"uuid-managed", "uuid-leftover"})
        actuator = _NvmlLikeActuator()
        self.assertTrue(hasattr(type(actuator), "restore_default_by_uuid"))
        self.assertFalse(hasattr(type(actuator), "managed_uuids"))

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._shutdown_cleanup(actuator)

        self.assertEqual(power_agent._previously_managed, {"uuid-leftover"})
        persist.assert_called_once_with({"uuid-leftover"})
        actuator.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
