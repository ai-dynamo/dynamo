# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Durability of the deferred cap-acquisition / retirement persistence.

The agent records a cap's ownership in TWO places: the in-memory
`_previously_managed` mirror (consulted by shutdown / orphan recovery) and the
durable `managed_gpus.json`. These can't be written atomically, so a persist
failure must not be lost:

  * ACQUISITION — `_record_managed_gpu_by_uuid` adds the UUID in memory, then
    persists. If the persist fails it queues the UUID in `_pending_acquisition`
    (NOT re-raising — the cap is already live) so the reconcile-loop flush
    retries the durable write. Without the queue, the membership guard would
    suppress every future persist for that UUID and an ungraceful exit would
    strand the live cap with no recovery record.
  * RETIREMENT — `_commit_release` drops the UUID in memory, then persists; a
    failure queues it in `_pending_retirement`.

Both queues are flushed (persistence ONLY, never a hardware re-apply/re-restore)
at the TOP of every reconcile cycle, BEFORE the Kubernetes pod list, so they
retry even during an apiserver outage (the retry touches only the state volume).
"""

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import (
    PowerAgent,
    _commit_release,
    _flush_pending_acquisitions,
    _flush_pending_retirements,
    _record_managed_gpu_by_uuid,
)


class _PendingTestBase(unittest.TestCase):
    """Reset all module-level state + stub persistence so tests never touch disk."""

    def setUp(self):
        power_agent._previously_managed.clear()
        power_agent._managed_gpu_indices.clear()
        power_agent._pending_acquisition.clear()
        power_agent._pending_retirement.clear()
        self._persist_patch = patch("power_agent._persist_managed_gpus")
        self.persist = self._persist_patch.start()

    def tearDown(self):
        self._persist_patch.stop()
        power_agent._previously_managed.clear()
        power_agent._managed_gpu_indices.clear()
        power_agent._pending_acquisition.clear()
        power_agent._pending_retirement.clear()


class TestAcquisitionPersistence(_PendingTestBase):
    def test_initial_persist_failure_queues_uuid_without_raising(self):
        """A failed durable ADD records ownership in memory + queues the UUID,
        and must NOT propagate (the cap write already succeeded)."""
        self.persist.side_effect = OSError("read-only state volume")

        # Must not raise.
        _record_managed_gpu_by_uuid("GPU-A")

        # In-memory ownership recorded so shutdown/orphan paths see it as managed…
        self.assertIn("GPU-A", power_agent._previously_managed)
        # …and the failed durable write is queued for retry.
        self.assertIn("GPU-A", power_agent._pending_acquisition)

    def test_failed_flush_retains_pending_entry(self):
        power_agent._previously_managed.add("GPU-A")
        power_agent._pending_acquisition.add("GPU-A")
        self.persist.side_effect = OSError("still down")

        _flush_pending_acquisitions()

        # Persist retried against the authoritative set, but it failed → keep it.
        self.persist.assert_called_once_with(power_agent._previously_managed)
        self.assertIn("GPU-A", power_agent._pending_acquisition)

    def test_successful_flush_persists_authoritative_set_and_clears(self):
        power_agent._previously_managed.update({"GPU-A", "GPU-B"})
        power_agent._pending_acquisition.add("GPU-A")

        _flush_pending_acquisitions()

        # The WHOLE authoritative in-memory set is written (not just the queued
        # UUID), and the queue is cleared.
        self.persist.assert_called_once_with(power_agent._previously_managed)
        self.assertEqual(power_agent._pending_acquisition, set())

    def test_flush_is_noop_when_nothing_pending(self):
        _flush_pending_acquisitions()
        self.persist.assert_not_called()

    def test_membership_guard_does_not_mask_a_still_pending_persist(self):
        """The membership guard (`if uuid in _previously_managed: return`) must
        not swallow a re-record while the UUID is still pending: the earlier
        persist failed, so the UUID sits in `_pending_acquisition` and the flush
        remains responsible for it (the guard skips the redundant in-line write,
        which is correct — the flush is the retry path)."""
        self.persist.side_effect = OSError("down")
        _record_managed_gpu_by_uuid("GPU-A")
        self.persist.reset_mock()
        self.persist.side_effect = OSError("still down")

        # Second record for the same UUID: guarded out (already in memory), so
        # no in-line persist — but the pending entry is untouched and will be
        # retried by the flush.
        _record_managed_gpu_by_uuid("GPU-A")
        self.persist.assert_not_called()
        self.assertIn("GPU-A", power_agent._pending_acquisition)


class TestAcquisitionThenRetirementWhilePersistUnavailable(_PendingTestBase):
    def test_acquire_then_release_resolves_after_volume_recovers(self):
        """Acquire a cap, then release it, both while the state volume is down.
        The UUID ends up queued on BOTH sides; once the volume recovers a flush
        persists the authoritative (now empty) set and clears the queues — the
        UUID must NOT survive on disk."""
        actuator = MagicMock()  # NVML-like: no retire_managed_uuid on the type
        self.persist.side_effect = OSError("down")

        _record_managed_gpu_by_uuid("GPU-A")  # acquire (persist fails → queued)
        power_agent._managed_gpu_indices.add(0)
        _commit_release(actuator, 0, "GPU-A")  # release (persist fails → queued)

        # In memory the release won: GPU-A is no longer managed.
        self.assertNotIn("GPU-A", power_agent._previously_managed)
        self.assertIn("GPU-A", power_agent._pending_retirement)

        # Volume recovers; the reconcile-loop flushes both queues.
        self.persist.side_effect = None
        _flush_pending_retirements()
        _flush_pending_acquisitions()

        # Every persist wrote the authoritative (empty) set, and both queues are
        # clear — GPU-A does not linger on disk.
        for call in self.persist.call_args_list:
            self.assertEqual(call.args[0], power_agent._previously_managed)
        self.assertEqual(power_agent._previously_managed, set())
        self.assertEqual(power_agent._pending_retirement, set())
        self.assertEqual(power_agent._pending_acquisition, set())


class TestFlushRunsDuringK8sOutage(_PendingTestBase):
    """The flushes sit at the TOP of `reconcile_once`, before the pod list, so a
    deferred durable write is retried even when the apiserver is unreachable (the
    retry touches only the state volume)."""

    @staticmethod
    def _make_agent(core_v1, device_count: int = 2) -> PowerAgent:
        agent = object.__new__(PowerAgent)
        agent._core_v1 = core_v1
        agent.node_name = "node-under-test"
        agent.k8s_namespace = None
        agent.device_count = device_count
        actuator = MagicMock()
        actuator.device_count.return_value = device_count
        agent._actuator = actuator
        agent.metrics = MagicMock()
        return agent

    def test_acquisition_flush_runs_when_pod_list_fails(self):
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError("apiserver down")
        agent = self._make_agent(core_v1)
        agent._reconcile_gpu = MagicMock()

        power_agent._previously_managed.add("GPU-A")
        power_agent._pending_acquisition.add("GPU-A")

        agent.reconcile_once()

        # The cycle skipped per-GPU work (list failed)…
        agent._reconcile_gpu.assert_not_called()
        # …but the deferred acquisition persist still ran and cleared.
        self.persist.assert_called_with(power_agent._previously_managed)
        self.assertEqual(power_agent._pending_acquisition, set())


if __name__ == "__main__":
    unittest.main()
