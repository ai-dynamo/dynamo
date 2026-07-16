# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the single-copy managed-state invariant.

In production the daemon entrypoint runs ``power_agent.py`` as the
top-level ``__main__`` module, while ``actuator.py`` reaches it via
``import power_agent`` — two distinct module objects in ``sys.modules``.
If the managed-GPU sets or acquisition retry queue lived in ``power_agent.py``
they would fork into two independent copies: the actuator would record
freshly-capped GPUs into one while the SIGTERM/reconcile paths read from the
other (always empty), so caps could leak past graceful shutdown or lose their
deferred durable-state retry.

The fix hosts that state in ``managed_state`` (imported by canonical name
from both sites) and aliases ``power_agent._managed_gpu_indices`` /
``power_agent._previously_managed`` to it. These tests lock in the
invariants that make the fix work:

  1. the ``power_agent`` names ARE the ``managed_state`` objects (identity);
  2. a cap recorded straight into ``managed_state`` (as the actuator's
     separate module copy would) is visible to the SIGTERM restore loop; and
  3. a failed acquisition persist queued by the actuator module is visible to
     the reconcile-loop flush; and
  4. startup orphan-recovery reloads state IN PLACE — it must never rebind
     the alias, which would re-introduce the dual-copy bug.
"""

import unittest
from unittest.mock import MagicMock, patch

import managed_state
import power_agent


class TestSharedStateIdentity(unittest.TestCase):
    """The ``power_agent`` module attributes must be the very objects that
    live in ``managed_state`` — not copies, not re-exports of equal value."""

    def test_managed_gpu_indices_is_shared_object(self):
        self.assertIs(
            power_agent._managed_gpu_indices, managed_state.managed_gpu_indices
        )

    def test_previously_managed_is_shared_object(self):
        self.assertIs(power_agent._previously_managed, managed_state.previously_managed)

    def test_pending_acquisition_is_shared_object(self):
        self.assertIs(
            power_agent._pending_acquisition, managed_state.pending_acquisition
        )

    def test_state_path_matches(self):
        self.assertEqual(
            power_agent._MANAGED_STATE_PATH, managed_state.MANAGED_STATE_PATH
        )


class TestActuatorWritesVisibleToShutdown(unittest.TestCase):
    """A managed GPU recorded into ``managed_state`` (the way the actuator's
    own ``import power_agent`` copy records it) must be restored by the
    shutdown cleanup. This is the exact path that silently broke before the
    state was hoisted into ``managed_state``."""

    def setUp(self):
        managed_state.managed_gpu_indices.clear()
        managed_state.previously_managed.clear()
        power_agent._shutdown.clear()

    def tearDown(self):
        managed_state.managed_gpu_indices.clear()
        managed_state.previously_managed.clear()
        power_agent._shutdown.clear()

    def test_index_recorded_in_managed_state_is_restored_on_cleanup(self):
        # Simulate the actuator recording a cap by mutating the canonical
        # shared set directly (its `power_agent._managed_gpu_indices` is the
        # same object as `managed_state.managed_gpu_indices`).
        managed_state.managed_gpu_indices.add(3)

        actuator = MagicMock()
        actuator.name = "nvml"
        actuator.get_uuid.return_value = "GPU-3"

        with patch.object(power_agent, "_persist_managed_gpus"):
            power_agent._shutdown_cleanup(actuator)

        actuator.restore_default.assert_called_once_with(3)
        actuator.shutdown.assert_called_once()


class TestActuatorPendingAcquisitionVisibleToReconcile(unittest.TestCase):
    """A failed durable ADD queued through the actuator's module copy must be
    flushed by the running daemon's reconcile loop."""

    def setUp(self):
        managed_state.previously_managed.clear()
        managed_state.pending_acquisition.clear()

    def tearDown(self):
        managed_state.previously_managed.clear()
        managed_state.pending_acquisition.clear()

    def test_pending_acquisition_recorded_in_managed_state_flushes(self):
        managed_state.previously_managed.add("GPU-x")
        managed_state.pending_acquisition.add("GPU-x")

        with patch.object(power_agent, "_persist_managed_gpus") as persist:
            power_agent._flush_pending_acquisitions()

        persist.assert_called_once_with(power_agent._previously_managed)
        self.assertEqual(managed_state.pending_acquisition, set())


class TestOrphanReloadKeepsAlias(unittest.TestCase):
    """Startup orphan recovery reloads the persisted UUID set. It must mutate
    the shared set in place rather than rebinding ``_previously_managed`` — a
    rebind would split the alias and re-break cross-module sharing."""

    def setUp(self):
        managed_state.previously_managed.clear()

    def tearDown(self):
        managed_state.previously_managed.clear()

    def test_reload_does_not_rebind_alias(self):
        before = power_agent._previously_managed

        # GPU-x is present and BUSY, so the reload is RETAINED (the running-PID
        # pre-filter skips it) — this test asserts the alias is not rebound and
        # the reload lands in the shared set, not prune semantics (those are
        # covered in test_orphan_recovery.py::TestAbsentPersistedUuid). Present +
        # busy keeps GPU-x without depending on any prune branch.
        actuator = MagicMock()
        actuator.device_count.return_value = 1
        actuator.get_uuid.side_effect = lambda idx: "GPU-x"
        # Startup recovery consumes ONE conclusive identity snapshot; GPU-x is
        # present and BUSY, so it is retained via the running-PID pre-filter.
        actuator.scan_uuid_index_map.return_value = ({"GPU-x": 0}, True)
        actuator.list_running_pids.side_effect = lambda idx, expected_uuid=None: [
            1234
        ]  # busy → retained

        with patch.object(
            power_agent,
            "_read_managed_gpus_state",
            return_value=({"GPU-x"}, True),
        ):
            with patch.object(power_agent, "_persist_managed_gpus"):
                power_agent._restore_orphaned_gpus_on_startup(actuator)

        # Same object before and after, and still the managed_state set.
        self.assertIs(power_agent._previously_managed, before)
        self.assertIs(power_agent._previously_managed, managed_state.previously_managed)
        # And the reload landed in that shared object (retained because busy).
        self.assertEqual(managed_state.previously_managed, {"GPU-x"})
        actuator.restore_default_by_uuid.assert_not_called()


if __name__ == "__main__":
    unittest.main()
