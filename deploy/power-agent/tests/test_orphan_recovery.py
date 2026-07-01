# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `_restore_orphaned_gpus_on_startup`.

An earlier version used inline `pynvml` calls and was structurally
untested (the agent's startup path executes it but no unit test
exercised the branches). It now sits on the `Actuator` surface so
it works on both NVML and DCGM paths; this file covers the branches
that the migration had to preserve:

  1. UUID-gating  — touch only GPUs whose UUID is in `_previously_managed`.
  2. Workload-busy skip — if `list_running_pids` returns anything,
     defer to the normal reconcile.
  3. Delegation to `restore_default_by_uuid` — the below-default /
     at-default / gone decision (and the identity-stable, atomic
     limits+UUID read behind it) lives in the actuator. The caller must
     NOT re-do the current_w/default_w/identity checks itself: doing so
     reintroduced an A->B->A re-enumeration hole where current_w came
     from GPU-A, default_w from GPU-B, and the identity recheck landed
     back on A — falsely concluding "at default" and retiring ownership
     of a still-capped GPU. The caller now just
     interprets the return: True/None -> retire, False -> retain.
  4. Per-GPU exception isolation — one GPU failing doesn't abort the
     loop for the others.
  5. Conclusive results (True/None) remove the UUID from
     `_previously_managed`; the inconclusive False keeps it.

All tests use a MagicMock that satisfies `Actuator`, sidestepping the
NVML/DCGM split — orphan recovery is now actuator-agnostic by design.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import _restore_orphaned_gpus_on_startup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_actuator(uuids, current_w, default_w, pids=None):
    """Build a mock Actuator with per-gpu_idx behaviour.

    `uuids`, `current_w`, `default_w` are dicts keyed by gpu_idx.
    `pids` defaults to no running processes; pass a dict to inject
    workload-busy behaviour per gpu_idx.
    """
    pids = pids or {}
    actuator = MagicMock()
    actuator.device_count.return_value = len(uuids)
    actuator.get_uuid.side_effect = lambda idx: uuids[idx]
    actuator.current_w.side_effect = lambda idx: current_w[idx]
    actuator.default_w.side_effect = lambda idx: default_w[idx]
    actuator.list_running_pids.side_effect = lambda idx, expected_uuid=None: pids.get(
        idx, []
    )
    return actuator


class _OrphanTestBase(unittest.TestCase):
    """Reset module-level state + stub persistence so tests don't touch disk."""

    def setUp(self):
        power_agent._previously_managed.clear()
        self._persist_patch = patch("power_agent._persist_managed_gpus")
        self._persist_patch.start()
        # _load_previously_managed_gpus reads from disk; pre-seed via
        # the global directly so the per-test load returns our set.
        self._load_patch = patch(
            "power_agent._load_previously_managed_gpus",
            side_effect=lambda: set(self._managed_uuids),
        )
        self._load_patch.start()
        self._managed_uuids: set[str] = set()

    def tearDown(self):
        self._load_patch.stop()
        self._persist_patch.stop()
        power_agent._previously_managed.clear()


# ---------------------------------------------------------------------------
# UUID-gating
# ---------------------------------------------------------------------------


class TestUuidGating(_OrphanTestBase):
    def test_unmanaged_uuid_is_skipped(self):
        """A GPU whose UUID isn't in managed_gpus.json must NOT be touched.

        Critical guard against stepping on caps applied by other
        workflows (different DGD, manual nvidia-smi -pl, vendor
        firmware defaults). An earlier inline NVML version did this
        via `if uuid not in _previously_managed: continue`; the
        current version preserves the same branch via
        `actuator.get_uuid(gpu_idx)`.
        """
        # We previously managed only GPU-a; GPU-b is foreign.
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a", 1: "GPU-b"},
            current_w={0: 400, 1: 400},
            default_w={0: 700, 1: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        # GPU-a (managed) → restore attempted (it'll proceed because
        # current<default and no PIDs). The write goes through the
        # UUID-stable path, so it is keyed by UUID, not loop index.
        # GPU-b (unmanaged) → restore NEVER attempted.
        restore_calls = [
            c.args for c in actuator.restore_default_by_uuid.call_args_list
        ]
        self.assertIn(("GPU-a",), restore_calls)
        self.assertNotIn(("GPU-b",), restore_calls)
        # The index-keyed restore_default must NOT be used by cold-start
        # orphan recovery (it can't self-verify before apply_cap runs).
        actuator.restore_default.assert_not_called()

    def test_no_managed_uuids_means_no_writes(self):
        """Empty managed_gpus.json → no GPU touched, regardless of state."""
        self._managed_uuids = set()
        actuator = _make_actuator(
            uuids={0: "GPU-a", 1: "GPU-b"},
            current_w={0: 200, 1: 200},
            default_w={0: 700, 1: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_not_called()
        actuator.restore_default.assert_not_called()


# ---------------------------------------------------------------------------
# Workload-busy skip
# ---------------------------------------------------------------------------


class TestWorkloadBusySkip(_OrphanTestBase):
    def test_gpu_with_running_pids_is_skipped(self):
        """If processes are running, the normal reconcile loop will
        handle this GPU. Don't fight it from the orphan-recovery path.
        """
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 400},
            default_w={0: 700},
            pids={0: [1234, 5678]},  # workload running
        )

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_not_called()
        actuator.restore_default.assert_not_called()
        # current_w / default_w must not have been queried either —
        # the busy check is the cheap exit.
        actuator.current_w.assert_not_called()


# ---------------------------------------------------------------------------
# Delegation to restore_default_by_uuid
# ---------------------------------------------------------------------------


class TestDelegatesToRestoreByUuid(_OrphanTestBase):
    """The caller must delegate the whole decision to
    `restore_default_by_uuid` and NOT re-do current_w/default_w/identity
    checks itself (those non-atomic reads reopened an
    A->B->A re-enumeration hole). The below-default / at-default / gone
    decision and its identity-stable, one-snapshot read live in the actuator.
    """

    def test_idle_managed_gpu_delegates_to_restore_by_uuid(self):
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 400},
            default_w={0: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        # UUID-stable: resolved from the confirmed UUID, not the loop index.
        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
        # The index-keyed restore_default must NOT be used by cold-start
        # orphan recovery (it can't self-verify before apply_cap runs).
        actuator.restore_default.assert_not_called()

    def test_caller_does_not_read_current_or_default(self):
        """The caller must not perform its own current_w/default_w reads —
        that non-atomic precheck is exactly the ABA hole the delegation
        closes. The actuator reads limits + identity in one snapshot."""
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 700},
            default_w={0: 700},
        )
        actuator.restore_default_by_uuid.return_value = None

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.current_w.assert_not_called()
        actuator.default_w.assert_not_called()
        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")


# ---------------------------------------------------------------------------
# Per-GPU exception isolation
# ---------------------------------------------------------------------------


class TestPerGpuExceptionIsolation(_OrphanTestBase):
    def test_one_gpu_failure_does_not_abort_others(self):
        """A NVMLError / DCGMError while handling GPU 0 must not prevent
        GPU 1 from being recovered. The implementation has a per-GPU
        try/except; preserve that.
        """
        self._managed_uuids = {"GPU-a", "GPU-b"}

        actuator = MagicMock()
        actuator.device_count.return_value = 2
        actuator.get_uuid.side_effect = lambda idx: {0: "GPU-a", 1: "GPU-b"}[idx]
        actuator.list_running_pids.return_value = []

        # GPU 0's restore raises → caught + warning. GPU 1: clean restore.
        def restore(uuid):
            if uuid == "GPU-a":
                raise RuntimeError("simulated DCGM failure on GPU 0")
            return True

        actuator.restore_default_by_uuid.side_effect = restore

        _restore_orphaned_gpus_on_startup(actuator)

        # Both were attempted; GPU-a raised (isolated), GPU-b restored.
        self.assertEqual(
            sorted(c.args[0] for c in actuator.restore_default_by_uuid.call_args_list),
            ["GPU-a", "GPU-b"],
        )
        # GPU-b's conclusive True retired it; GPU-a stayed (its iteration raised).
        self.assertNotIn("GPU-b", power_agent._previously_managed)
        self.assertIn("GPU-a", power_agent._previously_managed)


# ---------------------------------------------------------------------------
# Successful restore removes UUID from managed set
# ---------------------------------------------------------------------------


class TestManagedSetPruning(_OrphanTestBase):
    def test_successful_restore_discards_uuid(self):
        """After a clean orphan-restore, the UUID is no longer "managed"
        — the next startup must not re-attempt the restore.
        """
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 400},
            default_w={0: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        # _previously_managed must have GPU-a removed.
        self.assertNotIn("GPU-a", power_agent._previously_managed)

    def test_at_default_gpu_retires_ownership(self):
        """Idle + managed + already at default → ownership is RETIRED
        (UUID discarded), not retained.

        Retaining a UUID whose cap is conclusively gone re-arms the
        clobber the UUID-ownership guard exists to prevent: a later
        unrelated workflow could cap this same physical GPU, and a future
        startup would then "restore" (clobber) that foreign cap because
        the stale UUID still marks the GPU as ours. `restore_default_by_uuid`
        returns None for the at/above-default case (reconfirmed in one
        snapshot), and the caller retires ownership on that conclusive None.
        """
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 700},
            default_w={0: 700},
        )
        # At/above default is reported by the actuator as a conclusive None.
        actuator.restore_default_by_uuid.return_value = None

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
        actuator.restore_default.assert_not_called()
        # Conclusive None → ownership retired.
        self.assertNotIn("GPU-a", power_agent._previously_managed)

    def test_retired_uuid_is_not_clobbered_on_next_startup(self):
        """End-to-end of the stale-UUID exploit.

        Startup 1: our GPU is idle + at default → ownership retired.
        Between startups: an UNRELATED workflow caps that same physical GPU
        below default (e.g. a different DGD, a manual `nvidia-smi -pl`).
        Startup 2: because the UUID was retired, the GPU is no longer "ours",
        so orphan recovery must NOT restore/clobber that foreign cap.

        Before the fix the at-default UUID was retained, so startup 2 would see
        a managed UUID + below-default + idle and reset the foreign cap.
        """
        # --- Startup 1: idle + at default → retire ownership. The actuator
        # reports at/above default as a conclusive None. ---
        self._managed_uuids = {"GPU-a"}
        actuator1 = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 700},
            default_w={0: 700},
        )
        actuator1.restore_default_by_uuid.return_value = None
        _restore_orphaned_gpus_on_startup(actuator1)
        self.assertNotIn("GPU-a", power_agent._previously_managed)

        # --- Startup 2: a foreign workflow has since capped GPU-a below
        # default. The persisted set no longer lists GPU-a, so the UUID gate
        # skips it entirely (no read of its power state, no restore). ---
        self._managed_uuids = set()  # persisted state after startup 1's retire
        actuator2 = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 300},  # foreign cap, below default
            default_w={0: 700},
        )
        _restore_orphaned_gpus_on_startup(actuator2)

        actuator2.restore_default_by_uuid.assert_not_called()
        actuator2.restore_default.assert_not_called()
        actuator2.current_w.assert_not_called()  # UUID gate short-circuits


# ---------------------------------------------------------------------------
# UUID-stable restore: prune decision follows the by-UUID return contract
# ---------------------------------------------------------------------------


class TestUuidStableRestoreReturn(_OrphanTestBase):
    """The write resolves identity from the UUID at
    write time (so a DCGM reconnect/re-enumeration between probe and write
    can't restore/prune the wrong GPU), and the prune decision keys off the
    by-UUID return value, not the loop index.
    """

    def _capped_actuator(self):
        # Managed, idle, below-default at the probe index → the loop reaches
        # the restore_default_by_uuid call.
        self._managed_uuids = {"GPU-a"}
        return _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 400},
            default_w={0: 700},
        )

    def test_true_return_discards_uuid(self):
        """A live cap was restored at the resolved index → prune the UUID."""
        actuator = self._capped_actuator()
        actuator.restore_default_by_uuid.return_value = True

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
        self.assertNotIn("GPU-a", power_agent._previously_managed)

    def test_none_return_retires_ownership(self):
        """None = CONCLUSIVE nothing-of-ours: restore_default_by_uuid
        reconfirmed the GPU is already at/above default, or proved it gone
        on a clean scan. Retire ownership —
        a stale UUID whose cap is proven gone would let a future startup
        clobber a later unrelated cap on the same physical GPU."""
        actuator = self._capped_actuator()
        actuator.restore_default_by_uuid.return_value = None

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
        self.assertNotIn("GPU-a", power_agent._previously_managed)

    def test_false_return_retains_uuid(self):
        """False = the UUID could not be located conclusively (a probe raised,
        e.g. a transient DCGM outage), so the GPU may still carry our cap.
        Keep the UUID so cold-start orphan recovery retries on the next boot
        instead of leaking the cap by pruning prematurely."""
        actuator = self._capped_actuator()
        actuator.restore_default_by_uuid.return_value = False

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
        self.assertIn("GPU-a", power_agent._previously_managed)


if __name__ == "__main__":
    unittest.main()
