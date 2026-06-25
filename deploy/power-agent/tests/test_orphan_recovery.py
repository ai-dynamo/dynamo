# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for `_restore_orphaned_gpus_on_startup` after the v1.5 migration.

Pre-v1.5 the function used inline `pynvml` calls and was structurally
untested (the agent's startup path executes it but no unit test
exercised the branches). v1.5 lifted it onto the `Actuator` surface so
it works on both NVML and DCGM paths; this file covers the branches
that the migration had to preserve:

  1. UUID-gating  — touch only GPUs whose UUID is in `_previously_managed`.
  2. Workload-busy skip — if `list_running_pids` returns anything,
     defer to the normal reconcile.
  3. The current_w < default_w GUARD — only write if the cap is
     actually below default. The Protocol gained `current_w` and
     `default_w` expressly so this guard survives the migration.
  4. Per-GPU exception isolation — one GPU failing doesn't abort the
     loop for the others.
  5. Successful restore removes the UUID from `_previously_managed`
     (so the next startup doesn't re-attempt it).

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
    actuator.list_running_pids.side_effect = lambda idx: pids.get(idx, [])
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
        firmware defaults). Pre-v1.5 the inline NVML version did this
        via `if uuid not in _previously_managed: continue`; the
        migrated version preserves the same branch via
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
# The current_w < default_w GUARD
# ---------------------------------------------------------------------------


class TestCurrentVsDefaultGuard(_OrphanTestBase):
    """The load-bearing v1.5 Protocol-extension test.

    Pre-v1.5 NVML implementation had this guard inline; if the
    migration to actuator.restore_default had been done without
    extending the Protocol, the guard would have silently disappeared
    and orphan recovery would issue a redundant privileged write on
    every startup for every previously-managed GPU.
    """

    def test_current_below_default_triggers_restore(self):
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 400},  # well below
            default_w={0: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        # UUID-stable: resolved from the confirmed UUID, not the loop index.
        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
        actuator.restore_default.assert_not_called()

    def test_current_equal_to_default_skips_restore(self):
        """Cap already at default → no write, no audit-log churn."""
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 700},
            default_w={0: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_not_called()
        actuator.restore_default.assert_not_called()

    def test_current_above_default_skips_restore(self):
        """Defensive — `current > default` shouldn't be physically
        possible, but if it ever happens (firmware bug, vendor
        override), the guard still skips the write rather than
        clobbering with a smaller value the operator may have set
        deliberately."""
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 800},
            default_w={0: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_not_called()
        actuator.restore_default.assert_not_called()


# ---------------------------------------------------------------------------
# Per-GPU exception isolation
# ---------------------------------------------------------------------------


class TestPerGpuExceptionIsolation(_OrphanTestBase):
    def test_one_gpu_failure_does_not_abort_others(self):
        """A NVMLError / DCGMError on GPU 0 must not prevent GPU 1
        from being recovered. Pre-v1.5 implementation had a per-GPU
        try/except; preserve that.
        """
        self._managed_uuids = {"GPU-a", "GPU-b"}

        actuator = MagicMock()
        actuator.device_count.return_value = 2
        actuator.get_uuid.side_effect = lambda idx: {0: "GPU-a", 1: "GPU-b"}[idx]
        actuator.list_running_pids.return_value = []

        # GPU 0: current_w raises → caught + warning. GPU 1: clean restore.
        def current_w(idx):
            if idx == 0:
                raise RuntimeError("simulated DCGM read failure on GPU 0")
            return 400

        actuator.current_w.side_effect = current_w
        actuator.default_w.side_effect = lambda idx: 700

        _restore_orphaned_gpus_on_startup(actuator)

        # GPU 0 never reached the restore (exception aborted its iteration).
        # GPU 1 was handled normally — restored by its UUID.
        actuator.restore_default_by_uuid.assert_called_once_with("GPU-b")


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

    def test_skipped_gpu_retains_managed_uuid(self):
        """If we didn't actually restore (e.g. current==default),
        the UUID stays in _previously_managed so the next startup
        can re-check.
        """
        self._managed_uuids = {"GPU-a"}
        actuator = _make_actuator(
            uuids={0: "GPU-a"},
            current_w={0: 700},
            default_w={0: 700},
        )

        _restore_orphaned_gpus_on_startup(actuator)

        self.assertIn("GPU-a", power_agent._previously_managed)


# ---------------------------------------------------------------------------
# UUID-stable restore: prune decision follows the by-UUID return contract
# ---------------------------------------------------------------------------


class TestUuidStableRestoreReturn(_OrphanTestBase):
    """The PR9790-review fix: the write resolves identity from the UUID at
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

    def test_none_return_retains_uuid(self):
        """None = nothing of ours to restore (already at default, or the GPU
        moved/left on a clean scan). Do NOT prune — leave it for the next
        boot to re-check rather than dropping a UUID we might still own."""
        actuator = self._capped_actuator()
        actuator.restore_default_by_uuid.return_value = None

        _restore_orphaned_gpus_on_startup(actuator)

        actuator.restore_default_by_uuid.assert_called_once_with("GPU-a")
        self.assertIn("GPU-a", power_agent._previously_managed)

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
