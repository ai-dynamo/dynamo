# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile fail-safe on pod-listing failure (PR #9682 review follow-up).

`_list_pods_on_node` uses an *explicit* result contract so the failure path
cannot silently drift into the empty-node path:

  * success  → a (possibly empty) list of pods
  * failure  → ``None``

`reconcile_once` keys its fail-safe off that ``None``: on a listing failure it
SKIPS the cycle, leaving every GPU at its last-known-good cap, rather than
proceeding with an empty pod view that would re-derive caps from a zero-pod
snapshot. An empty list (node genuinely has no pods) is NOT a failure and must
still drive a normal reconcile pass.
"""

import unittest
from unittest.mock import MagicMock

import power_agent
from power_agent import PowerAgent


def _make_agent(core_v1, device_count: int = 2) -> PowerAgent:
    """Build a PowerAgent without touching NVML / K8s (bypass __init__)."""
    agent = object.__new__(PowerAgent)
    agent._core_v1 = core_v1
    agent.node_name = "node-under-test"
    agent.k8s_namespace = None  # exercise the list_pod_for_all_namespaces path
    agent.device_count = device_count
    # reconcile_once re-snapshots the count from the actuator each cycle.
    actuator = MagicMock()
    actuator.device_count.return_value = device_count
    agent._actuator = actuator
    # reconcile_once ticks k8s_list_failures_total on the skip path.
    agent.metrics = MagicMock()
    return agent


class TestListPodsExplicitResult(unittest.TestCase):
    def test_returns_none_on_api_error(self):
        """An apiserver error yields the explicit ``None`` sentinel, never []."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError("boom")
        agent = _make_agent(core_v1)

        self.assertIsNone(agent._list_pods_on_node())

    def test_returns_empty_list_when_node_has_no_pods(self):
        """A genuinely empty node returns ``[]`` — distinct from failure."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)

        result = agent._list_pods_on_node()
        self.assertIsNotNone(result)
        self.assertEqual(result, [])

    def test_cluster_scoped_list_served_from_watch_cache(self):
        """The cluster-scoped LIST passes resource_version="0" so the apiserver
        can serve it from its watch cache instead of an etcd read (PR #9682
        scale mitigation)."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)

        agent._list_pods_on_node()

        _, kwargs = core_v1.list_pod_for_all_namespaces.call_args
        self.assertEqual(kwargs.get("resource_version"), "0")
        # Do not add a client-side read timeout here. A timed-out LIST can still
        # be expensive server-side, and retrying every reconcile cycle can make
        # apiserver load worse under priority-and-fairness throttling. Rely on
        # apiserver-side throttling/backpressure instead.
        self.assertNotIn("_request_timeout", kwargs)
        core_v1.list_namespaced_pod.assert_not_called()

    def test_namespaced_list_served_from_watch_cache(self):
        """The namespace-scoped LIST also passes resource_version="0"."""
        core_v1 = MagicMock()
        core_v1.list_namespaced_pod.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1)
        agent.k8s_namespace = "dynamo"

        agent._list_pods_on_node()

        _, kwargs = core_v1.list_namespaced_pod.call_args
        self.assertEqual(kwargs.get("resource_version"), "0")
        self.assertEqual(kwargs.get("namespace"), "dynamo")
        self.assertNotIn("_request_timeout", kwargs)
        core_v1.list_pod_for_all_namespaces.assert_not_called()


class TestReconcileFailSafe(unittest.TestCase):
    def test_listing_failure_skips_reconcile_and_preserves_caps(self):
        """On listing failure, ``reconcile_once`` must not touch any GPU —
        no per-GPU reconcile runs, so existing caps are frozen in place."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError("boom")
        agent = _make_agent(core_v1, device_count=4)
        agent._reconcile_gpu = MagicMock()

        agent.reconcile_once()

        agent._reconcile_gpu.assert_not_called()

    def test_empty_node_still_reconciles_every_gpu(self):
        """An empty (but successful) listing is NOT a failure: reconcile
        proceeds normally and still visits every GPU, instead of skipping the
        whole cycle as it does on a listing failure. (Each GPU then early-
        returns in ``_reconcile_gpu`` since no listed pod owns it — no
        safe-default is applied on a genuinely empty node.)"""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=4)
        agent._reconcile_gpu = MagicMock()

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 4)


class TestReconcileStopsOnShutdown(unittest.TestCase):
    """A SIGTERM arriving mid-cycle must stop the per-GPU loop promptly so
    run()'s finally can restore caps within the pod's termination grace period,
    rather than grinding through every remaining GPU's DCGM/NVML calls first."""

    def setUp(self):
        power_agent._shutdown.clear()

    def tearDown(self):
        power_agent._shutdown.clear()

    def test_gpu_loop_breaks_when_shutdown_requested_midcycle(self):
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=4)

        processed = []

        def fake_reconcile(gpu_idx, uid_to_annotation):
            processed.append(gpu_idx)
            power_agent._shutdown.set()  # SIGTERM lands while handling GPU 0

        agent._reconcile_gpu = fake_reconcile

        agent.reconcile_once()

        # Only GPU 0 ran; the loop broke before GPUs 1-3 on the shutdown check.
        self.assertEqual(processed, [0])

    def test_no_gpu_processed_when_shutdown_already_set(self):
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=3)
        agent._reconcile_gpu = MagicMock()
        power_agent._shutdown.set()

        agent.reconcile_once()

        agent._reconcile_gpu.assert_not_called()


class TestReconcileDeviceCountRefresh(unittest.TestCase):
    """The reconcile loop must re-snapshot the
    GPU count from the actuator each cycle, not trust the value cached at
    startup. A DCGM hostengine reconnect can change the discovered-GPU set at
    runtime."""

    def test_grown_count_after_reconnect_is_picked_up(self):
        """Cached count is 2; a reconnect grew the set to 4. All 4 GPUs must be
        reconciled — a startup-frozen count would silently ignore the new
        GPUs."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=2)
        agent._reconcile_gpu = MagicMock()
        agent._actuator.device_count.return_value = 4

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 4)
        self.assertEqual(agent.device_count, 4)

    def test_shrunk_count_after_reconnect_avoids_stale_indices(self):
        """Cached count is 8; a reconnect shrank the set to 3. Only the 3 live
        indices are iterated — iterating the stale larger range would raise
        per-index errors on the removed GPUs."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=8)
        agent._reconcile_gpu = MagicMock()
        agent._actuator.device_count.return_value = 3

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 3)
        self.assertEqual(agent.device_count, 3)

    def test_count_refresh_failure_uses_last_known(self):
        """A transient device_count() read failure must not skip enforcement:
        fall back to the last-known count for this cycle."""
        core_v1 = MagicMock()
        core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])
        agent = _make_agent(core_v1, device_count=5)
        agent._reconcile_gpu = MagicMock()
        agent._actuator.device_count.side_effect = RuntimeError("hostengine down")

        agent.reconcile_once()

        self.assertEqual(agent._reconcile_gpu.call_count, 5)
        self.assertEqual(agent.device_count, 5)


if __name__ == "__main__":
    unittest.main()
