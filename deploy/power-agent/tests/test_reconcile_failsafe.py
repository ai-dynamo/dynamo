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

from power_agent import PowerAgent


def _make_agent(core_v1, device_count: int = 2) -> PowerAgent:
    """Build a PowerAgent without touching NVML / K8s (bypass __init__)."""
    agent = object.__new__(PowerAgent)
    agent._core_v1 = core_v1
    agent.node_name = "node-under-test"
    agent.k8s_namespace = None  # exercise the list_pod_for_all_namespaces path
    agent.device_count = device_count
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


if __name__ == "__main__":
    unittest.main()
