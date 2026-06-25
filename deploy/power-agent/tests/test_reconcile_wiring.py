# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the v1.6 actuator wiring in `PowerAgent._reconcile_gpu`.

Pre-v1.6 the reconcile loop hard-coded `pynvml.nvmlDeviceGetHandleByIndex`
and the module-level `_apply_cap(handle, ...)` — so `agent.actuator=dcgm`
only affected cold-start orphan recovery; steady-state cap writes silently
used NVML regardless of actuator selection. That was the load-bearing bug
in review comment #4.

v1.6 routes `_reconcile_gpu` through `self._actuator.list_running_pids`
and `self._actuator.apply_cap`. This file pins that contract:

  - PID enumeration goes through the actuator (so DcgmActuator's UUID-
    keyed identity map runs and PID reads land on the correct physical
    GPU).
  - Cap writes go through the actuator (so DcgmActuator's
    `dcgmConfigSet` actually runs on the DCGM path).
  - Raw `pynvml.nvmlDeviceGetHandleByIndex` is NOT called from
    `_reconcile_gpu` — if a future refactor reintroduces it, this test
    will fail and surface the regression.
  - Early-exit branches (no PIDs / no K8s PIDs) preserve their no-op
    behaviour.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import PowerAgent


def _make_agent_with_actuator(actuator):
    """Build a PowerAgent without exercising __init__'s NVML/K8s deps."""
    agent = object.__new__(PowerAgent)
    agent._actuator = actuator
    agent.metrics = MagicMock()
    agent.safe_default_watts = 500
    agent.device_count = 1
    return agent


class TestReconcileGpuRoutesViaActuator(unittest.TestCase):
    """The four wiring guarantees from v1.6."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()

    def test_pid_enumeration_goes_through_actuator(self):
        """list_running_pids must be called on self._actuator, not pynvml."""
        actuator = MagicMock()
        actuator.list_running_pids.return_value = []  # no workload, early exit
        agent = _make_agent_with_actuator(actuator)

        mock_nvml = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            agent._reconcile_gpu(3, {})

        actuator.list_running_pids.assert_called_once_with(3)
        # CRITICAL: raw NVML must NOT be touched for PID enumeration.
        # Pre-v1.6 the inline path called nvmlDeviceGetHandleByIndex +
        # nvmlDeviceGetComputeRunningProcesses; if either re-appears the
        # actuator's UUID-keyed identity map is bypassed and PID reads
        # land on the wrong physical GPU.
        mock_nvml.nvmlDeviceGetHandleByIndex.assert_not_called()
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.assert_not_called()

    def test_no_pids_skips_apply_cap(self):
        actuator = MagicMock()
        actuator.list_running_pids.return_value = []
        agent = _make_agent_with_actuator(actuator)

        with patch.object(power_agent, "pynvml", MagicMock()):
            agent._reconcile_gpu(0, {})

        actuator.apply_cap.assert_not_called()

    def test_non_k8s_pids_skip_apply_cap(self):
        """PIDs that don't resolve to a pod UID (e.g. host daemons) don't
        trigger an apply_cap. Same behaviour as pre-v1.6."""
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [9999]
        agent = _make_agent_with_actuator(actuator)

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch("power_agent._extract_pod_uid_from_cgroup", return_value=None):
                agent._reconcile_gpu(0, {"some-uid": "300"})

        actuator.apply_cap.assert_not_called()

    def test_apply_cap_goes_through_actuator_with_resolved_watts(self):
        """The load-bearing test for review comment #4.

        Pre-v1.6 the cap write was `_apply_cap(handle, gpu_idx, cap_w, metrics)`
        — module-level NVML, ignoring `self._actuator`. Setting
        `agent.actuator=dcgm` was therefore a no-op for the steady-state
        cap path. This test pins the v1.6 contract: the actuator
        receives the apply_cap call directly.
        """
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        agent = _make_agent_with_actuator(actuator)

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch(
                "power_agent._extract_pod_uid_from_cgroup",
                return_value="pod-uid-1",
            ):
                agent._reconcile_gpu(2, {"pod-uid-1": "350"})

        actuator.apply_cap.assert_called_once_with(2, 350)
        # No module-level _apply_cap, no raw NVML write.
        # (We can't easily assert _apply_cap wasn't called because it's
        # a module function, but the actuator.apply_cap call is the
        # positive proof — if _apply_cap had run too, the test for
        # comment #8 (double clamp) would catch the doubling.)

    def test_safe_default_used_when_annotation_is_none(self):
        """Pod has no annotation → safe_default_watts via actuator."""
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        agent = _make_agent_with_actuator(actuator)
        agent.safe_default_watts = 450

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch(
                "power_agent._extract_pod_uid_from_cgroup",
                return_value="pod-uid-1",
            ):
                agent._reconcile_gpu(0, {"pod-uid-1": None})

        actuator.apply_cap.assert_called_once_with(0, 450)


class TestReconcileGpuDedupesByPodUid(unittest.TestCase):
    """Per PR #9682 CodeRabbit review on power_agent.py:636.

    The pre-fix `_reconcile_gpu` appended `(uid, annotation)` once per
    PID, so one pod running N GPU processes (which is the common
    pattern for TP/PP/EP workloads, ranks-per-GPU, helper
    workers / profilers) showed up as N rows in `pod_annotations`.
    Downstream `_resolve_cap_for_gpu` uses `len(pod_annotations) > 1`
    to detect the multi-pod-per-GPU misconfig, so a one-pod / N-PID
    GPU would:

      * Spuriously WARN "N pods all agree on cap …" even though
        only one pod was on the GPU.
      * On a pod whose annotation was missing or invalid, take the
        conflict-resolution branch (because `len(unique) > 1` is
        possible only with multi-pod, but the safe-default fallback
        in `_resolve_cap_for_gpu` was reached via `len > 1 +
        agree-or-conflict`).
      * Bump `multi_pod_gpu_total{disposition="agree"}`, polluting the
        operator dashboard's multi-pod-misconfig alert.

    The fix dedupes by pod UID before the policy resolver sees the
    list. This file pins:

      1. One pod / two PIDs → one `pod_annotations` entry → no spurious
         multi-pod WARNING or counter bump.
      2. Two pods / two PIDs each (four PIDs total) → two
         `pod_annotations` entries (one per pod) — multi-pod policy
         still fires correctly.
      3. Two pods that disagree (each with multiple PIDs) → the
         conflict branch still fires once and resolves to safe-default.
    """

    def setUp(self):
        power_agent._managed_gpu_indices.clear()

    def test_one_pod_with_multiple_pids_counts_once(self):
        """The load-bearing dedup test. Three PIDs from the same pod
        must produce ONE entry in the policy resolver's input, not
        three. Otherwise the resolver thinks three pods agreed on
        the same cap and bumps the multi-pod counter."""
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1111, 2222, 3333]
        agent = _make_agent_with_actuator(actuator)

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch(
                "power_agent._extract_pod_uid_from_cgroup",
                # All three PIDs resolve to the same pod UID — the
                # realistic TP=3 ranks-per-GPU pattern.
                side_effect=lambda pid: "pod-tp3",
            ):
                with patch(
                    "power_agent._resolve_cap_for_gpu",
                    return_value=400,
                ) as resolver:
                    agent._reconcile_gpu(0, {"pod-tp3": "400"})

        resolver.assert_called_once()
        # Second positional arg is `pod_annotations`. Must contain
        # exactly one entry, even though three PIDs were enumerated.
        pod_annotations = resolver.call_args.args[1]
        self.assertEqual(
            len(pod_annotations),
            1,
            f"Expected dedup to one pod entry; got {pod_annotations!r}",
        )
        self.assertEqual(pod_annotations[0], ("pod-tp3", "400"))
        actuator.apply_cap.assert_called_once_with(0, 400)

    def test_one_pod_multi_pid_does_not_bump_multi_pod_counter(self):
        """End-to-end through the real resolver: a single pod with
        multiple PIDs must NOT bump multi_pod_gpu_total{agree} — that
        counter is meant for the operator-misconfig topology where
        two distinct pods share a GPU."""
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1111, 2222]
        agent = _make_agent_with_actuator(actuator)

        # Real metrics object so we can assert against its counters.
        from tests.test_multi_pod_policy import _FakeMetrics

        agent.metrics = _FakeMetrics()

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch(
                "power_agent._extract_pod_uid_from_cgroup",
                side_effect=lambda pid: "pod-tp2",
            ):
                agent._reconcile_gpu(0, {"pod-tp2": "350"})

        # Single-pod path → agree counter STAYS at zero.
        self.assertEqual(agent.metrics.multi_pod_agree, 0)
        self.assertEqual(agent.metrics.multi_pod_conflict, 0)
        actuator.apply_cap.assert_called_once_with(0, 350)

    def test_two_pods_with_multiple_pids_each_counts_as_two(self):
        """Genuine multi-pod-per-GPU topology: two pods, each with
        two PIDs (four PIDs total). After dedup, the resolver must
        see exactly TWO entries — one per pod — and the multi-pod
        WARNING / counter must still fire."""
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1, 2, 3, 4]
        agent = _make_agent_with_actuator(actuator)

        from tests.test_multi_pod_policy import _FakeMetrics

        agent.metrics = _FakeMetrics()

        def cgroup(pid):
            return {1: "pod-A", 2: "pod-A", 3: "pod-B", 4: "pod-B"}[pid]

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch("power_agent._extract_pod_uid_from_cgroup", side_effect=cgroup):
                agent._reconcile_gpu(0, {"pod-A": "400", "pod-B": "400"})

        # Genuine multi-pod-agree → counter ticks once (not twice).
        self.assertEqual(agent.metrics.multi_pod_agree, 1)
        actuator.apply_cap.assert_called_once_with(0, 400)

    def test_two_disagreeing_pods_each_multi_pid_still_resolves_to_safe_default(
        self,
    ):
        """Conflict branch survives the dedup: two pods with
        different caps each contributing multiple PIDs → safe default
        wins, conflict counter ticks once."""
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [10, 11, 20, 21]
        agent = _make_agent_with_actuator(actuator)
        agent.safe_default_watts = 500

        from tests.test_multi_pod_policy import _FakeMetrics

        agent.metrics = _FakeMetrics()

        def cgroup(pid):
            return {10: "pod-A", 11: "pod-A", 20: "pod-B", 21: "pod-B"}[pid]

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch("power_agent._extract_pod_uid_from_cgroup", side_effect=cgroup):
                agent._reconcile_gpu(0, {"pod-A": "300", "pod-B": "600"})

        actuator.apply_cap.assert_called_once_with(0, 500)
        self.assertEqual(agent.metrics.multi_pod_conflict, 1)
        # Critical: conflict tick ONCE, not once-per-PID.
        self.assertEqual(agent.metrics.safe_default_applied, 1)


class TestListPodsOnNodeReturnsNoneOnError(unittest.TestCase):
    """Per PR9790 Codex adversarial review (finding #3), unified with the
    PR #9682-reviewed contract.

    Pre-fix `_list_pods_on_node` swallowed every API error with
    `except Exception: return []`. That made a transient apiserver
    outage, RBAC regression, or network blip indistinguishable from a
    genuinely empty node. After the fix the method returns an *explicit*
    ``None`` sentinel on failure (never ``[]``), and `reconcile_once`
    keys its skip-this-cycle policy off that ``None``.
    """

    def _make_agent(self, node="node-a", namespace=None):
        agent = object.__new__(PowerAgent)
        agent.node_name = node
        agent.k8s_namespace = namespace
        agent._core_v1 = MagicMock()
        return agent

    def test_returns_none_on_namespaced_list_exception(self):
        agent = self._make_agent(namespace="dynamo")
        agent._core_v1.list_namespaced_pod.side_effect = RuntimeError(
            "503 ServiceUnavailable from apiserver"
        )
        with self.assertLogs("power_agent", level="WARNING") as cm:
            self.assertIsNone(agent._list_pods_on_node())
        self.assertIn("ServiceUnavailable", "\n".join(cm.output))

    def test_returns_none_on_cluster_wide_list_exception(self):
        agent = self._make_agent(namespace=None)
        agent._core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError(
            "RBAC: pods list forbidden"
        )
        with self.assertLogs("power_agent", level="WARNING") as cm:
            self.assertIsNone(agent._list_pods_on_node())
        self.assertIn("forbidden", "\n".join(cm.output))

    def test_returns_items_on_success(self):
        agent = self._make_agent(namespace=None)
        fake_pod = MagicMock()
        agent._core_v1.list_pod_for_all_namespaces.return_value = MagicMock(
            items=[fake_pod]
        )
        self.assertEqual(agent._list_pods_on_node(), [fake_pod])


class TestReconcileOnceK8sListFailure(unittest.TestCase):
    """When `_list_pods_on_node` raises, `reconcile_once` must:
    1. Log at ERROR level (so operators see the outage in pod logs).
    2. Increment `metrics.k8s_list_failures_total` (for alerting).
    3. NOT call `_reconcile_gpu` (which would build pod_annotations
       from an effectively empty uid_to_annotation map and could
       allow new workloads to run uncapped).
    4. Return cleanly so the next reconcile cycle gets a chance.
    """

    def _make_agent(self):
        agent = object.__new__(PowerAgent)
        agent._actuator = MagicMock()
        agent.metrics = MagicMock()
        agent.safe_default_watts = 500
        agent.device_count = 4
        agent.node_name = "node-a"
        agent.k8s_namespace = None
        agent._core_v1 = MagicMock()
        return agent

    def test_list_failure_skips_cycle_and_ticks_metric(self):
        agent = self._make_agent()
        agent._core_v1.list_pod_for_all_namespaces.side_effect = RuntimeError(
            "apiserver down"
        )

        with patch.object(agent, "_reconcile_gpu") as reconcile_gpu:
            # Capture at WARNING so both the underlying-error WARNING (from
            # `_list_pods_on_node`) and the skip ERROR (from reconcile_once)
            # are visible.
            with self.assertLogs("power_agent", level="WARNING") as cm:
                agent.reconcile_once()

        # Per-GPU reconcile MUST be skipped — otherwise an empty
        # uid_to_annotation map would let new workloads run uncapped.
        reconcile_gpu.assert_not_called()
        # Metric ticks exactly once per failed cycle for alerting.
        agent.metrics.k8s_list_failures_total.inc.assert_called_once()
        # Operator-visible logs include the underlying error and the skip.
        joined = "\n".join(cm.output)
        self.assertIn("apiserver down", joined)
        self.assertIn("skipping reconcile cycle", joined)

    def test_list_success_runs_per_gpu_reconcile(self):
        """Pin the happy path: when the list succeeds (even if empty),
        reconcile_once still calls _reconcile_gpu for each device. This
        distinguishes the new failure path from the genuinely-empty
        case the pre-fix code conflated."""
        agent = self._make_agent()
        agent._core_v1.list_pod_for_all_namespaces.return_value = MagicMock(items=[])

        with patch.object(agent, "_reconcile_gpu") as reconcile_gpu:
            agent.reconcile_once()

        self.assertEqual(reconcile_gpu.call_count, agent.device_count)
        agent.metrics.k8s_list_failures_total.inc.assert_not_called()


class TestReconcileGpuPolicyResolution(unittest.TestCase):
    """The v1.6 wiring must preserve the multi-pod-per-GPU resolution
    contract (`_resolve_cap_for_gpu`). Two pods that agree → use that
    value; two pods that disagree → safe default."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()

    def test_two_pods_agree(self):
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1111, 2222]
        agent = _make_agent_with_actuator(actuator)

        def cgroup(pid):
            return {1111: "pod-A", 2222: "pod-B"}[pid]

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch("power_agent._extract_pod_uid_from_cgroup", side_effect=cgroup):
                agent._reconcile_gpu(0, {"pod-A": "400", "pod-B": "400"})

        actuator.apply_cap.assert_called_once_with(0, 400)

    def test_two_pods_disagree_uses_safe_default(self):
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1111, 2222]
        agent = _make_agent_with_actuator(actuator)
        agent.safe_default_watts = 500

        def cgroup(pid):
            return {1111: "pod-A", 2222: "pod-B"}[pid]

        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch("power_agent._extract_pod_uid_from_cgroup", side_effect=cgroup):
                agent._reconcile_gpu(0, {"pod-A": "300", "pod-B": "600"})

        actuator.apply_cap.assert_called_once_with(0, 500)


if __name__ == "__main__":
    unittest.main()
