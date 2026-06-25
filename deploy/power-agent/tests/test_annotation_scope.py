# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in scope: the agent only caps GPUs owned by annotated pods (PR #9682 review).

`_build_uid_to_annotation` is scope-by-annotation-key: a pod is in scope only
if it carries ``dynamo.nvidia.com/gpu-power-limit``. A GPU running only
unannotated pods — a co-located non-Dynamo workload, or a Dynamo worker the
planner has not yet annotated — must be left at its hardware default and never
written, instead of being silently capped to the safe default.

A pod that *does* carry the key but with a malformed value stays in scope so
the safe-default fail-safe still protects a genuinely-managed pod.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import POWER_ANNOTATION_KEY, PowerAgent

SAFE_DEFAULT = 500


def _pod(uid: str, annotations):
    """Fake K8s pod object exposing ``metadata.uid`` / ``metadata.annotations``."""
    return types.SimpleNamespace(
        metadata=types.SimpleNamespace(uid=uid, annotations=annotations)
    )


def _make_agent(device_count: int = 1) -> PowerAgent:
    """Build a PowerAgent without touching NVML / K8s (bypass __init__)."""
    agent = object.__new__(PowerAgent)
    agent.node_name = "node-under-test"
    agent.k8s_namespace = None
    agent.device_count = device_count
    agent.safe_default_watts = SAFE_DEFAULT
    agent.metrics = MagicMock()
    return agent


class TestBuildUidToAnnotationScope(unittest.TestCase):
    def test_unannotated_pods_are_omitted(self):
        agent = _make_agent()
        pods = [
            _pod("annotated", {POWER_ANNOTATION_KEY: "480"}),
            _pod("no-annotations-at-all", None),
            _pod("other-annotations", {"team.example.com/foo": "bar"}),
        ]
        mapping = agent._build_uid_to_annotation(pods)
        self.assertEqual(mapping, {"annotated": "480"})
        self.assertNotIn("no-annotations-at-all", mapping)
        self.assertNotIn("other-annotations", mapping)

    def test_malformed_value_stays_in_scope(self):
        """Key present but value broken/empty → kept, so the fail-safe still fires."""
        agent = _make_agent()
        pods = [
            _pod("bad", {POWER_ANNOTATION_KEY: "not-a-number"}),
            _pod("empty", {POWER_ANNOTATION_KEY: ""}),
        ]
        mapping = agent._build_uid_to_annotation(pods)
        self.assertEqual(mapping, {"bad": "not-a-number", "empty": ""})


class TestReconcileScope(unittest.TestCase):
    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def test_unannotated_gpu_active_pod_is_left_untouched(self):
        """A GPU whose only live process belongs to an unannotated pod gets
        NO cap write — not even the safe default. Reconcile is actuator-routed
        (v1.6), so the assertion is on the actuator, not raw NVML."""
        agent = _make_agent(device_count=1)
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        agent._actuator = actuator
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ):
            agent._reconcile_gpu(0, uid_to_annotation)

        # Not opted-in and not previously managed → no cap, no release write.
        actuator.apply_cap.assert_not_called()
        actuator.restore_default.assert_not_called()

    def test_annotated_gpu_active_pod_is_capped(self):
        """Happy path still works: an annotated pod's GPU is capped to its
        value through the active actuator."""
        agent = _make_agent(device_count=1)
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        agent._actuator = actuator
        pods = [_pod("worker", {POWER_ANNOTATION_KEY: "480"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch("power_agent._extract_pod_uid_from_cgroup", return_value="worker"):
            agent._reconcile_gpu(0, uid_to_annotation)

        # Cap write flows through the actuator at the annotated value.
        actuator.apply_cap.assert_called_once_with(0, 480)


class TestReleaseOnReuse(unittest.TestCase):
    """A previously-managed GPU now running only unannotated work is released
    back to default, instead of stranding a stale cap on the new tenant."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def _run_reconcile_with_unannotated_pod(self, current_w, default_w):
        agent = _make_agent(device_count=1)
        actuator = MagicMock()
        actuator.list_running_pids.return_value = [1234]
        actuator.get_uuid.return_value = "GPU-A"
        actuator.default_w.return_value = default_w
        actuator.current_w.return_value = current_w
        agent._actuator = actuator
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ), patch("power_agent._persist_managed_gpus"):
            agent._reconcile_gpu(0, uid_to_annotation)

        return actuator

    def test_previously_managed_gpu_is_released_to_default(self):
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=400, default_w=700
        )

        # Restored to default via the actuator, never re-capped, and unmanaged.
        actuator.restore_default.assert_called_once_with(0)
        actuator.apply_cap.assert_not_called()
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)

    def test_previously_managed_across_restart_is_released(self):
        """After a restart `_managed_gpu_indices` is empty; the persisted UUID
        set is the only signal. A busy GPU we capped before the restart must
        still be released (startup orphan recovery skips busy GPUs)."""
        # No _managed_gpu_indices entry (cleared on restart); only persisted UUID.
        power_agent._previously_managed.add("GPU-A")

        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=400, default_w=700
        )

        actuator.restore_default.assert_called_once_with(0)
        actuator.apply_cap.assert_not_called()
        self.assertNotIn("GPU-A", power_agent._previously_managed)

    def test_never_managed_gpu_is_not_touched(self):
        # Neither in _managed_gpu_indices nor _previously_managed → not ours.
        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=400, default_w=700
        )

        actuator.restore_default.assert_not_called()
        actuator.apply_cap.assert_not_called()

    def test_release_unmanages_even_if_already_at_default(self):
        """If the cap was already cleared externally, still drop it from the
        managed set (no redundant restore write)."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        actuator = self._run_reconcile_with_unannotated_pod(
            current_w=700, default_w=700
        )

        actuator.restore_default.assert_not_called()
        self.assertNotIn(0, power_agent._managed_gpu_indices)


class _FakeDcgmActuator:
    """Actuator exposing the dcgm-only ``managed_uuid_for_idx`` helper.

    The MagicMock-based cases above stay on the NVML-equivalent branch because
    ``MagicMock``'s *type* has no ``managed_uuid_for_idx`` attribute, so they
    cannot exercise re-enumeration-aware pruning or the ``restore_default`` ->
    ``False`` skip. This fake makes both branches reachable.
    """

    name = "dcgm"

    def __init__(
        self,
        *,
        current_uuid,
        managed_uuid,
        current_w,
        default_w,
        restore_result=True,
    ):
        self._current_uuid = current_uuid
        self._managed_uuid = managed_uuid
        self._current_w = current_w
        self._default_w = default_w
        self._restore_result = restore_result
        self.list_running_pids = MagicMock(return_value=[1234])
        self.apply_cap = MagicMock()
        self.restore_calls = []

    def get_uuid(self, gpu_idx):
        return self._current_uuid

    def managed_uuid_for_idx(self, gpu_idx):
        return self._managed_uuid

    def default_w(self, gpu_idx):
        return self._default_w

    def current_w(self, gpu_idx):
        return self._current_w

    def restore_default(self, gpu_idx):
        self.restore_calls.append(gpu_idx)
        return self._restore_result


class TestReleaseDcgmReenumeration(unittest.TestCase):
    """dcgm-path release correctness: a hostengine re-enumeration must not let
    ``_release_managed_gpu`` prune the wrong UUID or drop retry state when the
    actuator could not conclusively restore the cap (PR #9790 review)."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def _run(self, actuator):
        agent = _make_agent(device_count=1)
        agent._actuator = actuator
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)
        persist = MagicMock()
        with patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ), patch("power_agent._persist_managed_gpus", persist):
            agent._reconcile_gpu(0, uid_to_annotation)
        return persist

    def test_relocation_prunes_managed_uuid_not_current_occupant(self):
        """idx 0 now hosts GPU-B (re-enumerated), but we capped GPU-A. The
        successful restore must prune GPU-A — pruning the current occupant
        GPU-B would strand GPU-A's cap in managed_gpus.json."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-B",
            managed_uuid="GPU-A",
            current_w=400,
            default_w=700,
            restore_result=True,
        )

        persist = self._run(actuator)

        self.assertEqual(actuator.restore_calls, [0])
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)
        persist.assert_called_once()

    def test_relocation_restores_even_when_current_index_at_default(self):
        """idx 0 now hosts GPU-B sitting at default, but we capped GPU-A which
        re-enumeration moved elsewhere (still live). The current-occupant watts
        read "already at default", yet we MUST still call restore_default (it
        relocates by UUID to GPU-A) before pruning GPU-A — otherwise GPU-A's
        cap is stranded permanently."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-B",
            managed_uuid="GPU-A",
            current_w=700,
            default_w=700,
            restore_result=True,
        )

        persist = self._run(actuator)

        # Restore attempted despite the current index reading at-default, then
        # GPU-A (the relocated managed GPU) pruned — not GPU-B.
        self.assertEqual(actuator.restore_calls, [0])
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)
        persist.assert_called_once()

    def test_restore_skipped_keeps_ownership_for_retry(self):
        """``restore_default`` returning False means the cap is still live but
        could not be located. State must be preserved (not pruned/persisted) so
        a later reconcile or startup orphan recovery retries."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")
        actuator = _FakeDcgmActuator(
            current_uuid="GPU-A",
            managed_uuid="GPU-A",
            current_w=400,
            default_w=700,
            restore_result=False,
        )

        persist = self._run(actuator)

        self.assertEqual(actuator.restore_calls, [0])
        self.assertIn(0, power_agent._managed_gpu_indices)
        self.assertIn("GPU-A", power_agent._previously_managed)
        persist.assert_not_called()


if __name__ == "__main__":
    unittest.main()
