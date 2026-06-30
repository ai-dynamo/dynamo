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


def _proc(pid: int):
    return types.SimpleNamespace(pid=pid)


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
        NO NVML write — not even the safe default."""
        agent = _make_agent(device_count=1)
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch("power_agent.pynvml") as mock_nvml, patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ), patch("power_agent._apply_cap") as mock_apply:
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle-0"
            mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [_proc(1234)]

            agent._reconcile_gpu(0, uid_to_annotation)

        mock_apply.assert_not_called()

    def test_annotated_gpu_active_pod_is_capped(self):
        """Happy path still works: an annotated pod's GPU is capped to its value."""
        agent = _make_agent(device_count=1)
        pods = [_pod("worker", {POWER_ANNOTATION_KEY: "480"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch("power_agent.pynvml") as mock_nvml, patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="worker"
        ), patch("power_agent._apply_cap") as mock_apply:
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle-0"
            mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [_proc(1234)]

            agent._reconcile_gpu(0, uid_to_annotation)

        mock_apply.assert_called_once()
        # _apply_cap(handle, gpu_idx, cap_w, metrics)
        args = mock_apply.call_args.args
        self.assertEqual(args[1], 0)
        self.assertEqual(args[2], 480)


class TestReleaseOnReuse(unittest.TestCase):
    """A previously-managed GPU now running only unannotated work is released
    back to default, instead of stranding a stale cap on the new tenant."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def tearDown(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def _run_reconcile_with_unannotated_pod(self, current_mw, default_mw):
        agent = _make_agent(device_count=1)
        pods = [_pod("bystander", {"team.example.com/foo": "bar"})]
        uid_to_annotation = agent._build_uid_to_annotation(pods)

        with patch("power_agent.pynvml") as mock_nvml, patch(
            "power_agent._extract_pod_uid_from_cgroup", return_value="bystander"
        ), patch("power_agent._apply_cap") as mock_apply, patch(
            "power_agent._persist_managed_gpus"
        ):
            mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle-0"
            mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [_proc(1234)]
            mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = default_mw
            mock_nvml.nvmlDeviceGetPowerManagementLimit.return_value = current_mw
            mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-A"

            agent._reconcile_gpu(0, uid_to_annotation)

        return mock_nvml, mock_apply

    def test_previously_managed_gpu_is_released_to_default(self):
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        mock_nvml, mock_apply = self._run_reconcile_with_unannotated_pod(
            current_mw=400_000, default_mw=700_000
        )

        # Restored to default, never re-capped, and unmanaged.
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            "handle-0", 700_000
        )
        mock_apply.assert_not_called()
        self.assertNotIn(0, power_agent._managed_gpu_indices)
        self.assertNotIn("GPU-A", power_agent._previously_managed)

    def test_previously_managed_across_restart_is_released(self):
        """After a restart `_managed_gpu_indices` is empty; the persisted UUID
        set is the only signal. A busy GPU we capped before the restart must
        still be released (startup orphan recovery skips busy GPUs)."""
        # No _managed_gpu_indices entry (cleared on restart); only persisted UUID.
        power_agent._previously_managed.add("GPU-A")

        mock_nvml, mock_apply = self._run_reconcile_with_unannotated_pod(
            current_mw=400_000, default_mw=700_000
        )

        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            "handle-0", 700_000
        )
        mock_apply.assert_not_called()
        self.assertNotIn("GPU-A", power_agent._previously_managed)

    def test_never_managed_gpu_is_not_touched(self):
        # Neither in _managed_gpu_indices nor _previously_managed → not ours.
        mock_nvml, mock_apply = self._run_reconcile_with_unannotated_pod(
            current_mw=400_000, default_mw=700_000
        )

        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        mock_apply.assert_not_called()

    def test_release_unmanages_even_if_already_at_default(self):
        """If the cap was already cleared externally, still drop it from the
        managed set (no redundant NVML write)."""
        power_agent._managed_gpu_indices.add(0)
        power_agent._previously_managed.add("GPU-A")

        mock_nvml, _ = self._run_reconcile_with_unannotated_pod(
            current_mw=700_000, default_mw=700_000
        )

        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        self.assertNotIn(0, power_agent._managed_gpu_indices)


if __name__ == "__main__":
    unittest.main()
