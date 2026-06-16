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

import pytest
from power_agent import POWER_ANNOTATION_KEY, PowerAgent

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

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


if __name__ == "__main__":
    unittest.main()
