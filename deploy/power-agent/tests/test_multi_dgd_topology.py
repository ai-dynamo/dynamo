# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-DGD / multi-framework reconcile coverage — Tier-1 unit tests.

Covers this well-formed mixed-framework topology::

    Node:  8 × B200 GPUs, single power-agent pod
      GPU 0     DGD-V (vLLM disagg)     pod V-P  cap=450W
      GPU 1                             pod V-D  cap=425W
      GPU 2,3   DGD-T (TRTLLM disagg)   pod T-P  cap=480W   (tp=2)
      GPU 4,5                           pod T-D  cap=450W   (tp=2)
      GPU 6,7   DGD-S (SGLang agg)      pod S-A  cap=500W   (tp=2)

Asserted invariants (well-formed topology):

  * Exactly 8 ``actuator.apply_cap(gpu_idx, cap_w)`` calls per reconcile,
    matching the topology table — 4 distinct watt values across 8 GPUs
    (450 W appears on GPUs 0, 4, 5).
  * ``multi_pod_gpu_total`` (both ``agree`` and ``conflict`` dispositions),
    ``apply_failures_total`` and ``safe_default_applied_total`` all stay
    at zero — the topology is well-formed, no misconfig should be
    triggered.
  * TP-rank multi-PID dedup: T-P and T-D each run 2 PIDs per GPU (rank
    main + helper), S-A likewise. The dedup added in commit ``ba3b5803a9``
    collapses them to one ``pod_annotations`` entry per GPU, so the
    multi-pod ``agree`` counter must NOT fire.

**Branch portability.** This file requires the UID-dedup in
``_reconcile_gpu`` introduced by ``ba3b5803a9`` on
``pr1b/power-agent-dcgm-actuator`` (#9790). On ``pr1a/power-agent``
(#9682), the multi-PID dedup is not present and the TP-rank PIDs would
spuriously trigger the multi-pod ``agree`` counter — only the
single-PID-per-pod subset of this file is back-portable to #9682.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import PowerAgent

from tests.test_multi_pod_policy import _FakeMetrics

# ---------------------------------------------------------------------------
# Canonical multi-DGD topology
# ---------------------------------------------------------------------------

# Pod UIDs — the pod-uid namespace is opaque to the agent; only equality
# matters. We use semantic names so the test output reads cleanly.
UID_V_PREFILL = "uid-vllm-prefill"
UID_V_DECODE = "uid-vllm-decode"
UID_T_PREFILL = "uid-trtllm-prefill"
UID_T_DECODE = "uid-trtllm-decode"
UID_S_AGG = "uid-sglang-agg"

# Per-pod NVML cap (watts). The planner's `_apply_power_annotations` would
# have stamped these as `dynamo.nvidia.com/gpu-power-limit` annotations.
ANNOTATIONS = {
    UID_V_PREFILL: "450",
    UID_V_DECODE: "425",
    UID_T_PREFILL: "480",
    UID_T_DECODE: "450",
    UID_S_AGG: "500",
}

# PID → pod-UID map. TP=2 pods (T-P, T-D, S-A) run two PIDs per GPU (rank
# main + helper); the dedup contract collapses them per-GPU.
PID_TO_UID: dict[int, str] = {
    # GPU 0  — vLLM prefill (1 GPU, 1 PID)
    1100: UID_V_PREFILL,
    # GPU 1  — vLLM decode (1 GPU, 1 PID)
    1200: UID_V_DECODE,
    # GPU 2,3 — TRTLLM prefill (tp=2, 2 PIDs per GPU)
    2200: UID_T_PREFILL,
    2201: UID_T_PREFILL,
    2300: UID_T_PREFILL,
    2301: UID_T_PREFILL,
    # GPU 4,5 — TRTLLM decode (tp=2, 2 PIDs per GPU)
    2400: UID_T_DECODE,
    2401: UID_T_DECODE,
    2500: UID_T_DECODE,
    2501: UID_T_DECODE,
    # GPU 6,7 — SGLang aggregated (tp=2, 2 PIDs per GPU)
    5600: UID_S_AGG,
    5601: UID_S_AGG,
    5700: UID_S_AGG,
    5701: UID_S_AGG,
}

# `actuator.list_running_pids(gpu_idx)` → list of PIDs on that physical GPU.
PIDS_PER_GPU: dict[int, list[int]] = {
    0: [1100],
    1: [1200],
    2: [2200, 2201],
    3: [2300, 2301],
    4: [2400, 2401],
    5: [2500, 2501],
    6: [5600, 5601],
    7: [5700, 5701],
}

# Expected `apply_cap(gpu_idx, watts)` calls — matches the topology table.
# Note: 4 distinct watt values across 8 calls; 450 W repeats on GPUs 0, 4, 5.
EXPECTED_CAPS: list[tuple[int, int]] = [
    (0, 450),
    (1, 425),
    (2, 480),
    (3, 480),
    (4, 450),
    (5, 450),
    (6, 500),
    (7, 500),
]


def _make_eight_gpu_agent(metrics) -> PowerAgent:
    """Build a PowerAgent for an 8-GPU node without touching NVML / K8s.

    Mirrors the helper pattern in ``test_reconcile_wiring.py``: bypass
    ``__init__`` (which would call ``nvmlInit()`` and ``k8s_config.*``)
    and inject a mocked actuator + metrics object directly.
    """
    actuator = MagicMock()

    def _pids(gpu_idx: int, expected_uuid=None) -> list[int]:
        return PIDS_PER_GPU.get(gpu_idx, [])

    actuator.list_running_pids.side_effect = _pids
    # apply_cap is a no-op MagicMock — we assert on its call list.

    agent = object.__new__(PowerAgent)
    agent._actuator = actuator
    agent.metrics = metrics
    agent.safe_default_watts = 350
    agent.device_count = 8
    return agent


def _reconcile_all_gpus(agent: PowerAgent) -> None:
    """Drive ``_reconcile_gpu`` for all 8 GPU indices with one shared
    ``uid_to_annotation`` map — the same input shape that
    ``reconcile_once()`` builds from ``_build_uid_to_annotation`` in
    production. Calling ``_reconcile_gpu`` directly avoids the K8s API
    dependency in ``reconcile_once()``.
    """
    uid_to_annotation = dict(ANNOTATIONS)
    with patch.object(power_agent, "pynvml", MagicMock()):
        with patch(
            "power_agent._extract_pod_uid_from_cgroup",
            side_effect=lambda pid: PID_TO_UID.get(pid),
        ):
            for gpu_idx in range(agent.device_count):
                agent._reconcile_gpu(gpu_idx, uid_to_annotation)


class TestMultiFrameworkReconcileHappyPath(unittest.TestCase):
    """8-GPU node hosting vLLM disagg + TRTLLM disagg + SGLang aggregated.

    The well-formed mixed topology. Every counter must stay at zero;
    every cap-write call must match the topology table.
    """

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        self.metrics = _FakeMetrics()
        self.agent = _make_eight_gpu_agent(self.metrics)

    def test_eight_cap_calls_match_topology_table(self):
        """All 8 GPUs receive exactly one ``apply_cap`` call with the
        ``(gpu_idx, watts)`` from the topology table — 4 distinct watt
        values, 8 calls total, value 450 W repeated on GPUs 0, 4, 5."""
        _reconcile_all_gpus(self.agent)

        actual_calls = [
            (call.args[0], call.args[1])
            for call in self.agent._actuator.apply_cap.call_args_list
        ]
        self.assertEqual(
            sorted(actual_calls),
            sorted(EXPECTED_CAPS),
            f"apply_cap call set mismatch.\n"
            f"  expected: {sorted(EXPECTED_CAPS)}\n"
            f"  actual:   {sorted(actual_calls)}",
        )
        # Sanity: 4 distinct watt values across 8 calls.
        self.assertEqual(len({w for _, w in actual_calls}), 4)

    def test_no_misconfig_counters_in_well_formed_topology(self):
        """The positive-invariant test: ``multi_pod_gpu_total{agree}``,
        ``{conflict}``, ``apply_failures_total`` and
        ``safe_default_applied_total`` all stay at zero across the full
        8-GPU reconcile. This is the inversion of
        ``test_multi_pod_policy.py``'s misconfig coverage — it pins that
        a correctly-deployed mixed topology produces a clean dashboard.
        """
        _reconcile_all_gpus(self.agent)

        self.assertEqual(self.metrics.multi_pod_agree, 0)
        self.assertEqual(self.metrics.multi_pod_conflict, 0)
        self.assertEqual(self.metrics.apply_failures, 0)
        self.assertEqual(self.metrics.safe_default_applied, 0)

    def test_tp_rank_helper_pids_deduped_by_uid(self):
        """TP=2 pods (T-P, T-D, S-A) each have 2 PIDs per GPU (rank main
        + helper). Without the UID dedup, ``_resolve_cap_for_gpu`` would
        see ``len(pod_annotations) == 2`` and tick the multi-pod
        ``agree`` counter on GPUs 2-7.

        This test asserts the dedup contract under realistic
        multi-framework conditions, not just the synthetic single-pod
        case in ``test_reconcile_wiring.py``.
        """
        _reconcile_all_gpus(self.agent)

        # GPUs 2-7 each had 2 PIDs sharing one pod UID. Without dedup
        # this would be 6 increments on the `agree` counter.
        self.assertEqual(
            self.metrics.multi_pod_agree,
            0,
            "TP-rank dedup leaked — multi_pod_gpu_total{agree} fired "
            "even though every GPU has exactly one pod-uid.",
        )

    def test_each_gpu_resolves_one_pod_uid(self):
        """The resolver receives exactly one ``(uid, value)`` entry per
        GPU after dedup — never two from the same pod's helper PIDs.
        Inspect ``_resolve_cap_for_gpu``'s argument list directly so a
        future regression in the dedup contract surfaces as a precise
        diff (count + identity) rather than just a counter bump.
        """
        with patch(
            "power_agent._resolve_cap_for_gpu",
            side_effect=lambda gpu_idx, pod_annotations, *_, **__: int(
                pod_annotations[0][1]
            ),
        ) as resolver:
            _reconcile_all_gpus(self.agent)

        expected_uid_per_gpu = {
            0: UID_V_PREFILL,
            1: UID_V_DECODE,
            2: UID_T_PREFILL,
            3: UID_T_PREFILL,
            4: UID_T_DECODE,
            5: UID_T_DECODE,
            6: UID_S_AGG,
            7: UID_S_AGG,
        }
        for call in resolver.call_args_list:
            gpu_idx = call.args[0]
            pod_annotations = call.args[1]
            self.assertEqual(
                len(pod_annotations),
                1,
                f"GPU {gpu_idx} resolver got "
                f"{len(pod_annotations)} pod entries; expected 1",
            )
            self.assertEqual(pod_annotations[0][0], expected_uid_per_gpu[gpu_idx])


class TestMultiFrameworkReconcileMisconfigBlastRadius(unittest.TestCase):
    """B5 — deliberate cross-DGD misconfig on one GPU must NOT bleed
    into the other 7 GPUs.

    Scenario: a stray DGD-T helper pod gets scheduled onto GPU 0
    alongside DGD-V's V-P prefill, both claiming the GPU with different
    annotations. The multi-pod conflict policy applies safe-default on
    GPU 0 only; GPUs 1-7 remain at their correctly-resolved caps.
    """

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        self.metrics = _FakeMetrics()
        self.agent = _make_eight_gpu_agent(self.metrics)
        self.agent.safe_default_watts = 350  # distinguishable from any topology value

    def test_conflict_on_gpu0_does_not_disturb_other_gpus(self):
        """A stray T-P helper PID on GPU 0 with annotation 480 W,
        alongside V-P's PID with annotation 450 W. Per the multi-pod
        policy (``test_multi_pod_policy.py::test_two_pods_conflict``),
        GPU 0 applies ``safe_default_watts``; GPUs 1-7 are untouched
        by the conflict.
        """
        # Inject a T-P stray PID onto GPU 0 alongside V-P's 1100.
        STRAY_PID = 9999
        custom_pids_per_gpu = dict(PIDS_PER_GPU)
        custom_pids_per_gpu[0] = [1100, STRAY_PID]
        custom_pid_to_uid = dict(PID_TO_UID)
        custom_pid_to_uid[STRAY_PID] = UID_T_PREFILL

        self.agent._actuator.list_running_pids.side_effect = (
            lambda gpu_idx, expected_uuid=None: custom_pids_per_gpu.get(gpu_idx, [])
        )

        uid_to_annotation = dict(ANNOTATIONS)
        with patch.object(power_agent, "pynvml", MagicMock()):
            with patch(
                "power_agent._extract_pod_uid_from_cgroup",
                side_effect=lambda pid: custom_pid_to_uid.get(pid),
            ):
                for gpu_idx in range(self.agent.device_count):
                    self.agent._reconcile_gpu(gpu_idx, uid_to_annotation)

        # GPU 0 went to safe-default (450 W and 480 W are a genuine
        # conflict per _resolve_cap_for_gpu).
        gpu0_cap_call = [
            call
            for call in self.agent._actuator.apply_cap.call_args_list
            if call.args[0] == 0
        ]
        self.assertEqual(len(gpu0_cap_call), 1)
        self.assertEqual(gpu0_cap_call[0].args[1], 350)

        # GPUs 1-7 untouched by the conflict — their caps match the
        # topology table exactly.
        for gpu_idx, expected_w in EXPECTED_CAPS:
            if gpu_idx == 0:
                continue  # safe-default case above
            calls = [
                call
                for call in self.agent._actuator.apply_cap.call_args_list
                if call.args[0] == gpu_idx
            ]
            self.assertEqual(len(calls), 1, f"GPU {gpu_idx} cap miscount")
            self.assertEqual(
                calls[0].args[1],
                expected_w,
                f"GPU {gpu_idx} got {calls[0].args[1]} W; expected {expected_w} W "
                f"(conflict on GPU 0 leaked).",
            )

        # Exactly one conflict event; safe-default applied exactly once.
        self.assertEqual(self.metrics.multi_pod_conflict, 1)
        self.assertEqual(self.metrics.safe_default_applied, 1)
        self.assertEqual(self.metrics.multi_pod_agree, 0)


if __name__ == "__main__":
    unittest.main()
