# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for multi-pod-per-GPU cap resolution policy.

Cases:
  - 1 pod: apply pod annotation
  - 2+ pods, all agree: apply agreed value + WARNING
  - 2+ pods, conflict: apply safe_default + ERROR
  - Parse failure on warm GPU (prior cap in effect): keep prior cap (no NVML call)
  - Parse failure on cold GPU (no prior cap): apply safe_default
"""

import unittest

from power_agent import _resolve_cap_for_gpu


class _FakeMetrics:
    def __init__(self):
        self.multi_pod_agree = 0
        self.multi_pod_conflict = 0
        self.apply_failures = 0
        self.safe_default_applied = 0

    @property
    def multi_pod_gpu_total(self):
        class _L:
            def __init__(self, m):
                self._m = m

            def labels(self, disposition):
                class _C:
                    def __init__(self, m, d):
                        self._m, self._d = m, d

                    def inc(self):
                        if self._d == "agree":
                            self._m.multi_pod_agree += 1
                        else:
                            self._m.multi_pod_conflict += 1

                return _C(self._m, disposition)

        return _L(self)

    @property
    def apply_failures_total(self):
        class _C:
            def __init__(self, m):
                self._m = m

            def inc(self):
                self._m.apply_failures += 1

        return _C(self)

    @property
    def safe_default_applied_total(self):
        class _C:
            def __init__(self, m):
                self._m = m

            def inc(self):
                self._m.safe_default_applied += 1

        return _C(self)


SAFE_DEFAULT = 500


class TestMultiPodPolicy(unittest.TestCase):
    def setUp(self):
        self.m = _FakeMetrics()

    def test_single_pod_apply_annotation(self):
        cap = _resolve_cap_for_gpu(0, [("uid-1", "480")], SAFE_DEFAULT, self.m)
        self.assertEqual(cap, 480)
        self.assertEqual(self.m.multi_pod_agree, 0)
        self.assertEqual(self.m.multi_pod_conflict, 0)
        self.assertEqual(self.m.safe_default_applied, 0)

    def test_two_pods_agree(self):
        cap = _resolve_cap_for_gpu(
            0, [("uid-1", "480"), ("uid-2", "480")], SAFE_DEFAULT, self.m
        )
        self.assertEqual(cap, 480)
        self.assertEqual(self.m.multi_pod_agree, 1)
        self.assertEqual(self.m.multi_pod_conflict, 0)
        self.assertEqual(self.m.safe_default_applied, 0)

    def test_two_pods_conflict(self):
        cap = _resolve_cap_for_gpu(
            0, [("uid-1", "480"), ("uid-2", "350")], SAFE_DEFAULT, self.m
        )
        self.assertEqual(cap, SAFE_DEFAULT)
        self.assertEqual(self.m.multi_pod_conflict, 1)
        self.assertEqual(self.m.safe_default_applied, 1)

    def test_no_parseable_annotation(self):
        cap = _resolve_cap_for_gpu(
            0, [("uid-1", None), ("uid-2", None)], SAFE_DEFAULT, self.m
        )
        self.assertEqual(cap, SAFE_DEFAULT)
        # Policy-fallback ticks safe_default_applied only.
        # apply_failures_total is reserved for actuator write failures
        # ("cap NOT live"); the caller will apply the cap at the
        # safe-default value, so the cap WILL be live.
        self.assertEqual(self.m.apply_failures, 0)
        self.assertEqual(self.m.safe_default_applied, 1)

    def test_invalid_annotation_value(self):
        cap = _resolve_cap_for_gpu(0, [("uid-1", "not-a-number")], SAFE_DEFAULT, self.m)
        self.assertEqual(cap, SAFE_DEFAULT)
        # Same rationale as test_no_parseable_annotation: parse
        # failure is policy-fallback, not an apply failure.
        self.assertEqual(self.m.apply_failures, 0)
        self.assertEqual(self.m.safe_default_applied, 1)

    def test_three_pods_all_agree(self):
        cap = _resolve_cap_for_gpu(
            0,
            [("uid-1", "300"), ("uid-2", "300"), ("uid-3", "300")],
            SAFE_DEFAULT,
            self.m,
        )
        self.assertEqual(cap, 300)
        self.assertEqual(self.m.multi_pod_agree, 1)

    def test_multipod_missing_annotation_on_one_pod_is_conflict(self):
        """A pod missing the annotation makes the GPU a conflict.

        Pre-fix the resolver filtered None values BEFORE computing the
        agree-set, so a GPU shared by pod A (cap 480) and pod B (no
        annotation) silently resolved to 480 — pod A's annotation
        governed pod B's GPU usage. This is the exact cross-workload
        policy failure the multi-pod guard exists to contain.

        After the fix: any missing annotation in multi-pod is a
        conflict → safe_default + conflict counter.
        """
        cap = _resolve_cap_for_gpu(
            0, [("uid-1", None), ("uid-2", "480")], SAFE_DEFAULT, self.m
        )
        self.assertEqual(cap, SAFE_DEFAULT)
        self.assertEqual(self.m.multi_pod_conflict, 1)
        self.assertEqual(self.m.multi_pod_agree, 0)
        self.assertEqual(self.m.safe_default_applied, 1)
        # Policy-fallback, NOT actuator write failure.
        self.assertEqual(self.m.apply_failures, 0)

    def test_multipod_invalid_annotation_on_one_pod_is_conflict(self):
        """Same logic as missing-on-one-pod: an unparseable annotation
        from one pod must not let another pod's cap silently win."""
        cap = _resolve_cap_for_gpu(
            0,
            [("uid-1", "not-a-number"), ("uid-2", "480")],
            SAFE_DEFAULT,
            self.m,
        )
        self.assertEqual(cap, SAFE_DEFAULT)
        self.assertEqual(self.m.multi_pod_conflict, 1)
        self.assertEqual(self.m.multi_pod_agree, 0)
        self.assertEqual(self.m.safe_default_applied, 1)

    def test_multipod_all_missing_is_conflict(self):
        """Two pods both with None — already worked, pin the behavior
        through the multi-pod branch (was reached via the values-empty
        early-return pre-fix; now reaches the multi-pod conflict path).
        Either way → safe_default."""
        cap = _resolve_cap_for_gpu(
            0, [("uid-1", None), ("uid-2", None)], SAFE_DEFAULT, self.m
        )
        self.assertEqual(cap, SAFE_DEFAULT)
        self.assertEqual(self.m.safe_default_applied, 1)


if __name__ == "__main__":
    unittest.main()
