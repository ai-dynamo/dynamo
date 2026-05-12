# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the NVML cap application path.

Covers:
  - PowerAgent._build_uid_to_annotation: pod UID → annotation value mapping
  - _clamp_to_constraints: SKU min/max clamping before any NVML write
  - _apply_cap: milliwatt conversion, managed-index tracking, metric update,
                NVMLError handling, and the clamping→NVML write integration
  - _nvml_uuid: handles both bytes (legacy pynvml) and str (nvidia-ml-py)
"""

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from power_agent import (
    POWER_ANNOTATION_KEY,
    PowerAgent,
    _apply_cap,
    _clamp_to_constraints,
    _nvml_uuid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> PowerAgent:
    """Return a raw PowerAgent instance, bypassing __init__ (needs NVML + K8s)."""
    return object.__new__(PowerAgent)


def _make_pod(uid: str, annotation_value=None) -> MagicMock:
    """Return a minimal mock pod object."""
    pod = MagicMock()
    pod.metadata.uid = uid
    pod.metadata.annotations = (
        {POWER_ANNOTATION_KEY: annotation_value} if annotation_value is not None else {}
    )
    return pod


def _make_nvml(
    min_mw: int = 100_000,
    max_mw: int = 700_000,
    uuid: bytes = b"GPU-test-uuid-0000",
) -> MagicMock:
    """Return a mock pynvml module with sensible defaults."""
    mock = MagicMock()
    mock.NVMLError = Exception
    mock.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (min_mw, max_mw)
    mock.nvmlDeviceGetUUID.return_value = uuid
    return mock


# ---------------------------------------------------------------------------
# _build_uid_to_annotation
# ---------------------------------------------------------------------------


class TestBuildUidToAnnotation(unittest.TestCase):
    def test_pod_with_annotation_present(self):
        agent = _make_agent()
        pod = _make_pod("uid-1", "300")
        result = agent._build_uid_to_annotation([pod])
        self.assertEqual(result, {"uid-1": "300"})

    def test_pod_without_annotation_returns_none(self):
        agent = _make_agent()
        pod = _make_pod("uid-1")  # annotation dict present but key absent
        result = agent._build_uid_to_annotation([pod])
        self.assertEqual(result, {"uid-1": None})

    def test_pod_with_annotations_field_none(self):
        """pod.metadata.annotations is None (no annotations block at all)."""
        agent = _make_agent()
        pod = MagicMock()
        pod.metadata.uid = "uid-1"
        pod.metadata.annotations = None
        result = agent._build_uid_to_annotation([pod])
        self.assertEqual(result, {"uid-1": None})

    def test_multiple_pods_correct_mapping(self):
        agent = _make_agent()
        pods = [
            _make_pod("uid-1", "300"),
            _make_pod("uid-2", "480"),
            _make_pod("uid-3"),  # no annotation → None
        ]
        result = agent._build_uid_to_annotation(pods)
        self.assertEqual(result, {"uid-1": "300", "uid-2": "480", "uid-3": None})

    def test_empty_pod_list_returns_empty_dict(self):
        agent = _make_agent()
        self.assertEqual(agent._build_uid_to_annotation([]), {})

    def test_annotation_with_unrelated_keys_ignored(self):
        """Other annotation keys on the pod are not included."""
        agent = _make_agent()
        pod = MagicMock()
        pod.metadata.uid = "uid-1"
        pod.metadata.annotations = {
            "some.other/key": "irrelevant",
            POWER_ANNOTATION_KEY: "350",
        }
        result = agent._build_uid_to_annotation([pod])
        self.assertEqual(result, {"uid-1": "350"})


# ---------------------------------------------------------------------------
# _clamp_to_constraints
# ---------------------------------------------------------------------------


class TestClampToConstraints(unittest.TestCase):
    def setUp(self):
        self.handle = MagicMock()
        self.metrics = MagicMock()

    def test_within_range_passes_through(self):
        mock_nvml = _make_nvml(min_mw=200_000, max_mw=700_000)
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _clamp_to_constraints(self.handle, 400, 0, self.metrics)
        self.assertEqual(result, 400)
        self.metrics.cap_clamped_total.labels.assert_not_called()

    def test_below_min_clamped_up(self):
        mock_nvml = _make_nvml(min_mw=200_000, max_mw=700_000)
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _clamp_to_constraints(self.handle, 100, 0, self.metrics)
        self.assertEqual(result, 200)
        self.metrics.cap_clamped_total.labels.assert_called_once_with(direction="min")
        self.metrics.cap_clamped_total.labels.return_value.inc.assert_called_once()

    def test_above_max_clamped_down(self):
        mock_nvml = _make_nvml(min_mw=200_000, max_mw=700_000)
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _clamp_to_constraints(self.handle, 900, 0, self.metrics)
        self.assertEqual(result, 700)
        self.metrics.cap_clamped_total.labels.assert_called_once_with(direction="max")
        self.metrics.cap_clamped_total.labels.return_value.inc.assert_called_once()

    def test_exactly_at_min_not_clamped(self):
        mock_nvml = _make_nvml(min_mw=200_000, max_mw=700_000)
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _clamp_to_constraints(self.handle, 200, 0, self.metrics)
        self.assertEqual(result, 200)
        self.metrics.cap_clamped_total.labels.assert_not_called()

    def test_exactly_at_max_not_clamped(self):
        mock_nvml = _make_nvml(min_mw=200_000, max_mw=700_000)
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _clamp_to_constraints(self.handle, 700, 0, self.metrics)
        self.assertEqual(result, 700)
        self.metrics.cap_clamped_total.labels.assert_not_called()

    def test_nvml_error_returns_requested_unchanged(self):
        """If GetPowerManagementLimitConstraints fails, pass the value through."""
        mock_nvml = MagicMock()
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlDeviceGetPowerManagementLimitConstraints.side_effect = Exception(
            "nvml"
        )
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _clamp_to_constraints(self.handle, 300, 0, self.metrics)
        self.assertEqual(result, 300)
        self.metrics.cap_clamped_total.labels.assert_not_called()


# ---------------------------------------------------------------------------
# _apply_cap
# ---------------------------------------------------------------------------


class TestApplyCap(unittest.TestCase):
    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def test_calls_nvml_with_milliwatts(self):
        """Watts must be converted to milliwatts before the NVML call."""
        mock_nvml = _make_nvml()
        handle = MagicMock()
        metrics = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            with patch("power_agent._persist_managed_gpus"):
                _apply_cap(handle, 0, 300, metrics)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            handle, 300_000
        )

    def test_adds_gpu_to_managed_indices(self):
        mock_nvml = _make_nvml()
        handle = MagicMock()
        metrics = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            with patch("power_agent._persist_managed_gpus"):
                _apply_cap(handle, 2, 300, metrics)
        self.assertIn(2, power_agent._managed_gpu_indices)

    def test_updates_applied_limit_gauge(self):
        """metrics.applied_limit_watts.labels(gpu=...).set(watts) must be called."""
        mock_nvml = _make_nvml()
        handle = MagicMock()
        metrics = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            with patch("power_agent._persist_managed_gpus"):
                _apply_cap(handle, 1, 450, metrics)
        metrics.applied_limit_watts.labels.assert_called_once_with(gpu="1")
        metrics.applied_limit_watts.labels.return_value.set.assert_called_once_with(450)

    def test_nvml_error_increments_failure_counter_and_does_not_raise(self):
        mock_nvml = _make_nvml()
        mock_nvml.nvmlDeviceSetPowerManagementLimit.side_effect = Exception("nvml fail")
        handle = MagicMock()
        metrics = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            _apply_cap(handle, 0, 300, metrics)  # must not propagate
        metrics.apply_failures_total.inc.assert_called_once()
        metrics.applied_limit_watts.labels.assert_not_called()

    def test_nvml_error_does_not_add_to_managed_indices(self):
        mock_nvml = _make_nvml()
        mock_nvml.nvmlDeviceSetPowerManagementLimit.side_effect = Exception("nvml fail")
        handle = MagicMock()
        metrics = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            _apply_cap(handle, 3, 300, metrics)
        self.assertNotIn(3, power_agent._managed_gpu_indices)

    def test_clamp_applied_before_nvml_call(self):
        """Requested 900 W but SKU max is 700 W → NVML receives 700_000 mW."""
        mock_nvml = _make_nvml(min_mw=100_000, max_mw=700_000)
        handle = MagicMock()
        metrics = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            with patch("power_agent._persist_managed_gpus"):
                _apply_cap(handle, 0, 900, metrics)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            handle, 700_000
        )

    def test_clamp_below_min_uses_min(self):
        """Requested 50 W but SKU min is 100 W → NVML receives 100_000 mW."""
        mock_nvml = _make_nvml(min_mw=100_000, max_mw=700_000)
        handle = MagicMock()
        metrics = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            with patch("power_agent._persist_managed_gpus"):
                _apply_cap(handle, 0, 50, metrics)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            handle, 100_000
        )


# ---------------------------------------------------------------------------
# _nvml_uuid: pynvml package-version compatibility shim
# ---------------------------------------------------------------------------


class TestNvmlUuid(unittest.TestCase):
    """Regression guard for `'str' object has no attribute 'decode'` crash.

    Legacy ``pynvml`` (NVIDIA bindings, deprecated) returns ``bytes`` from
    ``nvmlDeviceGetUUID``; ``nvidia-ml-py`` (the supported successor that
    pip's ``pynvml`` now installs by default) returns ``str``.  An
    unconditional ``.decode("ascii")`` crashes the orphan-restore loop on
    every iteration on hosts running the new bindings — this test pins the
    shim so neither variant breaks.
    """

    def test_returns_str_when_nvml_returns_bytes(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetUUID.return_value = b"GPU-bytes-uuid-0000"
        handle = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _nvml_uuid(handle)
        self.assertEqual(result, "GPU-bytes-uuid-0000")
        self.assertIsInstance(result, str)

    def test_returns_str_when_nvml_returns_str(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-str-uuid-0000"
        handle = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            result = _nvml_uuid(handle)
        self.assertEqual(result, "GPU-str-uuid-0000")
        self.assertIsInstance(result, str)


if __name__ == "__main__":
    unittest.main()
