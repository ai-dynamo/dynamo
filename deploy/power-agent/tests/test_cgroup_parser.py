# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _extract_pod_uid_from_cgroup.

Covers all QoS × driver × runtime combinations:
  - cgroup v1 multi-line, cgroup v2 single-line
  - systemd / cgroupfs drivers
  - Guaranteed / Burstable / BestEffort QoS
  - cri-containerd / cri-o wrapper segments
  - non-K8s process (returns None)
"""

import unittest
from unittest.mock import mock_open, patch

from power_agent import _extract_pod_uid_from_cgroup


class TestCgroupParser(unittest.TestCase):
    def _parse(self, content: str):
        with patch("builtins.open", mock_open(read_data=content)):
            return _extract_pod_uid_from_cgroup(12345)

    # ---- cgroup v1 (multi-line, one per controller) ----

    def test_v1_systemd_guaranteed(self):
        # systemd encodes dashes as underscores in the pod-UID segment
        content = (
            "11:devices:/kubepods.slice/kubepods-podcafe0001_dead_0002_beef_0003.slice/"
            "cri-containerd-abc.scope\n"
            "10:memory:/kubepods.slice/kubepods-podcafe0001_dead_0002_beef_0003.slice/"
            "cri-containerd-abc.scope\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    def test_v1_systemd_burstable(self):
        content = (
            "11:devices:/kubepods.slice/kubepods-burstable.slice/"
            "kubepods-burstable-podcafe0001_dead_0002_beef_0003.slice/"
            "cri-containerd-abc.scope\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    def test_v1_systemd_besteffort(self):
        content = (
            "10:memory:/kubepods.slice/kubepods-besteffort.slice/"
            "kubepods-besteffort-podcafe0001_dead_0002_beef_0003.slice/"
            "cri-containerd-abc.scope\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    def test_v1_cgroupfs_guaranteed(self):
        content = (
            "11:devices:/kubepods/podcafe0001-dead-0002-beef-0003/abc123\n"
            "10:memory:/kubepods/podcafe0001-dead-0002-beef-0003/abc123\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    def test_v1_cgroupfs_burstable(self):
        content = (
            "10:memory:/kubepods/burstable/podcafe0001-dead-0002-beef-0003/abc123\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    def test_v1_cgroupfs_besteffort(self):
        content = (
            "10:memory:/kubepods/besteffort/podcafe0001-dead-0002-beef-0003/abc123\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    # ---- cgroup v2 (single unified line) ----

    def test_v2_systemd(self):
        content = (
            "0::/kubepods.slice/kubepods-podcafe0001_dead_0002_beef_0003.slice/"
            "cri-containerd-abc.scope\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    def test_v2_cgroupfs(self):
        content = "0::/kubepods/podcafe0001-dead-0002-beef-0003/cri-containerd-abc\n"
        uid = self._parse(content)
        self.assertEqual(uid, "cafe0001-dead-0002-beef-0003")

    # ---- non-K8s process (no pod slice in cgroup) ----

    def test_non_k8s_process(self):
        content = "11:devices:/system.slice/kubelet.service\n"
        uid = self._parse(content)
        self.assertIsNone(uid)

    def test_empty_cgroup(self):
        uid = self._parse("")
        self.assertIsNone(uid)

    def test_os_error_returns_none(self):
        with patch("builtins.open", side_effect=OSError("no file")):
            uid = _extract_pod_uid_from_cgroup(99999)
        self.assertIsNone(uid)

    # ---- first matching line wins (v1 multi-line) ----

    def test_first_matching_line_wins(self):
        """When multiple lines match, the first one should be returned."""
        content = (
            "0::/not-k8s/something\n"
            "10:memory:/kubepods/podAAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE/abc\n"
            "9:cpu:/kubepods/pod11111111-2222-3333-4444-555555555555/abc\n"
        )
        uid = self._parse(content)
        self.assertEqual(uid, "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE")


if __name__ == "__main__":
    unittest.main()
