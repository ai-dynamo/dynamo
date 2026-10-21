# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
MINIMUM = (
    re.search(
        r"(?m)^  dynamo\.nvidia\.com/grove-version: &groveVersion (\S+)",
        (ROOT / "Chart.yaml").read_text(),
    )
    .group(1)
    .strip("\"'")
)
SCRIPT = ROOT / "files/grove-preflight/check.py"
SPEC = importlib.util.spec_from_file_location("preflight", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class GrovePreflightTest(unittest.TestCase):
    def test_orders_alpha_versions_numerically_before_stable(self):
        self.assertLess(
            preflight.version("v1.2.3-alpha.9"), preflight.version("v1.2.3-alpha.10")
        )
        self.assertLess(
            preflight.version("v1.2.3-alpha.10"), preflight.version("v1.2.3")
        )

    def test_uses_the_supplied_minimum(self):
        higher = f"v{self.major + 1}.0.0"
        with self.assertRaisesRegex(ValueError, re.escape(higher) + " or later"):
            preflight.check(self.get, higher)

    def setUp(self):
        self.major, minor, patch, _ = preflight.version(MINIMUM)
        self.stable = f"v{self.major}.{minor}.{patch}"
        self.container = {
            "name": "grove-operator",
            "image": "mirror/grove:" + MINIMUM,
        }
        self.controller = {
            "metadata": {"generation": 1},
            "spec": {"template": {"spec": {"containers": [self.container]}}},
            "status": dict(
                observedGeneration=1, replicas=1, updatedReplicas=1, availableReplicas=1
            ),
        }
        self.get = Mock(return_value={"items": [self.controller]})

    def test_accepts_release_and_rejects_old_or_unverifiable_images(self):
        for tag in (
            MINIMUM,
            MINIMUM + "@sha256:abc",
            self.stable,
            f"v{self.major + 1}.0.0-alpha.1",
        ):
            with self.subTest(tag=tag):
                self.container["image"] = "registry:5000/grove:" + tag
                preflight.check(self.get, MINIMUM)
        for image in (
            "grove:v0.0.0-alpha.0",
            "grove:v0.0.0",
            "grove:latest",
            "grove:" + MINIMUM + "-rc1",
            "grove@sha256:abc",
            "registry:5000/grove",
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                self.container["image"] = image
                preflight.check(self.get, MINIMUM)

    def test_blocks_missing_or_ambiguous_controllers_and_incomplete_rollouts(self):
        for controllers in ([], [self.controller, self.controller]):
            with self.subTest(count=len(controllers)), self.assertRaises(ValueError):
                preflight.check(lambda path: {"items": controllers}, MINIMUM)
        for field in ("observedGeneration", "updatedReplicas", "availableReplicas"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "rollout"),
            ):
                self.controller["status"][field] = 0
                preflight.check(self.get, MINIMUM)
            self.controller["status"][field] = 1

    def test_reads_only_deployments_and_blocks_api_errors(self):
        preflight.check(self.get, MINIMUM)
        self.get.assert_called_once_with("/apis/apps/v1/deployments")
        self.get.side_effect = PermissionError("HTTP 403")
        with self.assertRaises(PermissionError):
            preflight.check(self.get, MINIMUM)

    def test_explicit_controller_reference(self):
        self.get.return_value = self.controller
        preflight.check(self.get, MINIMUM, "grove-system", "grove-operator")
        self.assertEqual(
            self.get.call_args_list[0].args,
            ("/apis/apps/v1/namespaces/grove-system/deployments/grove-operator",),
        )
