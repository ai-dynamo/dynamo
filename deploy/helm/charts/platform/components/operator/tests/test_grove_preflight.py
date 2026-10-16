# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock

SCRIPT = Path(__file__).resolve().parents[1] / "files/grove-preflight/check.py"
SPEC = importlib.util.spec_from_file_location("preflight", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class GrovePreflightTest(unittest.TestCase):
    def setUp(self):
        self.container = {
            "name": "grove-operator",
            "image": "mirror/grove:v0.1.0-alpha.15",
        }
        self.controller = {
            "metadata": {"generation": 1},
            "spec": {"template": {"spec": {"containers": [self.container]}}},
            "status": dict(
                observedGeneration=1, replicas=1, updatedReplicas=1, availableReplicas=1
            ),
        }
        self.crd = {
            "metadata": {},
            "status": {"conditions": [{"type": "Established", "status": "True"}]},
        }
        self.get = Mock(
            side_effect=lambda path: (
                self.crd
                if "customresourcedefinitions" in path
                else {"items": [self.controller]}
            )
        )

    def test_accepts_release_and_rejects_old_or_unverifiable_images(self):
        for tag in (
            "v0.1.0-alpha.15",
            "v0.1.0-alpha.16@sha256:abc",
            "v0.1.0",
            "v0.2.0-alpha.1",
        ):
            with self.subTest(tag=tag):
                self.container["image"] = "registry:5000/grove:" + tag
                preflight.check(self.get)
        for image in (
            "grove:v0.1.0-alpha.9",
            "grove:v0.1.0-alpha.14",
            "grove:latest",
            "grove:v0.1.0-alpha.15-rc1",
            "grove@sha256:abc",
            "registry:5000/grove",
        ):
            with self.subTest(image=image), self.assertRaises(ValueError):
                self.container["image"] = image
                preflight.check(self.get)

    def test_blocks_missing_or_ambiguous_controllers_and_incomplete_rollouts(self):
        for controllers in ([], [self.controller, self.controller]):
            with self.subTest(count=len(controllers)), self.assertRaises(ValueError):
                preflight.check(lambda path: {"items": controllers})
        for field in ("observedGeneration", "updatedReplicas", "availableReplicas"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "rollout"),
            ):
                self.controller["status"][field] = 0
                preflight.check(self.get)
            self.controller["status"][field] = 1

    def test_reads_each_crd_and_blocks_unestablished_crds_or_api_errors(self):
        preflight.check(self.get)
        self.assertEqual(len(self.get.call_args_list), 1 + len(preflight.CRDS))
        self.crd["status"]["conditions"] = []
        with self.assertRaisesRegex(ValueError, "not established"):
            preflight.check(self.get)
        self.get.side_effect = PermissionError("HTTP 403")
        with self.assertRaises(PermissionError):
            preflight.check(self.get)

    def test_explicit_controller_reference(self):
        self.get.side_effect = lambda path: (
            self.crd if "customresourcedefinitions" in path else self.controller
        )
        preflight.check(self.get, "grove-system", "grove-operator")
        self.assertEqual(
            self.get.call_args_list[0].args,
            ("/apis/apps/v1/namespaces/grove-system/deployments/grove-operator",),
        )
