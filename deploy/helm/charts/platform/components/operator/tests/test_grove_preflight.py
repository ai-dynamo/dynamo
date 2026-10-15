# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run with python3 -m unittest discover -s components/operator/tests -p 'test_*.py'."""

import importlib.util
import json
import re
import unittest
from copy import deepcopy
from pathlib import Path

FILES = Path(__file__).resolve().parents[1] / "files" / "grove-preflight"
SPEC = importlib.util.spec_from_file_location("grove_preflight", FILES / "check.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class GrovePreflightTest(unittest.TestCase):
    def setUp(self):
        self.deployment = {
            "metadata": {
                "name": "grove-operator",
                "namespace": "grove-system",
                "generation": 2,
            },
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "grove-operator",
                                "image": "example.com/grove-operator:v0.1.0-alpha.15",
                            },
                        ]
                    }
                },
            },
            "status": {
                "observedGeneration": 2,
                "replicas": 1,
                "updatedReplicas": 1,
                "availableReplicas": 1,
            },
        }
        self.crd = {
            "metadata": {"name": "podcliquesets.grove.io"},
            "spec": {
                "versions": [
                    {
                        "name": "v1alpha1",
                        "served": True,
                        "schema": {
                            "openAPIV3Schema": {
                                "type": "object",
                                "properties": {"spec": {"type": "object"}},
                            },
                        },
                    }
                ]
            },
            "status": {"conditions": [{"type": "Established", "status": "True"}]},
        }
        self.contract = {
            "minimumVersion": "v0.1.0-alpha.15",
            "crdSchemaDigests": {
                "podcliquesets.grove.io": [preflight.schema_digest(self.crd)]
            },
        }
        self.paths = []
        self.deployments = [self.deployment]

    def get(self, path):
        self.paths.append(path)
        if "customresourcedefinitions" in path:
            return self.crd
        if path.endswith("/deployments"):
            return {"items": self.deployments}
        return self.deployment

    def test_accepts_compatible_controller_and_crds(self):
        self.assertIn("are compatible", preflight.check(self.get, self.contract))

    def test_rejects_old_controller_even_with_current_crds(self):
        for version in ("v0.1.0-alpha.9", "v0.1.0-alpha.11", "v0.1.0-alpha.14"):
            with self.subTest(version=version):
                self.deployment["spec"]["template"]["spec"]["containers"][0][
                    "image"
                ] = ("grove:" + version)
                with self.assertRaisesRegex(
                    preflight.Incompatible, "Upgrade Grove and matching CRDs"
                ):
                    preflight.check(self.get, self.contract)

    def test_semver_prerelease_ordering_and_build_metadata(self):
        self.assertLess(
            preflight.version_key("0.1.0-alpha.9"),
            preflight.version_key("v0.1.0-alpha.15"),
        )
        self.assertLess(
            preflight.version_key("0.1.0-alpha.15"), preflight.version_key("0.1.0")
        )
        self.assertEqual(
            preflight.version_key("v0.1.0+build.1"), preflight.version_key("0.1.0")
        )
        for value in ("latest", "custom", "0.1.0-alpha.015", "01.1.0"):
            with self.subTest(value=value), self.assertRaises(preflight.Incompatible):
                preflight.version_key(value)

    def test_rejects_unknown_image_version(self):
        for image in (
            "registry:5000/grove",
            "grove@sha256:abcdef",
            "grove:latest",
            "grove:custom",
        ):
            with self.subTest(image=image):
                self.deployment["spec"]["template"]["spec"]["containers"][0][
                    "image"
                ] = image
                with self.assertRaisesRegex(preflight.Incompatible, "Unverifiable"):
                    preflight.check(self.get, self.contract)

    def test_allows_release_tag_with_digest(self):
        self.assertEqual(
            preflight.image_version(
                "registry:5000/grove:v0.1.0-alpha.15@sha256:abcdef"
            ),
            "v0.1.0-alpha.15",
        )

    def test_rejects_missing_or_ambiguous_controllers(self):
        for deployments in ([], [self.deployment, deepcopy(self.deployment)]):
            with self.subTest(count=len(deployments)):
                self.deployments = deployments
                with self.assertRaisesRegex(
                    preflight.Incompatible, "identify exactly one controller"
                ):
                    preflight.check(self.get, self.contract)

    def test_scopes_discovery_to_configured_namespace(self):
        preflight.check(self.get, self.contract, namespace="grove-system")
        self.assertEqual(
            self.paths[0], "/apis/apps/v1/namespaces/grove-system/deployments"
        )

    def test_explicit_controller_needs_no_deployment_list(self):
        preflight.check(
            self.get,
            self.contract,
            namespace="grove-system",
            deployment_name="grove-operator",
        )
        self.assertEqual(
            self.paths[0],
            "/apis/apps/v1/namespaces/grove-system/deployments/grove-operator",
        )

    def test_rejects_wrong_explicit_container(self):
        with self.assertRaisesRegex(preflight.Incompatible, "controller container"):
            preflight.check(
                self.get,
                self.contract,
                namespace="grove-system",
                deployment_name="grove-operator",
                container_name="wrong",
            )

    def test_rejects_incomplete_controller_rollout(self):
        for status in (
            {"observedGeneration": 1},
            {"replicas": 2},
            {"updatedReplicas": 0},
            {"availableReplicas": 0},
        ):
            with self.subTest(status=status):
                original = deepcopy(self.deployment["status"])
                self.deployment["status"].update(status)
                with self.assertRaisesRegex(
                    preflight.Incompatible, "rollout is incomplete"
                ):
                    preflight.check(self.get, self.contract)
                self.deployment["status"] = original

    def test_rejects_old_or_unknown_crd_schema(self):
        self.crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"] = {}
        with self.assertRaisesRegex(
            preflight.Incompatible, "Unverifiable external Grove CRD"
        ):
            preflight.check(self.get, self.contract)

    def test_crd_descriptions_do_not_change_compatibility(self):
        self.crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"][
            "description"
        ] = "Updated documentation"
        preflight.check(self.get, self.contract)

    def test_property_named_description_is_part_of_schema(self):
        self.crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"][
            "description"
        ] = {"type": "string"}
        with self.assertRaisesRegex(
            preflight.Incompatible, "Unverifiable external Grove CRD"
        ):
            preflight.check(self.get, self.contract)

    def test_rejects_unserved_or_unestablished_crds(self):
        self.crd["spec"]["versions"][0]["served"] = False
        with self.assertRaises(preflight.Incompatible):
            preflight.check(self.get, self.contract)
        self.crd["status"]["conditions"] = []
        with self.assertRaisesRegex(preflight.Incompatible, "not established"):
            preflight.check(self.get, self.contract)

    def test_api_errors_cannot_become_success(self):
        def forbidden(path):
            raise preflight.Incompatible("HTTP 403")

        with self.assertRaisesRegex(preflight.Incompatible, "HTTP 403"):
            preflight.check(forbidden, self.contract)

    def test_contract_covers_required_grove_crds(self):
        contract = json.loads((FILES / "compatibility.json").read_text())
        self.assertEqual(
            set(contract["crdSchemaDigests"]),
            {
                "podcliquesets.grove.io",
                "podcliques.grove.io",
                "podcliquescalinggroups.grove.io",
                "clustertopologybindings.grove.io",
                "podgangmaps.grove.io",
            },
        )
        for digests in contract["crdSchemaDigests"].values():
            for digest in digests:
                self.assertRegex(digest, r"^[a-f0-9]{64}$")

    def test_contract_minimum_tracks_bundled_grove_release(self):
        chart = FILES.parents[3] / "Chart.yaml"
        bundled = re.search(
            r"name: grove-charts\s+alias: grove\s+version: (\S+)", chart.read_text()
        )
        self.assertIsNotNone(bundled)
        contract = json.loads((FILES / "compatibility.json").read_text())
        self.assertEqual(contract["minimumVersion"], bundled[1])


if __name__ == "__main__":
    unittest.main()
