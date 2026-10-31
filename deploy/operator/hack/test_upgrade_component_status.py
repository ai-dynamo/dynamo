# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component lowering goldens must tolerate runtime status races, not intent changes."""

import copy
import unittest

import generate_upgrade_fixtures as fixtures
import yaml


class UpgradeComponentStatusTest(unittest.TestCase):
    def test_absent_workload_status_is_not_manufactured(self):
        path = (
            fixtures.FIXTURES / "latest/sglang/aggregated/multi/component/output.yaml"
        )
        objects = list(yaml.safe_load_all(path.read_text()))
        workload = next(obj for obj in objects if obj["kind"] == "LeaderWorkerSet")
        workload.pop("status", None)

        # No LWS controller runs at the immediate-workload boundary.
        normalized = fixtures.golden_objects(objects, latest=True)
        self.assertNotIn(
            "status",
            next(obj for obj in normalized if obj["kind"] == "LeaderWorkerSet"),
        )

        # An observed runtime status is ignored without changing the historical capture.
        workload["status"] = {"readyReplicas": 0}
        normalized = fixtures.golden_objects(objects, latest=True)
        self.assertEqual(
            next(obj for obj in normalized if obj["kind"] == "LeaderWorkerSet")[
                "status"
            ],
            {"$ignore": True},
        )
        historical = fixtures.golden_objects(objects)
        self.assertEqual(
            next(obj for obj in historical if obj["kind"] == "LeaderWorkerSet")[
                "status"
            ],
            {"readyReplicas": 0},
        )

    def test_runtime_status_does_not_change_component_golden(self):
        path = (
            fixtures.FIXTURES / "latest/sglang/aggregated/single/component/output.yaml"
        )
        objects = list(yaml.safe_load_all(path.read_text()))
        for obj in objects:
            if obj["kind"] == "DynamoComponentDeployment":
                components = [obj["status"]["component"]]
            elif obj["kind"] == "DynamoGraphDeployment":
                components = obj["status"]["components"].values()
            else:
                continue
            for component in components:
                component.pop("$strict", None)
                component.update(gpusPerEngine=0, gpusPerReplica=0, readyReplicas=0)
        original = copy.deepcopy(objects)
        expected = fixtures.golden_objects(objects, latest=True)
        self.assertEqual(original, objects, "normalization must not mutate its input")
        actual = copy.deepcopy(objects)
        for obj in actual:
            if obj["kind"] == "DynamoComponentDeployment":
                components = [obj["status"]["component"]]
                for condition in obj["status"]["conditions"]:
                    if condition["type"] == "Available":
                        condition["reason"] = "Reconciling"
                        condition["message"] = "the object has been modified"
            elif obj["kind"] == "DynamoGraphDeployment":
                components = obj["status"]["components"].values()
            else:
                continue
            for component in components:
                # Exercise omitted GPU observations and independently changing replica counts.
                component.pop("gpusPerEngine", None)
                component.pop("gpusPerReplica", None)
                component.update(readyReplicas=1, availableReplicas=1, replicas=2)
        self.assertEqual(expected, fixtures.golden_objects(actual, latest=True))

        # Historical capture keeps the complete observed status, including transient errors.
        historical = fixtures.golden_objects(actual)
        for obj in historical:
            if obj["kind"] == "DynamoComponentDeployment":
                self.assertEqual(obj["status"]["component"]["readyReplicas"], 1)
                self.assertNotIn("$strict", obj["status"]["component"])
                self.assertEqual(
                    obj["status"]["conditions"][0]["reason"], "Reconciling"
                )

    def test_lowering_identity_and_spec_remain_exact(self):
        path = (
            fixtures.FIXTURES / "latest/sglang/aggregated/single/component/output.yaml"
        )
        objects = list(yaml.safe_load_all(path.read_text()))
        expected = fixtures.golden_objects(objects, latest=True)
        for kind, field, value in (
            ("Deployment", ("spec", "replicas"), 99),
            ("Deployment", ("metadata", "ownerReferences", 0, "name"), "other"),
            (
                "Deployment",
                ("metadata", "annotations", "nvidia.com/last-applied-generation"),
                "99",
            ),
            ("DynamoComponentDeployment", ("status", "observedGeneration"), 99),
            (
                "DynamoComponentDeployment",
                ("status", "component", "componentNames"),
                ["other"],
            ),
            (
                "DynamoComponentDeployment",
                ("status", "component", "runtimeNamespace"),
                "other",
            ),
        ):
            with self.subTest(kind=kind, field=field):
                actual = copy.deepcopy(objects)
                obj = next(obj for obj in actual if obj["kind"] == kind)
                parent = obj
                for key in field[:-1]:
                    parent = parent[key]
                parent[field[-1]] = value
                self.assertNotEqual(
                    expected, fixtures.golden_objects(actual, latest=True)
                )
