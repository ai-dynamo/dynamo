# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep each independently captured compatibility graph separate from authored intent."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import capture_compatibility as capture
import yaml

FIXTURES = Path(__file__).resolve().parents[1] / "internal/controller/testdata/upgrade"


class CompatibilityFixtureTest(unittest.TestCase):
    def fixture(self, scenario):
        directory = FIXTURES / "compatibility" / scenario.name
        data = (directory / "input.yaml").read_bytes()
        return (
            directory,
            data,
            list(yaml.safe_load_all(data)),
            json.loads((directory / "provenance.json").read_text()),
        )

    def test_every_scenario_is_captured_and_locked(self):
        lineage = json.loads((FIXTURES / "compatibility.lock.json").read_text())
        self.assertEqual(
            sorted(lineage["inputs"]),
            sorted(f"compatibility/{name}/input.yaml" for name in capture.SCENARIOS),
            "every scenario has a locked input and every locked input has a scenario",
        )
        for scenario in capture.SCENARIOS.values():
            with self.subTest(scenario=scenario.name):
                _, data, objects, provenance = self.fixture(scenario)
                digest = hashlib.sha256(data).hexdigest()
                self.assertEqual(
                    digest,
                    lineage["inputs"][f"compatibility/{scenario.name}/input.yaml"],
                )
                last = list(scenario.releases)[-1]
                if lineage["reconciledVersion"] == last:
                    self.assertEqual(digest, provenance["inputSHA256"])
                self.assertEqual(
                    list(provenance["observations"]), list(scenario.releases)
                )
                self.assertEqual(provenance["releases"], scenario.releases)
                dgd = capture.dgd_of(objects)
                self.assertEqual(
                    scenario.origin,
                    dgd["metadata"]["annotations"].get(capture.ORIGIN_ANNOTATION),
                )

    def test_every_scenario_has_a_golden_with_literal_identity(self):
        for scenario in capture.SCENARIOS.values():
            with self.subTest(scenario=scenario.name):
                directory, _, objects, _ = self.fixture(scenario)
                golden = list(
                    yaml.safe_load_all((directory / "output.yaml").read_text())
                )
                self.assertEqual(
                    sorted((o["kind"], o["metadata"]["name"]) for o in objects),
                    sorted((o["kind"], o["metadata"]["name"]) for o in golden),
                    "the golden covers exactly the released graph",
                )
                dgd = capture.dgd_of(golden)
                annotations = dgd["metadata"]["annotations"]
                self.assertEqual(
                    scenario.origin, annotations.get(capture.ORIGIN_ANNOTATION)
                )
                self.assertNotIn("$var", annotations[capture.WORKER_HASH_ANNOTATION])

    def test_authored_intent_matches_the_scenario(self):
        for scenario in capture.SCENARIOS.values():
            with self.subTest(scenario=scenario.name):
                directory, _, _, _ = self.fixture(scenario)
                authored = yaml.safe_load((directory / "create.yaml").read_text())
                self.assertEqual(scenario.create(), authored)

    def test_missing_origin_keeps_one_worker_hash_across_releases(self):
        _, _, _, provenance = self.fixture(capture.SCENARIOS["missing-origin"])
        self.assertEqual(
            len({item["workerHash"] for item in provenance["observations"].values()}), 1
        )

    def test_frontend_sidecar_probes_keep_the_field_merge(self):
        scenario = capture.SCENARIOS["frontend-sidecar-probes"]
        _, _, objects, provenance = self.fixture(scenario)
        self.assertEqual(capture.MERGED_PROBES, capture.sidecar_probes(objects))
        self.assertEqual(
            capture.MERGED_PROBES,
            provenance["observations"]["1.6.0"]["sidecarProbes"],
        )
        worker = next(
            c for c in scenario.create()["spec"]["components"] if c["name"] == "worker"
        )
        sidecar = next(
            c
            for c in worker["podTemplate"]["spec"]["containers"]
            if c["name"] == capture.SIDECAR
        )
        self.assertEqual(
            {"httpGet": {"path": "/health", "port": "http"}, "initialDelaySeconds": 30},
            sidecar["readinessProbe"],
        )
        self.assertEqual(
            {"httpGet": {"path": "/live", "port": "http"}, "periodSeconds": 30},
            sidecar["livenessProbe"],
        )

    def test_final_export_ignores_server_metadata_only(self):
        _, _, objects, _ = self.fixture(capture.SCENARIOS["missing-origin"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("first", "second"):
                observed = copy.deepcopy(objects)
                for obj in observed:
                    metadata = obj["metadata"]
                    metadata.update(
                        uid=value, resourceVersion=value, creationTimestamp=value
                    )
                    metadata["managedFields"] = [{"manager": value}]
                    for ref in metadata.get("ownerReferences", []):
                        ref["uid"] = value
                source = root / f"{value}.json"
                source.write_text(json.dumps(observed))
                capture.export_snapshot(source, root / f"{value}.yaml")
            self.assertEqual(
                (root / "first.yaml").read_bytes(), (root / "second.yaml").read_bytes()
            )
            self.assertEqual(
                list(yaml.safe_load_all((root / "first.yaml").read_text())),
                objects,
                "only API-assigned metadata may change; preserve intent, status and origin",
            )


if __name__ == "__main__":
    unittest.main()
