# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the independently captured legacy graph separate from authored test intent."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import capture_missing_origin as capture
import yaml


class MissingOriginFixtureTest(unittest.TestCase):
    def test_captured_input_matches_provenance(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "internal/controller/testdata/upgrade/compatibility/missing-origin"
        )
        provenance = json.loads((fixture / "provenance.json").read_text())
        data = (fixture / "input.yaml").read_bytes()
        lineage = json.loads(
            (fixture.parents[1] / "compatibility.lock.json").read_text()
        )
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            lineage["inputs"]["compatibility/missing-origin/input.yaml"],
        )
        if lineage["reconciledVersion"] == "1.6.0":
            self.assertEqual(
                hashlib.sha256(data).hexdigest(), provenance["inputSHA256"]
            )
        objects = list(yaml.safe_load_all(data))
        dgd = next(o for o in objects if o["kind"] == "DynamoGraphDeployment")
        self.assertNotIn(
            "nvidia.com/dynamo-operator-origin-version", dgd["metadata"]["annotations"]
        )
        self.assertEqual(list(provenance["observations"]), ["1.4.0", "1.5.0", "1.6.0"])
        self.assertEqual(
            len({item["workerHash"] for item in provenance["observations"].values()}), 1
        )

    def test_final_export_ignores_server_metadata_only(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "internal/controller/testdata/upgrade/compatibility/missing-origin/input.yaml"
        )
        objects = list(yaml.safe_load_all(fixture.read_text()))
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
                "only API-assigned metadata may change; preserve intent, status and absent origin",
            )
