# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the released 1.6.0 frontend-sidecar probe rendering separate from authored intent."""

import hashlib
import json
import unittest
from pathlib import Path

import capture_frontend_sidecar_probes as capture
import yaml


class FrontendSidecarProbesFixtureTest(unittest.TestCase):
    def setUp(self):
        self.fixture = (
            Path(__file__).resolve().parents[1]
            / "internal/controller/testdata/upgrade/compatibility/frontend-sidecar-probes"
        )
        self.provenance = json.loads((self.fixture / "provenance.json").read_text())
        self.data = (self.fixture / "input.yaml").read_bytes()
        self.objects = list(yaml.safe_load_all(self.data))

    def test_captured_input_matches_provenance_and_lock(self):
        lineage = json.loads(
            (self.fixture.parents[1] / "compatibility.lock.json").read_text()
        )
        digest = hashlib.sha256(self.data).hexdigest()
        self.assertEqual(
            digest,
            lineage["inputs"]["compatibility/frontend-sidecar-probes/input.yaml"],
        )
        if lineage["reconciledVersion"] == capture.RELEASE:
            self.assertEqual(digest, self.provenance["inputSHA256"])
        self.assertEqual(list(self.provenance["observations"]), [capture.RELEASE])

    def test_released_operator_stamped_its_origin(self):
        dgd = next(o for o in self.objects if o["kind"] == "DynamoGraphDeployment")
        self.assertEqual(
            capture.RELEASE,
            dgd["metadata"]["annotations"]["nvidia.com/dynamo-operator-origin-version"],
        )

    def test_input_keeps_the_field_merged_sidecar_probes(self):
        self.assertEqual(capture.MERGED_PROBES, capture.sidecar_probes(self.objects))
        self.assertEqual(
            capture.MERGED_PROBES,
            self.provenance["observations"][capture.RELEASE]["sidecarProbes"],
        )

    def test_authored_intent_is_partial(self):
        authored = yaml.safe_load((self.fixture / "create.yaml").read_text())
        worker = next(
            c for c in authored["spec"]["components"] if c["name"] == "worker"
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
        self.assertEqual(authored, capture.authored_dgd())


if __name__ == "__main__":
    unittest.main()
