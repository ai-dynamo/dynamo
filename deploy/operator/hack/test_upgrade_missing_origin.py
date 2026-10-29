# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep the independently captured legacy graph separate from authored test intent."""

import hashlib
import json
import unittest
from pathlib import Path

import yaml


class MissingOriginFixtureTest(unittest.TestCase):
    def test_captured_input_matches_provenance(self):
        fixture = (
            Path(__file__).resolve().parents[1]
            / "internal/controller/testdata/upgrade-compatibility/missing-origin"
        )
        provenance = json.loads((fixture / "provenance.json").read_text())
        data = (fixture / "input.yaml").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), provenance["inputSHA256"])
        objects = list(yaml.safe_load_all(data))
        dgd = next(o for o in objects if o["kind"] == "DynamoGraphDeployment")
        self.assertNotIn(
            "nvidia.com/dynamo-operator-origin-version", dgd["metadata"]["annotations"]
        )
        self.assertEqual(list(provenance["observations"]), ["1.4.0", "1.5.0", "1.6.0"])
        for observation in provenance["observations"].values():
            self.assertEqual(
                observation["workerHash"],
                dgd["metadata"]["annotations"]["nvidia.com/current-worker-hash-v2"],
            )
