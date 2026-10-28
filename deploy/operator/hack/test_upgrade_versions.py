# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release gate must reject skipped capture even when both versions parse."""

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_upgrade_fixtures as fixtures
import yaml
from generate_upgrade_fixtures import app_version, validate_versions


class UpgradeVersionTest(unittest.TestCase):
    def test_operator_app_version_is_authoritative(self):
        self.assertEqual(app_version('version: 99.0.0\nappVersion: "1.6.0"\n'), "1.6.0")

    def test_adjacent_development_and_release_boundary(self):
        for current, reconciled, previous in [
            ("1.6.0", "1.5.0", "1.6.0"),
            ("1.6.0", "1.6.0", "1.6.0"),
            ("1.7.0", "1.6.0", "1.6.0"),
            ("1.7.0", "1.5.0", "1.6.0"),
            ("1.7.0", "1.5.0", None),
        ]:
            with self.subTest(current=current, reconciled=reconciled):
                validate_versions(current, reconciled, previous)

    def test_premature_skipped_and_major_bumps_fail(self):
        for current, reconciled, previous in [
            ("1.7.0", "1.4.0", None),
            ("1.8.0", "1.6.0", None),
            ("1.8.0", "1.5.0", "1.7.0"),
            ("1.6.0", "1.7.0", None),
            ("1.8.0", "1.7.0", "1.6.0"),
            ("2.0.0", "1.9.0", "1.9.0"),
        ]:
            with (
                self.subTest(current=current, reconciled=reconciled),
                self.assertRaises(ValueError),
            ):
                validate_versions(current, reconciled, previous)


class UpgradeFixtureLifecycleTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tree = self.root / "fixtures"
        shutil.copytree(fixtures.FIXTURES, self.tree)
        chart = self.root / fixtures.CHART
        chart.parent.mkdir(parents=True)
        chart.write_text("appVersion: 1.6.0\n")
        for name, value in (("ROOT", self.root), ("FIXTURES", self.tree)):
            replacement = patch.object(fixtures, name, value)
            replacement.start()
            self.addCleanup(replacement.stop)

    def test_additional_scenario_is_valid_but_incomplete_release_is_blocked(self):
        shutil.copytree(
            self.tree / "latest/sglang-1.5", self.tree / "latest/additional"
        )
        for path in (self.tree / "latest/additional").iterdir():
            path.write_text(
                path.read_text().replace(
                    "upgrade-latest-sglang-1-5", "upgrade-latest-additional"
                )
            )
        fixtures.check()
        before = {
            p.relative_to(self.tree): p.read_bytes()
            for p in self.tree.rglob("*")
            if p.is_file()
        }
        with patch.object(fixtures.subprocess, "run") as capture:
            with self.assertRaisesRegex(ValueError, "Incomplete release matrix"):
                fixtures.regenerate()
            capture.assert_not_called()
        self.assertEqual(
            before,
            {
                p.relative_to(self.tree): p.read_bytes()
                for p in self.tree.rglob("*")
                if p.is_file()
            },
        )

    def test_regeneration_cannot_skip_a_minor(self):
        (self.root / fixtures.CHART).write_text("appVersion: 1.7.0\n")
        (self.tree / "RECONCILED_VERSION").write_text("1.5.0\n")
        fixtures.check()
        with patch.object(fixtures.subprocess, "run") as capture:
            with self.assertRaisesRegex(ValueError, "skips a minor"):
                fixtures.regenerate(release_tag="v1.7.0")
            capture.assert_not_called()

    def capture_api_outputs(self, command, *, env, **kwargs):
        # Model the capture boundary only; production admission/reconciliation is tested by clustertest.
        version = fixtures.app_version((self.root / fixtures.CHART).read_text())
        root = Path(env["DYNAMO_UPGRADE_CAPTURE_DIR"])
        for role in ("history", "latest"):
            for source in fixtures.fixture_inputs(role):
                relative = source.relative_to(self.tree)
                objects = list(yaml.safe_load_all(source.read_text()))
                if role == "history":
                    destination = root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(yaml.safe_dump_all(objects))
                    continue
                for destination in (
                    root / relative,
                    root / "history" / version / Path(*relative.parts[1:]),
                ):
                    namespace = re.sub(
                        r"[^a-z0-9]+",
                        "-",
                        "upgrade-" + destination.parent.relative_to(root).as_posix(),
                    )
                    for obj in objects:
                        obj["metadata"].update(namespace=namespace, uid=namespace)
                        obj["metadata"]["annotations"][
                            "nvidia.com/dynamo-operator-origin-version"
                        ] = version
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(yaml.safe_dump_all(objects))

    def test_complete_matrix_survives_two_release_cycles(self):
        shutil.rmtree(self.tree / "latest")
        matrix = json.loads((self.tree / "required-matrix.json").read_text())[
            "requiredScenarios"
        ]
        for scenario in matrix + ["additional"]:
            path = self.tree / "latest" / scenario / "input.yaml"
            path.parent.mkdir(parents=True)
            path.write_text(
                yaml.safe_dump(
                    {
                        "apiVersion": "nvidia.com/v1beta1",
                        "kind": "DynamoGraphDeployment",
                        "metadata": {
                            "name": "workers",
                            "namespace": "upgrade-latest-" + scenario.replace("/", "-"),
                            "annotations": {
                                "nvidia.com/dynamo-operator-origin-version": "0.0.0"
                            },
                        },
                        "spec": {"image": "runtime:fixed"},
                    }
                )
            )
            fixtures.write_golden(
                path.with_name("output.yaml"),
                [yaml.safe_load(path.read_text())],
                latest=True,
            )

        # Exercise publication and revalidation across two versions, including every required path.
        with (
            patch.object(
                fixtures.subprocess, "run", side_effect=self.capture_api_outputs
            ),
            patch.object(
                fixtures,
                "capture_source",
                side_effect=lambda current, *args: {
                    "sourceKind": "release-tag",
                    "sourceRef": f"v{current}",
                    "sourceRevision": "a" * 40,
                },
            ),
        ):
            fixtures.regenerate()
            fixtures.check()
            with self.assertRaisesRegex(ValueError, "already reconciled"):
                fixtures.regenerate()
            latest = {
                p.relative_to(self.tree): p.read_bytes()
                for p in (self.tree / "latest").rglob("output.yaml")
            }
            (self.root / fixtures.CHART).write_text("appVersion: 1.7.0\n")
            with patch.object(
                fixtures.subprocess, "check_output", return_value="appVersion: 1.6.0\n"
            ):
                fixtures.check("base")
            fixtures.regenerate()
            fixtures.check()
        self.assertEqual(
            latest,
            {
                p.relative_to(self.tree): p.read_bytes()
                for p in (self.tree / "latest").rglob("output.yaml")
            },
        )
        namespaces = []
        for path in fixtures.fixture_inputs("history") + fixtures.fixture_inputs(
            "latest"
        ):
            namespaces.append(
                next(yaml.safe_load_all(path.read_text()))["metadata"]["namespace"]
            )
        self.assertEqual(len(namespaces), len(set(namespaces)))
        self.assertEqual(
            (self.tree / "RECONCILED_VERSION").read_text().strip(), "1.7.0"
        )
        for path in (self.tree / "latest").rglob("output.yaml"):
            self.assertIn("$var:operator-version", path.read_text())
            self.assertIn("runtime:fixed", path.read_text())
        for path in (self.tree / "history/1.6.0").rglob("input.yaml"):
            self.assertEqual(
                next(yaml.safe_load_all(path.read_text()))["metadata"]["annotations"][
                    "nvidia.com/dynamo-operator-origin-version"
                ],
                "1.6.0",
            )

    def test_candidate_capture_does_not_publish_history(self):
        (self.tree / "required-matrix.json").write_text(
            json.dumps({"requiredScenarios": ["sglang-1.5"]})
        )
        before = {
            p.relative_to(self.tree): p.read_bytes()
            for p in self.tree.rglob("*")
            if p.is_file()
        }
        source = {
            "sourceKind": "release-candidate",
            "sourceRef": "a" * 40,
            "sourceRevision": "a" * 40,
        }
        with (
            patch.object(fixtures, "capture_source", return_value=source),
            patch.object(
                fixtures.subprocess, "run", side_effect=self.capture_api_outputs
            ),
        ):
            output = self.root / "candidate"
            fixtures.regenerate(candidate_sha="a" * 40, candidate_output=output)
        self.assertEqual(
            before,
            {
                p.relative_to(self.tree): p.read_bytes()
                for p in self.tree.rglob("*")
                if p.is_file()
            },
        )
        self.assertEqual(
            json.loads((output / "capture-source.json").read_text())["sourceKind"],
            "release-candidate",
        )
        self.assertTrue((output / "history/1.6.0/sglang-1.5/input.yaml").is_file())

    def test_failed_capture_does_not_publish_history(self):
        (self.tree / "required-matrix.json").write_text(
            json.dumps({"requiredScenarios": ["sglang-1.5"]})
        )
        before = {
            p.relative_to(self.tree): p.read_bytes()
            for p in self.tree.rglob("*")
            if p.is_file()
        }
        with (
            patch.object(fixtures, "capture_source", return_value={}),
            patch.object(
                fixtures.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "go test"),
            ),
            self.assertRaises(subprocess.CalledProcessError),
        ):
            fixtures.regenerate()
        self.assertEqual(
            before,
            {
                p.relative_to(self.tree): p.read_bytes()
                for p in self.tree.rglob("*")
                if p.is_file()
            },
        )


if __name__ == "__main__":
    unittest.main()
