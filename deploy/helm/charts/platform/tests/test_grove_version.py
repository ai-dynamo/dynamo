# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("jq"), "requires jq CLI")
class GroveWebhookConfigurationTest(unittest.TestCase):
    def test_alpha_requests_are_covered_by_fail_closed_webhooks(self):
        """Accept equivalent beta rules without accepting fail-open or exact beta rules."""
        action = (
            ROOT.parents[3] / ".github/actions/setup-dynamo-operator/action.yml"
        ).read_text()
        query = re.search(
            r"(?s)if ! jq -e --arg webhook_name.*? '\n(.*?)\n          ' <<<",
            action,
        ).group(1)
        webhook = {
            "name": "pcs.defaulting.webhooks.grove.io",
            "failurePolicy": "Fail",
            "rules": [
                {
                    "apiGroups": ["grove.io"],
                    "apiVersions": ["v1alpha1"],
                    "resources": ["podcliquesets"],
                    "operations": ["CREATE"],
                }
            ],
        }
        for version, policy, failure, accepted in (
            ("v1alpha1", "Exact", "Fail", True),
            ("v1beta1", "Equivalent", "Fail", True),
            ("v1beta1", None, "Fail", True),
            ("v1beta1", "Exact", "Fail", False),
            ("v1beta1", "Equivalent", "Ignore", False),
            ("v1beta2", "Equivalent", "Fail", False),
        ):
            with self.subTest(version=version, policy=policy, failure=failure):
                candidate = copy.deepcopy(webhook)
                candidate["failurePolicy"] = failure
                if policy is not None:
                    candidate["matchPolicy"] = policy
                candidate["rules"][0]["apiVersions"] = [version]
                result = subprocess.run(
                    ["jq", "-e", "--arg", "webhook_name", webhook["name"], query],
                    input=json.dumps({"webhooks": [candidate]}),
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                self.assertEqual(result.returncode, 0 if accepted else 1, result.stderr)


@unittest.skipUnless(shutil.which("helm"), "requires Helm CLI")
class GroveVersionTest(unittest.TestCase):
    def helm(self, *args):
        result = subprocess.run(
            ["helm", *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_dependency_and_preflight_use_same_chart_version(self):
        with tempfile.TemporaryDirectory() as directory:
            # Use Helm's YAML parser to resolve anchors in the production Chart.yaml.
            verifier = Path(directory) / "verifier"
            (verifier / "templates").mkdir(parents=True)
            (verifier / "Chart.yaml").write_text(
                "apiVersion: v2\nname: verifier\nversion: 0.0.0\n"
            )
            (verifier / "templates/verify.yaml").write_text(
                "{{- $chart := .Values.platformChart | fromYaml -}}\n"
                '{{- $minimum := index $chart.annotations "dynamo.nvidia.com/grove-version" '
                '| required "Grove version annotation is missing" -}}\n'
                '{{- $dependency := "" -}}\n'
                "{{- range $chart.dependencies -}}\n"
                '{{- if eq .name "grove-charts" -}}\n'
                "{{- $dependency = .version -}}\n"
                "{{- end -}}\n"
                "{{- end -}}\n"
                '{{- $dependency = required "Grove dependency is missing" $dependency -}}\n'
                "{{- if ne $minimum $dependency -}}\n"
                '{{- fail (printf "Grove dependency %s differs from preflight minimum %s" '
                "$dependency $minimum) -}}\n"
                "{{- end }}\n"
                "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: grove-version-verifier\n"
            )

            self.helm(
                "template",
                "verifier",
                str(verifier),
                "--set-file",
                f"platformChart={ROOT / 'Chart.yaml'}",
            )

    def test_git_source_and_package_read_changed_dependency_without_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "platform"
            (root / "templates").mkdir(parents=True)
            shutil.copy(ROOT / "values.yaml", root)
            shutil.copy(ROOT / "templates/grove-preflight.yaml", root / "templates")
            shutil.copytree(ROOT / "files", root / "files", symlinks=True)
            operator = root / "charts/dynamo-operator"
            (operator / "templates").mkdir(parents=True)
            for name in ("Chart.yaml", "values.yaml", "templates/_helpers.tpl"):
                shutil.copy(ROOT / "components/operator" / name, operator / name)
            grove = root / "charts/grove"
            grove.mkdir()

            # Keep Grove disabled, as it is for an externally managed installation.
            pin = "v99.0.0"
            (root / "Chart.yaml").write_text(
                "apiVersion: v2\nname: platform\nversion: 0.0.0\n"
                "annotations:\n"
                f"  dynamo.nvidia.com/grove-version: &groveVersion {pin}\n"
                "dependencies:\n"
                '  - name: dynamo-operator\n    version: "*"\n'
                "    condition: dynamo-operator.enabled\n"
                "  - name: grove-charts\n    alias: grove\n"
                "    version: *groveVersion\n"
                "    condition: global.grove.install\n"
            )
            (grove / "Chart.yaml").write_text(
                f"apiVersion: v2\nname: grove-charts\nversion: {pin}\n"
            )
            self.assertFalse(any(path.is_symlink() for path in root.rglob("*")))

            source = self.helm(
                "template", "dynamo", str(root), "--set", "global.grove.enabled=true"
            )
            self.assertIn(f'            - "{pin}"\n', source)

            self.helm("package", str(root), "--destination", directory)
            archive = Path(directory) / "platform-0.0.0.tgz"
            packaged = self.helm(
                "template",
                "dynamo",
                str(archive),
                "--set",
                "global.grove.enabled=true",
            )
            self.assertIn(f'            - "{pin}"\n', packaged)
