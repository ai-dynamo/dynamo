# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
