# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GroveVersionTest(unittest.TestCase):
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

            def helm(*args):
                result = subprocess.run(
                    ["helm", *args],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

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

            source = helm(
                "template", "dynamo", str(root), "--set", "global.grove.enabled=true"
            )
            self.assertIn(f'            - "{pin}"\n', source)

            helm("package", str(root), "--destination", directory)
            archive = Path(directory) / "platform-0.0.0.tgz"
            packaged = helm(
                "template",
                "dynamo",
                str(archive),
                "--set",
                "global.grove.enabled=true",
            )
            self.assertIn(f'            - "{pin}"\n', packaged)
