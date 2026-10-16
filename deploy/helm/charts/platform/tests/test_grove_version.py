# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class GroveVersionTest(unittest.TestCase):
    def test_dependency_bump_requires_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "sync_grove_version.py"
            shutil.copy(ROOT / script.name, script)
            chart = root / "Chart.yaml"
            chart.write_text((ROOT / chart.name).read_text())
            output = (
                root / "components/operator/files/grove-preflight/minimum-version.yaml"
            )
            output.parent.mkdir(parents=True)

            def run(*args):
                return subprocess.run(
                    [sys.executable, str(script), *args], capture_output=True, text=True
                )

            self.assertNotEqual(run("--check").returncode, 0)
            self.assertEqual(run().returncode, 0)
            self.assertEqual(run("--check").returncode, 0)
            current = output.read_text().split("version: ")[1].strip()
            newer = "v99.0.0"
            chart.write_text(
                chart.read_text().replace(f"version: {current}", f"version: {newer}")
            )
            self.assertNotEqual(run("--check").returncode, 0)
            self.assertEqual(run().returncode, 0)
            self.assertIn(f"version: {newer}\n", output.read_text())
            self.assertEqual(run("--check").returncode, 0)
