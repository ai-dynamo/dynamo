#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("assert-clean-slate.sh")
FAKE_KUBECTL = """#!/usr/bin/env python3
import os
import sys

arguments = sys.argv[1:]
if arguments[:1] == ["--context"]:
    arguments = arguments[2:]
if arguments[:1] != ["get"]:
    raise SystemExit(2)
resource = arguments[1]
if os.environ.get("FAKE_KUBECTL_ERROR") == resource:
    raise SystemExit(7)
values = {
    "pods": ["pod/glm52-eval-runner"],
}
if os.environ.get("FAKE_EXISTING") == "service":
    values["services"] = ["service/glm52-vllm-serve"]
if os.environ.get("FAKE_EXISTING") == "frontend":
    values["pods"] = [
        "pod/glm52-eval-runner",
        "pod/glm52-dynamo-vllm-frontend-abc",
    ]
print("\\n".join(values.get(resource, [])))
"""


class CleanSlateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        fake_bin = Path(self.temporary.name)
        kubectl = fake_bin / "kubectl"
        kubectl.write_text(FAKE_KUBECTL)
        kubectl.chmod(0o755)
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{self.environment['PATH']}",
                "KUBE_CONTEXT": "synthetic-context",
                "NAMESPACE": "synthetic-namespace",
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_script(self, **environment: str) -> subprocess.CompletedProcess[str]:
        variables = self.environment | environment
        return subprocess.run(
            [str(SCRIPT)],
            env=variables,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_eval_runner_alone_is_a_clean_slate(self) -> None:
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Clean-slate PASS", result.stdout)

    def test_service_or_cpu_frontend_blocks_reuse(self) -> None:
        for existing in ("service", "frontend"):
            with self.subTest(existing=existing):
                result = self.run_script(FAKE_EXISTING=existing)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Refusing to reuse", result.stderr)

    def test_kubectl_read_failure_fails_closed(self) -> None:
        result = self.run_script(FAKE_KUBECTL_ERROR="services")
        self.assertEqual(result.returncode, 7)


if __name__ == "__main__":
    unittest.main()
