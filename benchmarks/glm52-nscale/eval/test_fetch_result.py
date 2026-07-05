#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("fetch-result.sh")
FAKE_KUBECTL = r"""#!/usr/bin/env python3
import os
import sys
import tarfile
from pathlib import Path

Path(os.environ["FAKE_KUBECTL_MARKER"]).touch()
arguments = sys.argv[1:]
if arguments[:1] == ["--context"]:
    arguments = arguments[2:]
command = arguments[arguments.index("--") + 1:]
separator = command.index("--")
remote_dir = command[separator + 1]
files = command[separator + 2:]
root = Path(os.environ["FAKE_REMOTE_ROOT"]) / remote_dir.removeprefix("/")
if not root.is_dir() or root.is_symlink():
    raise SystemExit(1)
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for name in files:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise SystemExit(1)
        archive.add(path, arcname=name, recursive=False)
"""


class FetchResultTests(unittest.TestCase):
    FILES = {
        "bfcl": {
            "summary.json",
            "complete-validation.json",
            "metadata.json",
            "endpoint-models.json",
            "runtime-continuity.json",
            "expected-ids.json",
            "failures.jsonl",
            "environment-lock.json",
            "environment.freeze.txt",
        },
        "swebench": {
            "score.json",
            "generation-summary.json",
            "run-metadata.json",
            "run-scope.json",
            "runtime-continuity.json",
            "environment.freeze.txt",
            "environment.normalized.freeze.txt",
        },
        "terminalbench": {
            "summary.json",
            "trials.csv",
            "task-images.json",
            "runtime-continuity.json",
        },
    }

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        kubectl = fake_bin / "kubectl"
        kubectl.write_text(FAKE_KUBECTL)
        kubectl.chmod(0o755)
        self.remote_root = self.root / "remote"
        self.remote_root.mkdir()
        self.marker = self.root / "kubectl-called"
        self.environment = os.environ.copy()
        self.environment.update(
            {
                "PATH": f"{fake_bin}{os.pathsep}{self.environment['PATH']}",
                "KUBE_CONTEXT": "synthetic-context",
                "NAMESPACE": "synthetic-namespace",
                "EVAL_RUNNER_POD": "synthetic-runner",
                "FAKE_REMOTE_ROOT": str(self.remote_root),
                "FAKE_KUBECTL_MARKER": str(self.marker),
            }
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_fetch(
        self, suite: str, remote: str, local: Path
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SCRIPT), suite, remote, str(local)],
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def populate(self, remote: str, files: set[str]) -> None:
        directory = self.remote_root / remote.removeprefix("/")
        directory.mkdir(parents=True)
        for name in files:
            (directory / name).write_text(f"{name}\n")
        (directory / "raw-trajectory.json").write_text("must not transfer\n")
        (directory / "generation.log").write_text("must not transfer\n")

    def test_fetches_only_allowlisted_compact_files(self) -> None:
        for suite, files in self.FILES.items():
            with self.subTest(suite=suite):
                remote = f"/artifacts/glm52-nscale/test/{suite}"
                local = self.root / f"local-{suite}"
                self.populate(remote, files)
                result = self.run_fetch(suite, remote, local)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({path.name for path in local.iterdir()}, files)

    def test_refuses_overwrite_and_remote_path_traversal(self) -> None:
        remote = "/artifacts/glm52-nscale/test/swebench"
        self.populate(remote, self.FILES["swebench"])
        local = self.root / "existing"
        local.mkdir()
        result = self.run_fetch("swebench", remote, local)
        self.assertNotEqual(result.returncode, 0)

        self.marker.unlink(missing_ok=True)
        result = self.run_fetch(
            "swebench",
            "/artifacts/glm52-nscale/test/../private",
            self.root / "traversal",
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.marker.exists())

    def test_missing_required_file_fails_without_partial_destination(self) -> None:
        remote = "/artifacts/glm52-nscale/test/terminal"
        files = self.FILES["terminalbench"] - {"trials.csv"}
        self.populate(remote, files)
        local = self.root / "incomplete"
        result = self.run_fetch("terminalbench", remote, local)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(local.exists())

    def test_remote_directory_symlink_is_rejected(self) -> None:
        target_remote = "/artifacts/glm52-nscale/test/real-swe"
        self.populate(target_remote, self.FILES["swebench"])
        link = self.remote_root / "artifacts/glm52-nscale/test/link-swe"
        link.symlink_to(self.remote_root / target_remote.removeprefix("/"))
        local = self.root / "symlinked"
        result = self.run_fetch(
            "swebench", "/artifacts/glm52-nscale/test/link-swe", local
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(local.exists())


if __name__ == "__main__":
    unittest.main()
