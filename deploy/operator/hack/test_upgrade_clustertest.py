# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Makefile/shell ownership boundary without touching Kubernetes."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class UpgradeClusterLifecycleTest(unittest.TestCase):
    def test_shared_setup_isolates_inherited_kubeconfig(self):
        operator = Path(__file__).resolve().parents[1]
        text = (operator / "Makefile").read_text()
        start = text.index(".PHONY: setup-clustertest-kind")
        end = text.index(".PHONY: setup-clustertest-crds-in-cluster", start)
        defaults = "\n".join(
            line for line in text.splitlines() if line.startswith("CLUSTERTEST_")
        )
        for reuse, override in ((False, False), (True, False), (True, True)):
            with self.subTest(reuse=reuse, override=override):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve()
                    (root / "Makefile").write_text(
                        "SHELL := /bin/bash\n.SHELLFLAGS := -ec\n"
                        "KIND := ./kind\nLOCALBIN := $(CURDIR)\n"
                        + defaults
                        + "\nkubectl kind:\n\t@true\n"
                        + text[start:end]
                    )
                    sentinel = root / "user.kubeconfig"
                    sentinel.write_text("user configuration\n")
                    (root / "kind").write_text(
                        """#!/bin/bash
set -eu
if [[ "$1 $2" == 'get clusters' ]]; then
    if [[ "$REUSE" == true ]]; then echo dynamo-clustertest-v136; fi
    exit 0
fi
echo "$1 $2" > operation
while [[ "$1" != --kubeconfig ]]; do shift; done
printf '%s\n' "$2" > config-path
echo 'kind configuration' > "$2"
"""
                    )
                    (root / "kind").chmod(0o755)
                    command = ["make", "setup-clustertest-kind"]
                    expected = root / "clustertest.kubeconfig"
                    if override:
                        expected = root / "explicit.kubeconfig"
                        command.append(f"CLUSTERTEST_KUBECONFIG={expected}")
                    result = subprocess.run(
                        command,
                        cwd=root,
                        env=dict(
                            os.environ,
                            KUBECONFIG=str(sentinel),
                            REUSE=str(reuse).lower(),
                        ),
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=20,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual(
                        (root / "operation").read_text().strip(),
                        "export kubeconfig" if reuse else "create cluster",
                    )
                    self.assertEqual(
                        (root / "config-path").read_text().strip(), str(expected)
                    )
                    self.assertEqual(sentinel.read_text(), "user configuration\n")
                    self.assertEqual(expected.read_text(), "kind configuration\n")

    def test_exclusive_cluster_ownership_and_cleanup(self):
        operator = Path(__file__).resolve().parents[1]
        for mode in ("success", "failure", "interrupt", "collision"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "bin").mkdir()
                (root / "hack").mkdir()
                shutil.copy(operator / "hack/run_upgrade_clustertest.sh", root / "hack")
                # Use the actual shared Kind recipe; replace only Kubernetes-dependent commands.
                text = (operator / "Makefile").read_text()
                start = text.index(".PHONY: setup-clustertest-kind")
                end = text.index(".PHONY: setup-clustertest-crds-in-cluster", start)
                (root / "Makefile").write_text(
                    "SHELL := /bin/bash\n.SHELLFLAGS := -ec\nKIND := ./bin/kind\n"
                    "kubectl kind:\n\t@true\n"
                    + text[start:end]
                    + "setup-clustertest-crds-in-cluster:\n\t@echo crds >> calls\n"
                    "contract:\n\t@./action\n"
                )
                (root / "bin/kind").write_text(
                    """#!/bin/bash
set -eu
echo "$*" >> calls
if [[ "$1 $2" == 'get clusters' && "$MODE" == collision ]]; then
    echo "$DYNAMO_CLUSTERTEST_CONTEXT" | sed 's/^kind-//'
fi
if [[ "$1 $2" == 'create cluster' ]]; then
    echo "$KUBECONFIG" > config-path
    echo "$DYNAMO_CLUSTERTEST_CONTEXT" > context
fi
"""
                )
                (root / "action").write_text(
                    """#!/bin/bash
set -eu
case "$MODE" in
    failure) exit 7 ;;
    interrupt) kill -TERM "$UPGRADE_WRAPPER_PID" ;;
esac
"""
                )
                for file in (root / "bin/kind", root / "action"):
                    file.chmod(0o755)
                # Record the wrapper PID for an interruption after it has created its cluster.
                wrapper = root / "hack/run_upgrade_clustertest.sh"
                wrapper.write_text(
                    wrapper.read_text().replace(
                        "set -euo pipefail",
                        "set -euo pipefail\nexport UPGRADE_WRAPPER_PID=$$",
                    )
                )
                result = subprocess.run(
                    ["bash", str(wrapper), "contract"],
                    cwd=root,
                    env=dict(
                        os.environ,
                        MODE=mode,
                        KUBECONFIG="/untouched-user-config",
                        DYNAMO_CLUSTERTEST_CONTEXT="untouched-user-context",
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                calls = (root / "calls").read_text()
                self.assertEqual(
                    result.returncode == 0, mode == "success", result.stderr
                )
                if mode == "collision":
                    self.assertNotIn("create cluster", calls)
                    self.assertNotIn("delete cluster", calls)
                else:
                    context = (root / "context").read_text().strip()
                    name = context.removeprefix("kind-")
                    self.assertIn("delete cluster --name " + name, calls)
                    config = Path((root / "config-path").read_text().strip())
                    self.assertNotEqual(str(config), "/untouched-user-config")
                    self.assertFalse(
                        config.parent.exists(), "temporary kubeconfig must be removed"
                    )
