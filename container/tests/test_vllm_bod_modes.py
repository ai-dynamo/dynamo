# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

CONTAINER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTAINER_DIR.parent
sys.path.insert(0, str(CONTAINER_DIR))

import render  # noqa: E402


class VllmBuildOnDemandModeTest(unittest.TestCase):
    installer = CONTAINER_DIR / "deps/vllm/install_custom_sources.sh"

    @classmethod
    def setUpClass(cls) -> None:
        with (CONTAINER_DIR / "context.yaml").open() as stream:
            cls.context = yaml.safe_load(stream)

    def run_installer_function(
        self,
        function: str,
        *args: str,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = 'source "$1"; shift; "$@"'
        function_env = os.environ.copy()
        function_env.update(
            {
                "CUDA_VERSION": "13.0",
                "FLASHINF_REF": "test",
            }
        )
        if env is not None:
            function_env.update(env)
        return subprocess.run(
            [
                "bash",
                "-c",
                command,
                "bash",
                str(self.installer),
                function,
                *args,
            ],
            cwd=cwd,
            env=function_env,
            check=False,
            capture_output=True,
            text=True,
        )

    def verify_fixture(
        self,
        extension_names: tuple[str, ...],
        exact_native: bool,
        unowned_extension_names: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_dir = root / "vllm"
            package_dir.mkdir()
            (package_dir / "__init__.py").touch()
            dist_info = root / "vllm-1.0.dist-info"
            dist_info.mkdir()
            (dist_info / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: vllm\nVersion: 1.0\n"
            )
            records = ["vllm/__init__.py,,"]
            for extension_name in extension_names:
                (package_dir / extension_name).touch()
                records.append(f"vllm/{extension_name},,")
            for extension_name in unowned_extension_names:
                (package_dir / extension_name).touch()
            records.extend(
                [
                    "vllm-1.0.dist-info/METADATA,,",
                    "vllm-1.0.dist-info/RECORD,,",
                ]
            )
            (dist_info / "RECORD").write_text("\n".join(records))

            packaging_dir = root / "packaging"
            packaging_dir.mkdir()
            (packaging_dir / "__init__.py").touch()
            (packaging_dir / "utils.py").write_text(
                "def canonicalize_name(name):\n"
                "    return name.lower().replace('_', '-').replace('.', '-')\n"
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            python = bin_dir / "python3"
            python.write_text(
                f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -S "$@"\n'
            )
            python.chmod(0o755)
            env = {
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "PYTHONPATH": str(root),
            }
            return self.run_installer_function(
                "verify_vllm_install",
                "",
                "exact-native" if exact_native else "",
                env=env,
            )

    def render_runtime(self, platform: str) -> str:
        args = SimpleNamespace(
            framework="vllm",
            device="cuda",
            target="runtime",
            platform=platform,
            cuda_version="13.0",
            make_efa=False,
        )
        environment = render._make_jinja_env(CONTAINER_DIR)
        template = environment.get_template("Dockerfile.template")
        return template.render(
            context=self.context,
            **render._render_context(args, self.context),
        )

    def test_blank_exact_wheel_commit_keeps_legacy_multiarch_render(self) -> None:
        self.assertEqual(self.context["vllm"]["vllm_precompiled_wheel_commit"], "")
        self.assertEqual(
            self.context["vllm"]["vllm_precompiled_wheel_variant"], "cu130"
        )

        dockerfile = self.render_runtime("multi")
        self.assertIn(
            "FROM --platform=linux/amd64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} "
            "AS vllm_runtime_amd64",
            dockerfile,
        )
        self.assertIn(
            "FROM --platform=linux/arm64 ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} "
            "AS vllm_runtime_arm64",
            dockerfile,
        )
        self.assertIn("ARG VLLM_PRECOMPILED_WHEEL_COMMIT=", dockerfile)
        self.assertIn("ARG VLLM_PRECOMPILED_WHEEL_VARIANT=cu130", dockerfile)

    def test_exact_wheel_workflow_selects_amd64_and_passes_args(self) -> None:
        workflow = (REPO_ROOT / ".github/workflows/build-on-demand.yml").read_text()
        self.assertIn(
            "inputs.vllm_precompiled_wheel_commit != '' && "
            "'linux/amd64' || 'linux/amd64,linux/arm64'",
            workflow,
        )
        self.assertIn(
            "VLLM_PRECOMPILED_WHEEL_COMMIT=${{ inputs.vllm_precompiled_wheel_commit }}",
            workflow,
        )
        self.assertIn(
            "VLLM_PRECOMPILED_WHEEL_VARIANT="
            "${{ inputs.vllm_precompiled_wheel_variant }}",
            workflow,
        )

        dockerfile = self.render_runtime("amd64")
        self.assertIn(
            "FROM ${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG} AS pre_runtime",
            dockerfile,
        )
        self.assertNotIn("AS vllm_runtime_arm64", dockerfile)

    def test_flashinfer_override_follows_exact_vllm_install(self) -> None:
        installer = self.installer.read_text()
        vllm_install = installer.index(
            "uv pip install --system '.[flashinfer,runai,otel]'"
        )
        flashinfer_install = installer.index(
            "uv pip install --system --force-reinstall --no-deps ."
        )
        self.assertLess(vllm_install, flashinfer_install)
        self.assertIn("git merge-base --is-ancestor", installer)
        self.assertIn("Exact-native mode only permits Python changes", installer)
        self.assertIn("source-provenance.txt", installer)

    def test_exact_native_verification_requires_stable_libtorch_extensions(
        self,
    ) -> None:
        result = self.verify_fixture(
            (
                "_C_stable_libtorch.abi3.so",
                "_moe_C_stable_libtorch.abi3.so",
            ),
            exact_native=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        result = self.verify_fixture(
            ("_C.abi3.so",),
            exact_native=True,
            unowned_extension_names=(
                "_C_stable_libtorch.abi3.so",
                "_moe_C_stable_libtorch.abi3.so",
            ),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("vllm/_C_stable_libtorch.abi3.so", result.stderr)

        result = self.verify_fixture(("_C.abi3.so",), exact_native=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_native_rejects_rename_from_outside_vllm_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q", repository], check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "commit.gpgSign", "false"],
                cwd=repository,
                check=True,
            )
            (repository / "csrc").mkdir()
            (repository / "csrc/kernel.cpp").write_text("// kernel\n")
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            subprocess.run(["git", "commit", "-qm", "base"], cwd=repository, check=True)
            base = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()

            (repository / "vllm").mkdir()
            subprocess.run(
                ["git", "mv", "csrc/kernel.cpp", "vllm/kernel.py"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-qm", "rename"], cwd=repository, check=True
            )
            head = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repository, text=True
            ).strip()

            result = self.run_installer_function(
                "validate_python_only_changes",
                base,
                head,
                cwd=repository,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("csrc/kernel.cpp", result.stderr)
        self.assertIn(
            "Exact-native mode only permits Python changes under vllm/",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
