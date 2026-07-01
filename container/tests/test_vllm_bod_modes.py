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

    def initialize_git_repository(
        self,
        repository: Path,
        files: dict[str, str] | None = None,
    ) -> str:
        subprocess.run(["git", "init", "-q", repository], check=True)
        for key, value in (
            ("user.email", "test@example.com"),
            ("user.name", "Test User"),
            ("commit.gpgSign", "false"),
        ):
            subprocess.run(
                ["git", "config", key, value],
                cwd=repository,
                check=True,
            )
        for relative_path, contents in (files or {}).items():
            path = repository / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "base", "--allow-empty"],
            cwd=repository,
            check=True,
        )
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()

    def commit_git_repository(self, repository: Path, message: str) -> str:
        subprocess.run(["git", "add", "."], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-qm", message],
            cwd=repository,
            check=True,
        )
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()

    def verify_fixture(
        self,
        extension_names: tuple[str, ...],
        exact_native: bool,
        full_source: bool = False,
        unowned_extension_names: tuple[str, ...] = (),
        vllm_version: str | None = None,
        torch_version: str = "2.11.0+cu130",
        checkpoint_hooks: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package_dir = root / "vllm"
            package_dir.mkdir()
            (package_dir / "__init__.py").touch()
            dist_info = root / "vllm-1.0.dist-info"
            dist_info.mkdir()
            if vllm_version is None:
                vllm_version = "1.0+precompiled" if exact_native else "1.0"
            (dist_info / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: vllm\nVersion: {vllm_version}\n"
            )
            records = ["vllm/__init__.py,,"]
            for extension_name in extension_names:
                extension = package_dir / extension_name
                extension.parent.mkdir(parents=True, exist_ok=True)
                extension.touch()
                records.append(f"vllm/{extension_name},,")
            for extension_name in unowned_extension_names:
                (package_dir / extension_name).touch()
            if checkpoint_hooks:
                worker_path = package_dir / "v1/worker/gpu_worker.py"
                worker_path.parent.mkdir(parents=True)
                worker_path.write_text("class GPUWorker:\n    pass\n")
                lifecycle_path = package_dir / "distributed/parallel_state.py"
                lifecycle_path.parent.mkdir(parents=True)
                lifecycle_path.write_text(
                    "def checkpoint_prepare_distributed_state():\n"
                    "    pass\n\n"
                    "def checkpoint_restore_distributed_state():\n"
                    "    pass\n"
                )
                records.append("vllm/v1/worker/gpu_worker.py,,")
                records.append("vllm/distributed/parallel_state.py,,")
            records.extend(
                [
                    "vllm-1.0.dist-info/METADATA,,",
                    "vllm-1.0.dist-info/RECORD,,",
                ]
            )
            (dist_info / "RECORD").write_text("\n".join(records))

            if exact_native or full_source:
                torch_dist_info = root / "torch-test.dist-info"
                torch_dist_info.mkdir()
                (torch_dist_info / "METADATA").write_text(
                    "Metadata-Version: 2.1\n"
                    "Name: torch\n"
                    f"Version: {torch_version}\n"
                )
                (torch_dist_info / "RECORD").write_text(
                    "torch-test.dist-info/METADATA,,\n"
                    "torch-test.dist-info/RECORD,,\n"
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
                "checkpoint-hooks" if checkpoint_hooks else "",
                "exact-native" if exact_native else "",
                "cu130" if exact_native or full_source else "",
                "cu130" if exact_native or full_source else "",
                "full-source" if full_source else "",
                torch_version if full_source else "",
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
            "FROM --platform=linux/amd64 ${VLLM_RUNTIME_BASE_IMAGE} "
            "AS vllm_runtime_amd64",
            dockerfile,
        )
        self.assertIn(
            "FROM --platform=linux/arm64 ${VLLM_RUNTIME_BASE_IMAGE} "
            "AS vllm_runtime_arm64",
            dockerfile,
        )
        self.assertIn(
            "ARG VLLM_RUNTIME_BASE_IMAGE=${RUNTIME_IMAGE}:${RUNTIME_IMAGE_TAG}",
            dockerfile,
        )
        self.assertIn("ARG VLLM_PRECOMPILED_WHEEL_COMMIT=", dockerfile)
        self.assertIn("ARG VLLM_PRECOMPILED_WHEEL_VARIANT=cu130", dockerfile)

    def test_exact_wheel_workflow_selects_amd64_and_passes_args(self) -> None:
        workflow = (REPO_ROOT / ".github/workflows/build-on-demand.yml").read_text()
        self.assertIn(
            "inputs.vllm_precompiled_wheel_commit != '') && "
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
            "FROM ${VLLM_RUNTIME_BASE_IMAGE} AS pre_runtime",
            dockerfile,
        )
        self.assertNotIn("AS vllm_runtime_arm64", dockerfile)

    def test_flashinfer_override_follows_exact_vllm_install(self) -> None:
        installer = self.installer.read_text()
        vllm_install = installer.index(
            "uv pip install --system '.[flashinfer,runai,otel]'"
        )
        flashinfer_requirements = installer.index(
            "uv pip install --system -r requirements.txt"
        )
        flashinfer_install = installer.index(
            "BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0",
            flashinfer_requirements,
        )
        pip_check = installer.index("uv pip check --system")
        self.assertLess(vllm_install, flashinfer_requirements)
        self.assertLess(flashinfer_requirements, flashinfer_install)
        self.assertLess(flashinfer_install, pip_check)
        self.assertIn("git merge-base --is-ancestor", installer)
        self.assertIn("requirements/cuda.txt", installer)
        self.assertIn("source-provenance.txt", installer)
        self.assertIn("VLLM_TARGET_DEVICE=cuda", installer)
        self.assertIn("VLLM_TORCH_BACKEND", installer)
        self.assertIn("VLLM_EXPECTED_TORCH_LOCAL_VERSION", installer)
        self.assertIn('--torch-backend="${VLLM_TORCH_BACKEND}"', installer)
        self.assertNotIn("--torch-backend=auto", installer)

        dockerfile = self.render_runtime("amd64")
        self.assertLess(
            dockerfile.index("install_custom_vllm_sources"),
            dockerfile.index("build_nccl_checkpoint"),
        )

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

        installer = self.installer.read_text()
        self.assertIn(
            "Do not import native modules during image builds because libcuda",
            installer,
        )
        self.assertIn(
            "Stable-libtorch modules still require libcuda at GPU runtime",
            installer,
        )

    def test_exact_native_verification_requires_cuda_distribution_metadata(
        self,
    ) -> None:
        extensions = (
            "_C_stable_libtorch.abi3.so",
            "_moe_C_stable_libtorch.abi3.so",
        )
        result = self.verify_fixture(
            extensions,
            exact_native=True,
            torch_version="2.11.0+cpu",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "variant cu130 requires torch local version +cu130",
            result.stderr,
        )
        self.assertIn("found torch 2.11.0+cpu", result.stderr)

        for vllm_version in ("1.0+cpu", "1.0"):
            with self.subTest(vllm_version=vllm_version):
                result = self.verify_fixture(
                    extensions,
                    exact_native=True,
                    vllm_version=vllm_version,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "requires a CUDA precompiled vLLM distribution",
                    result.stderr,
                )
                self.assertIn(f"found vllm {vllm_version}", result.stderr)

    def test_exact_native_verifies_generic_lifecycle_without_worker_override(
        self,
    ) -> None:
        result = self.verify_fixture(
            (
                "_C_stable_libtorch.abi3.so",
                "_moe_C_stable_libtorch.abi3.so",
            ),
            exact_native=True,
            checkpoint_hooks=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("no GPUWorker checkpoint override", result.stdout)

    def test_exact_native_rejects_unsupported_torch_backend_variants(self) -> None:
        for variant in ("cpu", "cu129", "rocm6.3", "cu130;true"):
            with self.subTest(variant=variant):
                result = self.run_installer_function(
                    "validate_exact_native_mode",
                    env={
                        "VLLM_PRECOMPILED_WHEEL_COMMIT": "a" * 40,
                        "VLLM_PRECOMPILED_WHEEL_VARIANT": variant,
                    },
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"Unsupported exact-native vLLM wheel variant: {variant}",
                    result.stderr,
                )
                self.assertIn("Supported CUDA variants: cu130", result.stderr)

    def test_exact_native_accepts_python_tests_and_cuda_requirements_changes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            base = self.initialize_git_repository(repository)
            for relative_path in (
                "vllm/checkpoint.py",
                "tests/distributed/test_pynccl.py",
                "requirements/cuda.txt",
            ):
                path = repository / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# custom source\n")
            head = self.commit_git_repository(repository, "allowed changes")

            result = self.run_installer_function(
                "validate_exact_native_changes",
                base,
                head,
                cwd=repository,
            )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exact_native_rejects_other_metadata_build_and_native_paths(
        self,
    ) -> None:
        rejected_paths = (
            "requirements/common.txt",
            "setup.py",
            "pyproject.toml",
            "csrc/kernel.cpp",
            "tests/distributed/pynccl_fixture.json",
        )
        for rejected_path in rejected_paths:
            with self.subTest(path=rejected_path):
                with tempfile.TemporaryDirectory() as directory:
                    repository = Path(directory)
                    base = self.initialize_git_repository(repository)
                    path = repository / rejected_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# disallowed\n")
                    head = self.commit_git_repository(repository, "disallowed change")

                    result = self.run_installer_function(
                        "validate_exact_native_changes",
                        base,
                        head,
                        cwd=repository,
                    )

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(rejected_path, result.stderr)

    def test_exact_native_rejects_rename_from_outside_vllm_python(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            base = self.initialize_git_repository(
                repository,
                {"csrc/kernel.cpp": "// kernel\n"},
            )
            (repository / "vllm").mkdir()
            subprocess.run(
                ["git", "mv", "csrc/kernel.cpp", "vllm/kernel.py"],
                cwd=repository,
                check=True,
            )
            head = self.commit_git_repository(repository, "rename")

            result = self.run_installer_function(
                "validate_exact_native_changes",
                base,
                head,
                cwd=repository,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("csrc/kernel.cpp", result.stderr)
        self.assertIn(
            "Exact-native mode only permits Python changes under vllm/ or tests/, "
            "or requirements/cuda.txt",
            result.stderr,
        )

    def test_nccl_checkpoint_build_runs_pynccl_binding_smoke_check(self) -> None:
        build_script = (
            CONTAINER_DIR / "deps/vllm/build_nccl_checkpoint.sh"
        ).read_text()
        smoke_check = (
            CONTAINER_DIR / "deps/vllm/validate_pynccl_checkpoint_binding.py"
        ).read_text()
        dockerfile = self.render_runtime("amd64")

        install = build_script.index("uv pip install --system --no-deps ./python")
        preload = build_script.index('NCCL_CHECKPOINT_SHIM="${SHIM}"')
        smoke = build_script.index('python3 "${PYNCCL_SMOKE_CHECK}"')
        self.assertLess(install, preload)
        self.assertLess(preload, smoke)
        self.assertIn('LD_PRELOAD="${SHIM}${LD_PRELOAD:+:${LD_PRELOAD}}"', build_script)
        self.assertIn("library = NCCLLibrary()", smoke_check)
        self.assertIn('library._funcs["ncclAllReduce"]', smoke_check)
        self.assertIn('library._funcs["ncclGetVersion"]', smoke_check)
        self.assertIn("library.real_lib.ncclGetVersion", smoke_check)
        self.assertIn("library.ncclGetVersion()", smoke_check)
        self.assertIn('require_distribution("nvidia-nccl-cu13"', smoke_check)
        self.assertIn('require_distribution("flashinfer-python"', smoke_check)
        self.assertIn('require_distribution("flashinfer-cubin"', smoke_check)
        self.assertIn("checkpoint_version.checkpoint_version", smoke_check)
        self.assertIn("checkpoint_version.nccl_version", smoke_check)
        self.assertIn("library.ncclGetRawVersion()", smoke_check)
        self.assertIn("torch.cuda.nccl.version()", smoke_check)
        self.assertIn("EXPECTED_SHIM_PATH", smoke_check)
        self.assertIn("flashinfer-source-version.txt", smoke_check)

        helper_copy = dockerfile.index("validate_pynccl_checkpoint_binding.py")
        build = dockerfile.index("build_nccl_checkpoint", helper_copy)
        checkpoint_env = dockerfile.index("ENV NCCL_CHECKPOINT_SHIM=", build)
        self.assertLess(helper_copy, build)
        self.assertLess(build, checkpoint_env)
        final_validation = dockerfile.index(
            "python3 /usr/local/lib/validate_pynccl_checkpoint_binding.py",
            checkpoint_env,
        )
        cache_cleanup = dockerfile.index(
            "rm -rf /home/dynamo/.cache/flashinfer", final_validation
        )
        self.assertLess(final_validation, cache_cleanup)

    def test_nccl_checkpoint_source_is_public_immutable_and_verified(self) -> None:
        build_script = (
            CONTAINER_DIR / "deps/vllm/build_nccl_checkpoint.sh"
        ).read_text()
        workflow = (REPO_ROOT / ".github/workflows/build-on-demand.yml").read_text()
        dockerfile = self.render_runtime("amd64")

        self.assertIn("https://github.com/NVIDIA/nccl.git", build_script)
        self.assertIn(
            'NCCL_CORE_SHA="b81d6a5a3d2fa95ad11f6453c51cd6a6ba19f9b8"',
            build_script,
        )
        self.assertIn(
            'NCCL_SHIM_SOURCE_SHA="' 'a2e67d265b3c52172f20452a909f7eaa6a0f1328"',
            build_script,
        )
        self.assertIn(
            'NCCL_CHECKPOINT_TREE="' 'f43d560f98e687b0f175350b6ea51f054cbcb654"',
            build_script,
        )
        self.assertNotIn("NCCL_CHECKPOINT_GIT_URL", workflow)
        self.assertNotIn("NCCL_CHECKPOINT_GIT_URL", build_script)
        self.assertNotIn("gitlab-master.nvidia.com", build_script)
        self.assertNotIn("dynamoci.azurecr.io/ai-dynamo/nccl-source", workflow)
        self.assertNotIn("NCCL_CHECKPOINT_SOURCE_IMAGE", dockerfile)
        self.assertIn(
            'fetch --no-tags --depth=1 origin "${NCCL_CORE_SHA}"', build_script
        )
        self.assertIn(
            '"${resolved_shim_sha}:contrib/nccl_checkpoint"',
            build_script,
        )
        self.assertIn('git -C "${SRC_DIR}" write-tree', build_script)

    def test_cuda_build_only_package_purge_does_not_autoremove(self) -> None:
        dockerfile = self.render_runtime("amd64")

        self.assertIn("cuda-libraries-dev-13-0", dockerfile)
        self.assertIn("cuda-sandbox-dev-13-0", dockerfile)
        self.assertIn("libnvfatbin-dev-13-0", dockerfile)
        commands = "\n".join(
            line
            for line in dockerfile.splitlines()
            if not line.lstrip().startswith("#")
        )
        self.assertNotIn("apt-get autoremove", commands)

    def test_full_source_mode_is_native_pinned_and_digest_based(self) -> None:
        installer = self.installer.read_text()
        workflow = (REPO_ROOT / ".github/workflows/build-on-demand.yml").read_text()
        dockerfile = self.render_runtime("amd64")

        self.assertIn("VLLM_INSTALL_MODE=${{ inputs.vllm_install_mode }}", workflow)
        self.assertIn(
            "VLLM_RUNTIME_BASE_IMAGE=${{ inputs.vllm_runtime_base_image }}",
            workflow,
        )
        self.assertIn(
            "VLLM_NCCL_VERSION=${{ inputs.vllm_nccl_version }}",
            workflow,
        )
        self.assertIn("validate_full_source_mode", installer)
        self.assertIn("VLLM_RUNTIME_BASE_IMAGE digest", installer)
        self.assertIn("VLLM_USE_PRECOMPILED=0", installer)
        self.assertIn("VLLM_USE_PRECOMPILED_RUST=0", installer)
        self.assertIn("--no-build-isolation --no-deps .", installer)
        self.assertIn("torchaudio==0", installer)
        self.assertIn("nvidia-nccl-cu13==${VLLM_NCCL_VERSION}", installer)
        checkout = installer.index('vllm_source_sha="${RESOLVED_SOURCE_SHA}"')
        full_source_dispatch = installer.index(
            'if [[ "${VLLM_INSTALL_MODE}" == "full-source" ]]',
            checkout,
        )
        build = installer.index("build_full_source_vllm", full_source_dispatch)
        exact_native = installer.index(
            'elif [[ -n "${VLLM_PRECOMPILED_WHEEL_COMMIT}" ]]',
            build,
        )
        self.assertLess(checkout, full_source_dispatch)
        self.assertLess(full_source_dispatch, build)
        self.assertLess(build, exact_native)
        self.assertIn("VLLM_NCCL_SO_PATH=/opt/dynamo/nccl/libnccl.so.2", dockerfile)
        self.assertIn(
            'LABEL ai.dynamo.vllm.base="${VLLM_RUNTIME_BASE_IMAGE}"', dockerfile
        )

    def test_full_source_registry_solve_excludes_source_flashinfer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "cuda.txt"
            destination = root / "runtime.txt"
            source.write_text(
                "torch==2.12.0\n"
                "flashinfer-python==0.6.14\n"
                "flashinfer-cubin==0.6.14\n"
                "apache-tvm-ffi==0.1.9\n"
            )

            result = self.run_installer_function(
                "write_full_source_runtime_requirements",
                str(source),
                str(destination),
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                destination.read_text(),
                "torch==2.12.0\napache-tvm-ffi==0.1.9\n",
            )

        installer = self.installer.read_text()
        self.assertIn(
            '-r "${VLLM_RUNTIME_REQUIREMENTS_FILE}"',
            installer,
        )
        self.assertIn("BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0", installer)

    def test_full_source_nccl_runtime_does_not_require_checkpoint_shim(self) -> None:
        installer = self.installer.read_text()
        dockerfile = self.render_runtime("amd64")

        self.assertIn(
            'VLLM_NCCL_VERSION="${VLLM_NCCL_VERSION:-' '${NCCL_CHECKPOINT_VERSION:-}}"',
            installer,
        )
        self.assertIn("VLLM_NCCL_VERSION=2.29.7", installer)
        self.assertIn(
            'if [ -f "${NCCL_CHECKPOINT_SHIM}" ]; then',
            dockerfile,
        )

    def test_full_source_verification_requires_current_main_extensions(self) -> None:
        extensions = (
            "_C_stable_libtorch.abi3.so",
            "_moe_C_stable_libtorch.abi3.so",
            "cumem_allocator.abi3.so",
            "vllm_flash_attn/_vllm_fa2_C.abi3.so",
        )
        result = self.verify_fixture(
            extensions,
            exact_native=False,
            full_source=True,
            torch_version="2.12.0+cu130",
            checkpoint_hooks=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

        result = self.verify_fixture(
            extensions[:-1],
            exact_native=False,
            full_source=True,
            torch_version="2.12.0+cu130",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so", result.stderr)


if __name__ == "__main__":
    unittest.main()
