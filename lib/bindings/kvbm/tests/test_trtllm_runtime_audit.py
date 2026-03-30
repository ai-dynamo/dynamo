# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import io
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "trtllm_runtime_audit.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("trtllm_runtime_audit", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeDistribution:
    def __init__(self, package_root: Path, *, version: str, requires: list[str]) -> None:
        self._package_root = package_root
        self.version = version
        self.requires = requires

    def locate_file(self, path: str) -> Path:
        if path != "tensorrt_llm":
            raise AssertionError(path)
        return self._package_root


class TrtllmRuntimeAuditTests(unittest.TestCase):
    def test_parse_expected_cuda_major_prefers_nccl_suffix(self) -> None:
        module = _load_module()

        major = module._parse_expected_cuda_major(
            [
                "cuda-python>=13",
                "nvidia-nccl-cu13<=2.28.9,>=2.27.7",
            ]
        )

        self.assertEqual(major, 13)

    def test_read_repo_declared_trtllm_version_from_pyproject(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject = Path(temp_dir) / "pyproject.toml"
            pyproject.write_text(
                "\n".join(
                    [
                        "[project.optional-dependencies]",
                        "trtllm =[",
                        '    "uvloop",',
                        '    "tensorrt-llm==1.3.0rc8",',
                        "]",
                    ]
                ),
                encoding="utf-8",
            )

            version = module._read_repo_declared_trtllm_version(pyproject)

        self.assertEqual(version, "1.3.0rc8")

    def test_read_repo_declared_trtllm_version_returns_none_for_invalid_toml(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            pyproject = Path(temp_dir) / "pyproject.toml"
            pyproject.write_text("[project.optional-dependencies\n", encoding="utf-8")

            version = module._read_repo_declared_trtllm_version(pyproject)

        self.assertIsNone(version)

    def test_build_runtime_report_flags_wheel_surface_and_cuda_mismatch(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            installed_root = root / "site-packages" / "tensorrt_llm"
            (installed_root / "_torch" / "pyexecutor").mkdir(parents=True)

            checkout_root = root / "trtllm-checkout" / "tensorrt_llm"
            (checkout_root / "_torch" / "disaggregation").mkdir(parents=True)
            (checkout_root / "_torch" / "pyexecutor").mkdir(parents=True)

            lib_dir = root / "libs"
            lib_dir.mkdir()
            (lib_dir / "libcublasLt.so.12").touch()
            (root / "pyproject.toml").write_text(
                "\n".join(
                    [
                        "[project.optional-dependencies]",
                        "trtllm =[",
                        '    "tensorrt-llm==1.3.0rc8",',
                        "]",
                    ]
                ),
                encoding="utf-8",
            )

            report = module.build_runtime_report(
                distribution=_FakeDistribution(
                    installed_root,
                    version="1.2.0",
                    requires=[
                        "cuda-python>=13",
                        "nvidia-nccl-cu13<=2.28.9,>=2.27.7",
                    ],
                ),
                pinned_checkout=checkout_root,
                repo_pyproject=root / "pyproject.toml",
                library_dirs=[lib_dir],
            )

        self.assertEqual(report["status"], "blocked")
        self.assertEqual(report["repo_declared_tensorrt_llm_version"], "1.3.0rc8")
        self.assertFalse(report["installed_tensorrt_llm"]["has_disaggregation"])
        self.assertTrue(report["pinned_checkout"]["has_disaggregation"])
        self.assertEqual(report["libraries"]["available_majors"], [12])
        self.assertEqual(len(report["findings"]), 3)
        self.assertIn(
            "does not match repo-declared trtllm extra version 1.3.0rc8",
            report["findings"][0],
        )
        self.assertIn("_torch.disaggregation", report["findings"][1])
        self.assertIn("expects CUDA major 13", report["findings"][2])

    def test_build_runtime_report_records_import_probe_failures(self) -> None:
        module = _load_module()

        def fake_runner(command, **kwargs):
            del kwargs
            rendered = " ".join(command)
            if "tensorrt_llm._torch.disaggregation.transceiver" in rendered:
                return types.SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="*** An error occurred in MPI_Init_thread\nPMIx server's listener thread failed to start",
                )
            return types.SimpleNamespace(
                returncode=0,
                stdout='{"module": "tensorrt_llm", "file": "/tmp/site-packages/tensorrt_llm/__init__.py"}\n',
                stderr="",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            installed_root = root / "site-packages" / "tensorrt_llm"
            installed_root.mkdir(parents=True)
            (root / "pyproject.toml").write_text(
                "\n".join(
                    [
                        "[project.optional-dependencies]",
                        "trtllm =[",
                        '    "tensorrt-llm==1.3.0rc8",',
                        "]",
                    ]
                ),
                encoding="utf-8",
            )
            checkout_root = root / "trtllm-checkout" / "tensorrt_llm"
            (checkout_root / "_torch" / "disaggregation").mkdir(parents=True)

            report = module.build_runtime_report(
                distribution=_FakeDistribution(
                    installed_root,
                    version="1.3.0rc8",
                    requires=[],
                ),
                pinned_checkout=checkout_root,
                repo_pyproject=root / "pyproject.toml",
                library_dirs=[],
                probe_imports=True,
                python_executable="/fake/python",
                probe_runner=fake_runner,
            )

        self.assertEqual(len(report["import_probes"]), 2)
        self.assertEqual(report["import_probes"][0]["status"], "ok")
        self.assertEqual(report["import_probes"][1]["status"], "error")
        self.assertIn(
            "installed tensorrt_llm package does not expose _torch.disaggregation",
            report["findings"][0],
        )
        self.assertIn(
            "subprocess import of tensorrt_llm._torch.disaggregation.transceiver failed",
            report["findings"][1],
        )
        self.assertIn("Open MPI / PMIx listener startup failed", report["findings"][1])

    def test_probe_python_import_command_exits_immediately_after_summary(self) -> None:
        module = _load_module()
        observed = {}

        def fake_runner(command, **kwargs):
            del kwargs
            observed["command"] = command
            return types.SimpleNamespace(
                returncode=0,
                stdout='{"module": "tensorrt_llm", "file": "/tmp/mod.py"}\n',
                stderr="",
            )

        probe = module._probe_python_import(
            python_executable="/fake/python",
            module="tensorrt_llm",
            runner=fake_runner,
        )

        self.assertEqual(probe["status"], "ok")
        self.assertIn("os._exit(0)", observed["command"][2])

    def test_build_probe_targets_prefers_installed_disaggregation_surface(self) -> None:
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            checkout_root = Path(temp_dir) / "trtllm-checkout" / "tensorrt_llm"
            (checkout_root / "_torch" / "disaggregation").mkdir(parents=True)

            targets = module._build_probe_targets(
                installed={
                    "installed": True,
                    "package_root": "/usr/local/lib/python3.12/dist-packages/tensorrt_llm",
                    "has_disaggregation": True,
                },
                pinned_checkout=checkout_root,
                probe_imports=True,
            )

        self.assertEqual(
            [(target["module"], target["python_path"]) for target in targets],
            [
                ("tensorrt_llm", None),
                ("tensorrt_llm._torch.disaggregation.transceiver", None),
                (
                    "tensorrt_llm._torch.disaggregation.transceiver",
                    checkout_root.parent,
                ),
            ],
        )

    def test_probe_python_import_reports_timeout(self) -> None:
        module = _load_module()

        def fake_runner(command, **kwargs):
            raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

        probe = module._probe_python_import(
            python_executable="/fake/python",
            module="tensorrt_llm",
            timeout_s=1.5,
            runner=fake_runner,
        )

        self.assertEqual(probe["status"], "timeout")
        self.assertIsNone(probe["returncode"])

    def test_probe_python_import_timeout_keeps_partial_stderr(self) -> None:
        module = _load_module()

        def fake_runner(command, **kwargs):
            raise subprocess.TimeoutExpired(
                command,
                timeout=kwargs["timeout"],
                stderr=b"PMIx server's listener thread failed to start\n",
            )

        probe = module._probe_python_import(
            python_executable="/fake/python",
            module="tensorrt_llm",
            timeout_s=1.5,
            runner=fake_runner,
        )

        self.assertEqual(probe["status"], "timeout")
        self.assertIn("PMIx server's listener thread failed to start", probe["stderr"])
        self.assertEqual(
            module._probe_failure_summary(probe),
            "Open MPI / PMIx listener startup failed during import",
        )

    def test_build_runtime_report_uses_requested_python_executable(self) -> None:
        module = _load_module()

        report = module.build_runtime_report(
            distribution=None,
            pinned_checkout=Path("/does/not/exist"),
            library_dirs=[],
            python_executable="/custom/python",
        )

        self.assertEqual(report["python_executable"], "/custom/python")

    def test_build_runtime_report_uses_requested_probe_timeout(self) -> None:
        module = _load_module()
        observed_timeouts = []

        def fake_runner(command, **kwargs):
            del command
            observed_timeouts.append(kwargs["timeout"])
            return types.SimpleNamespace(
                returncode=0,
                stdout='{"module": "ok", "file": "/tmp/mod.py"}\n',
                stderr="",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            installed_root = root / "site-packages" / "tensorrt_llm"
            installed_root.mkdir(parents=True)
            checkout_root = root / "trtllm-checkout" / "tensorrt_llm"
            (checkout_root / "_torch" / "disaggregation").mkdir(parents=True)

            report = module.build_runtime_report(
                distribution=_FakeDistribution(
                    installed_root,
                    version="1.3.0rc9",
                    requires=[],
                ),
                pinned_checkout=checkout_root,
                library_dirs=[],
                probe_imports=True,
                probe_timeout_s=7.5,
                probe_runner=fake_runner,
            )

        self.assertEqual(report["probe_timeout_s"], 7.5)
        self.assertEqual(observed_timeouts, [7.5, 7.5])

    def test_main_supports_cli_overrides_and_fail_on_blocked(self) -> None:
        module = _load_module()
        captured = {}

        def fake_build_runtime_report(**kwargs):
            captured.update(kwargs)
            return {"status": "blocked"}

        with mock.patch.object(module, "build_runtime_report", side_effect=fake_build_runtime_report):
            with mock.patch(
                "sys.argv",
                [
                    "trtllm_runtime_audit.py",
                    "--json",
                    "--probe-imports",
                    "--repo-pyproject",
                    "/custom/pyproject.toml",
                    "--python-executable",
                    "/custom/python",
                    "--probe-timeout-s",
                    "7.5",
                    "--fail-on-blocked",
                ],
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = module.main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(captured["repo_pyproject"], Path("/custom/pyproject.toml"))
        self.assertEqual(captured["python_executable"], "/custom/python")
        self.assertEqual(captured["probe_timeout_s"], 7.5)
        self.assertTrue(captured["probe_imports"])


if __name__ == "__main__":
    unittest.main()
