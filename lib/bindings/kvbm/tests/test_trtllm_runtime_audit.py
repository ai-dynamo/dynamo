# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import types
import unittest


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
                library_dirs=[lib_dir],
            )

        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["installed_tensorrt_llm"]["has_disaggregation"])
        self.assertTrue(report["pinned_checkout"]["has_disaggregation"])
        self.assertEqual(report["libraries"]["available_majors"], [12])
        self.assertEqual(len(report["findings"]), 2)
        self.assertIn("_torch.disaggregation", report["findings"][0])
        self.assertIn("expects CUDA major 13", report["findings"][1])

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
            checkout_root = root / "trtllm-checkout" / "tensorrt_llm"
            (checkout_root / "_torch" / "disaggregation").mkdir(parents=True)

            report = module.build_runtime_report(
                distribution=None,
                pinned_checkout=checkout_root,
                library_dirs=[],
                probe_imports=True,
                python_executable="/fake/python",
                probe_runner=fake_runner,
            )

        self.assertEqual(len(report["import_probes"]), 2)
        self.assertEqual(report["import_probes"][0]["status"], "ok")
        self.assertEqual(report["import_probes"][1]["status"], "error")
        self.assertIn(
            "subprocess import of tensorrt_llm._torch.disaggregation.transceiver failed",
            report["findings"][0],
        )
        self.assertIn("Open MPI / PMIx listener startup failed", report["findings"][0])

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


if __name__ == "__main__":
    unittest.main()
