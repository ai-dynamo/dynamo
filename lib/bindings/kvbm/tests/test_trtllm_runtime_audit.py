# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
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


if __name__ == "__main__":
    unittest.main()
