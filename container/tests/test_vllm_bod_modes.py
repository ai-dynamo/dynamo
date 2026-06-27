# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

CONTAINER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CONTAINER_DIR.parent
sys.path.insert(0, str(CONTAINER_DIR))

import render  # noqa: E402


class VllmBuildOnDemandModeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (CONTAINER_DIR / "context.yaml").open() as stream:
            cls.context = yaml.safe_load(stream)

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
        installer = (CONTAINER_DIR / "deps/vllm/install_custom_sources.sh").read_text()
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


if __name__ == "__main__":
    unittest.main()
