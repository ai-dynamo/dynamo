# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for profiler-side AIConfigurator compatibility helpers."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dynamo.profiler.utils.aic_compat import (  # noqa: E402
    normalize_aic_model_config,
    patched_aic_model_config,
)


class TestNormalizeAICModelConfig(unittest.TestCase):
    def test_blackwell_mxfp4_alias_normalizes_to_nvfp4(self) -> None:
        config = {
            "moe_mode": "w4a16_mxfp4",
            "moe_quant_mode": "w4a16_mxfp4",
            "max_position_embeddings": 131072,
        }

        normalized = normalize_aic_model_config(config, system_name="b200_sxm")

        self.assertEqual(normalized["moe_mode"], "nvfp4")
        self.assertEqual(normalized["moe_quant_mode"], "nvfp4")
        self.assertEqual(normalized["max_position_embeddings"], 131072)
        self.assertEqual(config["moe_mode"], "w4a16_mxfp4")

    def test_non_blackwell_system_is_unchanged(self) -> None:
        config = {"moe_mode": "w4a16_mxfp4"}

        normalized = normalize_aic_model_config(config, system_name="h200_sxm")

        self.assertIs(normalized, config)

    def test_unrecognized_values_are_preserved(self) -> None:
        config = {"moe_mode": "fp8"}

        normalized = normalize_aic_model_config(config, system_name="gb200_sxm")

        self.assertEqual(normalized["moe_mode"], "fp8")


class TestPatchedAICModelConfig(unittest.TestCase):
    MODULES = (
        "aiconfigurator",
        "aiconfigurator.sdk",
        "aiconfigurator.sdk.utils",
        "aiconfigurator.sdk.task",
        "aiconfigurator.generator",
        "aiconfigurator.generator.enumerate",
        "aiconfigurator.cli",
        "aiconfigurator.cli.main",
    )

    def setUp(self) -> None:
        self._saved = {name: sys.modules.get(name) for name in self.MODULES}

    def tearDown(self) -> None:
        for name in self.MODULES:
            original = self._saved[name]
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def _install_fake_aic(self):
        root = types.ModuleType("aiconfigurator")
        sdk = types.ModuleType("aiconfigurator.sdk")
        utils = types.ModuleType("aiconfigurator.sdk.utils")
        task = types.ModuleType("aiconfigurator.sdk.task")
        generator = types.ModuleType("aiconfigurator.generator")
        enumerate_mod = types.ModuleType("aiconfigurator.generator.enumerate")
        cli = types.ModuleType("aiconfigurator.cli")
        cli_main = types.ModuleType("aiconfigurator.cli.main")

        def loader(*_args, **_kwargs):
            return {"moe_mode": "w4a16_mxfp4"}

        utils.get_model_config_from_model_path = loader
        task.get_model_config_from_model_path = loader
        enumerate_mod.get_model_config_from_model_path = loader
        cli_main.get_model_config_from_model_path = loader

        root.sdk = sdk
        root.generator = generator
        root.cli = cli
        sdk.utils = utils
        sdk.task = task
        generator.enumerate = enumerate_mod
        cli.main = cli_main

        sys.modules.update(
            {
                "aiconfigurator": root,
                "aiconfigurator.sdk": sdk,
                "aiconfigurator.sdk.utils": utils,
                "aiconfigurator.sdk.task": task,
                "aiconfigurator.generator": generator,
                "aiconfigurator.generator.enumerate": enumerate_mod,
                "aiconfigurator.cli": cli,
                "aiconfigurator.cli.main": cli_main,
            }
        )
        return utils, task, enumerate_mod, cli_main, loader

    def test_context_normalizes_gpt_oss_blackwell_mxfp4_and_restores_loaders(self) -> None:
        utils, task, enumerate_mod, cli_main, original = self._install_fake_aic()

        with patched_aic_model_config("b200_sxm"):
            for module in (utils, task, enumerate_mod, cli_main):
                self.assertEqual(
                    module.get_model_config_from_model_path("openai/gpt-oss-120b")["moe_mode"],
                    "nvfp4",
                )

        for module in (utils, task, enumerate_mod, cli_main):
            self.assertIs(module.get_model_config_from_model_path, original)

    def test_non_blackwell_context_leaves_loaders_untouched(self) -> None:
        utils, task, enumerate_mod, cli_main, original = self._install_fake_aic()

        with patched_aic_model_config("h200_sxm"):
            for module in (utils, task, enumerate_mod, cli_main):
                self.assertIs(module.get_model_config_from_model_path, original)


class TestProfilerUsesAICCompat(unittest.TestCase):
    def test_rapid_wraps_aic_calls(self) -> None:
        source = Path("components/src/dynamo/profiler/rapid.py").read_text()
        self.assertIn("with patched_aic_model_config(system):", source)

    def test_thorough_wraps_aic_calls(self) -> None:
        source = Path("components/src/dynamo/profiler/thorough.py").read_text()
        self.assertIn("with patched_aic_model_config(system):", source)

    def test_profile_sla_wraps_support_checks(self) -> None:
        source = Path("components/src/dynamo/profiler/profile_sla.py").read_text()
        self.assertIn("with patched_aic_model_config(system):", source)


if __name__ == "__main__":
    unittest.main()
