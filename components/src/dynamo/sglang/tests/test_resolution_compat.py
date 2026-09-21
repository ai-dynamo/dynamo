# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise launcher API selection without importing SGLang's CUDA dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture(params=["backend", "gms"])
def load_launcher(request, monkeypatch):
    paths = {
        "backend": "components/src/dynamo/sglang/_compat.py",
        "gms": "lib/gpu_memory_service/integrations/sglang/__init__.py",
    }
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.utils",
        "sglang.srt.arg_groups",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)

    parser = ModuleType("sglang.srt.utils.server_args_config_parser")
    parser.ConfigArgumentMerger = object
    monkeypatch.setitem(sys.modules, parser.__name__, parser)

    loader = ModuleType("gpu_memory_service.integrations.sglang.model_loader")
    loader.GMSModelLoader = object
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    utils = ModuleType("gpu_memory_service.integrations.common.utils")
    utils.get_gms_lock_mode = lambda config: None
    utils.get_gms_ro_connect_timeout_ms = lambda config: None
    monkeypatch.setitem(sys.modules, utils.__name__, utils)

    def load(api):
        monkeypatch.setitem(sys.modules, "sglang.srt.arg_groups.overrides", api)
        spec = importlib.util.spec_from_file_location(
            f"_test_resolution_{request.param}", REPO_ROOT / paths[request.param]
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if request.param == "backend":
            return lambda args: module.override_server_args(
                args, "test.launcher", enable_memory_saver=True
            )
        return module.setup_gms

    return load


@pytest.mark.parametrize("release", ["0.5.19", "0.5.20"])
def test_launcher_selects_guarded_declaration_api(load_launcher, release):
    api = ModuleType("sglang.srt.arg_groups.overrides")
    guarded = Mock()
    if release == "0.5.19":
        api.declare_late_resolution = guarded
        api.declare_resolution = Mock(
            side_effect=AssertionError("unguarded 0.5.19 API selected")
        )
    else:
        api.declare_resolution = guarded
    launch = load_launcher(api)
    args = SimpleNamespace(enable_memory_saver=False)

    launch(args)

    assert args.enable_memory_saver is False
    guarded.assert_called_once()
    assert guarded.call_args.args[0] is args
    assert guarded.call_args.kwargs == {"enable_memory_saver": True}

    # A published config must fail rather than silently updating the raw record.
    guarded.side_effect = ValueError("config already published")
    with pytest.raises(ValueError, match="config already published"):
        launch(args)
    assert args.enable_memory_saver is False


def test_launcher_preserves_legacy_xpu_assignment(load_launcher):
    api = ModuleType("sglang.srt.arg_groups.overrides")
    launch = load_launcher(api)
    args = SimpleNamespace(enable_memory_saver=False)

    launch(args)

    assert args.enable_memory_saver is True
