# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/render_recipe_kustomize.py"

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def load_renderer_module():
    spec = importlib.util.spec_from_file_location(
        "render_recipe_kustomize", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_recipe_roots_includes_all_for_central_component_change():
    renderer = load_renderer_module()

    assert (
        renderer.select_recipe_roots(
            ["recipes/kustomize/components/disagg-workers/aws-efa/patch-dgd.yaml"]
        )
        == renderer.all_recipe_roots()
    )
