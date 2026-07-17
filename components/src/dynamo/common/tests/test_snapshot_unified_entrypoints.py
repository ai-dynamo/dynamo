# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm"])
def test_restore_standby_runs_before_vendor_import(backend: str) -> None:
    module_name = f"dynamo.{backend}.unified_main"
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None
    module = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    standby_call = next(
        index
        for index, node in enumerate(main.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "maybe_run_restore_standby_mode"
    )
    vendor_import = next(
        index
        for index, node in enumerate(main.body)
        if isinstance(node, ast.ImportFrom)
        and node.module == f"dynamo.{backend}.llm_engine"
    )

    assert standby_call < vendor_import
