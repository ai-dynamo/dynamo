#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util


def check_module_available(module_name: str) -> bool:
    """For tests / pre-commit"""
    try:
        if importlib.util.find_spec(module_name) is None:
            return False
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False
