# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest.py for planner tests to disable automatic test logging.
This overrides the autouse logger fixture from the parent conftest.py.
"""

import pytest


@pytest.fixture(autouse=True)
def logger(request):
    """Dummy logger fixture that does nothing - overrides the parent one."""
    yield
