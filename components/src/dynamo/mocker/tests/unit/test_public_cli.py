# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_public_mocker_module_cli_is_removed() -> None:
    assert importlib.util.find_spec("dynamo.mocker.__main__") is None
    assert importlib.util.find_spec("dynamo.mocker._worker") is not None
