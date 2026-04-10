# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.mocker.args import resolve_planner_profile_data

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.unit,
]


def test_resolve_planner_profile_data_none_does_not_require_planner_deps():
    result = resolve_planner_profile_data(None)

    assert result.npz_path is None
