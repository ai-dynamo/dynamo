# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from case_runner import GoldenCase, discover_cases, run_case

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.timeout(240),
]

_CASES = discover_cases()


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.case_id)
def test_cli_golden_case(case: GoldenCase, tmp_path: Path) -> None:
    run_case(case, tmp_path)
