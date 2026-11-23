# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from dynamo.aisimulate.tests.dgd.model_corpus import (
    CASES,
    assert_model_corpus_dgd_golden,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
    pytest.mark.timeout(240),
]


@pytest.mark.parametrize(
    "entry", CASES[1::2], ids=lambda entry: f"{entry.hardware}-{entry.case}"
)
def test_model_corpus_dgd_goldens_part2(entry, tmp_path: Path) -> None:
    assert_model_corpus_dgd_golden(entry, tmp_path)
