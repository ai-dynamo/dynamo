# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

import pytest

from benchmarks.multimodal.sweep.experiments.epd.run_experiment import build_cells

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    ("axis", "values"),
    [
        ("backend", ["vllm", "VLLM"]),
        ("image_count", ["1", "01"]),
        ("image_token_budget", ["64", "064"]),
        ("isl", ["1", "01"]),
        ("osl", ["1", "01"]),
        ("qps", ["0.5", ".5"]),
    ],
)
def test_build_cells_rejects_duplicates_after_normalization(axis, values):
    args = Namespace(
        backend=["vllm"],
        topology=["aggregate"],
        image_count=["1"],
        image_token_budget=["-1"],
        isl=["1"],
        osl=["1"],
        qps=["0.5"],
    )
    setattr(args, axis, values)

    with pytest.raises(ValueError, match="duplicate .* after normalization"):
        build_cells(args)
