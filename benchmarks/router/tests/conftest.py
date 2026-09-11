# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

import pytest

_BENCHMARKS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCHMARKS_DIR))


@pytest.fixture(scope="session")
def benchmark_env(tmp_path_factory):
    return {
        **os.environ,
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path_factory.mktemp("router-matplotlib")),
        "PYTHONPATH": os.pathsep.join(
            (str(_BENCHMARKS_DIR), os.environ.get("PYTHONPATH", ""))
        ),
    }
