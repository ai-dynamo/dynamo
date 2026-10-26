# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_BENCHMARKS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCHMARKS_DIR))
