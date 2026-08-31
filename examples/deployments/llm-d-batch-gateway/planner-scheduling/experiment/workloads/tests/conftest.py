# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

WORKLOADS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKLOADS_DIR))
