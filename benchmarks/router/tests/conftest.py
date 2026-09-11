# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

_ROUTER_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROUTER_DIR.parent), str(_ROUTER_DIR)]
