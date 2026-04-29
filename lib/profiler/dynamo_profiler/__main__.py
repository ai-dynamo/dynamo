# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allow running as `python -m dynamo_profiler`."""

from .cli import main

raise SystemExit(main() or 0)
