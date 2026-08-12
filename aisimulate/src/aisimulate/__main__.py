# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the public AISimulate CLI with ``python -m aisimulate``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
