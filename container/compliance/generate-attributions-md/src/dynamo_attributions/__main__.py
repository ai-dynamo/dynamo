# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allow running as python3 -m dynamo_attributions."""

import sys

from .cli import main

sys.exit(main())
