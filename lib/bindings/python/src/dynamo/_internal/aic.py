# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import for :mod:`dynamo._internal.ais`."""

import sys

from . import ais

sys.modules[__name__] = ais
