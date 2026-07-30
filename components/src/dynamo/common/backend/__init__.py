# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Back-compat shim: `dynamo.common.backend` moved to `dynamo.backend`.

Removed in the follow-up migration commit. Import from `dynamo.backend` instead;
submodules (`engine`, `worker`, `run`, `publisher`, ...) alias to their new homes.
"""
from dynamo.backend import *  # noqa: F401,F403
