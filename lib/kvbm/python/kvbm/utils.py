# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


def is_dyn_runtime_enabled() -> bool:
    """
    Return True if DYN_RUNTIME_ENABLED_KVBM is set to '1' or 'true' (case-insensitive).
    DYN_RUNTIME_ENABLED_KVBM indicates if KVBM should use the existing DistributedRuntime
    in the current environment.
    """
    val = os.environ.get("DYN_RUNTIME_ENABLED_KVBM", "").strip().lower()
    return val in {"1", "true"}
