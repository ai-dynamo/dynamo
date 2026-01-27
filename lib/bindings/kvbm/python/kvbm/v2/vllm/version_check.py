# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

# API version constraints
VLLM_MIN_VERSION = (0, 11, 1)
VLLM_MAX_VERSION_TESTED = (0, 14, 0)


# The version of vLLM must be great minimum version and less than or equal to maximum version tests
# The max version is a soft limit and can be bypassed with KVBM_DISABLE_MAX_VERSION_CHECK
def version_check():
    from vllm.version import __version_tuple__

    if __version_tuple__ < VLLM_MIN_VERSION:
        raise ImportError(
            f"vLLM versions at or before {'.'.join(map(str, VLLM_MIN_VERSION))} are not supported"
        )

    # Check if max version check is disabled
    disable_max_version_check = os.environ.get(
        "KVBM_DISABLE_MAX_VERSION_CHECK", ""
    ).lower() in ("1", "true", "yes")

    if not disable_max_version_check and __version_tuple__ > VLLM_MAX_VERSION_TESTED:
        raise ImportError(
            f"vLLM versions after {'.'.join(map(str, VLLM_MAX_VERSION_TESTED))} are not yet validated. "
            f"Set KVBM_DISABLE_MAX_VERSION_CHECK=1 to bypass this check."
        )
