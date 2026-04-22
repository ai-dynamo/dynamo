# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single source of truth for the kvbm.v2 vllm version policy.

`version_check()` is invoked at `kvbm.v2.vllm` package import. Any
production caller (config.py, schedulers/, connectors/) that touches v2
goes through that init, so the policy is enforced exactly once per
process.
"""

import os

VLLM_MIN_VERSION = (0, 11, 1)
VLLM_MAX_VERSION_TESTED = (0, 19, 999)

_BYPASS_ENV_VAR = "KVBM_SKIP_VLLM_VERSION_CHECK"


def version_check() -> None:
    """Enforce the kvbm.v2 vllm version policy.

    Raises ImportError if the installed vllm is below `VLLM_MIN_VERSION`
    (hard floor — the v2 connector relies on APIs added in 0.11.1) or
    above `VLLM_MAX_VERSION_TESTED` (soft ceiling — bypassable via the
    `KVBM_SKIP_VLLM_VERSION_CHECK` env var when consciously running
    against an untested release).
    """
    from vllm.version import __version_tuple__

    if __version_tuple__ < VLLM_MIN_VERSION:
        raise ImportError(
            f"vLLM {'.'.join(map(str, __version_tuple__))} is below the "
            f"minimum supported version {'.'.join(map(str, VLLM_MIN_VERSION))}"
        )

    bypass = os.environ.get(_BYPASS_ENV_VAR, "").lower() in ("1", "true", "yes")
    if not bypass and __version_tuple__ > VLLM_MAX_VERSION_TESTED:
        raise ImportError(
            f"vLLM {'.'.join(map(str, __version_tuple__))} is above the "
            f"maximum tested version {'.'.join(map(str, VLLM_MAX_VERSION_TESTED))}. "
            f"Set {_BYPASS_ENV_VAR}=1 to bypass."
        )
