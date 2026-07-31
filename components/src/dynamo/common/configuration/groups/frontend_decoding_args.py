# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared `--frontend-decoding` flag helper.

Both `dynamo.vllm` and `dynamo.sglang` (and any future backend) expose the
same `--frontend-decoding` knob — when set, the Rust frontend decodes
multimodal images and ships pre-decoded pixels via NIXL RDMA instead of
letting the engine fetch + decode. The flag definition is identical
modulo the engine-specific env-var prefix (`DYN_VLLM_*` vs `DYN_SGL_*`),
so the per-backend `backend_args.py` modules call this helper instead of
duplicating the `add_negatable_bool_argument` block.

Engine-specific validation (e.g. sglang rejects FD combined with the EPD
topology) stays in each backend's `ConfigBase.validate()` — only the
flag plumbing is shared.
"""

from dynamo.common.configuration.utils import add_negatable_bool_argument


def add_frontend_decoding_arg(g, *, env_prefix: str) -> None:
    """Add the shared `--frontend-decoding` flag to an argparse group.

    Args:
        g: an argparse argument group (from `parser.add_argument_group(...)`).
        env_prefix: engine identifier inserted into the env var, e.g. `"VLLM"`
            yields `DYN_VLLM_FRONTEND_DECODING`, `"SGL"` yields
            `DYN_SGL_FRONTEND_DECODING`.
    """
    add_negatable_bool_argument(
        g,
        flag_name="--frontend-decoding",
        env_var=f"DYN_{env_prefix}_FRONTEND_DECODING",
        default=False,
        help=(
            "Enable frontend decoding of multimodal images. "
            "Images are decoded in the Rust frontend and transferred to the "
            "backend via NIXL RDMA, bypassing in-engine HTTP fetch + decode. "
            "Engine-specific topology constraints may apply — see backend "
            "validation."
        ),
    )
