# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast compatibility check between vLLM and vLLM-Omni.

vLLM-Omni imports vLLM internals that change across releases, so a
major/minor version mismatch crashes with an opaque ImportError. We check
up front and raise an actionable error instead.
"""

import importlib.metadata
import logging

logger = logging.getLogger(__name__)


class OmniVersionMismatchError(RuntimeError):
    """Raised when vLLM and vLLM-Omni major/minor versions are incompatible."""


def _major_minor(version: str) -> tuple[int, int] | None:
    """Return the ``(major, minor)`` tuple, or None if unparseable."""
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def check_vllm_omni_compatibility() -> None:
    """Raise if installed vLLM and vLLM-Omni major/minor versions differ."""
    try:
        vllm_version = importlib.metadata.version("vllm")
        vllm_omni_version = importlib.metadata.version("vllm-omni")
    except importlib.metadata.PackageNotFoundError as exc:
        # Missing metadata: let the downstream import raise its own error.
        logger.debug("Skipping vLLM/vLLM-Omni version check: %s", exc)
        return

    vllm_mm = _major_minor(vllm_version)
    omni_mm = _major_minor(vllm_omni_version)

    if vllm_mm is None or omni_mm is None:
        logger.debug(
            "Skipping vLLM/vLLM-Omni version check: unparseable versions "
            "(vllm=%s, vllm-omni=%s)",
            vllm_version,
            vllm_omni_version,
        )
        return

    if vllm_mm != omni_mm:
        raise OmniVersionMismatchError(
            f"vLLM {vllm_version} and vLLM-Omni {vllm_omni_version} have "
            f"mismatched versions; vLLM-Omni must be "
            f"{vllm_mm[0]}.{vllm_mm[1]}.x. Rebuild the runtime image with "
            "aligned versions (see container/context.yaml: vllm_omni_ref)."
        )
