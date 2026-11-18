# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source owners of measured cache counters in the qualified vLLM release."""

import hashlib
import logging
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path

logger = logging.getLogger(__name__)

# vLLM v0.33.0, commit 4ba787ec9959e566d3b587a796dff07673324da5.
CACHE_SOURCE_BLOBS = (
    ("vllm/config/scheduler.py", "5ad38c0a46f0446c9f4b43a7452eaa64483a33da"),
    ("vllm/outputs.py", "2149cf7df5e5447270d08e0f1d0561899647de04"),
    (
        "vllm/v1/core/sched/async_scheduler.py",
        "42906e2f19e51f301d8755218d20a71dcd7b6367",
    ),
    ("vllm/v1/core/sched/scheduler.py", "d35b7dfac42abd1335c827eff9f3c05728f3f248"),
    ("vllm/v1/engine/output_processor.py", "a1f99ad9932f5d884ccdbe558bd017dbe846df69"),
    ("vllm/v1/metrics/stats.py", "d30e857fa21b4833971d214c230d50782e571f30"),
    ("vllm/v1/request.py", "4b1b9ecc2eede6347d839ef52683f3d40a306af5"),
)


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_installed_sources(
    blobs: Iterable[tuple[str, str]], *, root: Path | None = None
) -> bool:
    """Inspect installed source without loading scheduler or inference modules."""
    try:
        if root is None:
            filename = import_module("vllm").__file__
            if filename is None:
                return False
            root = Path(filename).resolve().parent.parent
        for relative, expected in blobs:
            if git_blob_id((root / relative).read_bytes()) != expected:
                logger.warning(
                    "vLLM decisions disabled: cache source differs from qualified release"
                )
                return False
        return True
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
        logger.warning("vLLM decisions disabled: cache source unavailable")
        return False


def verify_cache_sources() -> bool:
    return verify_installed_sources(CACHE_SOURCE_BLOBS)
