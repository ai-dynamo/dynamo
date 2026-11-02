# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the cache-statistics owners required by causal decision accounting."""

import hashlib
import logging
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

from .decision_runtime_policy import is_exact_vllm_release

logger = logging.getLogger(__name__)

CAUSAL_RELEASE_COMMITS = MappingProxyType(
    {
        "0.32.0": "5df63a7203cdb9328d824a27b682e77c7b9928be",
        "0.33.0": "4ba787ec9959e566d3b587a796dff07673324da5",
    }
)
CAUSAL_CACHE_SOURCE_BLOBS = MappingProxyType(
    {
        "0.32.0": (
            ("vllm/config/scheduler.py", "1dbe4b72eb7ce43deaf7cb1e9e93176b8219f028"),
            ("vllm/outputs.py", "145b7d510f715fdbe32e04b96f4be941ba002826"),
            (
                "vllm/v1/core/sched/async_scheduler.py",
                "290204d58c39cdec4c3cc1e58fe0697378aa5e98",
            ),
            (
                "vllm/v1/core/sched/scheduler.py",
                "cec96d2a49235fdade2768c427f1cff7868c6026",
            ),
            (
                "vllm/v1/engine/output_processor.py",
                "eb499f6cc78ba89b14fa862e4cb5801f0837323f",
            ),
            ("vllm/v1/metrics/stats.py", "87277f6545f01cb75c65426306111086f5feefaa"),
            ("vllm/v1/request.py", "4b1b9ecc2eede6347d839ef52683f3d40a306af5"),
        ),
        "0.33.0": (
            ("vllm/config/scheduler.py", "5ad38c0a46f0446c9f4b43a7452eaa64483a33da"),
            ("vllm/outputs.py", "2149cf7df5e5447270d08e0f1d0561899647de04"),
            (
                "vllm/v1/core/sched/async_scheduler.py",
                "42906e2f19e51f301d8755218d20a71dcd7b6367",
            ),
            (
                "vllm/v1/core/sched/scheduler.py",
                "d35b7dfac42abd1335c827eff9f3c05728f3f248",
            ),
            (
                "vllm/v1/engine/output_processor.py",
                "a1f99ad9932f5d884ccdbe558bd017dbe846df69",
            ),
            ("vllm/v1/metrics/stats.py", "d30e857fa21b4833971d214c230d50782e571f30"),
            ("vllm/v1/request.py", "4b1b9ecc2eede6347d839ef52683f3d40a306af5"),
        ),
    }
)


def git_blob_id(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def verify_installed_sources(
    blobs: Iterable[tuple[str, str]], *, root: Path | None = None
) -> bool:
    """Check installed source owners without importing optional implementation code."""
    try:
        if root is None:
            filename = import_module("vllm").__file__
            if filename is None:
                return False
            root = Path(filename).resolve().parent.parent
        for relative, expected in blobs:
            if git_blob_id((root / relative).read_bytes()) != expected:
                logger.warning("vLLM decisions disabled: pinned source mismatch")
                return False
        return True
    except (
        ImportError,
        OSError,
        RuntimeError,
        AttributeError,
        TypeError,
        ValueError,
    ):
        logger.warning("vLLM decisions disabled: pinned source unavailable")
        return False


def qualified_cache_runtime(installed: str) -> bool:
    """Fail closed for missing or modified cache-provenance owners, not generation."""
    return any(
        is_exact_vllm_release(installed, release) and verify_installed_sources(sources)
        for release, sources in CAUSAL_CACHE_SOURCE_BLOBS.items()
    )
