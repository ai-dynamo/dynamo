# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TRT-LLM remote-code cache compatibility for GMS deployments.

TRT-LLM serializes Hugging Face remote-code config loading with a global
``_remote_code.lock`` under ``HF_MODULES_CACHE``. In Kubernetes recipes that
mount the HF cache from a RWX PVC, that path can be backed by a filesystem that
does not support POSIX file locks and returns ENOLCK. GMS only needs the model
weights to stay shared; the generated remote-code module cache can be local to
the pod.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterator

import filelock

logger = logging.getLogger(__name__)

_patched = False


def patch_remote_code_cache() -> None:
    """Move TRT-LLM/HF remote-code cache locking to pod-local storage."""
    global _patched
    if _patched:
        return

    cache_dir = Path(
        os.environ.get("GMS_TRTLLM_REMOTE_CODE_CACHE_DIR", "")
        or Path(tempfile.gettempdir()) / "trtllm-hf-modules-cache"
    )
    lock_dir = Path(os.environ.get("GMS_TRTLLM_CONFIG_LOCK_DIR", "") or cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_MODULES_CACHE"] = str(cache_dir)

    _patch_transformers_constant("transformers.utils.hub", cache_dir)
    _patch_transformers_constant("transformers.dynamic_module_utils", cache_dir)

    try:
        import tensorrt_llm._torch.model_config as model_config
    except Exception as exc:  # pragma: no cover - depends on optional TRT-LLM.
        logger.debug("[GMS] TRT-LLM model_config unavailable: %s", exc)
        return

    model_config.HF_MODULES_CACHE = str(cache_dir)
    model_config.config_file_lock = _make_local_config_file_lock(lock_dir)
    _patched = True
    logger.info(
        "[GMS] Patched TRT-LLM remote-code cache: cache=%s lock=%s",
        cache_dir,
        lock_dir,
    )


def _patch_transformers_constant(module_name: str, cache_dir: Path) -> None:
    try:
        module = __import__(module_name, fromlist=["HF_MODULES_CACHE"])
    except Exception:  # pragma: no cover - optional import path.
        return
    if hasattr(module, "HF_MODULES_CACHE"):
        module.HF_MODULES_CACHE = str(cache_dir)


def _make_local_config_file_lock(lock_dir: Path):
    @contextlib.contextmanager
    def local_config_file_lock(timeout: int = 10) -> Iterator[None]:
        lock_path = lock_dir / "_remote_code.lock"
        lock = filelock.FileLock(str(lock_path), timeout=timeout)
        yielded = False
        try:
            with lock:
                yielded = True
                yield
        except filelock.Timeout:
            logger.warning(
                "[GMS] TRT-LLM remote-code lock timed out at %s; proceeding unlocked",
                lock_path,
            )
            yield
        except (OSError, PermissionError) as exc:
            if yielded:
                logger.warning(
                    "[GMS] TRT-LLM remote-code lock release failed at %s: %s",
                    lock_path,
                    exc,
                )
                return
            logger.warning(
                "[GMS] TRT-LLM remote-code lock unavailable at %s: %s; proceeding unlocked",
                lock_path,
                exc,
            )
            yield

    return local_config_file_lock
