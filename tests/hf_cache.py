# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import tempfile
import textwrap
from pathlib import Path

_mistral_patch_applied: bool = False


def _enable_offline_with_mistral_patch():
    """Set HF_HUB_OFFLINE=1 and work around a transformers 4.57.3 regression.

    transformers 4.57.3 (PR #42389) introduced _patch_mistral_regex which calls
    huggingface_hub.model_info() unconditionally for every tokenizer load — even
    non-Mistral models with fully cached weights. This API call fails when
    HF_HUB_OFFLINE=1.

    Since tests launch TRT-LLM workers as subprocesses that inherit env vars but
    not in-process monkey-patches, we inject the fix via a sitecustomize.py on
    PYTHONPATH so every subprocess auto-applies it at startup.

    _mistral_patch_applied guards the class-level patch and PYTHONPATH injection
    so they run at most once per enable/disable cycle. _disable_offline_with_mistral_patch
    resets the flag so a subsequent enable call re-injects PYTHONPATH; the class-level
    re-application on that second call is harmless — it adds one extra try/except layer
    that behaves identically to the first.

    Upstream bug: https://github.com/huggingface/transformers/issues/44843

    TODO: Remove this workaround once transformers ships a fix and TRT-LLM (or
    any other dependency) upgrades to that fixed version.
    """
    global _mistral_patch_applied
    os.environ["HF_HUB_OFFLINE"] = "1"
    if _mistral_patch_applied:
        return  # class patch and sitecustomize already applied for this cycle

    # Resolve OfflineModeIsEnabled before touching transformers. If huggingface_hub
    # predates the .errors module, transformers 4.57.3+ imports OfflineModeIsEnabled
    # lazily inside _patch_mistral_regex, so that call itself raises ImportError under
    # offline mode — using ImportError as the fallback catches that exact error.
    try:
        from huggingface_hub.errors import OfflineModeIsEnabled
    except ImportError:
        OfflineModeIsEnabled = ImportError  # type: ignore[assignment,misc]

    # Apply the patch in this process
    try:
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase

        original = PreTrainedTokenizerBase._patch_mistral_regex

        @classmethod  # type: ignore[misc]
        def _safe_patch(cls, tokenizer, *args, **kwargs):
            try:
                return original.__func__(cls, tokenizer, *args, **kwargs)
            except OfflineModeIsEnabled:
                return tokenizer

        PreTrainedTokenizerBase._patch_mistral_regex = _safe_patch
    except (ImportError, AttributeError):
        return  # transformers version without _patch_mistral_regex — nothing to do

    # Write a sitecustomize.py so subprocesses also get the patch.
    # Use a per-worker dir under xdist to avoid write races.
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    patch_dir = os.path.join(tempfile.gettempdir(), f"dynamo_test_hf_patch_{worker_id}")
    os.makedirs(patch_dir, exist_ok=True)
    with open(os.path.join(patch_dir, "sitecustomize.py"), "w") as f:
        f.write(
            textwrap.dedent(
                """\
            import os
            if os.environ.get('HF_HUB_OFFLINE') == '1':
                try:
                    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _T
                    try:
                        from huggingface_hub.errors import OfflineModeIsEnabled as _E
                    except ImportError:
                        _E = ImportError
                    _orig = _T._patch_mistral_regex
                    @classmethod
                    def _safe(cls, tokenizer, *a, **kw):
                        try:
                            return _orig.__func__(cls, tokenizer, *a, **kw)
                        except _E:
                            return tokenizer
                    _T._patch_mistral_regex = _safe
                except (ImportError, AttributeError):
                    pass
        """
            )
        )
    existing_entries = [e for e in os.environ.get("PYTHONPATH", "").split(":") if e]
    os.environ["PYTHONPATH"] = ":".join([patch_dir] + existing_entries)
    logging.info(
        "Enabled HF_HUB_OFFLINE with _patch_mistral_regex workaround "
        "(see https://github.com/huggingface/transformers/issues/44843)"
    )
    _mistral_patch_applied = True


def _disable_offline_with_mistral_patch():
    """Undo _enable_offline_with_mistral_patch."""
    global _mistral_patch_applied
    os.environ.pop("HF_HUB_OFFLINE", None)
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "main")
    patch_dir = os.path.join(tempfile.gettempdir(), f"dynamo_test_hf_patch_{worker_id}")
    pythonpath = os.environ.get("PYTHONPATH", "")
    result = ":".join(e for e in pythonpath.split(":") if e and e != patch_dir)
    if result:
        os.environ["PYTHONPATH"] = result
    else:
        os.environ.pop("PYTHONPATH", None)
    shutil.rmtree(patch_dir, ignore_errors=True)
    _mistral_patch_applied = False


# Keys managed by _apply_models_dir_env / _restore_models_dir_env.
# PYTHONPATH is intentionally excluded: _disable_offline_with_mistral_patch()
# removes its entry by exact-match list filtering (idempotent, needs no snapshot).
_TRANSFORMERS_CACHE_OVERRIDE_KEYS = (
    "TRANSFORMERS_CACHE",
    "PYTORCH_TRANSFORMERS_CACHE",
    "PYTORCH_PRETRAINED_BERT_CACHE",
)

_MODELS_DIR_ENV_KEYS = (
    "HF_HUB_CACHE",
    "HF_HOME",
    *_TRANSFORMERS_CACHE_OVERRIDE_KEYS,
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "DYNAMO_MODELS_DIR",
)


def _apply_models_dir_env(models_dir: str) -> dict:
    """Set HF env vars for read-only cache mode. Returns original env values."""
    orig = {k: os.environ.get(k) for k in _MODELS_DIR_ENV_KEYS}
    if (Path(models_dir) / "hub").is_dir():
        logging.warning(
            "--models-dir: detected HF_HOME layout (hub/ subdirectory found). "
            "If this is wrong (e.g. you have a model named hub/), rename hub/ "
            "or pass a bare HF_HUB_CACHE directory instead."
        )
        os.environ.pop("HF_HUB_CACHE", None)  # clear so HF_HOME takes effect
        os.environ["HF_HOME"] = models_dir
    else:
        logging.info("--models-dir: detected bare HF_HUB_CACHE layout")
        os.environ.pop("HF_HOME", None)  # clear for consistency
        os.environ["HF_HUB_CACHE"] = models_dir
    for key in _TRANSFORMERS_CACHE_OVERRIDE_KEYS:
        os.environ.pop(key, None)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DYNAMO_MODELS_DIR"] = models_dir
    _enable_offline_with_mistral_patch()  # activates sitecustomize for Mistral tokenizer workaround
    return orig


def _restore_models_dir_env(orig: dict) -> None:
    """Undo _apply_models_dir_env. Call after fixture yield."""
    # _disable pops HF_HUB_OFFLINE; the loop below then restores the original value
    # (no-op if orig was None, set-back if orig had a pre-existing value). Safe.
    _disable_offline_with_mistral_patch()  # pops HF_HUB_OFFLINE + cleans sitecustomize
    for k, v in orig.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
