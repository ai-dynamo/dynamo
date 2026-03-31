# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TTS/audio utility functions for the vLLM-Omni backend."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_dummy_tokenizer_for_tts(model: str):
    """Create a minimal tokenizer.json for TTS models that lack one.

    Audio/TTS models (e.g., Qwen3-TTS) use a custom speech tokenizer and don't
    ship the standard tokenizer.json expected by the Rust ModelDeploymentCard
    loader. This writes a placeholder so register_model doesn't fail.

    This is a short-term workaround. The long-term fix is making TokenizerKind
    optional in ModelDeploymentCard::from_repo_checkout().
    """
    from huggingface_hub import scan_cache_dir

    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == model:
            for revision in repo.revisions:
                tokenizer_path = Path(revision.snapshot_path) / "tokenizer.json"
                if not tokenizer_path.exists():
                    logger.warning(
                        "TTS model %s has no tokenizer.json; "
                        "creating a minimal placeholder at %s",
                        model,
                        tokenizer_path,
                    )
                    tokenizer_path.write_text(json.dumps({"version": "1.0"}))
            return
