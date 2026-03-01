# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample handler that returns a fixed short reply, streamed token by token."""

import logging
from typing import Any, AsyncGenerator, Dict, List

from dynamo._core import Context
from dynamo.common.backend import BaseHandler

logger = logging.getLogger(__name__)

# Fallback when tokenizer is unavailable (tokenizer-specific; may decode to other text).
_FALLBACK_REPLY_IDS: List[int] = [0]

_REPLY_TEXT = "Hello World!"


def _encode_reply(model: str) -> List[int]:
    """Encode reply text using the model's tokenizer so it decodes to 'Hello World!'."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return tokenizer.encode(_REPLY_TEXT, add_special_tokens=False)
    except Exception as e:
        logger.debug(
            "Could not load tokenizer for %s: %s; using fallback reply ids", model, e
        )
        return _FALLBACK_REPLY_IDS


class MyEngineHandler(BaseHandler):
    """Returns a fixed short reply (e.g. 'Hi!') streamed token by token."""

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        model = request.get("model", "")
        reply_ids = _encode_reply(model) if model else _FALLBACK_REPLY_IDS
        if not reply_ids:
            reply_ids = _FALLBACK_REPLY_IDS
        total = len(reply_ids)
        for i, token_id in enumerate(reply_ids):
            out: Dict[str, Any] = {"token_ids": [token_id]}
            if i == total - 1:
                out["finish_reason"] = "stop"
            yield out
