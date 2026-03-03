# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample handler that returns a fixed short reply, streamed token by token."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from dynamo.common import Context
from dynamo.backend import Handler

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


class ExampleHandler(Handler):
    """Returns a fixed 'Hello World!' reply streamed token by token."""

    def __init__(
        self,
        component: Optional[Any] = None,
        shutdown_event: Optional[asyncio.Event] = None,
        token_delay: float = 0.0,
    ) -> None:
        super().__init__(component=component, shutdown_event=shutdown_event)
        self.token_delay = token_delay

    async def generate(
        self, request: Dict[str, Any], context: Context
    ) -> AsyncGenerator[Dict[str, Any], None]:
        model = request.get("model", "")
        reply_ids = _encode_reply(model) if model else _FALLBACK_REPLY_IDS
        if not reply_ids:
            reply_ids = _FALLBACK_REPLY_IDS
        total = len(reply_ids)
        async with self._cancellation_monitor(context):
            for i, token_id in enumerate(reply_ids):
                if context.is_stopped() or context.is_killed():
                    break
                if self.token_delay > 0:
                    await asyncio.sleep(self.token_delay)
                out: Dict[str, Any] = {"token_ids": [token_id]}
                if i == total - 1:
                    out["finish_reason"] = "stop"
                yield out
