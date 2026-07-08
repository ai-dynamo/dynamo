# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine
except ImportError:
    pytest.skip("tensorrt_llm backend not available", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]


def _stored_kv_event(cache_salt: str | None = "tenant-a") -> dict:
    return {
        "event_id": 1,
        "attention_dp_rank": 0,
        "data": {
            "type": "stored",
            "parent_hash": None,
            "blocks": [
                {
                    "type": "stored_block",
                    "block_hash": 123,
                    "cache_salt": cache_salt,
                    "tokens": [
                        {"token_id": 1},
                        {"token_id": 2},
                        {"token_id": 3},
                        {"token_id": 4},
                    ],
                }
            ],
        },
    }


def test_dispatch_kv_event_forwards_per_block_cache_salt() -> None:
    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    publisher = MagicMock()
    engine._kv_publishers = {0: publisher}
    engine._last_event_id_by_rank = {}
    engine._warned_unknown_dp_rank = False
    engine._additional_metrics = None
    engine._partial_block_hashes_by_rank = {}
    engine.kv_block_size = 4

    engine._dispatch_kv_event(_stored_kv_event())

    publisher.publish_stored.assert_called_once()
    assert publisher.publish_stored.call_args.kwargs["cache_salt"] == "tenant-a"
