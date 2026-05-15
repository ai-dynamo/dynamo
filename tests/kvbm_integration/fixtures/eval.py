# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer C: eval (DeterminismTester binding).

The eval layer is already decoupled in `tests/kvbm_integration/common.py` —
this module exposes a thin factory and pytest fixture that bind a
`DeterminismTester` (or `AggDeterminismTester` for agg-mode prefix-cache
reset semantics) to a server handle yielded by `kvbm_server`.
"""

import os
from typing import Optional

import pytest
import requests

from ..common import DeterminismTester, ServerType


class AggDeterminismTester(DeterminismTester):
    """Aggregated-mode determinism tester with `/reset_prefix_cache` support.

    Moved out of `test_determinism_agg.py` so the fixture layer owns it.
    Behavior is unchanged from the previous inline definition.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id, server_type)

    def reset_prefix_cache(self) -> None:
        print("Resetting prefix cache...")
        if self.server_type == ServerType.trtllm:
            # TRTLLM has no reset_prefix_cache endpoint; evict via 300 Shakespeare requests.
            shakespeare_count = 300
            for seq_idx in range(1, shakespeare_count + 1):
                start_word = (seq_idx - 1) * self.word_count
                content = self.get_shakespeare_content(start_word)
                if content:
                    print(
                        f"Resetting Shakespeare sequence {seq_idx} "
                        f"(words {start_word}-{start_word + self.word_count - 1})..."
                    )
                    try:
                        self.make_request(content)
                    except Exception as e:
                        print(f"Resetting request failed: {e}")
        else:
            response = requests.post(
                f"{self.base_url}/reset_prefix_cache",
                timeout=int(os.environ.get("KVBM_HTTP_TIMEOUT", "30")),
            )
            response.raise_for_status()
        print("Cache reset done")


def make_determinism_tester(server_handle) -> AggDeterminismTester:
    """Build an `AggDeterminismTester` bound to a `kvbm_server` handle.

    The handle is duck-typed — both `KvbmServerManager` (spawn mode) and
    `_ExternalServer` (attach mode) expose `base_url`, `model_config`, and
    `server_type`.
    """
    tester = AggDeterminismTester(
        base_url=server_handle.base_url,
        model_id=server_handle.model_config.model_id,
        server_type=server_handle.server_type,
    )
    tester.download_shakespeare_text()
    return tester


@pytest.fixture(scope="function")
def kvbm_tester(kvbm_server) -> AggDeterminismTester:
    """Build the agg-mode determinism tester for the active server."""
    return make_determinism_tester(kvbm_server)
