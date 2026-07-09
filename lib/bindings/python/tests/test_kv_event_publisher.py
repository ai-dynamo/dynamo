# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.llm import KvEventPublisher
from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_publish_batch_rejects_complete_malformed_python_list(
    runtime: DistributedRuntime,
) -> None:
    endpoint = runtime.endpoint("test.kv-publisher.generate")
    publisher = KvEventPublisher(
        endpoint,
        worker_id=1,
        kv_block_size=1,
        enable_local_indexer=True,
    )

    try:
        with pytest.raises(ValueError, match="invalid KV event batch"):
            publisher.publish_batch(
                [
                    {"type": "removed", "block_hashes": [10]},
                    {
                        "type": "stored",
                        "token_ids": [1],
                        "num_block_tokens": [1],
                    },
                ]
            )

        # A rejected list never reaches the publisher channel; a subsequent
        # fully valid list remains independently publishable.
        publisher.publish_batch(
            [
                {"type": "removed", "block_hashes": [10]},
                {
                    "type": "stored",
                    "token_ids": [1],
                    "num_block_tokens": [1],
                    "block_hashes": [11],
                },
            ]
        )
    finally:
        publisher.shutdown()
