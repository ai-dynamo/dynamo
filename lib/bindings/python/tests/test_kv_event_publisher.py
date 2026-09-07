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
@pytest.mark.timeout(30)
@pytest.mark.parametrize("request_plane", ["tcp"], indirect=True)
async def test_publish_batch_depythonizes_complete_typed_python_list(
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

        with pytest.raises(ValueError, match="unknown field.*cache_sal"):
            publisher.publish_batch(
                [
                    {
                        "type": "stored",
                        "token_ids": [1],
                        "num_block_tokens": [1],
                        "block_hashes": [11],
                        "cache_sal": "tenant-a",
                    }
                ]
            )

        # A rejected list never reaches the publisher channel; a subsequent
        # fully valid mixed list exercises the real Python-object depythonize
        # path and remains independently publishable.
        publisher.publish_batch(
            [
                {"type": "removed", "block_hashes": [10]},
                {
                    "type": "stored",
                    "token_ids": [1],
                    "num_block_tokens": [1],
                    "block_hashes": [11],
                    "parent_hash": 9,
                    "block_mm_infos": [None],
                    "lora_name": "adapter-a",
                    "is_eagle": True,
                    "cache_salt": "tenant-a",
                },
            ]
        )
    finally:
        publisher.shutdown()
