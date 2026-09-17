# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract coverage for the s3fs version installed by the vLLM runtime."""

from unittest.mock import AsyncMock, MagicMock

import pytest

s3fs = pytest.importorskip(
    "s3fs", reason="s3fs is installed and tested by the vLLM runtime lane"
)
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_s3fs_create_mode_maps_to_atomic_conditional_put() -> None:
    payload = b"artifact-bytes"
    filesystem = s3fs.S3FileSystem(asynchronous=True, skip_instance_cache=True)
    filesystem._call_s3 = AsyncMock(return_value={})
    filesystem.invalidate_cache = MagicMock()

    await filesystem._pipe_file(
        "artifacts/run/output.dynexp",
        payload,
        mode="create",
        chunksize=64 * 1024 * 1024,
    )

    filesystem._call_s3.assert_awaited_once_with(
        "put_object",
        Bucket="artifacts",
        Key="run/output.dynexp",
        Body=payload,
        IfNoneMatch="*",
    )
