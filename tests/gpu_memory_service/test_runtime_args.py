# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.gpu_memory_service.common.runtime import _replace_cli_option


def test_replace_cli_option_preserves_profile_byte_cap():
    args = [
        "--kv-cache-memory-bytes",
        "67108864",
        "--gpu-memory-utilization",
        "0.01",
    ]

    assert _replace_cli_option(args, "--gpu-memory-utilization", "0.22") == [
        "--kv-cache-memory-bytes",
        "67108864",
        "--gpu-memory-utilization",
        "0.22",
    ]
    assert args[-1] == "0.01"
