# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
    pytest.mark.timeout(60),
]


def test_worker_lifecycle_import_does_not_require_pooling_bootstrap_api():
    # Use a separate interpreter so removing a native helper cannot affect peers.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from vllm.v1.engine import utils; "
                "utils.__dict__.pop('get_engine_zmq_addresses', None); "
                "from dynamo.vllm.embedding_worker_processes "
                "import is_embedding_process_child; "
                "assert callable(is_embedding_process_child)"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, result.stderr
