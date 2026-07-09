# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path

from .fences import (
    _assert_generation_fence_state,
    _assert_worker_wide_fence_state,
    _exercise_generation_fence,
    _exercise_worker_wide_fences,
    _wait_for_aof_rewrite,
)
from .support import _command, _find_free_port, _start, _stop



def run_aof_rewrite(server: str, module: str) -> None:
    """Ensure an AOF rewrite preserves a conditional-replace generation."""
    with tempfile.TemporaryDirectory(prefix="dynkv-aof-") as path:
        directory = Path(path)
        port = _find_free_port()
        process = _start(server, module, directory, port)
        try:
            # Force the rewrite to use AOF commands rather than a base RDB so
            # router_index_aof_rewrite emits and reloads DYNKV.RESTORE.
            assert (
                _command(port, b"CONFIG", b"SET", b"aof-use-rdb-preamble", b"no")
                == b"OK"
            )
            _exercise_generation_fence(port)
            _exercise_worker_wide_fences(port)
            _wait_for_aof_rewrite(port)
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port)
        try:
            _assert_generation_fence_state(port)
            _assert_worker_wide_fence_state(port)
        finally:
            _stop(process, port)
