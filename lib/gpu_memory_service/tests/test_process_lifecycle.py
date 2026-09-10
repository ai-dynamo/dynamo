# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import select
import signal
import sys

import pytest
from gpu_memory_service.integrations.common.process_lifecycle import (
    arm_parent_death_signal,
)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PDEATHSIG test")
def test_shared_writer_dies_with_its_parent():
    read_fd, write_fd = os.pipe()
    supervisor = os.fork()
    if supervisor == 0:
        os.close(read_fd)
        writer = os.fork()
        if writer == 0:
            arm_parent_death_signal()
            os.write(write_fd, b"ready")
            signal.pause()
            os._exit(1)
        os.close(write_fd)
        signal.pause()
        os._exit(1)

    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 2.0)
        assert readable and os.read(read_fd, 5) == b"ready"
        os.kill(supervisor, signal.SIGKILL)
        os.waitpid(supervisor, 0)
        supervisor = 0
        readable, _, _ = select.select([read_fd], [], [], 2.0)
        assert readable and os.read(read_fd, 1) == b""
    finally:
        os.close(read_fd)
        if supervisor:
            os.kill(supervisor, signal.SIGKILL)
            os.waitpid(supervisor, 0)
