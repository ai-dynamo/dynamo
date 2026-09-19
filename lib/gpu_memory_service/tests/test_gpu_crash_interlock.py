# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level tests for the async-signal-safe GPU crash interlock."""

import os
import select
import signal
import struct

import pytest
from gpu_memory_service.integrations.common.gpu_quiescence import (
    arm_gpu_crash_interlock,
)

gms_rust_ring = pytest.importorskip("gms_rust_ring")

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.none]


def test_native_handler_reports_signal_and_stops_until_authority_kills():
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            arm_gpu_crash_interlock(write_fd, backend_name="vllm")
            os.kill(os.getpid(), signal.SIGABRT)
        except (ImportError, OSError, RuntimeError, ValueError):
            os._exit(2)
        os._exit(3)

    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 2)
        assert readable == [read_fd], "native crash record was not delivered"
        record = os.read(read_fd, struct.calcsize("=IIii"))
        assert struct.unpack("=IIii", record) == (
            0x47534D43,
            1,
            signal.SIGABRT,
            child,
        )

        waited, status = os.waitpid(child, os.WUNTRACED)
        assert waited == child
        assert os.WIFSTOPPED(status)
        assert os.WSTOPSIG(status) == signal.SIGSTOP
    finally:
        os.close(read_fd)
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass


def test_native_handler_leaves_sigterm_for_graceful_shutdown():
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            arm_gpu_crash_interlock(write_fd, backend_name="vllm")
            os.kill(os.getpid(), signal.SIGTERM)
        except (ImportError, OSError, RuntimeError, ValueError):
            os._exit(2)
        os._exit(3)

    os.close(write_fd)
    try:
        waited, status = os.waitpid(child, os.WUNTRACED)
        assert waited == child
        assert os.WIFSIGNALED(status)
        assert os.WTERMSIG(status) == signal.SIGTERM
        assert os.read(read_fd, 1) == b""
    finally:
        os.close(read_fd)
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass
