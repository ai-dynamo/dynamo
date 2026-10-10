# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level tests for the async-signal-safe GPU crash interlock."""

import os
import select
import signal
import socket
import struct
import time

import pytest

gms_rust_ring = pytest.importorskip("gms_rust_ring")


def arm_gpu_crash_interlock(notification_fd, *, backend_name):
    # Test the native extension in a Python process that has not imported torch
    # or initialized background CUDA/framework threads before fork.
    gms_rust_ring.install_gpu_crash_interlock(
        notification_fd,
        [signal.SIGABRT, signal.SIGSEGV, signal.SIGBUS, signal.SIGILL, signal.SIGFPE],
    )


pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]


def test_broken_notification_pipe_still_parks_reporting_thread():
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    child = os.fork()
    if child == 0:
        arm_gpu_crash_interlock(write_fd, backend_name="vllm")
        signal.raise_signal(signal.SIGABRT)
        os._exit(3)
    os.close(write_fd)
    try:
        time.sleep(0.05)
        assert os.waitpid(child, os.WNOHANG | os.WUNTRACED) == (0, 0)
    finally:
        try:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
        except (ProcessLookupError, ChildProcessError):
            pass


def test_proactive_command_parks_listener_without_stopping_cuda_client():
    daemon, client = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    ack_read, ack_write = os.pipe()
    child = os.fork()
    if child == 0:
        daemon.close()
        os.close(ack_read)
        arm_gpu_crash_interlock(client.detach(), backend_name="vllm")
        # The command listener, not this Python thread, must enter the handler.
        while True:
            os.write(ack_write, b"r")
            time.sleep(0.05)
    client.close()
    os.close(ack_write)
    try:
        daemon.sendall(b"\x01")
        readable, _, _ = select.select([daemon], [], [], 2)
        assert readable == [daemon]
        assert struct.unpack("=IIii", daemon.recv(16)) == (
            0x47534D43,
            1,
            signal.SIGABRT,
            child,
        )
        os.set_blocking(ack_read, False)
        while True:
            try:
                if not os.read(ack_read, 4096):
                    break
            except BlockingIOError:
                break
        readable, _, _ = select.select([ack_read], [], [], 1)
        assert readable == [ack_read], "native command stopped the whole CUDA client"
        assert os.waitpid(child, os.WNOHANG | os.WUNTRACED) == (0, 0)
    finally:
        os.close(ack_read)
        daemon.close()
        try:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
        except (ProcessLookupError, ChildProcessError):
            pass


def test_invalid_signal_installation_leaves_previous_handler_intact():
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        signal.signal(signal.SIGABRT, lambda *_args: os._exit(42))
        try:
            gms_rust_ring.install_gpu_crash_interlock(write_fd, [signal.SIGABRT, 9999])
        except (ValueError, OSError):
            os.kill(os.getpid(), signal.SIGABRT)
        os._exit(3)
    os.close(write_fd)
    try:
        _, status = os.waitpid(child, os.WUNTRACED)
        assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 42
    finally:
        os.close(read_fd)
        try:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
        except (ProcessLookupError, ChildProcessError):
            pass


def test_native_handler_reports_signal_and_parks_until_authority_kills():
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

        assert os.waitpid(child, os.WNOHANG | os.WUNTRACED) == (0, 0)
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
