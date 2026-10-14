# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import signal

import pytest

from dynamo.frontend.shutdown import install_signal_handlers

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class FakeLoop:
    def __init__(self):
        self.handlers = {}

    def add_signal_handler(self, sig, handler):
        self.handlers[sig] = handler


class FakeRuntime:
    def __init__(self):
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


class FakeFrontendShutdown:
    def __init__(self):
        self.shutdowns = 0

    def shutdown(self):
        self.shutdowns += 1


def test_http_signal_stages_first_signal_and_forces_on_second():
    loop = FakeLoop()
    runtime = FakeRuntime()
    frontend_shutdown = FakeFrontendShutdown()
    install_signal_handlers(loop, runtime, frontend_shutdown)

    loop.handlers[signal.SIGTERM]()
    assert frontend_shutdown.shutdowns == 1
    assert runtime.shutdowns == 0

    loop.handlers[signal.SIGINT]()
    assert frontend_shutdown.shutdowns == 1
    assert runtime.shutdowns == 1

    loop.handlers[signal.SIGTERM]()
    assert frontend_shutdown.shutdowns == 1
    assert runtime.shutdowns == 1


def test_non_http_signal_shuts_runtime_down_immediately():
    loop = FakeLoop()
    runtime = FakeRuntime()
    install_signal_handlers(loop, runtime, None)

    loop.handlers[signal.SIGTERM]()
    assert runtime.shutdowns == 1

    loop.handlers[signal.SIGINT]()
    assert runtime.shutdowns == 1
