# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import signal
from unittest.mock import Mock, call

import pytest

from dynamo.frontend.shutdown import install_signal_handlers

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class FakeLoop:
    def __init__(self):
        self.handlers = {}

    def add_signal_handler(self, sig, handler):
        self.handlers[sig] = handler


@pytest.mark.parametrize(
    "first,second", [(signal.SIGTERM, signal.SIGINT), (signal.SIGINT, signal.SIGTERM)]
)
def test_first_signal_shuts_down_and_second_forces(first, second):
    loop = FakeLoop()
    runtime = Mock(spec=["shutdown", "force_shutdown"])
    install_signal_handlers(loop, runtime)

    loop.handlers[first]()
    assert runtime.method_calls == [call.shutdown()]

    loop.handlers[second]()
    loop.handlers[first]()
    assert runtime.method_calls == [call.shutdown(), call.force_shutdown()]
