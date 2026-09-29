# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for capture-independent, context-free server readiness."""

import pytest
from pagebroker_server import context_state, parse_args

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


class Driver:
    def __init__(self, current=0, active=0, failure=0):
        self.current = current
        self.active = active
        self.failure = failure
        self.calls = []

    def cuCtxGetCurrent(self):
        self.calls.append("current")
        return self.failure, self.current

    def cuDeviceGet(self, device):
        self.calls.append(("device", device))
        return 0, 7

    def cuDevicePrimaryCtxGetState(self, device):
        self.calls.append(("primary", device))
        return 0, 0, self.active


def test_context_probe_only_queries_state():
    driver = Driver()
    state = context_state(driver)
    assert state["current_context"] == 0
    assert state["primary_context_active"] is False
    assert state["primary_context_flags"] == 0
    assert driver.calls == ["current", ("device", 0), ("primary", 7)]


@pytest.mark.parametrize("driver", [Driver(current=42), Driver(active=1)])
def test_context_probe_rejects_current_or_primary_context(driver):
    with pytest.raises(RuntimeError, match="unexpectedly acquired"):
        context_state(driver)


def test_context_query_error_cannot_look_like_context_free():
    with pytest.raises(RuntimeError, match="cuCtxGetCurrent failed"):
        context_state(Driver(failure=3))


def test_server_starts_without_capture_or_load_generation(monkeypatch):
    def unexpected_read(*_args, **_kwargs):
        pytest.fail("service identity must not read a capture or manifest")

    monkeypatch.setattr("pathlib.Path.read_text", unexpected_read)
    args = parse_args(
        [
            "--rank",
            "3",
            "--expected-uuid",
            "GPU-test-destination",
            "--service-generation",
            "service-lifetime-A",
        ]
    )
    assert args.rank == 3
    assert args.service_generation == "service-lifetime-A"
    assert not hasattr(args, "capture_id")
    assert not hasattr(args, "generation")


@pytest.mark.parametrize("option", ["--capture-id", "--generation", "--artifact-root"])
def test_server_cli_rejects_restore_binding(option):
    with pytest.raises(SystemExit) as error:
        parse_args(
            [
                "--rank",
                "0",
                "--expected-uuid",
                "GPU-test-destination",
                "--service-generation",
                "service-lifetime-A",
                option,
                "capture-specific-value",
            ]
        )
    assert error.value.code == 2
