# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.kv_dc_relay import __main__ as relay_main

pytestmark = [pytest.mark.gpu_0, pytest.mark.pre_merge, pytest.mark.unit]


def test_runtime_shutdown_is_graceful() -> None:
    relay_main._handle_relay_shutdown({"host_last_error": None})


def test_terminal_host_failure_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as error:
        relay_main._handle_relay_shutdown({"host_last_error": "injected failure"})

    assert error.value.code == 1


@pytest.mark.asyncio
async def test_late_host_failure_is_classified_after_shutdown() -> None:
    class LateFailureRelay:
        def __init__(self) -> None:
            self.host_last_error = None

        async def shutdown(self) -> None:
            self.host_last_error = "failure during host drain"

        async def health(self) -> dict[str, object]:
            return {"host_last_error": self.host_last_error}

    with pytest.raises(SystemExit) as error:
        await relay_main._shutdown_relay(
            LateFailureRelay(), classify_host_failure=True
        )

    assert error.value.code == 1


@pytest.mark.asyncio
async def test_endpoint_failure_is_not_masked_by_late_host_failure() -> None:
    class LateFailureRelay:
        async def shutdown(self) -> None:
            raise RuntimeError("secondary shutdown failure")

        async def health(self) -> dict[str, object]:
            return {"host_last_error": "secondary host failure"}

    await relay_main._shutdown_relay(
        LateFailureRelay(), classify_host_failure=False
    )
