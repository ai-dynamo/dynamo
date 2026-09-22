# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from tests.utils.http_checks import (
    check_health_generate,
    check_health_ready,
    check_http_ok,
    check_model_registered,
    model_registered,
    models_available,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class StubResponse:
    def __init__(
        self,
        payload: Any = None,
        *,
        status_code: int = 200,
        error: ValueError | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._error = error

    def json(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._payload


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (StubResponse({"status": "ready"}), True),
        (StubResponse({"status": "starting"}), False),
        (StubResponse({"status": "ready"}, status_code=503), False),
        (StubResponse(error=ValueError("invalid JSON")), False),
        (StubResponse([{"status": "ready"}]), False),
    ],
)
def test_check_health_ready(response: StubResponse, expected: bool) -> None:
    assert check_health_ready(response) is expected  # type: ignore[arg-type]


def test_model_checks_require_a_valid_model_list() -> None:
    response = StubResponse({"data": [{"id": "model-a"}, {"id": "model-b"}]})

    assert models_available(response)  # type: ignore[arg-type]
    assert model_registered(response, model="model-b")  # type: ignore[arg-type]
    assert not model_registered(response, model="missing")  # type: ignore[arg-type]
    assert not models_available(StubResponse({"data": {"id": "model-a"}}))  # type: ignore[arg-type]


def test_compatibility_model_check_keeps_stabilization_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[int] = []
    monkeypatch.setattr("tests.utils.http_checks.time.sleep", sleeps.append)

    assert check_model_registered(  # type: ignore[arg-type]
        StubResponse({"data": [{"id": "model-a"}]}), model="model-a"
    )
    assert not check_model_registered(  # type: ignore[arg-type]
        StubResponse({"data": [{"id": "model-a"}]}), model="missing"
    )
    assert sleeps == [1]


def test_generate_health_accepts_endpoint_and_instance_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("tests.utils.http_checks.time.sleep", lambda _: None)

    assert check_health_generate(StubResponse({"endpoints": ["backend.generate"]}))  # type: ignore[arg-type]
    assert check_health_generate(  # type: ignore[arg-type]
        StubResponse({"instances": [{"endpoint": "generate"}]})
    )
    assert not check_health_generate(StubResponse({"endpoints": ["metrics"]}))  # type: ignore[arg-type]


def test_http_ok_only_accepts_200() -> None:
    assert check_http_ok(StubResponse(status_code=200))  # type: ignore[arg-type]
    assert not check_http_ok(StubResponse(status_code=204))  # type: ignore[arg-type]
