# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from tests.utils.health_checks import check_health_ready

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


class _Response:
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, ValueError):
            raise self._body
        return self._body


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (200, {"status": "ready"}, True),
        (200, {"status": "starting"}, False),
        (503, {"status": "ready"}, False),
        (200, {}, False),
        (200, ["ready"], False),
    ],
)
def test_check_health_ready(status_code, body, expected):
    assert check_health_ready(_Response(status_code, body)) is expected


def test_check_health_ready_rejects_invalid_json():
    assert check_health_ready(_Response(200, ValueError("invalid JSON"))) is False
