# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Protocol


class HealthResponse(Protocol):
    status_code: int

    def json(self) -> Any:
        ...


def check_health_ready(response: HealthResponse) -> bool:
    """Return whether an HTTP health response reports a ready component."""
    try:
        if response.status_code != 200:
            return False
        body = response.json()
        return isinstance(body, Mapping) and body.get("status") == "ready"
    except ValueError:
        return False
