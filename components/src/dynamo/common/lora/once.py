# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of the `OnceLock` class, which is a small thread-safe utility for initializing a value once
and then providing read-only access to it.
"""

import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class OnceLock(Generic[T]):
    def __init__(self) -> None:
        self._value: T | None = None
        self._lock = threading.Lock()

    def get_or_init(self, initializer: Callable[[], T]) -> T:
        value = self._value
        if value is not None:
            return value
        with self._lock:
            value = self._value
            if value is None:
                value = initializer()
                self._value = value
            return value

    def get(self) -> T | None:
        return self._value
