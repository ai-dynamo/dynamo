# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom FoM providers loaded by the ``python:<module>:<Class>`` tests."""


class FakeProvider:
    """A provider that satisfies EPD's ``get_perf() -> float`` contract."""

    def __init__(self) -> None:
        self.calls = 0

    def get_perf(self) -> float:
        self.calls += 1
        return 42.0


class NoPerfProvider:
    """A class without ``get_perf``; the loader must reject it."""


class NegativeProvider:
    """Returns a value EPD cannot use (negative); the probe must flag it."""

    def get_perf(self) -> float:
        return -1.0
