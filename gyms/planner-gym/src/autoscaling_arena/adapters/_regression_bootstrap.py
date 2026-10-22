# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared no-op regression-bootstrap contract for Arena rival engines."""

from __future__ import annotations

from typing import Any, ClassVar, Optional


class _NoopRegressionBootstrap:
    """Accept Planner replay bootstrap calls without consuming the FPMs.

    Rival policies do not use the Planner's fitted throughput regressions.  The
    method completes the replay adapter's de-facto engine contract, while the
    capability flag lets the Arena avoid generating AIS benchmark points that
    would only be discarded.
    """

    supports_ais_bootstrap: ClassVar[bool] = False

    def install_regressions_from_fpms(
        self,
        *,
        prefill_fpms: Optional[list[Any]] = None,
        decode_fpms: Optional[list[Any]] = None,
        agg_fpms: Optional[list[Any]] = None,
    ) -> None:
        pass


__all__ = ["_NoopRegressionBootstrap"]
