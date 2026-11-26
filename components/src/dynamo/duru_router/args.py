# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Duru router CLI parsing and config assembly."""

from __future__ import annotations

import argparse
from typing import Optional

from dynamo.thunderagent_router.args import ThunderAgentArgGroup as BaseArgGroup
from dynamo.thunderagent_router.args import ThunderAgentRouterConfig
from dynamo.thunderagent_router.router import ThunderAgentConfig


class DuruRouterConfig(ThunderAgentRouterConfig):
    # The shared config requires a value, but Duru disables forced timeout resumes.
    resume_timeout_seconds: float = ThunderAgentConfig.resume_timeout_seconds


class DuruArgGroup(BaseArgGroup):
    # Duru resumes by score/capacity, never by the upstream forced-resume timer.
    include_resume_timeout = False


def parse_args(argv: Optional[list[str]] = None) -> DuruRouterConfig:
    parser = argparse.ArgumentParser(
        description="Dynamo Duru Router: program-level scheduler with "
        "tool-boundary pause/resume on top of native KV-aware routing",
        epilog=(
            "DURU_DISPATCH_ACK_TIMEOUT_SECONDS (default: 300) bounds the wait for "
            "a worker acknowledgement after uncertain dispatch failure or cancellation. "
            "Expiry stops new admissions until recovery; it does not release "
            "unacknowledged work. Accounting continues to drain late worker events."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    DuruArgGroup().add_arguments(parser)
    args = parser.parse_args(argv)
    config = DuruRouterConfig.from_cli_args(args)
    config.validate()
    return config


__all__ = [
    "DuruArgGroup",
    "DuruRouterConfig",
    "parse_args",
]
