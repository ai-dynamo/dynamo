# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay


def run_trace_replay(
    trace_file,
    *,
    extra_engine_args=None,
    num_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
):
    return _run_mocker_trace_replay(
        trace_file,
        extra_engine_args=extra_engine_args,
        num_workers=num_workers,
        replay_concurrency=replay_concurrency,
        replay_mode=replay_mode,
    )


__all__ = ["run_trace_replay"]
