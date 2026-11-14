# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for endpoint profiling CLI argument forwarding."""

import inspect
import logging
import runpy
import sys
from unittest.mock import patch

import pytest

from dynamo.profiler.utils.profile_prefill import profile_prefill

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.planner,
]


@pytest.mark.parametrize("attention_dp_size", [None, 1, 4])
def test_prefill_cli_passes_attention_dp_size(attention_dp_size, monkeypatch, tmp_path):
    argv = [
        "profile_endpoint",
        "--mode",
        "prefill",
        "--model_name",
        "test-model",
        "--url",
        "http://test-endpoint",
        "--num_gpus",
        "4",
        "--work_dir",
        str(tmp_path),
    ]
    if attention_dp_size is not None:
        argv.extend(["--attention_dp_size", str(attention_dp_size)])
    monkeypatch.setattr(sys, "argv", argv)

    # The CLI configures the __main__ logger on every invocation.
    logger = logging.getLogger("__main__")
    with (
        patch(
            "dynamo.profiler.utils.profile_prefill.profile_prefill",
            autospec=True,
        ) as mock_profile_prefill,
        patch.object(logger, "handlers", []),
        patch.object(logger, "level", logger.level),
    ):
        runpy.run_module("dynamo.profiler.profile_endpoint", run_name="__main__")

    mock_profile_prefill.assert_called_once()
    arguments = inspect.signature(profile_prefill).bind(
        *mock_profile_prefill.call_args.args,
        **mock_profile_prefill.call_args.kwargs,
    )
    arguments.apply_defaults()
    assert arguments.arguments["attention_dp_size"] == (
        1 if attention_dp_size is None else attention_dp_size
    )
