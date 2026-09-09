# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
from pathlib import Path

import pytest
from router.common import add_common_args, add_synthesis_args, prepare_trace_dataset

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.mark.parametrize(
    ("synthesize", "output_field"),
    [(False, "output_length"), (False, "output_tokens"), (True, "output_length")],
)
def test_expected_osl_matches_trace_output_length(tmp_path, synthesize, output_field):
    trace = tmp_path / "source.jsonl"
    rows = [
        {
            "timestamp": timestamp,
            "input_length": 128,
            output_field: 37,
            "hash_ids": [1, 2],
            "nvext": {"agent_hints": {"priority": 2}, "ignore_eos": True},
        }
        for timestamp in (0, 1)
    ]
    if output_field == "output_length":
        for row in rows:
            row["output_tokens"] = 999  # Canonical length wins over the legacy alias.
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_synthesis_args(parser)
    cli_args = [
        "--input-dataset",
        str(trace),
        "--use-expected-osl",
        "--block-size",
        "64",
        "--seed",
        "42",
    ]
    if synthesize:
        cli_args.extend(["--num-requests", "5", "--osl-multiplier", "2"])
    args = parser.parse_args(cli_args)

    requests, output_path = prepare_trace_dataset(
        args, tmp_path, logging.getLogger(__name__)
    )

    assert len(requests) == (5 if synthesize else 2)
    expected_osl = 74 if synthesize else 37
    assert all(row["nvext"]["agent_hints"]["osl"] == expected_osl for row in requests)
    if not synthesize:
        assert all(row["nvext"]["agent_hints"]["priority"] == 2 for row in requests)
        assert all(row["nvext"]["ignore_eos"] for row in requests)
    saved_requests = [
        json.loads(line) for line in Path(output_path).read_text().splitlines()
    ]
    assert saved_requests == requests
    assert [json.loads(line) for line in trace.read_text().splitlines()] == rows
