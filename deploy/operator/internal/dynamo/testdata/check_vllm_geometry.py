# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the Go resolver's fixtures against native EngineArgs, without launching engines."""

import argparse
import json
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", help="Local model config, or a cached model ID")
args = parser.parse_args()
fixtures = json.loads(
    Path(__file__).with_name("enginegroup-native-geometry.json").read_text()
)

for fixture in fixtures["vllm"]:
    native_parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = native_parser.parse_args(
        ["--model", args.model, "--trust-remote-code", *fixture["args"]]
    )
    config = EngineArgs.from_cli_args(parsed).create_engine_config().parallel_config
    observed = {key: getattr(config, key) for key in fixture["native"]}
    assert observed == fixture["native"], (fixture["name"], observed, fixture["native"])
    print(json.dumps({"name": fixture["name"], "native": observed}), flush=True)
