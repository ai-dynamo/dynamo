# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check one fixture against SGLang's complete argument-resolution pipeline."""

import argparse
import json
from pathlib import Path

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.server_args import ServerArgs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", help="Local model config, or a cached model ID")
parser.add_argument("index", type=int, choices=(0, 1))
args = parser.parse_args()
fixtures = json.loads(
    Path(__file__).with_name("enginegroup-native-geometry.json").read_text()
)
fixture = fixtures["sglang"][args.index]

native_parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(native_parser)
parsed = native_parser.parse_args(
    ["--model-path", args.model, "--trust-remote-code", *fixture["args"]]
)
server_args = ServerArgs.from_cli_args(parsed)
server_args.resolve_once()
config = resolved_view(server_args)
observed = {key: getattr(config, key) for key in fixture["native"]}
assert observed == fixture["native"], (fixture["name"], observed, fixture["native"])
print(json.dumps({"name": fixture["name"], "native": observed}), flush=True)
