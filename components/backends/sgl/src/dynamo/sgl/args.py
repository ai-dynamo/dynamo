# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import socket
import logging
from enum import Enum
from argparse import Namespace

import sglang as sgl
from sglang.srt.server_args import ServerArgs

# dynamo args/keys should be added here 
DYNAMO_ARGS = ["namespace", "component", "endpoint", "migration-limit"]

class DynamoArgs:
    def __init__(self) -> None:
        pass

    @classmethod
    def create_dynamo_args(args) -> None:
        # for a in args
        # match against dynamo args
        # set them as self.<>
        # remove them from the args array before giving to sglang
        pass

class DisaggregationMode(Enum):
    """
    Set this instead of string matching
    """
    AGGREGATED = "agg"
    PREFILL = "prefill"
    DECODE = "decode"

# Configuration class to store SGL engine arguments and dynamo specific arguments
class Config:
    def __init__(self, server_args: ServerArgs, dynamo_args: DynamoArgs) -> None:
        self.server_args = server_args
        self.dynamo_args = dynamo_args
        self.serving_strategy = self._set_serving_strategy()

    def _set_serving_strategy(self):
        if self.server_args.disaggregation_mode == "null":
            return DisaggregationMode.AGGREGATED
        elif self.server_args.disaggregation_mode == "prefill":
            return DisaggregationMode.PREFILL
        elif self.server_args.disaggregation_mode == "decode":
            return DisaggregationMode.DECODE
        else:
            logging.error("Cannot ascertain serving strategy. This shouldn't happen")

# TODO: parse dynamo args first and figure out how to make it clean 
# like the vllm. Make add that style of parser in the DynamoArgs class
# and then do the stuff below after its been removed. Yes?
def parse_cmd_line_args(args: list[str]) -> Config:
    dynamo_args = DynamoArgs.create_dynamo_args(args)
    parser = argparse.ArgumentParser()
    bootstrap_port = _reserve_disaggregation_bootstrap_port()
    ServerArgs.add_cli_args(parser)
    parsed_args = parser.parse_args(args)
    if not any(arg.startswith("--disaggregation-bootstrap-port") for arg in args):
        args_dict = vars(parsed_args)
        args_dict["disaggregation_bootstrap_port"] = bootstrap_port
        parsed_args = Namespace(**args_dict)
    server_args = ServerArgs.from_cli_args(parsed_args)
    return Config(server_args, dynamo_args)

@contextlib.contextmanager
def reserve_free_port(host: str = "localhost"):
    """
    Find and reserve a free port until context exits.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, 0))
        _, port = sock.getsockname()
        yield port
    finally:
        sock.close()


def _reserve_disaggregation_bootstrap_port():
    """
    Each worker requires a unique port for disaggregation_bootstrap_port.
    We use an existing utility function that reserves a free port on your
    machine to avoid collisions.
    """
    with reserve_free_port() as port:
        return port


