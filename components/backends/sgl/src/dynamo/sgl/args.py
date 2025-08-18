# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import logging
import socket
from argparse import Namespace
from enum import Enum

from sglang.srt.server_args import ServerArgs

DEFAULT_ENDPOINT = "dyn://dynamo.backend.generate"
DYNAMO_ARGS = {
    "endpoint": {
        "flags": ["--endpoint"],
        "type": str,
        "default": DEFAULT_ENDPOINT,
        "help": f"Dynamo endpoint string in 'dyn://namespace.component.endpoint' format. Default: {DEFAULT_ENDPOINT}",
    },
    "migration-limit": {
        "flags": ["--migration-limit"],
        "type": int,
        "default": 0,
        "help": "Maximum number of times a request may be migrated to a different engine worker",
    },
}


class DynamoArgs:
    def __init__(self, endpoint: str, migration_limit: int) -> None:
        self.endpoint = endpoint
        self.migration_limit = migration_limit


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
        self.serving_mode = self._set_serving_strategy()

    def _set_serving_strategy(self):
        if self.server_args.disaggregation_mode == "null":
            return DisaggregationMode.AGGREGATED
        elif self.server_args.disaggregation_mode == "prefill":
            return DisaggregationMode.PREFILL
        elif self.server_args.disaggregation_mode == "decode":
            return DisaggregationMode.DECODE
        else:
            logging.error("Cannot ascertain serving strategy. This shouldn't happen")


def parse_args(args: list[str] = None) -> Config:
    """
    Parse all arguments and return Config with server_args and dynamo_args
    """
    parser = argparse.ArgumentParser()

    # Dynamo args
    for info in DYNAMO_ARGS.values():
        parser.add_argument(
            info["flags"],
            type=info["type"],
            default=info["default"],
            help=info["help"],
        )

    # SGLang args
    bootstrap_port = _reserve_disaggregation_bootstrap_port()
    ServerArgs.add_cli_args(parser)

    parsed_args = parser.parse_args(args)

    # Auto-set bootstrap port if not provided
    if not any(arg.startswith("--disaggregation-bootstrap-port") for arg in args):
        args_dict = vars(parsed_args)
        args_dict["disaggregation_bootstrap_port"] = bootstrap_port
        parsed_args = Namespace(**args_dict)

    # Create dynamo args from parsed values
    dynamo_args = DynamoArgs(
        endpoint=parsed_args.endpoint,
        migration_limit=parsed_args.migration_limit,
    )
    server_args = ServerArgs.from_cli_args(parsed_args)

    # Create config
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
