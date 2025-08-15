# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import contextlib
import logging
import socket
from argparse import Namespace

from sglang.srt.server_args import ServerArgs


class SkipTokenizerInitError(RuntimeError):
    def __init__(self):
        super().__init__("--skip-tokenizer-init flag is required")


def parse_sglang_args_inc(args: list[str]) -> ServerArgs:
    # Currently we only support Dynamo doing the tokenization, so we must give
    # sglang the skip-tokenizer-init flag. We don't default it because this is temporary.
    # Allow the --version and --help flags through.
    temp_need_tok = ["--skip-tokenizer-init", "--version", "--help", "-h"]
    if not any(w in args for w in temp_need_tok):
        raise SkipTokenizerInitError()

    parser = argparse.ArgumentParser()
    bootstrap_port = _reserve_disaggregation_bootstrap_port()
    ServerArgs.add_cli_args(parser)
    parsed_args = parser.parse_args(args)
    if not any(arg.startswith("--disaggregation-bootstrap-port") for arg in args):
        args_dict = vars(parsed_args)
        args_dict["disaggregation_bootstrap_port"] = bootstrap_port
        parsed_args = Namespace(**args_dict)
    return ServerArgs.from_cli_args(parsed_args)


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


async def graceful_shutdown(runtime):
    logging.info("Received shutdown signal, shutting down DistributedRuntime")
    runtime.shutdown()
    logging.info("DistributedRuntime shutdown complete")


def setup_native_endpoints(server_args, component, handler):
    """Setup sgl native endpoints"""
    # flush cache
    flush_endpoint = component.endpoint("flush_cache")
    tasks = []
    tasks.append(flush_endpoint.serve_endpoint(handler.flush_cache))

    # expert distribution endpoints
    if server_args.expert_distribution_recorder_mode is not None:
        start_expert_distribution_endpoint = component.endpoint(
            "start_expert_distribution_record"
        )
        stop_expert_distribution_endpoint = component.endpoint(
            "stop_expert_distribution_record"
        )
        dump_expert_distribution_endpoint = component.endpoint(
            "dump_expert_distribution_record"
        )

        tasks.append(
            start_expert_distribution_endpoint.serve_endpoint(
                handler.start_expert_distribution_record
            )
        )
        tasks.append(
            stop_expert_distribution_endpoint.serve_endpoint(
                handler.stop_expert_distribution_record
            )
        )
        tasks.append(
            dump_expert_distribution_endpoint.serve_endpoint(
                handler.dump_expert_distribution_record
            )
        )

    return tasks
