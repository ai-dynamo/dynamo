# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch a local SGLang OpenAI-compatible engine and the Dynamo backend worker."""

import argparse
import importlib.util
import shlex
import sys
from collections.abc import Sequence

import uvloop

from dynamo.openai_backend.launcher_common import (
    add_shared_launcher_args,
    build_health_url,
    build_worker_command,
    configure_logging,
    run_launcher,
    strip_remainder_separator,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch a local SGLang OpenAI-compatible engine and Dynamo worker "
            "together."
        )
    )
    add_shared_launcher_args(parser)
    parser.add_argument(
        "--router-port",
        type=int,
        default=None,
        help=(
            "Run an sglang-router between the worker and the engine on this "
            "port. The worker forwards to the router instead of the engine."
        ),
    )
    parser.add_argument(
        "--router-args",
        default="",
        help=(
            "Extra arguments for sglang_router.launch_router as a single "
            'shell-quoted string, e.g. "--dp-aware --policy manual '
            '--assignment-mode min_load".'
        ),
    )
    return parser


def _engine_command(args: argparse.Namespace) -> list[str]:
    if importlib.util.find_spec("sglang") is None:
        raise SystemExit(
            "SGLang launcher requested, but the 'sglang' Python package is not "
            "installed in this image."
        )

    served_model_name = args.served_model_name or args.model
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--served-model-name",
        served_model_name,
        "--host",
        args.engine_host,
        "--port",
        str(args.engine_port),
    ]
    command.extend(strip_remainder_separator(list(args.engine_args)))
    return command


def _router_command(args: argparse.Namespace) -> list[str] | None:
    if args.router_port is None:
        return None
    if importlib.util.find_spec("sglang_router") is None:
        raise SystemExit(
            "--router-port requested, but the 'sglang_router' Python package "
            "is not installed in this image."
        )

    command = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--host",
        args.engine_host,
        "--port",
        str(args.router_port),
        "--worker-urls",
        f"http://{args.engine_host}:{args.engine_port}",
    ]
    command.extend(shlex.split(args.router_args))
    return command


def _router_health_url(args: argparse.Namespace) -> str | None:
    if args.router_port is None:
        return None
    return f"http://{args.engine_host}:{args.router_port}/health"


def _worker_command(args: argparse.Namespace) -> list[str]:
    return build_worker_command(
        args,
        priority_multiplier=1,
        upstream_port=args.router_port,
    )


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    raise SystemExit(
        uvloop.run(
            run_launcher(
                engine_command=_engine_command(args),
                worker_command=_worker_command(args),
                health_url=build_health_url(args),
                router_command=_router_command(args),
                router_health_url=_router_health_url(args),
            )
        )
    )


if __name__ == "__main__":
    main()
