# SPDX-FileCopyrightText: Copyright (c) 2026 doubleword.ai
# SPDX-License-Identifier: MIT

"""Launch a local vLLM OpenAI-compatible engine and the Dynamo backend worker."""

import argparse
import shutil
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
            "Launch a local vLLM OpenAI-compatible engine and Dynamo worker "
            "together."
        )
    )
    return add_shared_launcher_args(parser)


def _engine_command(args: argparse.Namespace) -> list[str]:
    if shutil.which("vllm") is None:
        raise SystemExit(
            "vLLM launcher requested, but the 'vllm' executable is not present "
            "in this image."
        )

    served_model_name = args.served_model_name or args.model
    command = [
        "vllm",
        "serve",
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


def main(argv: Sequence[str] | None = None) -> None:
    configure_logging()
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    raise SystemExit(
        uvloop.run(
            run_launcher(
                engine_command=_engine_command(args),
                worker_command=build_worker_command(args),
                health_url=build_health_url(args),
            )
        )
    )


if __name__ == "__main__":
    main()
