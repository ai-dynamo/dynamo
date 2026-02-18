# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker entry point."""

import os
import sys

SUPPORTED_ENGINES = ("myengine",)


def _peek_engine_type() -> str:
    """Extract --engine-type from sys.argv without consuming other args."""
    for i, arg in enumerate(sys.argv):
        if arg == "--engine-type" and i + 1 < len(sys.argv):
            val = sys.argv.pop(i + 1)
            sys.argv.pop(i)
            return val
        if arg.startswith("--engine-type="):
            val = arg.split("=", 1)[1]
            sys.argv.pop(i)
            return val

    print(
        f"Error: --engine-type is required. Choose from: {SUPPORTED_ENGINES}",
        file=sys.stderr,
    )
    sys.exit(1)


async def _run_worker(engine_type: str):
    from dynamo.common.config_dump import dump_config

    if engine_type == "myengine":
        from dynamo.worker.myengine.args import parse_args
        from dynamo.worker.myengine.backends import MyEngineBackend

        config = parse_args()
        backend_cls = MyEngineBackend
    else:
        print(
            f"Error: Unknown engine type '{engine_type}'. "
            f"Choose from: {SUPPORTED_ENGINES}",
            file=sys.stderr,
        )
        sys.exit(1)

    dump_config(config.dump_config_to, config)

    model_path = getattr(config, "model", None)
    if not config.served_model_name and model_path:
        config.served_model_name = model_path

    if model_path and not os.path.exists(model_path):
        from dynamo.llm import fetch_llm
        await fetch_llm(model_path)

    await backend_cls(config=config).run()


def main():
    engine_type = _peek_engine_type()

    if engine_type not in SUPPORTED_ENGINES:
        print(
            f"Error: Unknown engine type '{engine_type}'. "
            f"Choose from: {SUPPORTED_ENGINES}",
            file=sys.stderr,
        )
        sys.exit(1)

    import uvloop

    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging()
    uvloop.run(_run_worker(engine_type))


if __name__ == "__main__":
    main()
