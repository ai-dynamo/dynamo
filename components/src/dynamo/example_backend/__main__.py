# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for running the example backend: python -m dynamo.example_backend"""

import os


async def _run() -> None:
    from dynamo.example_backend.args import parse_args
    from dynamo.example_backend.backends import ExampleBackend

    config = parse_args()

    model_path = getattr(config, "model", None)
    if not config.served_model_name and model_path:
        config.served_model_name = model_path

    if model_path and not os.path.exists(model_path):
        from dynamo.llm import fetch_llm

        await fetch_llm(model_path)
    elif not model_path:
        raise ValueError("Please specify a model or model path using --model.")

    await ExampleBackend(config=config).run()


def main() -> None:
    import uvloop

    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging()
    uvloop.run(_run())


if __name__ == "__main__":
    main()
