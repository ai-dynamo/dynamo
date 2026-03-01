# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo worker entry point."""

import importlib.util
import os
import sys

SUPPORTED_BACKENDS = ("myengine",)


def _get_single_backend_class(backends_mod) -> type:
    """Return the single Backend class from dynamo.worker.{backend_type}.backends."""
    from dynamo.common.backend import BaseBackend

    candidates = [
        v
        for v in vars(backends_mod).values()
        if isinstance(v, type) and issubclass(v, BaseBackend) and v is not BaseBackend
    ]
    if len(candidates) != 1:
        raise AttributeError(
            f"Expected exactly one BaseBackend subclass in {getattr(backends_mod, '__name__', backends_mod)}; found {len(candidates)}"
        )
    return candidates[0]


# Maps backend name to the Python package whose presence signals it.
_BACKEND_PACKAGE = {
    "trtllm": "tensorrt_llm",
    "vllm": "vllm",
    "sglang": "sglang",
}


def _detect_backend() -> str:
    """Auto-detect backend from installed packages.

    Checks for tensorrt_llm, vllm, and sglang (in that order).
    Returns the first match, or exits with an error if none are found.
    """
    for backend, package in _BACKEND_PACKAGE.items():
        if importlib.util.find_spec(package) is not None:
            print(
                f"Auto-detected backend '{backend}' (package '{package}' is installed)"
            )
            return backend

    print(
        "Error: Could not auto-detect backend. No supported package found "
        f"(checked: {', '.join(_BACKEND_PACKAGE.values())}). "
        f"Install one or pass --backend explicitly. Supported: {SUPPORTED_BACKENDS}",
        file=sys.stderr,
    )
    sys.exit(1)


def _peek_backend_type() -> str:
    """Extract --backend from sys.argv without consuming other args.

    Falls back to auto-detection if --backend is not provided.
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            val = sys.argv.pop(i + 1)
            sys.argv.pop(i)
            return val
        if arg.startswith("--backend="):
            val = arg.split("=", 1)[1]
            sys.argv.pop(i)
            return val

    return _detect_backend()


async def _run_worker(backend_type: str):
    if backend_type not in SUPPORTED_BACKENDS:
        print(
            f"Error: Unknown engine type '{backend_type}'. "
            f"Choose from: {SUPPORTED_BACKENDS}",
            file=sys.stderr,
        )
        sys.exit(1)

    parse_args = getattr(
        __import__(f"dynamo.worker.{backend_type}.args", fromlist=["parse_args"]),
        "parse_args",
    )
    backends_mod = __import__(
        f"dynamo.worker.{backend_type}.backends", fromlist=["backends"]
    )
    backend_cls = _get_single_backend_class(backends_mod)
    config = parse_args()

    # dump_config(config.dump_config_to, config)

    model_path = getattr(config, "model", None)
    if not config.served_model_name and model_path:
        config.served_model_name = model_path

    if model_path and not os.path.exists(model_path):
        from dynamo.llm import fetch_llm

        await fetch_llm(model_path)
    elif not model_path:
        raise ValueError("Please specify a model or model path using the --model.")

    await backend_cls(config=config).run()


def main():
    backend_type = _peek_backend_type()

    if backend_type not in SUPPORTED_BACKENDS:
        print(
            f"Error: Unknown engine type '{backend_type}'. "
            f"Choose from: {SUPPORTED_BACKENDS}",
            file=sys.stderr,
        )
        sys.exit(1)

    import uvloop

    from dynamo.runtime.logging import configure_dynamo_logging

    configure_dynamo_logging()
    uvloop.run(_run_worker(backend_type))


if __name__ == "__main__":
    main()
