# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One bounded stdin/stdout transaction. Fixture modes never use network IO."""

import argparse
import asyncio
import sys

from .coordinator import Counter, Pipeline
from .fixtures import ExtractiveModels
from .protocol import (
    MAX_INPUT_BYTES,
    MAX_OUTPUT_BYTES,
    Invalid,
    Request,
    Result,
    canonical,
    require,
    strict_json,
)


async def process(raw: bytes, pipeline: Pipeline) -> dict:
    try:
        require(len(raw) <= MAX_INPUT_BYTES, "input_bytes_exceeded")
        request = Request.parse(strict_json(raw))
        result = await pipeline.run(request)
        encoded = canonical(result.wire()).encode()
        require(
            len(encoded) <= request.budget.max_output_bytes, "output_bytes_exceeded"
        )
        return result.wire()
    except (Invalid, RecursionError):
        return Result(
            "rejected", None, None, (), "invalid_or_oversized_request", pipeline.mode
        ).wire()


async def execute(raw: bytes, mode: str) -> dict:
    counter = Counter("fixture_bytes", lambda text: len(text.encode()), False)
    if mode == "model":
        return Result(
            "rejected", None, None, (), "unqualified_model_profile", "model"
        ).wire()
    if mode == "fixture":
        return await process(raw, Pipeline(ExtractiveModels(), counter))
    require(mode == "fixture-sdk", "invalid_mode")
    import httpx

    from .clients import Profile, SDKModels, make_client
    from .fixtures import sdk_response

    async with make_client(
        "http://fixture.invalid/v1", httpx.MockTransport(sdk_response)
    ) as client:
        profile = Profile(
            "cpu-fixture-decision",
            "cpu-fixture-generation",
            counter,
            MAX_INPUT_BYTES,
            0.9,
            0.9,
        )
        return await process(
            raw, Pipeline(SDKModels(client, profile, fixture=True), counter)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("fixture", "fixture-sdk", "model"), required=True
    )
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    try:
        result = asyncio.run(execute(raw, args.mode))
    except Exception:
        # Keep evidence and credential-bearing exceptions off the wire and stderr.
        result = Result(
            "rejected",
            None,
            None,
            (),
            "helper_failure",
            "fixture" if args.mode.startswith("fixture") else "model",
        ).wire()
    encoded = canonical(result).encode()
    if len(encoded) > MAX_OUTPUT_BYTES:
        encoded = canonical(
            Result(
                "rejected", None, None, (), "output_bytes_exceeded", "fixture"
            ).wire()
        ).encode()
    sys.stdout.buffer.write(encoded + b"\n")


if __name__ == "__main__":
    main()
