# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One bounded stdin/stdout transaction. Fixture modes never use network IO."""

import argparse
import asyncio
import os
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


async def execute_model(raw, args):
    import httpx

    from .clients import Profile, SDKModels, make_client
    from .tokenization import PROFILE_MODEL, load_counter

    require(len(raw) <= MAX_INPUT_BYTES, "input_bytes_exceeded")
    require(
        args.tokenizer_path
        and args.base_url
        and args.decision_model
        and args.generation_model
        and args.max_context_tokens
        and args.min_probability is not None
        and args.min_confidence is not None,
        "incomplete_model_profile",
    )
    require(
        args.decision_model == PROFILE_MODEL
        and args.generation_model == PROFILE_MODEL
        and args.max_context_tokens == 8192,
        "unqualified_model_profile",
    )
    # The POC uses one pinned tokenizer/template for both served aliases.
    rendered = load_counter(args.tokenizer_path)
    counter = Counter(rendered.name, rendered, True)
    profile = Profile(
        args.decision_model,
        args.generation_model,
        counter,
        args.max_context_tokens,
        args.min_probability,
        args.min_confidence,
        generation_max_tokens=args.generation_max_tokens,
        usage_counter=rendered.usage,
    )
    credential = os.environ.get("DYNAMO_COMPACTION_API_KEY")
    async with make_client(
        args.base_url, httpx.AsyncHTTPTransport(retries=0), api_key=credential
    ) as client:
        return await process(
            raw,
            Pipeline(
                SDKModels(client, profile),
                counter,
                mode="model",
                minimum_selection_score=min(
                    profile.min_probability, profile.min_confidence
                ),
            ),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("fixture", "fixture-sdk", "model", "count"), required=True
    )
    parser.add_argument("--base-url")
    parser.add_argument("--decision-model")
    parser.add_argument("--generation-model")
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--generation-max-tokens", type=int, default=512)
    parser.add_argument("--min-probability", type=float)
    parser.add_argument("--min-confidence", type=float)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    try:
        if args.mode == "count":
            from .tokenization import count_request, load_counter

            require(
                len(raw) <= MAX_INPUT_BYTES and bool(args.tokenizer_path),
                "invalid_count_input",
            )
            result = count_request(strict_json(raw), load_counter(args.tokenizer_path))
        elif args.mode == "model":
            result = asyncio.run(execute_model(raw, args))
        else:
            result = asyncio.run(execute(raw, args.mode))
    except Invalid as error:
        result = Result(
            "rejected",
            None,
            None,
            (),
            str(error),
            "fixture" if args.mode.startswith("fixture") else "model",
        ).wire()
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
