# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check serial partial-prefix reuse through a running Dynamo HTTP frontend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass

PROMPT_TOKENS = 24_300
OUTPUT_TOKENS = 2


@dataclass
class ResponseSummary:
    phase: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    completion_token_ids: list[int]


def prompts(nonce: str, shared_prefix: int) -> tuple[list[int], list[int]]:
    """Use a distinct run prefix followed by a repeated synthetic token pattern."""
    digest = hashlib.shake_256(nonce.encode()).digest(256)
    run_prefix = [
        256 + int.from_bytes(digest[index : index + 2], "big") % 3840
        for index in range(0, len(digest), 2)
    ]
    pattern = list(range(512, 576))
    cold = (run_prefix + pattern * (PROMPT_TOKENS // len(pattern) + 1))[:PROMPT_TOKENS]
    warm = cold[:shared_prefix] + [
        256 + (token - 256 + 1) % 3840 for token in cold[shared_prefix:]
    ]
    return cold, warm


def complete(
    base_url: str,
    model: str,
    prompt: list[int],
    phase: str,
    expected_cached: int,
    timeout: float,
) -> ResponseSummary:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": OUTPUT_TOKENS,
        "ignore_eos": True,
        "stream": False,
        "n": 1,
        "nvext": {"extra_fields": ["completion_token_ids"]},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{phase}: HTTP {error.code}: {detail}") from error

    usage = result["usage"]
    summary = ResponseSummary(
        phase=phase,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        cached_tokens=usage["prompt_tokens_details"]["cached_tokens"],
        completion_token_ids=result["nvext"]["completion_token_ids"],
    )
    expected = (PROMPT_TOKENS, OUTPUT_TOKENS, expected_cached, OUTPUT_TOKENS)
    observed = (
        summary.prompt_tokens,
        summary.completion_tokens,
        summary.cached_tokens,
        len(summary.completion_token_ids),
    )
    if observed != expected:
        raise RuntimeError(
            f"{phase}: expected (prompt, output, cached, returned token IDs) "
            f"{expected}, got {observed}"
        )
    if result["choices"][0]["finish_reason"] != "length":
        raise RuntimeError(f"{phase}: unexpected finish reason: {result['choices']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--model", default=os.environ.get("MODEL_NAME", "state-cache-mocker")
    )
    parser.add_argument(
        "--shared-prefix",
        type=int,
        choices=[24192, 24191, 23700, 23040, 21504, 15360, 7680, 0],
        default=24192,
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Expect physical-block reuse without state cache",
    )
    parser.add_argument(
        "--nonce",
        default=None,
        help="Reproduce a token pattern; use a fresh worker when reusing a nonce",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    nonce = args.nonce if args.nonce is not None else uuid.uuid4().hex
    cold, warm = prompts(nonce, args.shared_prefix)
    if args.legacy:
        expected_cached = args.shared_prefix // 1536 * 1536
    elif args.shared_prefix < 23_040:
        expected_cached = 0
    elif args.shared_prefix < 24_192:
        expected_cached = 23_040
    else:
        expected_cached = 24_192
    # Await each complete HTTP response before submitting the next request.
    responses = [complete(args.base_url, args.model, cold, "cold", 0, args.timeout)]
    responses.append(
        complete(args.base_url, args.model, warm, "warm", expected_cached, args.timeout)
    )
    print(
        json.dumps(
            {
                "nonce": nonce,
                "state_cache": not args.legacy,
                "shared_prefix_tokens": args.shared_prefix,
                "requests": [asdict(response) for response in responses],
                "total_output_tokens": sum(
                    response.completion_tokens for response in responses
                ),
                "recomputed_prefill_tokens": sum(
                    response.prompt_tokens - response.cached_tokens
                    for response in responses
                ),
                "prefill_measurement": "inferred from HTTP usage, not measured committed work",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
