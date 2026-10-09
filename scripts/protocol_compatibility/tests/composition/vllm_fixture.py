# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract small, exact field schemas from the pinned vLLM HTTP capture.

The result is regression input, not a complete API contract. The full capture
stays in validation evidence. Run with --check to verify the committed fixture.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jsonpointer

PACKAGE = Path(__file__).resolve().parents[2]
PROFILE = PACKAGE / "frameworks/vllm-0.30.0.json"
FIXTURE = Path(__file__).with_name("vllm-0.30.0-fields.json")
POINTERS = {
    "chat_template_kwargs": "/components/schemas/ChatCompletionRequest/properties/chat_template_kwargs",
    "reasoning": "/components/schemas/CustomChatCompletionMessageParam/properties/reasoning",
    "chat_user": "/components/schemas/ChatCompletionRequest/properties/user",
    "completion_user": "/components/schemas/CompletionRequest/properties/user",
    "suffix": "/components/schemas/CompletionRequest/properties/suffix",
}


def extract(raw: bytes) -> dict:
    profile = json.loads(PROFILE.read_text())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != profile["reference_capture_sha256"]:
        raise ValueError("Capture checksum differs from the reviewed vLLM profile")
    document = json.loads(raw)
    return {
        "_generated": "Do not edit; regenerate with tests.composition.vllm_fixture",
        "scope": "Selected field schemas only; not a complete request contract",
        "image": profile["image"],
        "source_revision": profile["provenance"]["source_revision"],
        "capture_sha256": digest,
        "fields": {
            name: {
                "pointer": pointer,
                "schema": jsonpointer.resolve_pointer(document, pointer),
            }
            for name, pointer in POINTERS.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Retained full HTTP capture"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = (
        json.dumps(extract(args.source.read_bytes()), indent=2, sort_keys=True) + "\n"
    )
    if args.check:
        if FIXTURE.read_text() != rendered:
            parser.error(
                "Committed vLLM fixture is stale; regenerate from the reviewed capture"
            )
    else:
        FIXTURE.write_text(rendered)
