# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only extractive fixtures; not learned selection or quality evidence."""

from .protocol import canonical, strict_json


class ExtractiveModels:
    async def select(self, units, context, session_id):
        return {
            "selections": [
                {"id": unit.id, "action": "summarize", "score": 1.0} for unit in units
            ]
        }

    async def compact(self, units, context, session_id):
        return {
            "claims": [{"text": unit.text, "source_ids": [unit.id]} for unit in units]
        }

    async def verify(self, units, claims, context, session_id):
        known = {unit.id: unit.text for unit in units}
        unsupported = [
            index
            for index, claim in enumerate(claims)
            if len(claim["source_ids"]) != 1
            or claim["text"] != known.get(claim["source_ids"][0])
        ]
        return {"supported": not unsupported, "unsupported_claims": unsupported}


def sdk_response(request):
    import httpx

    payload = strict_json(request.content)
    if request.url.path == "/v1/decisions":
        body = {
            "model": payload["model"],
            "answers": [
                {
                    "type": "choice",
                    "name": question["name"],
                    "choice": "summarize",
                    "confidence": 1.0,
                    "probabilities": [
                        {"value": "keep", "probability": 0.0},
                        {"value": "summarize", "probability": 1.0},
                    ],
                }
                for question in payload["questions"]
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 0,
                "total_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        return httpx.Response(
            200,
            json=body,
            headers={
                "x-request-id": "cpu-fixture",
                "x-typesafe-request-id": "cpu-fixture",
            },
        )
    if request.url.path != "/v1/chat/completions":
        return httpx.Response(404)
    data = strict_json(payload["messages"][1]["content"])
    if "claims" in data:
        known = {unit["id"]: unit["text"] for unit in data["selected"]}
        unsupported = [
            index
            for index, claim in enumerate(data["claims"])
            if len(claim["source_ids"]) != 1
            or claim["text"] != known.get(claim["source_ids"][0])
        ]
        result = {"supported": not unsupported, "unsupported_claims": unsupported}
    else:
        result = {
            "claims": [
                {"text": unit["text"], "source_ids": [unit["id"]]}
                for unit in data["selected"]
            ]
        }
    return httpx.Response(
        200,
        json={
            "id": "cpu-fixture",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": canonical(result),
                        "refusal": None,
                    },
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
