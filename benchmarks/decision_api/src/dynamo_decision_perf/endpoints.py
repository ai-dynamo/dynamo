# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIPerf 0.13 adapters for structured, non-generative decision evaluation."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from aiperf.common.models import BaseResponseData, ErrorDetails, ParsedResponse
from aiperf.endpoints.base_endpoint import BaseEndpoint

from .contracts import (
    DecisionContractError,
    require,
    validate_request,
    validate_response,
)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def decode_json(value: str | bytes) -> dict:
    try:
        result = json.loads(value, object_pairs_hook=_pairs)
    except (TypeError, ValueError) as error:
        raise DecisionContractError("Invalid decision JSON") from error
    require(isinstance(result, dict), "Decision payload must be an object")
    return result


def payload_hash(body: dict) -> str:
    """Hash immutable input semantics, preserving meaningful question-map order."""
    return hashlib.sha256(
        json.dumps(
            body, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def request_body(request_info) -> dict:
    require(
        request_info is not None, "Missing request context for response correlation"
    )
    if request_info.payload_bytes is not None:
        return decode_json(request_info.payload_bytes)
    turns = getattr(request_info, "turns", [])
    require(
        len(turns) == 1 and turns[0].raw_payload is not None,
        "Exactly one raw decision payload is required",
    )
    return deepcopy(turns[0].raw_payload)


@dataclass(slots=True)
class DecisionResponseData(BaseResponseData):
    body: dict[str, Any]
    dialect: str


class DecisionEndpoint(BaseEndpoint):
    dialect: str

    def format_payload(self, request_info):
        body = request_body(request_info)
        validate_request(body, self.dialect)
        return body

    def get_endpoint_headers(self, request_info):
        # AIPerf's raw-payload and mmap paths bypass format_payload, but call
        # this hook first. Validate and customize only this request's copy.
        body = self.format_payload(request_info)
        request_id = request_info.x_request_id
        require(
            isinstance(request_id, str) and bool(request_id),
            "Missing measurement request ID",
        )
        headers = {
            key: value
            for key, value in super().get_endpoint_headers(request_info).items()
            if key.lower()
            not in (
                "x-request-id",
                "x-decision-measurement-id",
                "x-decision-payload-sha256",
            )
        }
        headers.update(
            {
                "X-Decision-Measurement-ID": request_id,
                "X-Decision-Payload-SHA256": payload_hash(body),
            }
        )
        if self.dialect == "native_score":
            body = {
                **body,
                "cache_salt": "decision-native-"
                + hashlib.sha256(request_id.encode()).hexdigest(),
            }
        request_info.payload_bytes = json.dumps(
            body, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        return headers

    def parse_response(self, response):
        text = response.get_text()
        if (
            self.dialect == "native_score"
            and text is not None
            and text.strip() == "[DONE]"
        ):
            return None
        body = decode_json(text)
        if self.dialect == "native_score" and "error" not in body:
            meta = body.get("meta_info")
            if isinstance(meta, dict) and meta.get("finish_reason") is None:
                require(
                    body.get("output_ids", []) == []
                    and meta.get("completion_tokens", 0) == 0,
                    "Native scoring emitted generated tokens before termination",
                )
                return None
        summary = validate_response(body, self.dialect)
        if self.dialect == "native_score":
            usage = {
                "prompt_tokens": summary.input_tokens,
                "completion_tokens": summary.output_tokens,
            }
            if summary.cached_tokens is not None:
                usage["prompt_tokens_details"] = {
                    "cached_tokens": summary.cached_tokens
                }
        else:
            usage = deepcopy(body.get("usage"))
        return ParsedResponse(
            perf_ns=response.perf_ns,
            data=DecisionResponseData(body=body, dialect=self.dialect),
            usage=usage,
            metadata={"decision_dialect": self.dialect, **asdict(summary)},
        )

    def extract_response_data(self, record):
        if getattr(record, "error", None) is not None:
            record._parsed_responses_cache = []
            return []
        if record._parsed_responses_cache is not None:
            return record._parsed_responses_cache
        try:
            return self._validated_response_data(record)
        except DecisionContractError as error:
            # Keep the real transport record; raising here makes the worker
            # replace it with a synthetic failure without wire evidence.
            record.error = ErrorDetails.from_exception(error)
            record._parsed_responses_cache = []
            return []

    def _validated_response_data(self, record):
        request = request_body(record.request_info)
        parsed = [
            item
            for response in record.responses
            if (item := self.parse_response(response)) is not None
        ]
        require(len(parsed) == 1, "Expected exactly one complete decision result")
        summary = validate_response(parsed[0].data.body, self.dialect, request)
        request_id = record.request_info.x_request_id
        server_request_id = None
        if self.dialect != "native_score":
            trace = getattr(record, "trace_data", None)
            headers = {
                key.lower(): value
                for key, value in (
                    getattr(trace, "response_headers", None) or {}
                ).items()
            }
            server_request_id = headers.get("x-request-id")
            require(
                isinstance(server_request_id, str)
                and bool(server_request_id.strip())
                and headers.get("x-typesafe-request-id") == server_request_id,
                "Missing or mismatched decision request-ID response headers",
            )
        parsed[0].metadata.update(
            {
                **asdict(summary),
                "request_id": request_id,
                "server_request_id": server_request_id,
            }
        )
        record._parsed_responses_cache = parsed
        return parsed

    def build_assistant_turn(self, record):
        return None


class SystemOneEndpoint(DecisionEndpoint):
    dialect = "systemone"


class OpenAIDecisionEndpoint(DecisionEndpoint):
    dialect = "oai"


class NativeScoreEndpoint(DecisionEndpoint):
    dialect = "native_score"
