# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check composed success-response schemas against shared Rust serialization fixtures.

Run with --spec pointing to a composed OpenAPI 3.2 export. Cases in
lib/llm/tests/fixtures/openapi/responses.json cover unary bodies and stream payloads;
the checks also remove required fields to catch overly permissive schemas.
Run lib/llm/tests/protocols/openapi_response_schema.rs at the same revision to
verify the fixtures still match serialization. This suite does not exercise a
server, error responses, or stream ordering.
"""

import argparse
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openapi_spec_validator import validate
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from scripts.protocol_compatibility.assessment.responses import (
    response_document,
    validate_sse_event,
)
from scripts.protocol_compatibility.composition.openai import MARKER


def check(spec: dict) -> int:
    """Validate actual Serde fixtures against unary schemas and decoded SSE data.

    `spec` must be the composed OpenAPI 3.2 export. Returns the number of fixtures;
    missing imports, unsupported success contracts or invalid payloads raise.
    This is success-payload fidelity, not error, ordering or real-server E2E coverage.
    """
    scoped = copy.deepcopy(spec)
    scoped["paths"] = {
        path: scoped["paths"][path]
        for path in ("/v1/chat/completions", "/v1/completions")
    }
    validate(scoped)
    if any(MARKER in schema for schema in spec["components"]["schemas"].values()):
        raise AssertionError("Unresolved dependency import")
    # These fixtures cover successful serialization only. Description-only error
    # responses remain coverage gaps in the full assessment, not in this suite.
    for path in scoped["paths"].values():
        path["post"]["responses"] = {"200": path["post"]["responses"]["200"]}
    projected, gaps = response_document(scoped)
    if gaps:
        raise AssertionError(gaps)
    resource = Resource.from_contents(projected, default_specification=DRAFT202012)
    registry = Registry().with_resource("urn:dynamo:response", resource)
    validators = {}
    for kind, endpoint, media in (
        ("chat", "/v1/chat/completions", "application/json"),
        ("stream", "/v1/chat/completions", "text/event-stream"),
        ("completion", "/v1/completions", "application/json"),
        ("completion_stream", "/v1/completions", "text/event-stream"),
    ):
        pointer = (
            "urn:dynamo:response#/paths/"
            + endpoint.replace("/", "~1")
            + "/post/responses/200/content/"
            + media.replace("/", "~1")
            + "/schema"
        )
        validators[kind] = Draft202012Validator({"$ref": pointer}, registry=registry)

    def payload(kind: str, output: dict) -> dict:
        """Wrap already-decoded Serde stream output in the comparison representation."""
        if kind in {"stream", "completion_stream"}:
            return {"data": {"encoding": "json", "value": output}}
        return output

    cases = json.loads(
        (
            Path(__file__).parents[4] / "lib/llm/tests/fixtures/openapi/responses.json"
        ).read_text()
    )["cases"]
    for case in cases:
        validator = validators[case["kind"]]
        validator.validate(payload(case["kind"], case["output"]))
        if case["kind"] in {"stream", "completion_stream", "completion"}:
            endpoint = (
                "/v1/chat/completions"
                if case["kind"] == "stream"
                else "/v1/completions"
            )
            item = spec["paths"][endpoint]["post"]["responses"]["200"]["content"][
                "text/event-stream"
            ]["itemSchema"]
            validate_sse_event(spec, item, {"data": json.dumps(case["output"])})
        if case["kind"] == "completion":
            validators["completion_stream"].validate(
                payload("completion_stream", case["output"])
            )
        # The schema must reject omission of the shared required envelope key.
        invalid = copy.deepcopy(case["output"])
        del invalid["id"]
        if validator.is_valid(payload(case["kind"], invalid)):
            raise AssertionError(case["name"])
    # Required-and-nullable is different from optional: these keys must exist.
    for case_index, pointer in (
        (0, ("choices", 0, "message", "refusal")),
        (0, ("choices", 0, "logprobs")),
        (1, ("choices", 0, "finish_reason")),
        (4, ("usage",)),
        (4, ("system_fingerprint",)),
    ):
        case = cases[case_index]
        invalid = copy.deepcopy(case["output"])
        parent = invalid
        for part in pointer[:-1]:
            parent = parent[part]
        del parent[pointer[-1]]
        if validators[case["kind"]].is_valid(payload(case["kind"], invalid)):
            raise AssertionError(pointer)
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Response schema/Serde fixtures passed: {check(json.loads(args.spec.read_text()))}"
    )
