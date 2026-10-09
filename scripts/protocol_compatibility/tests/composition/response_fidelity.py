# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the composed HTTP response schemas against real-Serde fixtures."""

import argparse
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openapi_spec_validator import validate
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from scripts.protocol_compatibility.composition.openai import MARKER


def check(spec: dict) -> int:
    scoped = copy.deepcopy(spec)
    scoped["paths"] = {
        path: spec["paths"][path]
        for path in ("/v1/chat/completions", "/v1/completions")
    }
    validate(scoped)
    assert not any(
        MARKER in schema for schema in spec["components"]["schemas"].values()
    ), "Unresolved dependency import"
    resource = Resource.from_contents(spec, default_specification=DRAFT202012)
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
    cases = json.loads(
        (
            Path(__file__).parents[4] / "lib/llm/tests/fixtures/openapi/responses.json"
        ).read_text()
    )["cases"]
    for case in cases:
        validator = validators[case["kind"]]
        validator.validate(case["output"])
        if case["kind"] == "completion":
            validators["completion_stream"].validate(case["output"])
        # The schema must reject omission of the shared required envelope key.
        invalid = copy.deepcopy(case["output"])
        del invalid["id"]
        assert not validator.is_valid(invalid), case["name"]
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
        assert not validators[case["kind"]].is_valid(invalid), pointer
    return len(cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Response schema/Serde fixtures passed: {check(json.loads(args.spec.read_text()))}"
    )
