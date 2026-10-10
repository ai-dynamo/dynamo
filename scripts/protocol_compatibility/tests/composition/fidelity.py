# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protect composed request-schema acceptance against shared Rust fixtures.

The companion lib/llm/tests/protocols/openapi_request_fidelity.rs protects Rust
deserialization acceptance/rejection according to `valid`. This suite checks the
same lib/llm/tests/fixtures/openapi/requests.json cases against `schema_valid`,
defaulting to `valid`, with documented exceptions in `gap`.
Both suites must pass against the same source revision to establish agreement
for aligned examples. Passing known-gap cases reproduces documented disagreements,
not conformance. Neither suite proves HTTP admission, backend support, inference,
or Dynamo/framework parity.

Run with --spec pointing to a composed HTTP export. This is an explicit
integration check, not a silently skipped unit test when an export is missing.
"""

import argparse
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
from openapi_spec_validator import validate
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012


def check(spec: dict) -> tuple[int, int]:
    # Other endpoints are retained in the composed artifact but are not part of
    # this check. The current raw server export omits path-parameter declarations
    # on unrelated batch routes; do not present scoped validation as a full-API pass.
    scoped = copy.deepcopy(spec)
    scoped["paths"] = {
        path: spec["paths"][path]
        for path in ("/v1/chat/completions", "/v1/completions")
    }
    validate(scoped)
    resource = Resource.from_contents(spec, default_specification=DRAFT202012)
    registry = Registry().with_resource("urn:dynamo:openapi", resource)
    validators = {}
    for name, endpoint in (
        ("chat", "/v1/chat/completions"),
        ("completion", "/v1/completions"),
    ):
        validators[name] = Draft202012Validator(
            {
                "$ref": "urn:dynamo:openapi#/paths/"
                + endpoint.replace("/", "~1")
                + "/post/requestBody/content/application~1json/schema"
            },
            registry=registry,
        )
    cases = json.loads(
        (
            Path(__file__).parents[4] / "lib/llm/tests/fixtures/openapi/requests.json"
        ).read_text()
    )["cases"]
    failures = []
    known_gaps = 0
    for case in cases:
        errors = list(validators[case["endpoint"]].iter_errors(case["body"]))
        expected = case.get("schema_valid", case["valid"])
        if expected != case["valid"]:
            if not case.get("gap"):
                raise AssertionError(
                    f"Unexplained schema/Serde disagreement: {case['name']}"
                )
            known_gaps += 1
        if (not errors) != expected:
            failures.append(
                f"{case['name']}: expected schema_valid={expected}; "
                + "; ".join(error.message for error in errors)
            )
    if failures:
        raise AssertionError("\n".join(failures))
    return len(cases) - known_gaps, known_gaps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    aligned, gaps = check(json.loads(args.spec.read_text()))
    print(
        f"Schema fidelity: {aligned} aligned cases passed; {gaps} known-gap witnesses reproduced"
    )
