# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM-specific regression fragments exercised through the generic comparator.

Only framework fragments come from the HTTP capture. The Dynamo-shaped fixture
is synthetic; runtime alias fidelity is covered by the base PR's Rust tests.
"""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.protocol_compatibility.acquisition.http import write_json
from scripts.protocol_compatibility.assessment.openapi import (
    alias_value_document,
    compare,
    match_input_aliases,
    normalize_nullable_strings,
    request_differences,
)
from scripts.protocol_compatibility.tests.composition.vllm_fixture import (
    FIXTURE,
    PROFILE,
    extract,
)


def documents():
    fields = json.loads(FIXTURE.read_text())["fields"]
    native = {
        "openapi": "3.1.0",
        "info": {"title": "Reduced vLLM fixture", "version": "0.30.0"},
        "paths": {},
    }
    chat = {
        "type": "object",
        "properties": {
            "chat_template_kwargs": fields["chat_template_kwargs"]["schema"],
            "user": fields["chat_user"]["schema"],
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reasoning": fields["reasoning"]["schema"],
                    },
                },
            },
        },
    }
    completion = {
        "type": "object",
        "properties": {
            "user": fields["completion_user"]["schema"],
            "suffix": fields["suffix"]["schema"],
        },
    }
    for endpoint, schema in (
        ("/v1/chat/completions", chat),
        ("/v1/completions", completion),
    ):
        native["paths"][endpoint] = {
            "post": {
                "requestBody": {"content": {"application/json": {"schema": schema}}},
                "responses": {"200": {"description": "Outside scope"}},
            }
        }
    own = copy.deepcopy(native)
    properties = own["paths"]["/v1/chat/completions"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    for container, alias, canonical in (
        (properties, "chat_template_kwargs", "chat_template_args"),
        (
            properties["messages"]["items"]["properties"],
            "reasoning",
            "reasoning_content",
        ),
    ):
        container[canonical] = container.pop(alias)
        container[canonical].update({"x-dynamo-input-aliases": [alias]})
    own, _ = normalize_nullable_strings(own)
    return own, native


class VllmCaptureTests(unittest.TestCase):
    def test_fixture_provenance_and_capture_guard(self):
        fixture, profile = json.loads(FIXTURE.read_text()), json.loads(
            PROFILE.read_text()
        )
        self.assertEqual(fixture["capture_sha256"], profile["reference_capture_sha256"])
        self.assertEqual(
            fixture["source_revision"], profile["provenance"]["source_revision"]
        )
        self.assertEqual(fixture["image"], profile["image"])
        with self.assertRaisesRegex(ValueError, "checksum"):
            extract(b"{}")

    def test_vllm_spellings_match_without_whole_contract_claim(self):
        own, native = documents()
        matches, gaps = match_input_aliases(own, native)
        self.assertFalse(gaps)
        self.assertEqual(
            {m["backend_name"] for m in matches}, {"reasoning", "chat_template_kwargs"}
        )
        self.assertTrue(
            all(m["compatibility"] == "not_established_by_name_match" for m in matches)
        )
        _, rewrites = normalize_nullable_strings(native)
        self.assertEqual(len(rewrites), 4)

    def test_real_comparator_keeps_value_changes_after_alias_match(self):
        if not os.environ.get("OASDIFF_BIN"):
            self.skipTest("OASDIFF_BIN is required for real comparator validation")
        own, native = documents()
        native, _ = normalize_nullable_strings(native)
        matches, _ = match_input_aliases(own, native)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for match in matches:
                before = alias_value_document(native, match["backend_location"])
                after = alias_value_document(own, match["dynamo_location"])
                write_json(root / "native.json", before)
                write_json(root / "own.json", after)
                diff, _ = compare(
                    Path(os.environ["OASDIFF_BIN"]),
                    root / "native.json",
                    root / "own.json",
                    flatten=True,
                )
                self.assertEqual(request_differences(diff), [])
                for operation in after["paths"].values():
                    operation["post"]["requestBody"]["content"]["application/json"][
                        "schema"
                    ] = {"type": "integer"}
                write_json(root / "own.json", after)
                diff, _ = compare(
                    Path(os.environ["OASDIFF_BIN"]),
                    root / "native.json",
                    root / "own.json",
                    flatten=True,
                )
                self.assertTrue(request_differences(diff))
