# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise SGLang capture guards and the shared decoded-event projection on CPU."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.protocol_compatibility.assessment.responses import (
    prepare_response_source,
    response_document,
)
from scripts.protocol_compatibility.frameworks import capture_sglang_openapi as capture

PROFILE = (
    Path(__file__).resolve().parents[2] / "frameworks/sglang-response-openapi.json"
)


def fixture():
    """Return a minimal declaration fixture with SGLang's inline error wrapper.

    This synthetic fixture isolates capture guards; real acquisition is validated
    separately against the pinned production app, never replaced by this fixture.
    """
    document = {
        "openapi": "3.1.0",
        "info": {"title": "fixture", "version": "1"},
        "paths": {},
        "components": {"schemas": {"ErrorResponse": {"type": "object"}}},
    }
    for endpoint, prefix in (
        ("/v1/chat/completions", "ChatCompletion"),
        ("/v1/completions", "Completion"),
    ):
        for suffix in ("Response", "StreamResponse"):
            document["components"]["schemas"][prefix + suffix] = {"type": "object"}
        document["paths"][endpoint] = {
            "post": {
                "responses": {
                    "200": {
                        "description": "ok",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "$ref": f"#/components/schemas/{prefix}Response"
                                }
                            },
                            "text/event-stream": {
                                "schema": {
                                    "type": "object",
                                    "required": ["data"],
                                    "properties": {
                                        "data": {
                                            "type": "string",
                                            "anyOf": [
                                                {"const": "[DONE]"},
                                                {
                                                    "contentMediaType": "application/json",
                                                    "contentSchema": {
                                                        "$ref": f"#/components/schemas/{prefix}StreamResponse"
                                                    },
                                                },
                                                {
                                                    "contentMediaType": "application/json",
                                                    "contentSchema": {
                                                        "type": "object",
                                                        "required": ["error"],
                                                        "properties": {
                                                            "error": {
                                                                "$ref": "#/components/schemas/ErrorResponse"
                                                            }
                                                        },
                                                    },
                                                },
                                            ],
                                        },
                                        "event": {"type": "string"},
                                        "id": {"type": "string"},
                                        "retry": {"type": "integer", "minimum": 0},
                                    },
                                }
                            },
                        },
                    }
                }
            }
        }
    return document


class SGLangCaptureTests(unittest.TestCase):
    def setUp(self):
        """Load the maintained profile and a fresh isolated declaration fixture."""
        self.profile = json.loads(PROFILE.read_text())
        self.document = fixture()

    def test_qualified_inline_error_survives_shared_projection(self):
        """Bind exact bytes and preserve nested error, terminator and both endpoints."""
        raw = json.dumps(self.document).encode()
        adapter = capture.make_adapter(raw, self.profile)
        selected = adapter["sources"]["framework"]
        self.assertEqual(selected["sha256"], hashlib.sha256(raw).hexdigest())
        prepared, streams, _ = prepare_response_source(
            self.document, selected, selected["sha256"]
        )
        projected, gaps = response_document(prepared, streams)
        self.assertFalse(gaps)
        self.assertEqual(set(projected["paths"]), set(self.document["paths"]))
        rendered = json.dumps(projected)
        self.assertIn("[DONE]", rendered)
        self.assertIn("error", rendered)
        self.assertIn("ErrorResponse", rendered)

    def test_broken_external_and_missing_references_fail(self):
        """Reject unresolved and remote refs, including nested contentSchema refs."""
        for reference in (
            "#/components/schemas/Missing",
            "https://example.invalid/schema",
        ):
            changed = copy.deepcopy(self.document)
            changed["components"]["schemas"]["ErrorResponse"] = {"$ref": reference}
            with self.subTest(reference=reference), self.assertRaises(
                (ValueError, KeyError)
            ):
                capture.make_adapter(json.dumps(changed).encode(), self.profile)

    def test_literal_reference_values_are_not_references(self):
        """Examples/defaults and map keys named $ref must remain ordinary data."""
        schema = self.document["components"]["schemas"]["ErrorResponse"]
        schema.update(
            default={"$ref": "literal"},
            example={"$ref": "literal"},
            properties={"$ref": {"type": "string"}},
        )
        capture.make_adapter(json.dumps(self.document).encode(), self.profile)

    def test_wrong_dialect_selector_and_flat_error_fail(self):
        """Reject vLLM-shaped flat error refs and incompatible profile changes."""
        for change in ("dialect", "selector", "error"):
            document, profile = copy.deepcopy(self.document), copy.deepcopy(
                self.profile
            )
            if change == "dialect":
                document["openapi"] = "3.2.0"
            elif change == "selector":
                profile["streams"][0]["interpretation"] = "decoded-chunk"
            else:
                event = capture.resolve(document, profile["streams"][0]["pointer"])
                event["properties"]["data"]["anyOf"][2]["contentSchema"] = {
                    "$ref": "#/components/schemas/ErrorResponse"
                }
            with self.subTest(change=change), self.assertRaises(ValueError):
                capture.make_adapter(json.dumps(document).encode(), profile)

    def test_source_revision_and_cleanliness_are_required(self):
        """Reject drift before import; a clean exact head passes the source gate."""
        with patch.object(
            capture, "git", side_effect=[self.profile["source"]["revision"], ""]
        ):
            capture.verify_source(Path("unused"), self.profile)
        for replies in (["wrong"], [self.profile["source"]["revision"], " M changed"]):
            with patch.object(capture, "git", side_effect=replies), self.assertRaises(
                ValueError
            ):
                capture.verify_source(Path("unused"), self.profile)

    def test_existing_output_is_never_overwritten(self):
        """The standalone CLI must reject an existing directory before source import."""
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "sentinel"
            sentinel.write_text("preserve")
            result = subprocess.run(
                [
                    sys.executable,
                    capture.__file__,
                    "--source",
                    directory,
                    "--profile",
                    str(PROFILE),
                    "--output-dir",
                    directory,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileExistsError", result.stderr)
            self.assertEqual(sentinel.read_text(), "preserve")
