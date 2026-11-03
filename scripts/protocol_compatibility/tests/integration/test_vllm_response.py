# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the current recipe's selectors with synthetic, not captured, inputs.

These tests qualify recipe integration with the generic comparator. They do not
import vLLM or attest the schema produced by the pinned upstream source.
"""

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.protocol_compatibility.assessment.openapi import compare
from scripts.protocol_compatibility.assessment.responses import (
    prepare_response_source,
    response_differences,
    response_document,
)
from scripts.protocol_compatibility.inputs import write_json
from scripts.protocol_compatibility.tests.assessment.test_responses import (
    streaming_document,
)

PROFILE = Path(__file__).resolve().parents[2] / "frameworks/vllm-response-openapi.json"


def recipe_inputs():
    """Return synthetic 3.1/3.2 event documents and a hash-qualified recipe selection.

    Reuse the parent fixture's payloads, add optional envelope fields, and apply
    the profile's real selectors. Source labels identify the synthetic fixture,
    not an upstream capture; profile loading and parsing errors propagate.
    """
    native = streaming_document()
    for path in native["paths"].values():
        event = path["post"]["responses"]["200"]["content"]["text/event-stream"][
            "itemSchema"
        ]
        event["properties"].update(
            {
                "event": {"type": "string"},
                "id": {"type": "string"},
                "retry": {"type": "integer", "minimum": 0},
            }
        )
    framework = copy.deepcopy(native)
    framework["openapi"] = "3.1.0"
    for path in framework["paths"].values():
        media = path["post"]["responses"]["200"]["content"]["text/event-stream"]
        media["schema"] = media.pop("itemSchema")
    config = {
        "sha256": hashlib.sha256(json.dumps(framework).encode()).hexdigest(),
        "source": "synthetic-vllm-recipe-fixture",
        "revision": "fixture-v1",
        "streams": json.loads(PROFILE.read_text())["response_contract"]["streams"],
    }
    return native, framework, config


class VllmResponseRecipeTests(unittest.TestCase):
    def test_recipe_selects_both_envelopes_without_changing_raw_input(self):
        """The recipe selects both synthetic endpoints and preserves their contracts."""
        native, framework, config = recipe_inputs()
        before = copy.deepcopy(framework)
        selected, streams, provenance = prepare_response_source(
            framework, config, config["sha256"]
        )
        self.assertEqual(len(streams), 2)
        self.assertEqual(
            [entry["pointer"] for entry in provenance["streams"]],
            [entry["pointer"] for entry in config["streams"]],
        )
        self.assertEqual(
            response_document(selected, streams), response_document(native)
        )
        self.assertEqual(framework, before)

    def test_recipe_requires_explicit_selection_and_matching_bytes(self):
        """Absent selection reports gaps; a wrong input hash rejects the adapter."""
        _, framework, config = recipe_inputs()
        self.assertTrue(response_document(framework)[1])
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare_response_source(framework, config, "0" * 64)

    def test_real_comparator_keeps_chunk_and_envelope_changes(self):
        """Selecting equivalent representations must not erase real schema changes."""
        if not os.environ.get("OASDIFF_BIN"):
            self.skipTest("OASDIFF_BIN is required for real comparator validation")
        native, framework, config = recipe_inputs()
        selected, streams, _ = prepare_response_source(
            framework, config, config["sha256"]
        )
        before, gaps = response_document(selected, streams)
        self.assertFalse(gaps)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "before.json", before)
            for change in ("equivalent", "chunk", "envelope"):
                with self.subTest(change=change):
                    altered = copy.deepcopy(native)
                    if change == "chunk":
                        altered["components"]["schemas"]["Response"]["properties"][
                            "id"
                        ] = {"type": "integer"}
                    elif change == "envelope":
                        for path in altered["paths"].values():
                            path["post"]["responses"]["200"]["content"][
                                "text/event-stream"
                            ]["itemSchema"]["properties"]["retry"]["minimum"] = 2
                    after, gaps = response_document(altered)
                    self.assertFalse(gaps)
                    write_json(root / "after.json", after)
                    diff, _ = compare(
                        Path(os.environ["OASDIFF_BIN"]),
                        root / "before.json",
                        root / "after.json",
                        flatten=True,
                    )
                    self.assertEqual(
                        bool(response_differences(diff)), change != "equivalent"
                    )
