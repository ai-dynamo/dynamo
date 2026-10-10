# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
import unittest

import yaml

from scripts.protocol_compatibility.composition.openai import MARKER, compose
from scripts.protocol_compatibility.composition.responses import (
    DIRECTORY,
    response_manifest,
)


class ResponseCompositionTests(unittest.TestCase):
    def test_response_command_uses_the_request_workflows_shared_manifest(self):
        base = yaml.safe_load((DIRECTORY / "async-openai-0.42.1.yaml").read_bytes())
        result = response_manifest()
        self.assertEqual(result, base)
        for name in (
            "FinishReason",
            "Role",
            "TopLogprobs",
            "CompletionUsage",
            "CreateCompletionResponse",
            "ChatCompletionResponseMessageAudio",
        ):
            self.assertIn(name, result["imports"])

    def test_chat_usage_and_legacy_usage_do_not_share_serialization_corrections(self):
        baseline = {
            "components": {
                "schemas": {
                    "Usage": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                    },
                    "Completion": {
                        "type": "object",
                        "properties": {"usage": {"$ref": "#/components/schemas/Usage"}},
                    },
                }
            }
        }
        data = json.dumps(baseline).encode()
        raw = {
            "paths": {
                "/v1/completions": {
                    "post": {
                        "responses": {
                            "200": {
                                "content": {
                                    "text/event-stream": {
                                        "schema": {
                                            "$ref": "#/components/schemas/Completion"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {
                "schemas": {
                    name: {MARKER: {"crate": "async-openai", "type": name}}
                    for name in ("ChatUsage", "Completion")
                }
            },
        }
        manifest = {
            "openai": {"sha256": hashlib.sha256(data).hexdigest()},
            "dependencies": {},
            "exclusions": [],
            "imports": {
                "ChatUsage": {
                    "pointer": "/components/schemas/Usage",
                    "redirect_references": False,
                },
                "Completion": {"pointer": "/components/schemas/Completion"},
            },
            "components": {
                "Usage": {
                    "rationale": "Legacy emits nullable count; chat omits it.",
                    "patch": [
                        {
                            "op": "test",
                            "path": "/properties/count/type",
                            "value": "integer",
                        },
                        {
                            "op": "replace",
                            "path": "/properties/count/type",
                            "value": ["integer", "null"],
                        },
                    ],
                }
            },
        }
        original = copy.deepcopy(raw)
        result = compose(raw, manifest, baseline_bytes=data)
        self.assertEqual(raw, original)
        self.assertEqual(result["paths"], raw["paths"])
        schemas = result["components"]["schemas"]
        self.assertEqual(schemas["ChatUsage"]["properties"]["count"]["type"], "integer")
        self.assertEqual(
            schemas["OpenAIBaseline.Usage"]["properties"]["count"]["type"],
            ["integer", "null"],
        )
