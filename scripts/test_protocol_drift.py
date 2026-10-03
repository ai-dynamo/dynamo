# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hermetic declaration-drift tests; run with unittest discover -s scripts."""

import unittest
from pathlib import Path

from protocol_drift import changes, commit, compare, module_contract, vllm_source


class ProtocolDriftTests(unittest.TestCase):
    def test_formatting_comments_and_docstrings_do_not_create_drift(self):
        before = 'class Request(Base):\n    """Old prose."""\n    top_k: int = Field(0, description="old")\n'
        after = 'class Request( Base ):\n    # new comment\n    """New prose."""\n    top_k : int = Field( 0, description="new")\n'
        self.assertEqual(module_contract(before), module_contract(after))

    def test_type_default_and_requiredness_are_distinct(self):
        variants = [
            "class Request(Base):\n    value: int\n",
            "class Request(Base):\n    value: int = None\n",
            "class Request(Base):\n    value: int = 0\n",
            "class Request(Base):\n    value: bool = False\n",
        ]
        for index, first in enumerate(variants):
            for second in variants[index + 1 :]:
                self.assertTrue(
                    changes(module_contract(first), module_contract(second))
                )

    def test_alias_constraints_and_factory_are_contract(self):
        for keyword in ["alias='old'", "ge=0", "default_factory=list", "exclude=True"]:
            before = "class Request(Base):\n    field: int = Field()\n"
            after = f"class Request(Base):\n    field: int = Field({keyword})\n"
            self.assertTrue(changes(module_contract(before), module_contract(after)))

    def test_validator_and_stream_behavior_changes_require_review(self):
        before = "class Request(Base):\n    def validate(self):\n        return self.value > 0\n"
        after = before.replace("> 0", ">= 0")
        delta = changes(module_contract(before), module_contract(after))
        self.assertEqual(
            delta[0]["path"], ["classes", "Request", "methods", "validate"]
        )
        before = "async def stream():\n    yield {'prompt_logprobs': None}\n"
        after = "async def stream():\n    yield {}\n"
        self.assertTrue(changes(module_contract(before), module_contract(after)))

    def test_add_remove_response_field_and_preserve_null(self):
        before = module_contract("class Response(Base):\n    id: str\n")
        after = module_contract(
            "class Response(Base):\n    id: str\n    prompt_logprobs: list | None = None\n"
        )
        added = changes(before, after)[0]
        self.assertEqual(added["kind"], "added")
        self.assertEqual(added["after"]["default"], "None")
        self.assertEqual(changes(after, before)[0]["kind"], "removed")
        self.assertEqual(changes({}, {"value": None})[0]["kind"], "added")

    def test_inheritance_alias_and_import_changes_are_visible(self):
        for before, after in [
            ("class Request(Base): pass", "class Request(OtherBase): pass"),
            ("Choice = Literal['auto']", "Choice = Literal['auto', 'none']"),
            ("from old import Base", "from new import Base"),
        ]:
            self.assertTrue(changes(module_contract(before), module_contract(after)))

    def test_source_hash_only_is_not_drift(self):
        def inventory(source_hash, contract):
            return {
                "target": "vllm",
                "modules": {
                    "protocol.py": {"source_sha256": source_hash, "contract": contract}
                },
            }

        self.assertEqual(compare(inventory("old", {}), inventory("new", {})), [])
        delta = compare(inventory("old", {}), inventory("new", {"field": "int"}))
        self.assertEqual(delta[0]["status"], "Unverified")
        self.assertEqual(delta[0]["runtime_evidence"], [])

    def test_source_selection_includes_both_upstream_layouts(self):
        for path in [
            "vllm/entrypoints/openai/protocol.py",
            "vllm/entrypoints/openai/serving_chat.py",
            "vllm/entrypoints/openai/chat_completion/protocol.py",
            "vllm/entrypoints/openai/completion/serving.py",
            "vllm/entrypoints/openai/engine/protocol.py",
            "vllm/entrypoints/serve/engine/protocol.py",
            "vllm/entrypoints/generate/base/protocol.py",
            "vllm/renderers/hf.py",
            "vllm/sampling_params.py",
        ]:
            self.assertTrue(vllm_source(path), path)
        self.assertFalse(vllm_source("vllm/entrypoints/openai/embedding/protocol.py"))

    def test_mutable_refs_and_git_options_rejected_before_git(self):
        for revision in ["main", "v0.30.0", "--help", "a" * 40 + "~1"]:
            with self.assertRaises(ValueError):
                commit(Path("/nonexistent"), revision)


if __name__ == "__main__":
    unittest.main()
