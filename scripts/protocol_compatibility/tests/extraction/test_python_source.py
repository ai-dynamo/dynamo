# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extraction regressions retained from the retired upstream-only workflow."""

import unittest
from pathlib import Path

from scripts.protocol_compatibility.common.source import commit
from scripts.protocol_compatibility.extraction.python_source import (
    module_contract,
    vllm_source,
)


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
                self.assertNotEqual(module_contract(first), module_contract(second))

    def test_alias_constraints_and_factory_are_contract(self):
        for keyword in ["alias='old'", "ge=0", "default_factory=list", "exclude=True"]:
            before = "class Request(Base):\n    field: int = Field()\n"
            after = f"class Request(Base):\n    field: int = Field({keyword})\n"
            self.assertNotEqual(module_contract(before), module_contract(after))

    def test_validator_and_stream_behavior_changes_require_review(self):
        before = "class Request(Base):\n    def validate(self):\n        return self.value > 0\n"
        after = before.replace("> 0", ">= 0")
        self.assertNotEqual(
            module_contract(before)["classes"]["Request"]["methods"],
            module_contract(after)["classes"]["Request"]["methods"],
        )
        before = "async def stream():\n    yield {'prompt_logprobs': None}\n"
        after = "async def stream():\n    yield {}\n"
        self.assertNotEqual(module_contract(before), module_contract(after))

    def test_add_remove_response_field_and_preserve_null(self):
        before = module_contract("class Response(Base):\n    id: str\n")
        after = module_contract(
            "class Response(Base):\n    id: str\n    prompt_logprobs: list | None = None\n"
        )
        self.assertNotIn("prompt_logprobs", before["classes"]["Response"]["fields"])
        self.assertEqual(
            after["classes"]["Response"]["fields"]["prompt_logprobs"]["default"], "None"
        )

    def test_inheritance_alias_and_import_changes_are_visible(self):
        for before, after in [
            ("class Request(Base): pass", "class Request(OtherBase): pass"),
            ("Choice = Literal['auto']", "Choice = Literal['auto', 'none']"),
            ("from old import Base", "from new import Base"),
        ]:
            self.assertNotEqual(module_contract(before), module_contract(after))

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
