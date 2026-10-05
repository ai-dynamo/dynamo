# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from generate_protocol_inventory import request_fields
from protocol_dependencies import ROOT_CLASSES, DependencyResolver
from protocol_drift import compare, module_contract

PROTOCOL = "vllm/entrypoints/openai/protocol.py"


def inventory(
    extra, declaration="value: Payload", prefix="from vllm.types import Payload"
):
    source = (
        prefix
        + "\n"
        + "\n".join(
            f"class {name}(BaseModel):\n    {declaration}\n"
            for name in sorted(ROOT_CLASSES)
        )
    )
    sources = {PROTOCOL: source, **extra}
    coverage = DependencyResolver(list(sources), sources.__getitem__).collect(
        [PROTOCOL]
    )
    return {
        "target": "vllm",
        "dependency_coverage": coverage,
        "modules": {
            path: {"contract": module_contract(sources[path])}
            for path in {PROTOCOL, *coverage["module_consumers"]}
        },
    }


class DependencyTests(unittest.TestCase):
    def test_nested_alias_reexport_and_relative_import_change(self):
        sources = {
            "vllm/types/__init__.py": "from .nested import Payload\n",
            "vllm/types/nested.py": "from .choices import Choice\nclass Payload:\n    choice: Choice\n",
            "vllm/types/choices.py": "from typing import Literal\nChoice = Literal['auto']\n",
        }
        before = inventory(sources)
        sources[
            "vllm/types/choices.py"
        ] = "from typing import Literal\nChoice = Literal['auto', 'none']\n"
        after = inventory(sources)
        self.assertTrue(after["dependency_coverage"]["complete"])
        delta = compare(before, after)
        self.assertEqual(len(delta), 1)
        self.assertEqual(delta[0]["path"], ["vllm/types/choices.py", "bindings"])
        self.assertIn(
            PROTOCOL + ":ChatCompletionRequest.value", delta[0]["reachable_consumers"]
        )

    def test_new_field_and_changed_type_in_previously_unscanned_model(self):
        before = inventory({"vllm/types.py": "class Payload:\n    old: int\n"})
        after = inventory(
            {"vllm/types.py": "class Payload:\n    old: str\n    new: bool\n"}
        )
        delta = compare(before, after)
        self.assertEqual(
            {tuple(item["path"][2:]) for item in delta},
            {("Payload", "fields", "old", "annotation"), ("Payload", "fields", "new")},
        )

    def test_module_alias_string_forward_refs_and_recursive_models(self):
        result = inventory(
            {"vllm/types.py": "class Payload:\n    children: list['Payload']\n"},
            "value: 'types.Payload'",
            "import vllm.types as types",
        )
        self.assertTrue(result["dependency_coverage"]["complete"])

    def test_external_dependency_is_explicit_not_complete(self):
        result = inventory({}, prefix="from external_sdk import Payload")
        coverage = result["dependency_coverage"]
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["unresolved"][0]["symbol"], "external_sdk.Payload")
        self.assertIn("external dependency", coverage["unresolved"][0]["reason"])

    def test_alias_cycles_and_conditional_dynamic_definitions_are_reported(self):
        for source, expected in [
            ("Payload = Other\nOther = Payload\n", "cyclic alias"),
            ("if FLAG:\n    Payload = int\nelse:\n    Payload = str\n", "conditional"),
            ("Payload = make_model('Payload')\n", "runtime evaluation"),
        ]:
            with self.subTest(source=source):
                coverage = inventory({"vllm/types.py": source})["dependency_coverage"]
                self.assertFalse(coverage["complete"])
                self.assertTrue(
                    any(expected in item["reason"] for item in coverage["unresolved"])
                )

    def test_type_checking_import_and_literal_string_not_forward_reference(self):
        result = inventory(
            {
                "vllm/types.py": "from typing import Literal\nPayload = Literal['not a type', 'auto']\n"
            },
            prefix="from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vllm.types import Payload",
        )
        self.assertTrue(result["dependency_coverage"]["complete"])

    def test_unused_imports_are_not_scanned_or_executed(self):
        sources = {
            "vllm/types.py": "raise RuntimeError('must not execute')\nclass Payload:\n    value: int\n",
            "vllm/unrelated.py": "deliberately not Python!",
        }
        result = inventory(
            sources, prefix="from vllm.types import Payload\nimport vllm.unrelated"
        )
        self.assertTrue(result["dependency_coverage"]["complete"])
        self.assertNotIn("vllm/unrelated.py", result["modules"])

    def test_imported_base_fields_are_followed(self):
        result = inventory(
            {
                "vllm/types.py": "from .base import Parent\nclass Payload(Parent): pass\n",
                "vllm/base.py": "class Parent:\n    value: int\n",
            }
        )
        self.assertIn("vllm/base.py", result["modules"])
        self.assertTrue(result["dependency_coverage"]["complete"])

    def test_inventory_resolves_renamed_imported_base_not_same_named_decoy(self):
        result = inventory(
            {
                "vllm/types.py": "from pydantic import BaseModel as BM\nclass Parent(BM):\n    inherited: int\nPayload = Parent\n"
            },
            prefix="from vllm.types import Parent as BaseModel\nclass Parent:\n    decoy: bool\n",
            declaration="own: int",
        )
        fields = request_fields(result, "ChatCompletionRequest")
        self.assertEqual(set(fields), {"inherited", "own"})


if __name__ == "__main__":
    unittest.main()
