# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-only native normalization; fixtures are not actual server behavior."""

import ast
import unittest

from protocol_drift import module_contract
from protocol_native_contract import NativeContractExtractor, UnknownFact


def extractor(sources):
    return NativeContractExtractor(
        list(sources),
        sources.__getitem__,
        {
            "modules": {
                path: {"contract": module_contract(source)}
                for path, source in sources.items()
            },
        },
    )


class NativeContractTests(unittest.TestCase):
    def test_inherited_config_changes_alias_acceptance_at_request_and_nested_levels(
        self,
    ):
        instance = extractor(
            {
                "vllm/protocol.py": """
from pydantic import BaseModel, ConfigDict, Field
class Parent(BaseModel):
    value: int = Field(alias="wire")
class ChatCompletionRequest(Parent):
    model_config = ConfigDict(populate_by_name=True)
class CompletionRequest(BaseModel):
    nested: ChatCompletionRequest
"""
            }
        )
        result = instance.extract("a" * 40)
        self.assertEqual(
            result.endpoints["/v1/chat/completions"].fields["value"].wire_names,
            ["value", "wire"],
        )
        nested = result.endpoints["/v1/completions"].fields["nested"].wire_type
        self.assertEqual(set(nested["properties"]), {"value", "wire"})

    def test_alias_config_flags_and_null_validation_alias(self):
        instance = extractor({"vllm/types.py": ""})
        field = ast.parse(
            'Field(alias="wire", validation_alias=None)', mode="eval"
        ).body
        for config, expected in (
            ({}, ["wire"]),
            ({"validate_by_name": True}, ["n", "wire"]),
            ({"validate_by_alias": False}, ["n"]),
        ):
            with self.subTest(config=config):
                self.assertEqual(
                    instance.field_facts("n", field, config=config)[0], expected
                )
        with self.assertRaises(UnknownFact):
            instance.field_facts(
                "n",
                field,
                config={"validate_by_name": False, "validate_by_alias": False},
            )

    def test_dynamic_alias_retains_field_with_unknown_names(self):
        instance = extractor(
            {
                "vllm/protocol.py": """
from pydantic import BaseModel, Field
class ChatCompletionRequest(BaseModel):
    model: str
    x: int = Field(alias=make_alias())
class CompletionRequest(BaseModel):
    prompt: str
"""
            }
        )
        result = instance.extract("a" * 40)
        fields = result.endpoints["/v1/chat/completions"].fields
        self.assertEqual(set(fields), {"model", "x"})
        self.assertEqual(fields["x"].wire_names, [])
        self.assertFalse(result.complete())

    def test_unknown_default_does_not_erase_known_input_names(self):
        instance = extractor(
            {
                "vllm/protocol.py": """
from pydantic import BaseModel, Field
class ChatCompletionRequest(BaseModel):
    request_id: str = Field(default_factory=random_uuid, alias="id")
class CompletionRequest(BaseModel):
    pass
"""
            }
        )
        result = instance.extract("a" * 40)
        field = result.endpoints["/v1/chat/completions"].fields["request_id"]
        self.assertEqual(field.wire_names, ["id"])
        self.assertIsNone(field.default)

    def test_unsupported_config_cannot_claim_known_wire_names(self):
        instance = extractor(
            {
                "vllm/protocol.py": """
from pydantic import BaseModel, ConfigDict
class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    field_name: str
class CompletionRequest(BaseModel):
    prompt: str
"""
            }
        )
        result = instance.extract("a" * 40)
        endpoint = result.endpoints["/v1/chat/completions"]
        self.assertFalse(endpoint.fields_complete)
        self.assertEqual(endpoint.fields["field_name"].wire_names, [])
        self.assertTrue(
            any(item.aspect == "configuration" for item in result.diagnostics)
        )

    def test_named_integer_bounds_follow_imports_without_execution(self):
        instance = extractor(
            {
                "vllm/constants.py": "MINIMUM = -(2 ** 63)\nMAXIMUM = (1 << 63) - 1",
                "vllm/types.py": "from .constants import MINIMUM, MAXIMUM",
            }
        )
        field = ast.parse("Field(default=0, ge=MINIMUM, le=MAXIMUM)", mode="eval").body
        self.assertEqual(
            instance.field_facts("n", field, module="vllm.types")[3],
            {"ge": -(2**63), "le": 2**63 - 1},
        )
        with self.assertRaises(UnknownFact):
            instance.constant("vllm.types", ast.parse("2 ** 1000000", mode="eval").body)

    def test_annotated_constraints_normalize_to_same_field_facts(self):
        template = "from pydantic import BaseModel, Field\nfrom typing import Annotated\nclass ChatCompletionRequest(BaseModel):\n    n: {annotation} = {default}\nclass CompletionRequest(BaseModel):\n    pass\n"
        annotated = extractor(
            {
                "vllm/protocol.py": template.format(
                    annotation="Annotated[int, Field(ge=0)]", default="3"
                )
            }
        ).extract("a" * 40)
        direct = extractor(
            {
                "vllm/protocol.py": template.format(
                    annotation="int", default="Field(default=3, ge=0)"
                )
            }
        ).extract("a" * 40)
        self.assertEqual(
            annotated.endpoints["/v1/chat/completions"].fields["n"].facts(),
            direct.endpoints["/v1/chat/completions"].fields["n"].facts(),
        )

    def test_shadowed_factory_and_annotated_alias_are_not_erased(self):
        instance = extractor({"vllm/types.py": "def list():\n    return [3]"})
        instance.resolver.index("vllm.types")
        with self.assertRaises(UnknownFact):
            instance.field_facts(
                "n",
                ast.parse("Field(default_factory=list)", mode="eval").body,
                module="vllm.types",
            )
        with self.assertRaises(UnknownFact):
            instance.wire_type(
                "vllm.types",
                ast.parse('Annotated[int, Field(alias="x")]', mode="eval").body,
            )

    def test_cross_language_normalization_uses_wire_primitives(self):
        instance = extractor({"vllm/types.py": ""})
        self.assertEqual(
            instance.wire_type("vllm.types", ast.parse("list[int]", mode="eval").body),
            {"type": "array", "items": {"type": "integer"}},
        )

    def test_imported_alias_and_nested_class_change_followed(self):
        sources = {
            "vllm/types.py": "from .nested import Config\nAlias = list[Config]",
            "vllm/nested.py": "from pydantic import BaseModel\nclass Config(BaseModel):\n    limit: int = 3",
        }
        node = ast.parse("Alias", mode="eval").body
        first = extractor(sources).wire_type("vllm.types", node)
        sources["vllm/nested.py"] = sources["vllm/nested.py"].replace("= 3", "= 4")
        second = extractor(sources).wire_type("vllm.types", node)
        self.assertNotEqual(first, second)
        self.assertEqual(second["items"]["properties"]["limit"]["default"]["value"], 4)

    def test_unpinned_external_dependency_is_unknown_not_any(self):
        instance = extractor({"vllm/types.py": "from external_sdk import Model"})
        with self.assertRaisesRegex(UnknownFact, "no pinned source"):
            instance.wire_type("vllm.types", ast.parse("Model", mode="eval").body)

    def test_alias_cycle_is_explicit(self):
        instance = extractor({"vllm/types.py": "A = B\nB = A"})
        with self.assertRaises(UnknownFact):
            instance.wire_type("vllm.types", ast.parse("A", mode="eval").body)

    def test_defaults_requiredness_constraints_aliases(self):
        instance = extractor({"vllm/types.py": ""})
        names, required, default, constraints = instance.field_facts(
            "n",
            ast.parse(
                'Field(default=3, ge=0, validation_alias=AliasChoices("n", "count"))',
                mode="eval",
            ).body,
        )
        self.assertCountEqual(names, ["n", "count"])
        self.assertFalse(required)
        self.assertEqual(default, {"kind": "value", "value": 3})
        self.assertEqual(constraints, {"ge": 0})
        self.assertTrue(instance.field_facts("n", None)[1])

    def test_full_endpoint_extraction_keeps_unknown_fields(self):
        instance = extractor(
            {
                "vllm/protocol.py": """
from pydantic import BaseModel
from external_sdk import Message
class ChatCompletionRequest(BaseModel):
    messages: list[Message]
    temperature: float | None = None
class CompletionRequest(BaseModel):
    prompt: str
"""
            }
        )
        contract = instance.extract("a" * 40)
        chat = contract.endpoints["/v1/chat/completions"]
        self.assertIn("messages", chat.fields)
        self.assertIsNone(chat.fields["messages"].wire_type)
        self.assertTrue(chat.fields["temperature"].nullable)
        self.assertFalse(contract.complete())
        self.assertEqual(contract.diagnostics[0].path, "messages")


if __name__ == "__main__":
    unittest.main()
