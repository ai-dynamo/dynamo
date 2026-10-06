# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source fixture tests; not a Rust runtime serde conformance claim."""

import unittest

from scripts.protocol_compatibility.assessment.policy import direct_compare
from scripts.protocol_compatibility.common.contracts import (
    Contract,
    EndpointContract,
    FieldContract,
)
from scripts.protocol_compatibility.extraction.dynamo import DynamoContractExtractor
from scripts.protocol_compatibility.extraction.rust_source import (
    RustSources,
    RustUnknown,
    tokens,
)


def extractor(source):
    sources = RustSources()
    sources.add("lib/llm/src/protocols/openai/fixture.rs", source)
    return DynamoContractExtractor(sources)


class DynamoContractTests(unittest.TestCase):
    def test_tree_sitter_rejects_recovered_syntax_without_partial_index(self):
        for source in (
            "struct Good {} struct Broken { pub value: }",
            "struct Broken { pub value: String",
            "/* unterminated",
        ):
            with self.subTest(source=source):
                sources = RustSources()
                with self.assertRaisesRegex(RustUnknown, "syntax"):
                    sources.add("fixture.rs", source)
                self.assertEqual(sources.items, [])
                self.assertEqual(sources.sources, {})

    def test_tree_sitter_keeps_const_generic_braces_inside_declaration(self):
        instance = extractor(
            """
            #[derive(Deserialize)] struct Request<const N: usize = { 2 + 2 }> {
                pub model: String,
            }
            """
        )
        fields, _ = instance.fields(instance.sources.resolve("Request", ""))
        self.assertEqual(set(fields), {"model"})
        self.assertEqual(fields["model"].wire_type, {"type": "string"})

    def test_tree_sitter_handles_lifetimes_unicode_and_raw_identifiers(self):
        instance = extractor(
            """
            #[derive(Deserialize)] struct Request {
                #[serde(rename = r#"模式"#)] pub r#type: String,
            }
            impl<'de> Deserialize<'de> for Other {
                fn deserialize() { let café = '{'; }
            }
            """
        )
        fields, _ = instance.fields(instance.sources.resolve("Request", ""))
        self.assertEqual(set(fields), {"模式"})
        method = next(item for item in instance.sources.items if item.kind == "fn")
        self.assertIn("Deserialize", method.owner)
        self.assertIn("café", method.body)
        self.assertIn("'{'", method.body)

    def test_tree_sitter_does_not_read_declarations_inside_macros_or_functions(self):
        instance = extractor(
            """
            macro_rules! define { () => { struct Fake {} }; }
            define!();
            fn helper() { struct Local {} }
            #[derive(Deserialize)] struct Real { pub n: u32 }
            """
        )
        names = {item.name for item in instance.sources.items}
        self.assertNotIn("Fake", names)
        self.assertNotIn("Local", names)
        self.assertIn("Real", names)

    def test_tuple_struct_does_not_become_empty_object(self):
        instance = extractor("#[derive(Deserialize)] struct Request(String);")
        with self.assertRaises(RustUnknown):
            instance.fields(instance.sources.resolve("Request", ""))

    def test_tree_sitter_literal_projection_preserves_empty_and_raw_strings(self):
        for source, expected in (
            ('r#""#', '""'),
            ('r###"a { b }"###', '"a { b }"'),
            ('"a { b }"', '"a { b }"'),
            ("'{'", "'{'"),
        ):
            with self.subTest(source=source):
                self.assertEqual(tokens(source), [expected])

    def test_nested_placement_survives_partial_schema_and_follows_aliases(self):
        instance = extractor(
            """
            type Extension = Options;
            #[derive(Deserialize)] struct Options {
                #[serde(rename = "temperature", alias = "temp")]
                pub internal_temperature: Option<f32>,
                pub unavailable: ExternalType,
            }
            #[derive(Deserialize)] struct NvCreateChatCompletionRequest {
                #[serde(rename = "nvidia", alias = "nvext")]
                pub extension: Option<Extension>,
            }
            #[derive(Deserialize)] struct NvCreateCompletionRequest { pub prompt: String }
            """
        )
        dynamo = instance.extract("a" * 40)
        endpoint = "/v1/chat/completions"
        chat = dynamo.endpoints[endpoint]
        self.assertIsNone(chat.fields["nvidia"].wire_type)
        locations = {tuple(item["wire_path"]) for item in chat.nested_inputs}
        self.assertIn(("nvext", "temperature"), locations)
        self.assertIn(("nvidia", "temp"), locations)
        self.assertIn(("nvext", "unavailable"), locations)
        native = Contract(
            "vllm",
            "b" * 40,
            {
                endpoint: EndpointContract(
                    {"temperature": FieldContract("temperature", ["temperature"])}, True
                )
            },
        )
        findings = direct_compare(native, dynamo)
        placement = next(
            item for item in findings if item.aspect == "placement_candidate"
        )
        self.assertEqual(placement.category, "coverage")
        self.assertIn("not established aliases", placement.observation)
        self.assertEqual(len(placement.dynamo["nested_inputs"]), 2)
        # A nearby nested declaration must not stand in for a root input slot.
        self.assertTrue(
            any(
                item.path == "temperature" and item.aspect == "input_slot"
                for item in findings
            )
        )
        self.assertFalse(dynamo.complete())

    def test_custom_deserializers_do_not_invent_nested_public_paths(self):
        for custom in ("field", "container"):
            with self.subTest(custom=custom):
                instance = extractor(
                    """
                    #[derive(Deserialize)] struct Options { pub temperature: f32 }
                    #[derive(Deserialize)] struct NvCreateChatCompletionRequest {
                    """
                    + (
                        '#[serde(deserialize_with = "custom")]'
                        if custom == "field"
                        else ""
                    )
                    + "pub nvext: Option<Options> }"
                    + (
                        "impl Deserialize for Options { fn deserialize() {} }"
                        if custom == "container"
                        else ""
                    )
                )
                result = instance.extract("a" * 40)
                self.assertEqual(
                    result.endpoints["/v1/chat/completions"].nested_inputs, []
                )

    def test_nested_paths_distinguish_literal_dotted_names_and_stop_recursion(self):
        instance = extractor(
            """
            #[derive(Deserialize)] struct Options {
                #[serde(rename = "literal.key")] pub value: String,
                pub next: Option<Options>,
            }
            #[derive(Deserialize)] struct NvCreateChatCompletionRequest {
                pub nvext: Options,
            }
            """
        )
        result = instance.extract("a" * 40)
        paths = [
            item["wire_path"]
            for item in result.endpoints["/v1/chat/completions"].nested_inputs
        ]
        self.assertEqual(paths, [["nvext", "literal.key"], ["nvext", "next"]])

    def test_remote_derive_keeps_declarations_without_claiming_wrapper_completeness(
        self,
    ):
        instance = extractor(
            """
            #[derive(Deserialize)] #[serde(remote = "Self")]
            struct NvCreateChatCompletionRequest { pub model: String }
            impl Deserialize for NvCreateChatCompletionRequest {
                fn deserialize() { Self::deserialize(input) }
            }
            #[derive(Deserialize)] struct NvCreateCompletionRequest { pub model: String }
        """
        )
        contract = instance.extract("a" * 40)
        chat = contract.endpoints["/v1/chat/completions"]
        self.assertIn("model", chat.fields)
        self.assertFalse(chat.fields_complete)
        self.assertTrue(
            any(item.aspect == "custom_deserialize" for item in contract.diagnostics)
        )

    def test_import_alias_reexport_and_duplicate_name_resolve_by_module(self):
        sources = RustSources()
        sources.add(
            "lib/llm/src/protocols/one.rs",
            "#[derive(Deserialize)] pub struct Value { pub n: u32 }",
        )
        sources.add(
            "lib/llm/src/protocols/two.rs",
            "#[derive(Deserialize)] pub struct Value { pub n: String }",
        )
        sources.add(
            "lib/llm/src/protocols/exports.rs", "pub use super::one::Value as Exported;"
        )
        sources.add(
            "lib/llm/src/protocols/root.rs",
            """
            use crate::protocols::exports::Exported as Selected;
            #[derive(Deserialize)] pub struct Request { pub value: Selected }
        """,
        )
        instance = DynamoContractExtractor(sources)
        fields, _ = instance.fields(sources.resolve("Request", ""))
        self.assertEqual(
            fields["value"].wire_type["properties"]["n"]["wire_type"],
            {"type": "integer"},
        )

    def test_unimported_same_named_type_is_not_used(self):
        sources = RustSources()
        sources.add("lib/llm/src/protocols/one.rs", "struct Thing {}")
        sources.add(
            "lib/llm/src/protocols/root.rs", "struct Request { pub thing: Thing }"
        )
        with self.assertRaises(RustUnknown):
            sources.resolve("Thing", "lib/llm/src/protocols/root.rs")

    def test_unknown_nested_field_prevents_complete_parent_schema(self):
        instance = extractor(
            """
            #[derive(Deserialize)] struct Nested { pub value: MissingType }
            #[derive(Deserialize)] struct Request { pub nested: Nested }
        """
        )
        fields, _ = instance.fields(instance.sources.resolve("Request", ""))
        self.assertIsNone(fields["nested"].wire_type)
        self.assertIn(
            "nested field contract incomplete", str(instance.structural_problems)
        )

    def test_rust_continued_string_does_not_consume_later_braces(self):
        source = 'fn example() { let text = "first \\\n second"; }'
        self.assertEqual(tokens(source).count("{"), 1)
        self.assertEqual(tokens(source).count("}"), 1)

    def test_qualified_crate_does_not_fall_back_to_wrong_definition(self):
        sources = RustSources()
        sources.add(
            "crate:async-openai/src/test.rs", "struct Request {}", "async_openai"
        )
        with self.assertRaises(RustUnknown):
            sources.resolve("dynamo_protocols::types::Request", "")

    def test_flatten_shared_fields_aliases_defaults_and_bounds(self):
        instance = extractor(
            """
        #[derive(Deserialize)]
        pub struct Base { pub model: String, pub temperature: Option<f32> }
        #[derive(Deserialize)]
        pub struct Common { #[serde(default)] pub min_tokens: Option<u32> }
        #[derive(Deserialize)]
        pub struct NvCreateChatCompletionRequest {
            #[serde(flatten)] pub inner: Base,
            #[serde(flatten)] pub common: Common,
            #[serde(alias = "chat_template_kwargs")]
            pub chat_template_args: Option<String>,
        }
        #[derive(Deserialize)]
        pub struct NvCreateCompletionRequest { #[serde(flatten)] pub inner: Base }
        """
        )
        result = instance.extract("a" * 40)
        chat = result.endpoints["/v1/chat/completions"]
        self.assertTrue(chat.fields_complete)
        self.assertEqual(
            set(chat.fields),
            {"model", "temperature", "min_tokens", "chat_template_args"},
        )
        self.assertTrue(chat.fields["model"].required)
        self.assertFalse(chat.fields["temperature"].required)
        self.assertEqual(
            chat.fields["min_tokens"].constraints, {"ge": 0, "le": 4294967295}
        )
        self.assertEqual(
            chat.fields["min_tokens"].default, {"kind": "value", "value": None}
        )
        self.assertIn(
            "chat_template_kwargs", chat.fields["chat_template_args"].wire_names
        )
        self.assertNotIn("min_tokens", result.endpoints["/v1/completions"].fields)

    def test_nested_alias_and_untagged_enum(self):
        instance = extractor(
            """
        type Text = String;
        #[derive(Deserialize)] #[serde(untagged)]
        enum Prompt { Text(Text), Tokens(Vec<u32>) }
        #[derive(Deserialize)] struct Request { pub prompt: Prompt }
        """
        )
        fields, _ = instance.fields(instance.sources.resolve("Request", ""))
        self.assertEqual(len(fields["prompt"].wire_type["any_of"]), 2)

    def test_flatten_map_is_not_proof_of_forwarding(self):
        instance = extractor(
            """
        #[derive(Deserialize)] struct Request {
            #[serde(flatten, deserialize_with = "custom")]
            pub unknown: HashMap<String, serde_json::Value>,
        }
        """
        )
        fields, wildcard = instance.fields(instance.sources.resolve("Request", ""))
        self.assertEqual(fields, {})
        self.assertFalse(wildcard.complete)
        self.assertEqual(wildcard.effects, [])

    def test_custom_deserializer_and_missing_type_are_unknown(self):
        instance = extractor(
            """
        #[derive(Deserialize)] struct Request {
            #[serde(deserialize_with = "custom")]
            pub value: u32,
            pub extra: ExternalType,
        }
        """
        )
        fields, _ = instance.fields(instance.sources.resolve("Request", ""))
        self.assertIsNone(fields["value"].wire_type)
        self.assertIsNone(fields["extra"].wire_type)
        self.assertEqual(len(instance.structural_problems), 2)

    def test_raw_strings_comments_and_test_definitions_do_not_pollute(self):
        instance = extractor(
            """
        // struct Fake { }
        const TEXT: &str = r#"struct Fake { \" }"#;
        /* nested /* struct Fake { } */ comment */
        #[cfg(test)] mod tests { struct Fake {} }
        #[derive(Deserialize)] struct Real { pub n: u32 }
        """
        )
        self.assertFalse(any(item.name == "Fake" for item in instance.sources.items))
        self.assertEqual(instance.sources.resolve("Real", "").kind, "struct")
        self.assertIn('"x { y"', tokens('r#"x { y"#'))


if __name__ == "__main__":
    unittest.main()
