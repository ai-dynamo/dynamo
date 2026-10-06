# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from scripts.protocol_compatibility.common.contracts import EndpointContract, Handling
from scripts.protocol_compatibility.extraction.dynamo_handling import apply_admission
from scripts.protocol_compatibility.extraction.rust_source import RustSources


def sources():
    result = RustSources()
    result.add(
        "lib/llm/src/protocols/openai/validate.rs",
        """
        pub const PASSTHROUGH_EXTRA_FIELDS: &[&str] = &["sampling_x"];
        fn validate_no_unsupported_fields_observed(unsupported_fields: Map) {
            let unknown = unsupported_fields.keys()
                .filter(|k| !PASSTHROUGH_EXTRA_FIELDS.contains(&k.as_str()));
            let known = unknown.iter().copied().filter_map(known_native_field).collect();
            if !known.is_empty() { return Err(UnsupportedField); }
        }
    """,
    )
    result.add(
        "lib/llm/src/protocols/openai/chat_completions.rs",
        """
        impl ValidateRequest for NvCreateChatCompletionRequest {
            fn validate(&self) { validate_no_unsupported_fields_for_endpoint(&self.extra); }
        }
    """,
    )
    result.add(
        "lib/llm/src/protocols/openai/compatibility/vllm_fields.rs",
        """
        const VLLM_REQUEST_FIELDS: &[&str] = &["sampling_x", "unhandled_y"];
    """,
    )
    result.add(
        "lib/llm/src/preprocessor.rs",
        """
        fn sampling_passthrough_args(request: &Request) {
            if let Some(fields) = request.unsupported_fields() {
                for key in ["sampling_x"] {
                    if let Some(value) = fields.get(key) { output.insert(key.to_string(), value.clone()); }
                }
            }
        }
    """,
    )
    result.add(
        "lib/llm/src/protocols/common/backend_extensions.rs",
        """
        const SAMPLING_FIELDS: &[&str] = &["sampling_x"];
        impl SamplingTarget {
            fn resolve() { if capability.schema_version != 1 { return Err(rejected()); } }
            fn validate_field() { if !worker.supports(field) { return Err(rejected()); } }
        }
        fn lower_sampling_passthrough_for_target() { target.validate_field(field)?; }
    """,
    )
    return result


class AdmissionExtractionTests(unittest.TestCase):
    def test_admission_vocabulary_does_not_grant_forwarding_to_every_field(self):
        contract = EndpointContract(
            fields_complete=True, additional_properties=Handling()
        )
        apply_admission(sources(), contract, "NvCreateChatCompletionRequest")
        self.assertEqual(
            contract.untyped_handling["sampling_x"].effects,
            ["forward", "interpret", "reject"],
        )
        self.assertFalse(contract.untyped_handling["sampling_x"].complete)
        self.assertEqual(contract.fields, {})
        self.assertEqual(contract.untyped_handling["unhandled_y"].effects, ["reject"])
        self.assertTrue(contract.untyped_handling["unhandled_y"].complete)
        self.assertNotIn("unhandled_y", contract.fields)
        self.assertTrue(
            any(
                condition.get("stage") == "worker_capability_and_legacy_target"
                for condition in contract.untyped_handling["sampling_x"].conditions
            )
        )

    def test_endpoint_without_validation_call_does_not_inherit_rules(self):
        contract = EndpointContract(
            fields_complete=True, additional_properties=Handling()
        )
        apply_admission(sources(), contract, "NvCreateCompletionRequest")
        self.assertEqual(contract.untyped_handling, {})
        self.assertEqual(contract.fields, {})

    def test_non_catchall_request_does_not_inherit_rules(self):
        contract = EndpointContract(fields_complete=True)
        apply_admission(sources(), contract, "NvCreateChatCompletionRequest")
        self.assertEqual(contract.fields, {})

    def test_capability_body_changes_are_fingerprinted(self):
        before, after = sources(), sources()
        rule = next(item for item in after.items if item.name == "resolve")
        rule.body.append("new_condition")
        contracts = [
            EndpointContract(fields_complete=True, additional_properties=Handling())
            for _ in range(2)
        ]
        for source, contract in zip((before, after), contracts):
            apply_admission(source, contract, "NvCreateChatCompletionRequest")
        self.assertNotEqual(
            contracts[0].untyped_handling["sampling_x"].facts(),
            contracts[1].untyped_handling["sampling_x"].facts(),
        )


if __name__ == "__main__":
    unittest.main()
