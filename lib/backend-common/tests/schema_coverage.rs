// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Coverage test for `dynamo_backend_common::schema::REQUEST_FIELDS`.
//!
//! Fails when a new serializable field is added to `PreprocessedRequest`
//! without a corresponding classification in the schema registry. The
//! check populates every `Option` field with a placeholder so
//! `#[serde(skip_serializing_if = "Option::is_none")]` does not hide
//! fields from the resulting JSON object — adding a Rust field without
//! categorizing it then trips this test in CI.

use std::collections::{HashMap, HashSet};

use dynamo_backend_common::schema::REQUEST_FIELDS;
use dynamo_kv_router::scheduling::config::RouterConfigOverride;
use dynamo_llm::agents::context::AgentContextBuilder;
use dynamo_llm::protocols::common::preprocessor::{
    BootstrapInfo, MmRoutingInfo, PrefillResult, PreprocessedRequest, RouterParams, RoutingHints,
    TraceLink,
};
use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

/// Build a request with every `Option` field populated so the JSON
/// serializer emits every key (defeats `skip_serializing_if`).
fn fully_populated_request() -> PreprocessedRequest {
    let agent_context = AgentContextBuilder::default()
        .session_type_id("type".to_string())
        .session_id("session".to_string())
        .trajectory_id("trajectory".to_string())
        .build()
        .expect("AgentContext builds with required fields");

    let prefill_result = PrefillResult {
        disaggregated_params: serde_json::Value::Null,
        prompt_tokens_details: None,
    };

    PreprocessedRequest::builder()
        .model("m".to_string())
        .token_ids(vec![1])
        .prompt_embeds(Some("base64".to_string()))
        .multi_modal_data(Some(HashMap::new()))
        .mm_routing_info(Some(MmRoutingInfo::default()))
        .stop_conditions(StopConditions::default())
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .eos_token_ids(vec![2])
        .mdc_sum(Some("checksum".to_string()))
        .annotations(vec!["tag".to_string()])
        .routing(Some(RoutingHints::default()))
        .router_config_override(Some(RouterConfigOverride::default()))
        .prefill_result(Some(prefill_result))
        .migration_link(Some(TraceLink {
            trace_id: "0".repeat(32),
            span_id: "0".repeat(16),
        }))
        .bootstrap_info(Some(BootstrapInfo::default()))
        .extra_args(Some(serde_json::Value::Null))
        .router(Some(RouterParams::default()))
        .agent_context(Some(agent_context))
        .mm_processor_kwargs(Some(serde_json::Value::Null))
        .request_timestamp_ms(Some(0.0))
        .is_probe(true)
        .build()
        .expect("fully-populated PreprocessedRequest builds")
}

#[test]
fn every_serializable_request_field_is_classified() {
    let request = fully_populated_request();
    let value = serde_json::to_value(&request).expect("PreprocessedRequest serializes");
    let map = value
        .as_object()
        .expect("PreprocessedRequest serializes as a JSON object");

    let rust_fields: HashSet<String> = map.keys().cloned().collect();
    let registered: HashSet<String> = REQUEST_FIELDS.iter().map(|f| f.name.to_string()).collect();

    let missing_from_registry: Vec<&String> = rust_fields.difference(&registered).collect();
    let stale_in_registry: Vec<&String> = registered.difference(&rust_fields).collect();

    assert!(
        missing_from_registry.is_empty(),
        "PreprocessedRequest has fields missing from schema::REQUEST_FIELDS: {missing_from_registry:?}. \
         Add each with a FieldStatus (Supported / Forwarded) in lib/backend-common/src/schema.rs."
    );
    assert!(
        stale_in_registry.is_empty(),
        "schema::REQUEST_FIELDS contains entries that no longer exist on PreprocessedRequest: {stale_in_registry:?}. \
         Remove them from lib/backend-common/src/schema.rs."
    );
}

#[test]
fn registry_has_no_duplicate_names() {
    let mut seen = HashSet::new();
    for f in REQUEST_FIELDS {
        assert!(
            seen.insert(f.name),
            "duplicate field name in REQUEST_FIELDS: {}",
            f.name
        );
    }
}
