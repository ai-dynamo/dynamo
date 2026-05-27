// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema registry classifying every [`PreprocessedRequest`] field as
//! `Supported` or `Forwarded` for the unified backend.
//!
//! [`check_request`] runs per-request in [`crate::adapter::EngineAdapter`]
//! and applies the configured [`UnsupportedFieldPolicy`]. Engines opt
//! into specific `Forwarded` fields by declaring [`Capability`] variants
//! in [`EngineConfig::capabilities`].
//!
//! The coverage test in `tests/schema_coverage.rs` fails CI if a Rust
//! field is added without classification.

use std::collections::HashSet;

use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;

use crate::error::{BackendError, DynamoError, ErrorType};

/// Lifecycle status of a request field on the unified backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldStatus {
    /// Documented in the unified contract; engines accept it without
    /// declaring a capability.
    Supported,
    /// Carried over the wire but consumed only by engines that declare
    /// the matching [`Capability`] in [`EngineConfig::capabilities`].
    Forwarded,
}

impl FieldStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FieldStatus::Supported => "supported",
            FieldStatus::Forwarded => "forwarded",
        }
    }
}

/// Capabilities a unified engine can declare in
/// [`EngineConfig::capabilities`]. Each variant corresponds 1-to-1 to a
/// [`FieldStatus::Forwarded`] field on [`PreprocessedRequest`]; the gate
/// permits the request when the engine has declared the matching
/// variant. Add a new Forwarded field → add a Capability variant →
/// the coverage test enforces both ends stay in lock-step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    PromptEmbeds,
    MultiModalData,
    MmRoutingInfo,
    MmProcessorKwargs,
    RouterConfigOverride,
    AgentContext,
    ExtraArgs,
}

impl Capability {
    /// The `PreprocessedRequest` field name this capability gates.
    /// Used in the `Reject` error message and pinned by the schema
    /// consistency test to catch typo'd `Capability` variants in the
    /// `forwarded!` registry entries.
    pub fn field_name(self) -> &'static str {
        match self {
            Capability::PromptEmbeds => "prompt_embeds",
            Capability::MultiModalData => "multi_modal_data",
            Capability::MmRoutingInfo => "mm_routing_info",
            Capability::MmProcessorKwargs => "mm_processor_kwargs",
            Capability::RouterConfigOverride => "router_config_override",
            Capability::AgentContext => "agent_context",
            Capability::ExtraArgs => "extra_args",
        }
    }
}

/// One entry in [`REQUEST_FIELDS`]. `capability` is `Some` iff the field
/// is `Forwarded`. `is_set` returns `true` iff the field carries a
/// meaningful value, so the gate only fires when the user populated it.
pub struct FieldDescriptor {
    pub name: &'static str,
    pub status: FieldStatus,
    pub capability: Option<Capability>,
    pub is_set: fn(&PreprocessedRequest) -> bool,
}

macro_rules! supported {
    ($name:literal, $is_set:expr) => {
        FieldDescriptor {
            name: $name,
            status: FieldStatus::Supported,
            capability: None,
            is_set: $is_set,
        }
    };
}

macro_rules! forwarded {
    ($name:literal, $cap:expr, $is_set:expr) => {
        FieldDescriptor {
            name: $name,
            status: FieldStatus::Forwarded,
            capability: Some($cap),
            is_set: $is_set,
        }
    };
}

/// Every serializable field on [`PreprocessedRequest`].
/// `tests/schema_coverage.rs` fails if a struct field is missing here.
pub const REQUEST_FIELDS: &[FieldDescriptor] = &[
    // Always-present / metadata.
    supported!("model", |r| !r.model.is_empty()),
    supported!("token_ids", |r| !r.token_ids.is_empty()),
    supported!("stop_conditions", |_| true),
    supported!("sampling_options", |_| true),
    supported!("output_options", |_| true),
    supported!("eos_token_ids", |r| !r.eos_token_ids.is_empty()),
    supported!("mdc_sum", |r| r.mdc_sum.is_some()),
    supported!("annotations", |r| !r.annotations.is_empty()),
    supported!("request_timestamp_ms", |r| r.request_timestamp_ms.is_some()),
    // Routing / disaggregation.
    supported!("routing", |r| r.routing.is_some()),
    supported!("router", |r| r.router.is_some()),
    supported!("prefill_result", |r| r.prefill_result.is_some()),
    supported!("bootstrap_info", |r| r.bootstrap_info.is_some()),
    // Framework-owned trace plumbing (engines neither read nor write).
    supported!("migration_link", |r| r.migration_link.is_some()),
    // Canary marker. Skips the gate entirely (see `check_request`).
    supported!("_HEALTH_CHECK", |r| r.is_probe),
    // Forwarded — engine must declare the matching Capability variant.
    forwarded!("prompt_embeds", Capability::PromptEmbeds, |r| r
        .prompt_embeds
        .is_some()),
    forwarded!("multi_modal_data", Capability::MultiModalData, |r| r
        .multi_modal_data
        .is_some()),
    forwarded!("mm_routing_info", Capability::MmRoutingInfo, |r| r
        .mm_routing_info
        .is_some()),
    forwarded!("mm_processor_kwargs", Capability::MmProcessorKwargs, |r| r
        .mm_processor_kwargs
        .is_some()),
    forwarded!(
        "router_config_override",
        Capability::RouterConfigOverride,
        |r| r.router_config_override.is_some()
    ),
    forwarded!("agent_context", Capability::AgentContext, |r| r
        .agent_context
        .is_some()),
    forwarded!("extra_args", Capability::ExtraArgs, |r| r
        .extra_args
        .is_some()),
];

/// Policy applied when a request sets a `Forwarded` field that the
/// engine has not declared the matching [`Capability`] for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnsupportedFieldPolicy {
    /// Return [`BackendError::InvalidArgument`].
    Reject,
    /// `warn!` and pass through.
    #[default]
    Warn,
    /// Silent pass-through.
    Ignore,
}

/// Apply `policy` to every `Forwarded` field set on `request` whose
/// [`Capability`] is not in `engine_capabilities`.
///
/// Health-check probes (`request.is_probe`) bypass the gate so an
/// operator-defined canary payload can exercise engine-specific
/// fields without tripping `Reject`.
pub fn check_request(
    request: &PreprocessedRequest,
    policy: UnsupportedFieldPolicy,
    engine_capabilities: &HashSet<Capability>,
) -> Result<(), DynamoError> {
    if request.is_probe {
        return Ok(());
    }
    for field in REQUEST_FIELDS {
        let Some(cap) = field.capability else {
            continue;
        };
        if !(field.is_set)(request) {
            continue;
        }
        if engine_capabilities.contains(&cap) {
            continue;
        }
        match policy {
            UnsupportedFieldPolicy::Reject => {
                let name = cap.field_name();
                return Err(DynamoError::builder()
                    .error_type(ErrorType::Backend(BackendError::InvalidArgument))
                    .message(format!(
                        "unified backend received request with `{name}` set; engine did \
                         not declare `Capability::{cap:?}`. Declare it from \
                         `LLMEngine::start()` or remove the field from the request.",
                    ))
                    .build());
            }
            UnsupportedFieldPolicy::Warn => {
                tracing::warn!(
                    field = field.name,
                    "unsupported request field set; engine has not declared matching capability"
                );
            }
            UnsupportedFieldPolicy::Ignore => {}
        }
    }
    Ok(())
}

/// Snapshot of `(name, status_str)` pairs for tooling/tests. The PyO3
/// binding re-exports this so Python tests can introspect the registry
/// without re-encoding the list.
pub fn list_request_fields() -> Vec<(&'static str, &'static str)> {
    REQUEST_FIELDS
        .iter()
        .map(|f| (f.name, f.status.as_str()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;
    use dynamo_llm::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

    fn empty_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    #[test]
    fn ignore_policy_allows_unsupported_field() {
        let mut req = empty_request();
        req.prompt_embeds = Some("base64-tensor".to_string());
        check_request(&req, UnsupportedFieldPolicy::Ignore, &HashSet::new()).unwrap();
    }

    #[test]
    fn supported_fields_never_trip_check() {
        let req = empty_request();
        for p in [UnsupportedFieldPolicy::Warn, UnsupportedFieldPolicy::Reject] {
            check_request(&req, p, &HashSet::new()).expect("supported-only request must pass");
        }
    }

    #[test]
    fn reject_fires_on_unsupported_forwarded_field() {
        let mut req = empty_request();
        req.prompt_embeds = Some("base64-tensor".to_string());
        let err = check_request(&req, UnsupportedFieldPolicy::Reject, &HashSet::new()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("prompt_embeds"), "got: {msg}");
        assert!(msg.contains("Capability::PromptEmbeds"), "got: {msg}");
    }

    #[test]
    fn capability_declaration_allows_forwarded_field() {
        let mut req = empty_request();
        req.prompt_embeds = Some("base64-tensor".to_string());
        let caps: HashSet<Capability> = [Capability::PromptEmbeds].into_iter().collect();
        check_request(&req, UnsupportedFieldPolicy::Reject, &caps)
            .expect("declared capability must allow the field");
    }

    #[test]
    fn warn_policy_does_not_error() {
        let mut req = empty_request();
        req.prompt_embeds = Some("base64-tensor".to_string());
        check_request(&req, UnsupportedFieldPolicy::Warn, &HashSet::new())
            .expect("warn policy must not return error");
    }

    #[test]
    fn policy_serde_uses_lowercase_variants() {
        // The PyO3 layer and any JSON tooling depends on the
        // serde-derived rename; pin one variant to lock it in.
        let s = serde_json::to_string(&UnsupportedFieldPolicy::Reject).unwrap();
        assert_eq!(s, "\"reject\"");
    }

    #[test]
    fn registry_capability_names_match_field_names() {
        // Catches a typo'd Capability variant in a `forwarded!` macro
        // invocation. Both ends are hand-authored from the same source
        // (the Rust struct field name); pin them in lock-step.
        for f in REQUEST_FIELDS {
            if let Some(cap) = f.capability {
                assert_eq!(cap.field_name(), f.name);
            }
        }
    }
}
