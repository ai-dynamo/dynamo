// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Schema registry classifying every [`PreprocessedRequest`] field as
//! `Supported` or `Forwarded` for the unified backend.
//!
//! No runtime enforcement: the registry is documentation + a single
//! coverage test (`tests/schema_coverage.rs`) that fails CI if a Rust
//! field is added to `PreprocessedRequest` without classification. The
//! distinction between `Supported` (engines accept it as part of the
//! unified contract) and `Forwarded` (engines may consume engine-
//! specific extensions) is informational — tooling and docs read it,
//! the request path does not gate on it.

use dynamo_llm::protocols::common::preprocessor::PreprocessedRequest;

/// Lifecycle status of a request field on the unified backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FieldStatus {
    /// Part of the unified contract; every engine handles it (or has
    /// permission to ignore it).
    Supported,
    /// Engine-specific extension carried verbatim over the wire. Whether
    /// an individual engine consumes it is engine documentation; the
    /// framework does not enforce.
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

/// One entry in [`REQUEST_FIELDS`]. `is_set` returns `true` iff the
/// field carries a meaningful value (used by introspection tooling
/// when reporting which fields a given request populated).
pub struct FieldDescriptor {
    pub name: &'static str,
    pub status: FieldStatus,
    pub is_set: fn(&PreprocessedRequest) -> bool,
}

macro_rules! supported {
    ($name:literal, $is_set:expr) => {
        FieldDescriptor {
            name: $name,
            status: FieldStatus::Supported,
            is_set: $is_set,
        }
    };
}

macro_rules! forwarded {
    ($name:literal, $is_set:expr) => {
        FieldDescriptor {
            name: $name,
            status: FieldStatus::Forwarded,
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
    // Canary marker stamped by HealthCheckManager.
    supported!("_HEALTH_CHECK", |r| r.is_probe),
    // Forwarded — engine-specific extensions; consumption is documented
    // per engine, not enforced here.
    forwarded!("prompt_embeds", |r| r.prompt_embeds.is_some()),
    forwarded!("multi_modal_data", |r| r.multi_modal_data.is_some()),
    forwarded!("mm_routing_info", |r| r.mm_routing_info.is_some()),
    forwarded!("mm_processor_kwargs", |r| r.mm_processor_kwargs.is_some()),
    forwarded!("router_config_override", |r| r
        .router_config_override
        .is_some()),
    forwarded!("agent_context", |r| r.agent_context.is_some()),
    forwarded!("extra_args", |r| r.extra_args.is_some()),
];

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

    #[test]
    fn registry_has_no_duplicate_names() {
        let mut seen = std::collections::HashSet::new();
        for f in REQUEST_FIELDS {
            assert!(seen.insert(f.name), "duplicate: {}", f.name);
        }
    }

    #[test]
    fn list_request_fields_emits_known_statuses() {
        // Sanity: every entry serializes as one of the documented strings.
        for (_name, status) in list_request_fields() {
            assert!(status == "supported" || status == "forwarded");
        }
    }
}
