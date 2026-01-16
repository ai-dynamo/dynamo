// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::{Validate, ValidationError};

pub use crate::protocols::common::timing::TimingInfo;

/// HTTP header for specifying decode/backend worker instance ID.
/// Used for both `backend_instance_id` (aggregated mode) and `decode_worker_id` (disaggregated mode).
pub const HEADER_WORKER_INSTANCE_ID: &str = "x-worker-instance-id";

/// HTTP header for specifying prefill worker instance ID (disaggregated mode).
pub const HEADER_PREFILL_INSTANCE_ID: &str = "x-prefill-instance-id";

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
    fn raw_prompt(&self) -> Option<String>;
}

/// Worker ID information for disaggregated serving
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WorkerIdInfo {
    /// The prefill worker ID that processed this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,

    /// The decode worker ID that processed this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,
}

/// NVIDIA LLM response extensions
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct NvExtResponse {
    /// Worker ID information (prefill and decode worker IDs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<WorkerIdInfo>,

    /// Per-request timing information
    /// Populated when client requests `extra_fields: ["timing"]`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingInfo>,

    /// Token IDs for GAIE Stage 1 query-only mode
    /// Contains the tokenized prompt for reuse in Stage 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,
}

/// NVIDIA LLM extensions to the OpenAI API
#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[validate(schema(function = "validate_nv_ext"))]
pub struct NvExt {
    /// If true, sampling will be forced to be greedy.
    /// The backend is responsible for selecting the correct backend-specific options to
    /// implement this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub greed_sampling: Option<bool>,

    /// If true, the preproessor will try to bypass the prompt template and pass the prompt directly to
    /// to the tokenizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub use_raw_prompt: Option<bool>,

    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,

    /// Targeted backend instance ID for the request
    /// If set, the request will be routed to backend instance with the given ID.
    /// If not set, the request will be routed to the best matching instance.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_instance_id: Option<u64>,

    /// Pre-tokenized data to use instead of tokenizing the prompt
    /// If provided along with backend_instance_id, these tokens will be used directly
    /// and tokenization will be skipped.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_data: Option<Vec<u32>>,

    /// Maximum number of thinking tokens allowed
    /// NOTE: Currently passed through to backends as a no-op for future implementation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub max_thinking_tokens: Option<u32>,

    /// Extra fields to be included in the response's nvext
    /// This is a list of field names that should be populated in the response
    /// Supported fields: "worker_id", "timing", which has a 1:1 mapping with the NvExtResponse names
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub extra_fields: Option<Vec<String>>,

    /// Targeted prefill worker ID for disaggregated serving (GAIE Stage 2)
    /// When set, the request will be routed to this specific prefill worker.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,

    /// Targeted decode worker ID for disaggregated serving (GAIE Stage 2)
    /// When set, the request will be routed to this specific decode worker.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,

    /// Controls whether the router should manage local bookkeeping (add_request,
    /// mark_prefill_completed, free) for this request.
    ///
    /// - `None` or `true`: Router handles bookkeeping locally (default behavior)
    /// - `false`: External caller (e.g., GAIE sidecar) handles bookkeeping via C FFI
    ///
    /// Set to `false` for GAIE Stage 2 when the EPP/sidecar manages request lifecycle.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_local_updates: Option<bool>,

    /// Expected number of output tokens for this request.
    /// Used as a hint for routing decisions to estimate resource requirements.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output_tokens: Option<u32>,
}

impl Default for NvExt {
    fn default() -> Self {
        NvExt::builder().build().unwrap()
    }
}

impl NvExt {
    pub fn builder() -> NvExtBuilder {
        NvExtBuilder::default()
    }

    /// Apply routing hints from HTTP headers, with headers taking priority over existing values.
    ///
    /// Header mappings:
    /// - `x-worker-instance-id` → `backend_instance_id` and `decode_worker_id`
    /// - `x-prefill-instance-id` → `prefill_worker_id`
    ///
    /// If a header is present and parseable as u64, it overrides the corresponding nvext field.
    /// If the header is absent or unparseable, the existing nvext value is preserved.
    pub fn apply_routing_headers(&mut self, headers: &HeaderMap) {
        // Extract worker instance ID from header (used for both backend_instance_id and decode_worker_id)
        if let Some(worker_id) = parse_u64_header(headers, HEADER_WORKER_INSTANCE_ID) {
            self.backend_instance_id = Some(worker_id);
            self.decode_worker_id = Some(worker_id);
        }

        // Extract prefill instance ID from header
        if let Some(prefill_id) = parse_u64_header(headers, HEADER_PREFILL_INSTANCE_ID) {
            self.prefill_worker_id = Some(prefill_id);
        }
    }
}

/// Parse a u64 value from an HTTP header.
/// Returns None if the header is missing or the value cannot be parsed.
fn parse_u64_header(headers: &HeaderMap, header_name: &str) -> Option<u64> {
    headers
        .get(header_name)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok())
}

/// Apply routing hints from HTTP headers to an optional NvExt.
///
/// If routing headers are present, this ensures an NvExt exists and applies the headers to it.
/// Headers take priority over any existing nvext values.
///
/// Returns the modified Option<NvExt>.
pub fn apply_routing_headers_to_nvext(nvext: Option<NvExt>, headers: &HeaderMap) -> Option<NvExt> {
    // Check if any routing headers are present
    let has_worker_header = headers.contains_key(HEADER_WORKER_INSTANCE_ID);
    let has_prefill_header = headers.contains_key(HEADER_PREFILL_INSTANCE_ID);

    if !has_worker_header && !has_prefill_header {
        // No routing headers, return nvext unchanged
        return nvext;
    }

    // Ensure NvExt exists and apply headers
    let mut ext = nvext.unwrap_or_default();
    ext.apply_routing_headers(headers);
    Some(ext)
}

fn validate_nv_ext(_nv_ext: &NvExt) -> Result<(), ValidationError> {
    Ok(())
}

impl NvExtBuilder {
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("stop should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use validator::Validate;

    use super::*;

    // Test default builder configuration
    #[test]
    fn test_nv_ext_builder_default() {
        let nv_ext = NvExt::builder().build().unwrap();
        assert_eq!(nv_ext.greed_sampling, None);
        assert_eq!(nv_ext.use_raw_prompt, None);
        assert_eq!(nv_ext.annotations, None);
        assert_eq!(nv_ext.backend_instance_id, None);
        assert_eq!(nv_ext.token_data, None);
        assert_eq!(nv_ext.max_thinking_tokens, None);
        assert_eq!(nv_ext.extra_fields, None);
        assert_eq!(nv_ext.prefill_worker_id, None);
        assert_eq!(nv_ext.decode_worker_id, None);
        assert_eq!(nv_ext.enable_local_updates, None);
        assert_eq!(nv_ext.expected_output_tokens, None);
    }

    // Test valid builder configurations
    #[test]
    fn test_nv_ext_builder_custom() {
        let nv_ext = NvExt::builder()
            .greed_sampling(true)
            .use_raw_prompt(true)
            .backend_instance_id(42)
            .token_data(vec![1, 2, 3, 4])
            .max_thinking_tokens(1024)
            .extra_fields(vec!["worker_id".to_string()])
            .build()
            .unwrap();

        assert_eq!(nv_ext.greed_sampling, Some(true));
        assert_eq!(nv_ext.use_raw_prompt, Some(true));
        assert_eq!(nv_ext.backend_instance_id, Some(42));
        assert_eq!(nv_ext.token_data, Some(vec![1, 2, 3, 4]));
        assert_eq!(nv_ext.max_thinking_tokens, Some(1024));
        assert_eq!(nv_ext.extra_fields, Some(vec!["worker_id".to_string()]));
        // Validate the built struct
        assert!(nv_ext.validate().is_ok());
    }

    // Test GAIE Stage 2 disaggregated worker IDs
    #[test]
    fn test_nv_ext_disagg_worker_ids() {
        let nv_ext = NvExt::builder()
            .prefill_worker_id(100)
            .decode_worker_id(200)
            .build()
            .unwrap();

        assert_eq!(nv_ext.prefill_worker_id, Some(100));
        assert_eq!(nv_ext.decode_worker_id, Some(200));
        assert!(nv_ext.validate().is_ok());
    }

    // Test routing headers override nvext values
    #[test]
    fn test_apply_routing_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_WORKER_INSTANCE_ID, "123".parse().unwrap());
        headers.insert(HEADER_PREFILL_INSTANCE_ID, "456".parse().unwrap());

        let mut nv_ext = NvExt::builder()
            .backend_instance_id(999)
            .prefill_worker_id(888)
            .decode_worker_id(777)
            .build()
            .unwrap();

        nv_ext.apply_routing_headers(&headers);

        // Headers should override existing values
        assert_eq!(nv_ext.backend_instance_id, Some(123));
        assert_eq!(nv_ext.decode_worker_id, Some(123));
        assert_eq!(nv_ext.prefill_worker_id, Some(456));
    }

    // Test routing headers with no existing nvext
    #[test]
    fn test_apply_routing_headers_to_none_nvext() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_WORKER_INSTANCE_ID, "100".parse().unwrap());

        let result = apply_routing_headers_to_nvext(None, &headers);

        assert!(result.is_some());
        let nv_ext = result.unwrap();
        assert_eq!(nv_ext.backend_instance_id, Some(100));
        assert_eq!(nv_ext.decode_worker_id, Some(100));
    }

    // Test no routing headers preserves nvext unchanged
    #[test]
    fn test_no_routing_headers_preserves_nvext() {
        let headers = HeaderMap::new();

        let nv_ext = NvExt::builder()
            .backend_instance_id(42)
            .build()
            .unwrap();

        let result = apply_routing_headers_to_nvext(Some(nv_ext), &headers);

        assert!(result.is_some());
        assert_eq!(result.unwrap().backend_instance_id, Some(42));
    }

    // Test invalid header values are ignored
    #[test]
    fn test_invalid_header_values_ignored() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_WORKER_INSTANCE_ID, "not_a_number".parse().unwrap());

        let nv_ext = NvExt::builder()
            .backend_instance_id(42)
            .build()
            .unwrap();

        // Even though header exists, invalid value should not override
        let result = apply_routing_headers_to_nvext(Some(nv_ext), &headers);

        // Since header key exists but value is unparseable, nvext is still created/modified
        // but the invalid value doesn't override
        assert!(result.is_some());
        let ext = result.unwrap();
        assert_eq!(ext.backend_instance_id, Some(42)); // Original preserved
    }
}
