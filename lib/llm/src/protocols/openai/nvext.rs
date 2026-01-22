// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::{Validate, ValidationError};

pub use crate::protocols::common::timing::TimingInfo;

pub const HEADER_WORKER_INSTANCE_ID: &str = "x-worker-instance-id";
pub const HEADER_PREFILL_INSTANCE_ID: &str = "x-prefill-instance-id";

/// Apply routing overrides from HTTP headers to nvext.
///
/// Header mappings:
/// - `x-worker-instance-id` -> `backend_instance_id` and `decode_worker_id`
/// - `x-prefill-instance-id` -> `prefill_worker_id`
///
/// Headers take priority over existing nvext values when present.
/// If no headers are present, returns the original nvext unchanged.
pub fn apply_header_routing_overrides(nvext: Option<NvExt>, headers: &HeaderMap) -> Option<NvExt> {
    let worker_id = headers
        .get(HEADER_WORKER_INSTANCE_ID)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let prefill_id = headers
        .get(HEADER_PREFILL_INSTANCE_ID)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    if worker_id.is_none() && prefill_id.is_none() {
        return nvext;
    }

    let mut ext = nvext.unwrap_or_default();
    if let Some(id) = worker_id {
        ext.backend_instance_id = Some(id);
        ext.decode_worker_id = Some(id);
    }
    if let Some(id) = prefill_id {
        ext.prefill_worker_id = Some(id);
    }
    Some(ext)
}

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

    // Test apply_header_routing_overrides - worker header present, prefill header absent
    #[test]
    fn test_apply_header_routing_overrides() {
        use axum::http::HeaderMap;

        // Only HEADER_WORKER_INSTANCE_ID is in the header
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_WORKER_INSTANCE_ID, "123".parse().unwrap());
        // Note: HEADER_PREFILL_INSTANCE_ID is NOT in the header

        let nvext = NvExt::builder()
            .backend_instance_id(999)
            .decode_worker_id(888)
            .prefill_worker_id(777)
            .build()
            .unwrap();

        let result = apply_header_routing_overrides(Some(nvext), &headers).unwrap();

        // Header should override backend_instance_id and decode_worker_id
        assert_eq!(result.backend_instance_id, Some(123));
        assert_eq!(result.decode_worker_id, Some(123));
        // prefill_worker_id should remain from original nvext (not overwritten by header)
        assert_eq!(result.prefill_worker_id, Some(777));
    }
}
