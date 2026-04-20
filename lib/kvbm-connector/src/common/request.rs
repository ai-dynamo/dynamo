// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request types for the scheduler and connector.

use derive_builder::Builder;
use dynamo_tokens::{Tokens, compute_hash_v2};
use serde::Serialize;

/// Metadata for KVBM request integration.
///
/// Holds optional data forwarded from the scheduler (e.g. a vLLM
/// `Request`) into the connector layer. Today this carries the raw
/// `kv_transfer_params` JSON as an opaque `serde_json::Value`; a
/// follow-up change will introduce a typed `TransferParams` that
/// parses this lazily on demand.
#[derive(Debug, Clone, Default)]
pub struct RequestMetadata {
    /// Connector-specific KV transfer parameters, as received from the
    /// scheduler protocol. `None` when the upstream request did not
    /// supply any (the common case for non-disaggregated requests).
    pub kv_transfer_params: Option<serde_json::Value>,
}

impl RequestMetadata {
    /// Construct metadata carrying only a `kv_transfer_params` JSON payload.
    pub fn with_kv_transfer_params(value: serde_json::Value) -> Self {
        Self {
            kv_transfer_params: Some(value),
        }
    }
}

/// Minimal representation of a scheduler slot request.
///
/// # Builder Pattern
///
/// Use [`Request::builder()`] for a cleaner API:
///
/// ```ignore
/// let request = Request::builder()
///     .request_id("req-1")
///     .tokens(vec![1, 2, 3])
///     .max_tokens(200)
///     .build()
///     .unwrap();
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(
    pattern = "owned",
    build_fn(private, name = "build_internal", error = "RequestBuilderError"),
    setter(into)
)]
pub struct Request {
    /// Unique identifier for this request.
    pub request_id: String,

    /// Input tokens (prompt).
    pub tokens: Tokens,

    /// Optional LoRA adapter name.
    #[builder(default)]
    pub lora_name: Option<String>,

    /// Hash computed from salt and lora_name for prefix cache isolation.
    /// Use the builder's `.salt()` method to set the salt string.
    #[builder(default = "0", setter(skip))]
    pub salt_hash: u64,

    /// Minimum number of output tokens before the request is eligible for eviction.
    ///
    /// When set, the scheduler guarantees that this request will generate at least
    /// `min_tokens` output tokens before it can be preempted/evicted. This is used
    /// by the projection analysis system to ensure every request makes meaningful
    /// progress before being considered for eviction.
    ///
    /// If `None`, the scheduler uses a default based on block alignment:
    /// `min(tokens_to_boundary + 2 * block_size, 3 * block_size)`
    #[builder(default)]
    pub min_tokens: Option<usize>,

    /// Maximum number of output tokens this request can generate.
    ///
    /// When set, the request will finish when it reaches this many output tokens.
    /// Used by the projection system to estimate worst-case block requirements.
    #[builder(default)]
    pub max_tokens: Option<usize>,

    /// User-defined priority for eviction ordering.
    ///
    /// Higher values indicate higher priority (less likely to be evicted).
    /// If `None`, the request has the lowest priority and will be evicted first
    /// when memory pressure requires preemption.
    ///
    /// Requests that are restarted after preemption automatically get their
    /// priority bumped to avoid repeated eviction of the same request.
    #[builder(default)]
    pub priority: Option<usize>,

    /// Number of times this request has been restarted after preemption.
    ///
    /// Used to automatically bump priority after restarts to prevent the same
    /// request from being repeatedly evicted. Each restart increments this
    /// counter and increases the effective priority.
    #[builder(default = "0")]
    pub restart_count: usize,

    /// Optional metadata for connector integration.
    /// This field is completely optional - the scheduler and connector
    /// work correctly without it.
    #[builder(default)]
    pub metadata: Option<RequestMetadata>,
}

/// Error type for RequestBuilder.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RequestBuilderError {
    #[error("Uninitialized field: {0}")]
    UninitializedField(&'static str),
}

impl From<derive_builder::UninitializedFieldError> for RequestBuilderError {
    fn from(e: derive_builder::UninitializedFieldError) -> Self {
        Self::UninitializedField(e.field_name())
    }
}

impl From<String> for RequestBuilderError {
    fn from(s: String) -> Self {
        Self::UninitializedField(Box::leak(s.into_boxed_str()))
    }
}

impl RequestBuilder {
    /// Build the Request, computing salt_hash from the optional salt string.
    ///
    /// # Arguments
    /// * `salt` - Optional salt string for prefix cache isolation (combined with lora_name)
    pub fn build(self, salt: Option<&str>) -> Result<Request, RequestBuilderError> {
        // Compute salt_hash
        #[derive(Serialize)]
        struct SaltPayload<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<&'a str>,
        }

        let lora_ref = self.lora_name.as_ref().and_then(|l| l.as_deref());

        let payload = SaltPayload {
            salt,
            lora_name: lora_ref,
        };
        let salt_bytes = serde_json::to_vec(&payload).expect("failed to serialize salt payload");
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        // Build with default salt_hash, then set the computed value
        let mut request = self.build_internal()?;
        request.salt_hash = salt_hash;
        Ok(request)
    }
}

impl Request {
    /// Create a new builder for Request.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let request = Request::builder()
    ///     .request_id("req-1")
    ///     .tokens(vec![1, 2, 3])
    ///     .max_tokens(200)
    ///     .build(None)
    ///     .unwrap();
    /// ```
    pub fn builder() -> RequestBuilder {
        RequestBuilder::default()
    }

    /// Create a new request without metadata.
    pub fn new(
        request_id: impl Into<String>,
        tokens: impl Into<Tokens>,
        lora_name: Option<String>,
        salt: Option<String>,
        max_tokens: Option<usize>,
    ) -> Self {
        Self::with_token_limits(request_id, tokens, lora_name, salt, None, max_tokens, None)
    }

    /// Create a new request with min/max token limits.
    pub fn with_token_limits(
        request_id: impl Into<String>,
        tokens: impl Into<Tokens>,
        lora_name: Option<String>,
        salt: Option<String>,
        min_tokens: Option<usize>,
        max_tokens: Option<usize>,
        metadata: Option<RequestMetadata>,
    ) -> Self {
        let mut builder = Request::builder()
            .request_id(request_id)
            .tokens(tokens)
            .lora_name(lora_name)
            .min_tokens(min_tokens)
            .max_tokens(max_tokens)
            .metadata(metadata);
        // Builder returns Option via strip_option setters on other fields; priority
        // remains None by default.
        builder = builder.priority(None);
        builder
            .build(salt.as_deref())
            .expect("Request builder requires request_id and tokens")
    }

    /// Create a new request with optional metadata (backwards compatibility).
    #[deprecated(since = "0.1.0", note = "Use with_token_limits instead")]
    pub fn with_metadata(
        request_id: impl Into<String>,
        tokens: impl Into<Tokens>,
        lora_name: Option<String>,
        salt: Option<String>,
        max_tokens: Option<usize>,
        metadata: Option<RequestMetadata>,
    ) -> Self {
        Self::with_token_limits(
            request_id, tokens, lora_name, salt, None, max_tokens, metadata,
        )
    }

    /// Clone the request without metadata.
    ///
    /// This creates a copy of the request with all fields except metadata,
    /// which is set to None. Use this when you need a copy but don't need
    /// to preserve the metadata.
    pub fn clone_without_metadata(&self) -> Self {
        Self {
            request_id: self.request_id.clone(),
            tokens: self.tokens.clone(),
            lora_name: self.lora_name.clone(),
            salt_hash: self.salt_hash,
            min_tokens: self.min_tokens,
            max_tokens: self.max_tokens,
            priority: self.priority,
            restart_count: self.restart_count,
            metadata: None,
        }
    }

    /// Bump priority after a restart to avoid repeated eviction.
    ///
    /// Each restart increments the restart_count and adds to the priority,
    /// making the request less likely to be evicted again.
    pub fn mark_restarted(&mut self) {
        self.restart_count += 1;
        // Bump priority: each restart adds 10 to the effective priority
        let current = self.priority.unwrap_or(0);
        self.priority = Some(current.saturating_add(self.restart_count * 10));
    }

    /// Get the effective priority for eviction ordering.
    ///
    /// Returns the user-defined priority if set, otherwise returns 0 (lowest priority).
    /// Used by the projection system to sort eviction candidates.
    pub fn effective_priority(&self) -> usize {
        self.priority.unwrap_or(0)
    }

    /// Get the metadata if present.
    pub fn metadata(&self) -> Option<&RequestMetadata> {
        self.metadata.as_ref()
    }

    /// Borrow the raw KV transfer params JSON, if any.
    ///
    /// Returns `None` when the upstream request did not supply
    /// `kv_transfer_params`. Callers that require the data decide
    /// locally whether absence is fatal.
    pub fn kv_transfer_params(&self) -> Option<&serde_json::Value> {
        self.metadata
            .as_ref()
            .and_then(|m| m.kv_transfer_params.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use serde_json::json;

    #[test]
    fn kv_transfer_params_round_trips() {
        let params = json!({
            "transfer_id": "abc-123",
            "do_remote_prefill": true,
            "remote_block_ids": [1, 2, 3],
        });
        let request = Request::with_token_limits(
            "req-1",
            vec![1_u32, 2, 3],
            None,
            None,
            None,
            Some(128),
            Some(RequestMetadata::with_kv_transfer_params(params.clone())),
        );
        assert_eq!(request.kv_transfer_params(), Some(&params));
    }

    #[test]
    fn kv_transfer_params_absent_when_no_metadata() {
        let request = Request::new("req-2", vec![1_u32, 2, 3], None, None, Some(64));
        assert!(request.kv_transfer_params().is_none());
    }

    #[test]
    fn kv_transfer_params_absent_when_metadata_has_none() {
        let request = Request::with_token_limits(
            "req-3",
            vec![1_u32, 2, 3],
            None,
            None,
            None,
            Some(64),
            Some(RequestMetadata::default()),
        );
        assert!(request.kv_transfer_params().is_none());
    }

    // -------- Property tests for the FFI boundary --------
    //
    // The Python side serializes `dict[str, Any]` with `json.dumps`; the
    // pyo3 shim deserializes with `serde_json::from_str::<serde_json::Value>`
    // and hands the result to `RequestMetadata::with_kv_transfer_params`.
    // These tests mirror that path over arbitrary JSON values that Python
    // could reasonably produce, and assert the value stored on the request
    // is byte-for-byte the one we put in.

    /// Strategy: arbitrary JSON value up to bounded depth/size. Deliberately
    /// keeps primitives inside what `json.dumps` on a plain Python dict
    /// would emit (no NaN/Infinity — `json.dumps` rejects those by default).
    fn arb_json_value() -> impl Strategy<Value = serde_json::Value> {
        let leaf = prop_oneof![
            Just(serde_json::Value::Null),
            any::<bool>().prop_map(serde_json::Value::Bool),
            any::<i64>().prop_map(|n| json!(n)),
            // finite f64s only — json.dumps does not emit NaN/Inf.
            any::<f64>()
                .prop_filter("finite", |f| f.is_finite())
                .prop_map(|f| json!(f)),
            ".*".prop_map(serde_json::Value::String),
        ];
        leaf.prop_recursive(
            4,  // up to 4 levels of nesting
            32, // target total node count
            8,  // max children per collection
            |inner| {
                prop_oneof![
                    prop::collection::vec(inner.clone(), 0..8).prop_map(serde_json::Value::Array),
                    prop::collection::hash_map(".*", inner, 0..8)
                        .prop_map(|m| serde_json::Value::Object(m.into_iter().collect())),
                ]
            },
        )
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, ..ProptestConfig::default() })]

        /// Any value that Python's `json.dumps` could emit must survive the
        /// FFI hop: serialize → parse → store on Request → retrieve.
        #[test]
        fn ffi_json_string_round_trip(value in arb_json_value()) {
            // Python side: json.dumps(dict)
            let wire = serde_json::to_string(&value).expect("serializable");
            // pyo3 shim side: serde_json::from_str
            let parsed: serde_json::Value =
                serde_json::from_str(&wire).expect("valid JSON round-trips");
            let request = Request::with_token_limits(
                "req-prop",
                vec![1_u32, 2, 3],
                None,
                None,
                None,
                Some(128),
                Some(RequestMetadata::with_kv_transfer_params(parsed.clone())),
            );
            prop_assert_eq!(request.kv_transfer_params(), Some(&parsed));
            // Transitivity: re-serializing what we stored should equal `wire`'s
            // canonical form (both sides parsed from the same bytes).
            prop_assert_eq!(
                serde_json::to_string(request.kv_transfer_params().unwrap()).unwrap(),
                serde_json::to_string(&parsed).unwrap()
            );
        }

        /// Object-shaped payloads (the common vLLM case: `dict[str, Any]`)
        /// round-trip and preserve every key. Scoped to top-level objects to
        /// match the vLLM protocol's shape.
        #[test]
        fn top_level_object_preserves_keys(
            m in prop::collection::hash_map(".{1,16}", arb_json_value(), 0..10)
        ) {
            let value = serde_json::Value::Object(m.clone().into_iter().collect());
            let wire = serde_json::to_string(&value).unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&wire).unwrap();

            let request = Request::with_token_limits(
                "req-obj",
                vec![1_u32],
                None,
                None,
                None,
                Some(1),
                Some(RequestMetadata::with_kv_transfer_params(parsed)),
            );
            let stored = request
                .kv_transfer_params()
                .expect("params present")
                .as_object()
                .expect("top-level object");
            for key in m.keys() {
                prop_assert!(stored.contains_key(key), "lost key: {key}");
            }
            prop_assert_eq!(stored.len(), m.len());
        }
    }

    // -------- Spot-check adversarial / boundary cases --------

    #[test]
    fn malformed_json_is_rejected_by_shim_path() {
        // Mirrors the pyo3 shim's failure branch without reaching into pyo3.
        for bad in ["not json", "{unterminated", "", "{\"k\": }"] {
            let err = serde_json::from_str::<serde_json::Value>(bad);
            assert!(err.is_err(), "expected {bad:?} to fail to parse");
        }
    }

    #[test]
    fn deeply_nested_json_round_trips() {
        // vLLM connectors occasionally nest config dicts a few layers deep.
        let value = json!({
            "a": {"b": {"c": {"d": [1, 2, {"e": "deep"}]}}},
        });
        let wire = serde_json::to_string(&value).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&wire).unwrap();
        let request = Request::with_token_limits(
            "req-nested",
            vec![1_u32],
            None,
            None,
            None,
            Some(1),
            Some(RequestMetadata::with_kv_transfer_params(parsed.clone())),
        );
        assert_eq!(request.kv_transfer_params(), Some(&parsed));
    }

    #[test]
    fn unicode_keys_and_values_round_trip() {
        let value = json!({"日本語": "🚀", "emoji-key-🔑": [1, 2, 3]});
        let wire = serde_json::to_string(&value).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&wire).unwrap();
        let request = Request::with_token_limits(
            "req-unicode",
            vec![1_u32],
            None,
            None,
            None,
            Some(1),
            Some(RequestMetadata::with_kv_transfer_params(parsed.clone())),
        );
        assert_eq!(request.kv_transfer_params(), Some(&parsed));
    }
}
