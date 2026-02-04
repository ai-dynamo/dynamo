// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request types for the scheduler and connector.

use derive_builder::Builder;
use dynamo_tokens::{Tokens, compute_hash_v2};
use serde::Serialize;

/// Metadata for KVBM request integration.
///
/// This struct holds optional metadata that can be passed from the scheduler
/// to the connector. Fields will be added as needed.
#[derive(Debug, Clone, Default)]
pub struct RequestMetadata {
    // Empty for now - will be extended in the future
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
        Self::with_priority(
            request_id,
            tokens,
            lora_name,
            salt,
            min_tokens,
            max_tokens,
            None, // priority
            metadata,
        )
    }

    /// Create a new request with all parameters including priority.
    pub fn with_priority(
        request_id: impl Into<String>,
        tokens: impl Into<Tokens>,
        lora_name: Option<String>,
        salt: Option<String>,
        min_tokens: Option<usize>,
        max_tokens: Option<usize>,
        priority: Option<usize>,
        metadata: Option<RequestMetadata>,
    ) -> Self {
        // Pack any data that needs to be included in the salt hash into [`SaltPayload`]
        #[derive(Serialize)]
        struct SaltPayload<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<&'a str>,
        }

        let request_id = request_id.into();
        let payload = SaltPayload {
            salt: salt.as_deref(),
            lora_name: lora_name.as_deref(),
        };
        let salt_bytes = serde_json::to_vec(&payload).expect("failed to serialize salt payload");
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        Self {
            request_id,
            tokens: tokens.into(),
            lora_name,
            salt_hash,
            min_tokens,
            max_tokens,
            priority,
            restart_count: 0,
            metadata,
        }
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
        Self::with_token_limits(request_id, tokens, lora_name, salt, None, max_tokens, metadata)
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
}
