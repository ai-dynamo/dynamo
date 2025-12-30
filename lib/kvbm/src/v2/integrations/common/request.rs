// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request types for the scheduler and connector.

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
#[derive(Debug, Clone)]
pub struct Request {
    pub request_id: String,
    pub tokens: Tokens,
    pub lora_name: Option<String>,
    pub salt_hash: u64,
    pub max_tokens: Option<usize>,
    /// Optional metadata for connector integration.
    /// This field is completely optional - the scheduler and connector
    /// work correctly without it.
    pub metadata: Option<RequestMetadata>,
}

impl Request {
    /// Create a new request without metadata.
    pub fn new(
        request_id: impl Into<String>,
        tokens: impl Into<Tokens>,
        lora_name: Option<String>,
        salt: Option<String>,
        max_tokens: Option<usize>,
    ) -> Self {
        Self::with_metadata(request_id, tokens, lora_name, salt, max_tokens, None)
    }

    /// Create a new request with optional metadata.
    pub fn with_metadata(
        request_id: impl Into<String>,
        tokens: impl Into<Tokens>,
        lora_name: Option<String>,
        salt: Option<String>,
        max_tokens: Option<usize>,
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
            max_tokens,
            metadata,
        }
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
            max_tokens: self.max_tokens,
            metadata: None,
        }
    }

    /// Get the metadata if present.
    pub fn metadata(&self) -> Option<&RequestMetadata> {
        self.metadata.as_ref()
    }
}
