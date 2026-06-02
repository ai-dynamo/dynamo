// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::Validate;

mod aggregator;
mod nvext;

pub use nvext::{NvExt, NvExtProvider};

#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateEmbeddingRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateEmbeddingRequest,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI unary chat completion response, embedded
///   using `serde(flatten)`.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateEmbeddingResponse {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateEmbeddingResponse,

    /// Internal-only handle for an embedding response delivered via POSIX
    /// shared memory (same-node fast path). When present, `inner.data` is
    /// empty and the vectors live in `/dev/shm/<name>`; the HTTP handler reads
    /// them directly, populates `inner.data`, and unlinks the segment. Always
    /// `None` in the client-facing response (skipped when serializing).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_shm: Option<EmbeddingShmHandle>,
}

/// Handle describing an embedding response staged in POSIX shared memory.
///
/// The worker writes a contiguous `[count, dim]` little-endian `f32` buffer to
/// the shared-memory object `name` (a tmpfs file at `/dev/shm/<name>` on Linux)
/// and sends only this small handle over the request plane. The frontend reads
/// the `count * dim * 4` bytes directly and unlinks the segment. This avoids
/// base64-encoding + JSON-serializing the vectors across the worker->frontend
/// hop on a co-located deployment.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingShmHandle {
    /// POSIX shared-memory object name (tmpfs file `/dev/shm/<name>` on Linux).
    pub name: String,
    /// Number of embedding vectors stored in the buffer.
    pub count: usize,
    /// Dimensionality (number of `f32` elements) of each vector.
    pub dim: usize,
}

impl NvCreateEmbeddingResponse {
    pub fn empty() -> Self {
        Self {
            inner: dynamo_protocols::types::CreateEmbeddingResponse {
                object: "list".to_string(),
                model: "embedding".to_string(),
                data: vec![],
                usage: dynamo_protocols::types::EmbeddingUsage {
                    prompt_tokens: 0,
                    total_tokens: 0,
                },
            },
            data_shm: None,
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateEmbeddingRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateEmbeddingRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateEmbeddingRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateEmbeddingRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}
