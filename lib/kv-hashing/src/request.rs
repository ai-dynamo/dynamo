// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Universal [`Request`] type used to derive block hashes.

use derive_builder::Builder;
use dynamo_tokens::{Token, TokenBlockMmInfo, validate_and_sort_mm_info};
use serde::{Deserialize, Serialize};

use crate::error::KvHashingError;

/// Multimodal placeholder run as carried on a [`Request`].
///
/// Mirrors [`dynamo_tokens::TokenBlockMmInfo`]; kept distinct so the public Request shape
/// is owned by the kv-hashing crate. `From` conversions are provided in both directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestMmObjectInfo {
    /// Hash identifying the multimodal object.
    pub mm_hash: u64,
    /// Start position of the placeholder run in the full token sequence (zero-based).
    pub offset: usize,
    /// Number of placeholder slots in the run.
    pub length: usize,
}

impl From<RequestMmObjectInfo> for TokenBlockMmInfo {
    fn from(v: RequestMmObjectInfo) -> Self {
        Self {
            mm_hash: v.mm_hash,
            offset: v.offset,
            length: v.length,
        }
    }
}

impl From<TokenBlockMmInfo> for RequestMmObjectInfo {
    fn from(v: TokenBlockMmInfo) -> Self {
        Self {
            mm_hash: v.mm_hash,
            offset: v.offset,
            length: v.length,
        }
    }
}

/// Canonical Request used to derive a deterministic sequence of block hashes.
///
/// Construction validates `mm_info` (no overlap, no out-of-bounds, no zero-length) and
/// sorts it by `offset`. The validated/sorted state is the only way to construct a
/// `Request`, so all downstream block-formation code can trust the invariant.
///
/// Built via the owned [`RequestBuilder`] (no clones on the build path):
///
/// ```ignore
/// let request = Request::builder()
///     .tokens(tokens)
///     .lora_name(Some("lora-x".into()))
///     .salt(Some("model-tag".into()))
///     .mm_info(vec![/* RequestMmObjectInfo ... */])
///     .build()?;
/// ```
#[derive(Debug, Clone, Builder)]
#[builder(
    pattern = "owned",
    build_fn(private, name = "build_internal", error = "KvHashingError"),
    derive(Debug)
)]
pub struct Request {
    /// Token IDs of the request.
    #[builder(setter(into))]
    pub(crate) tokens: Vec<Token>,
    /// Optional LoRA adapter name.
    #[builder(default, setter(into))]
    pub(crate) lora_name: Option<String>,
    /// Optional free-form caller salt mixed into the per-request `SaltHash`.
    #[builder(default, setter(into))]
    pub(crate) salt: Option<String>,
    /// Multimodal placeholder runs. Validated and sorted by `build()`.
    #[builder(default)]
    pub(crate) mm_info: Vec<RequestMmObjectInfo>,
}

impl Request {
    /// Returns a fresh owned [`RequestBuilder`].
    pub fn builder() -> RequestBuilder {
        RequestBuilder::default()
    }

    /// Returns the request tokens.
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Returns the LoRA adapter name, if any.
    pub fn lora_name(&self) -> Option<&str> {
        self.lora_name.as_deref()
    }

    /// Returns the free-form caller salt, if any.
    pub fn salt(&self) -> Option<&str> {
        self.salt.as_deref()
    }

    /// Returns the validated, sorted multimodal runs.
    pub fn mm_info(&self) -> &[RequestMmObjectInfo] {
        &self.mm_info
    }

    /// Returns `mm_info` projected to the dynamo-tokens type, ready for
    /// [`dynamo_tokens::TokenBlockSequence::new_with_mm`] (already sorted/validated).
    pub(crate) fn token_mm_info(&self) -> Vec<TokenBlockMmInfo> {
        self.mm_info.iter().copied().map(Into::into).collect()
    }
}

impl RequestBuilder {
    /// Builds the [`Request`], validating and sorting `mm_info` against the token length.
    pub fn build(self) -> Result<Request, KvHashingError> {
        let mut request = self.build_internal()?;
        // Validate against the actual token length, sort by offset, and write back.
        // No clones: we move out of `request.mm_info`, transform via Into, and replace.
        let token_mm: Vec<TokenBlockMmInfo> = std::mem::take(&mut request.mm_info)
            .into_iter()
            .map(Into::into)
            .collect();
        let validated = validate_and_sort_mm_info(&token_mm, request.tokens.len())?;
        request.mm_info = validated.into_iter().map(Into::into).collect();
        Ok(request)
    }
}

impl From<derive_builder::UninitializedFieldError> for KvHashingError {
    fn from(e: derive_builder::UninitializedFieldError) -> Self {
        KvHashingError::MissingField(e.field_name())
    }
}
