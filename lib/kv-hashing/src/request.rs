// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Universal [`Request`] type used to derive block hashes.

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
#[derive(Debug, Clone)]
pub struct Request {
    pub(crate) tokens: Vec<Token>,
    pub(crate) lora_name: Option<String>,
    pub(crate) salt: Option<String>,
    /// Validated, sorted multimodal placeholder runs.
    pub(crate) mm_info: Vec<RequestMmObjectInfo>,
}

impl Request {
    /// Builds a Request, validating and sorting `mm_info`.
    pub fn new(
        tokens: Vec<Token>,
        lora_name: Option<String>,
        salt: Option<String>,
        mm_info: Vec<RequestMmObjectInfo>,
    ) -> Result<Self, KvHashingError> {
        // Reuse the tokens-crate validator so behaviour matches downstream block formation.
        let token_mm: Vec<TokenBlockMmInfo> = mm_info.iter().copied().map(Into::into).collect();
        let validated = validate_and_sort_mm_info(&token_mm, tokens.len())?;
        let mm_info = validated.into_iter().map(Into::into).collect();
        Ok(Self {
            tokens,
            lora_name,
            salt,
            mm_info,
        })
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
