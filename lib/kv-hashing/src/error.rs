// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Error types for [`crate`].

use dynamo_tokens::{MmInfoError, TokenBlockError};

/// Errors raised while constructing a [`crate::Request`] or computing block hashes.
#[derive(Debug, thiserror::Error)]
pub enum KvHashingError {
    /// Multimodal info validation failed.
    #[error(transparent)]
    MmInfo(#[from] MmInfoError),

    /// An error originating from [`dynamo_tokens`] block formation.
    #[error(transparent)]
    TokenBlock(#[from] TokenBlockError),

    /// Salt payload could not be canonicalized for hashing.
    ///
    /// In practice this is unreachable for the field types we serialize, but the path is
    /// preserved so the public API never panics on serialization.
    #[error("failed to canonicalize salt payload: {0}")]
    SaltSerialization(#[from] serde_json::Error),

    /// A required field was not set on the [`crate::RequestBuilder`].
    #[error("RequestBuilder missing required field: {0}")]
    MissingField(&'static str),
}
