// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

//! Dynamo-owned overrides of the upstream `async-openai` embedding response
//! types.
//!
//! Upstream's `Embedding.embedding` is typed `Vec<f32>`, which can't
//! represent the `encoding_format=base64` shape from the OpenAI spec
//! (where each per-input embedding is a base64-encoded little-endian
//! f32 byte string). We own the narrowest subtree -- [`Embedding`] and
//! [`CreateEmbeddingResponse`] -- so the wire-level value can be either a
//! JSON array of floats or a JSON string.
//!
//! Everything else (request, [`EmbeddingInput`], [`EmbeddingUsage`],
//! [`EncodingFormat`], the builder types, the base64-specific upstream
//! types like `Base64Embedding`) is re-exported via the glob below;
//! the local [`Embedding`] and [`CreateEmbeddingResponse`] shadow their
//! upstream namesakes. This preserves the previous full re-export
//! surface for downstream crates that may have been importing those
//! other symbols.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
pub use async_openai::types::embeddings::*;

/// Per-input embedding payload.
///
/// The OpenAI `/v1/embeddings` spec returns either a vector of floats
/// (`encoding_format="float"`, the default) or a base64 string encoding
/// raw little-endian `f32` bytes (`encoding_format="base64"`).
///
/// Serializes as an untagged union: a JSON array for [`Float`] and a JSON
/// string for [`Base64`], matching the OpenAI wire format.
///
/// [`Float`]: EmbeddingVector::Float
/// [`Base64`]: EmbeddingVector::Base64
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingVector {
    Float(Vec<f32>),
    Base64(String),
}

impl EmbeddingVector {
    /// Number of float dimensions if this is the `Float` variant. Returns
    /// `None` for `Base64` -- the count is encoded in the byte length and
    /// only resolved on decode.
    pub fn dim(&self) -> Option<usize> {
        match self {
            EmbeddingVector::Float(v) => Some(v.len()),
            EmbeddingVector::Base64(_) => None,
        }
    }
}

impl From<Vec<f32>> for EmbeddingVector {
    fn from(v: Vec<f32>) -> Self {
        EmbeddingVector::Float(v)
    }
}

/// A single embedding object inside an embeddings response.
///
/// Mirrors `async_openai::types::embeddings::Embedding` except for the
/// `embedding` field, which is widened to [`EmbeddingVector`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Embedding {
    pub index: u32,
    pub object: String,
    pub embedding: EmbeddingVector,
}

/// Top-level `/v1/embeddings` response.
///
/// Mirrors `async_openai::types::embeddings::CreateEmbeddingResponse`
/// except the `data` field carries the Dynamo-owned [`Embedding`].
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CreateEmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<Embedding>,
    pub usage: EmbeddingUsage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_variant_serializes_as_array() {
        let v = EmbeddingVector::Float(vec![0.1, 0.2, 0.3]);
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "[0.1,0.2,0.3]");
    }

    #[test]
    fn base64_variant_serializes_as_string() {
        let v = EmbeddingVector::Base64("AAACOQ==".to_string());
        let json = serde_json::to_string(&v).unwrap();
        assert_eq!(json, "\"AAACOQ==\"");
    }

    #[test]
    fn deserializes_array_as_float() {
        let v: EmbeddingVector = serde_json::from_str("[0.1, 0.2, 0.3]").unwrap();
        assert_eq!(v, EmbeddingVector::Float(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn deserializes_string_as_base64() {
        let v: EmbeddingVector = serde_json::from_str("\"AAACOQ==\"").unwrap();
        assert_eq!(v, EmbeddingVector::Base64("AAACOQ==".to_string()));
    }

    #[test]
    fn round_trips_via_embedding_struct() {
        let e = Embedding {
            index: 0,
            object: "embedding".to_string(),
            embedding: EmbeddingVector::Float(vec![1.0, 2.0]),
        };
        let json = serde_json::to_string(&e).unwrap();
        let parsed: Embedding = serde_json::from_str(&json).unwrap();
        assert_eq!(e, parsed);

        let e = Embedding {
            index: 1,
            object: "embedding".to_string(),
            embedding: EmbeddingVector::Base64("AAACOQ==".to_string()),
        };
        let json = serde_json::to_string(&e).unwrap();
        let parsed: Embedding = serde_json::from_str(&json).unwrap();
        assert_eq!(e, parsed);
    }
}
