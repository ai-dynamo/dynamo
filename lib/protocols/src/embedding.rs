// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use crate::{
    Client,
    config::Config,
    error::OpenAIError,
    types::{CreateBase64EmbeddingResponse, CreateEmbeddingRequest, CreateEmbeddingResponse},
};

#[cfg(not(feature = "byot"))]
use crate::types::EncodingFormat;

/// Get a vector representation of a given input that can be easily
/// consumed by machine learning models and algorithms.
///
/// Related guide: [Embeddings](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
pub struct Embeddings<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Embeddings<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Creates an embedding vector representing the input text.
    ///
    /// byot: In serialized `request` you must ensure "encoding_format" is not "base64"
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn create(
        &self,
        request: CreateEmbeddingRequest,
    ) -> Result<CreateEmbeddingResponse, OpenAIError> {
        #[cfg(not(feature = "byot"))]
        {
            if matches!(request.encoding_format, Some(EncodingFormat::Base64)) {
                return Err(OpenAIError::InvalidArgument(
                    "When encoding_format is base64, use Embeddings::create_base64".into(),
                ));
            }
        }
        self.client.post("/embeddings", request).await
    }

    /// Creates an embedding vector representing the input text.
    ///
    /// The response will contain the embedding in base64 format.
    ///
    /// byot: In serialized `request` you must ensure "encoding_format" is "base64"
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn create_base64(
        &self,
        request: CreateEmbeddingRequest,
    ) -> Result<CreateBase64EmbeddingResponse, OpenAIError> {
        #[cfg(not(feature = "byot"))]
        {
            if !matches!(request.encoding_format, Some(EncodingFormat::Base64)) {
                return Err(OpenAIError::InvalidArgument(
                    "When encoding_format is not base64, use Embeddings::create".into(),
                ));
            }
        }
        self.client.post("/embeddings", request).await
    }
}
