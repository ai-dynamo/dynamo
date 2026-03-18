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
    types::responses::{CreateResponse, Response, ResponseStream},
};

/// Given text input or a list of context items, the model will generate a response.
///
/// Related guide: [Responses](https://platform.openai.com/docs/api-reference/responses)
pub struct Responses<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Responses<'c, C> {
    /// Constructs a new Responses client.
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Creates a model response for the given input.
    #[crate::byot(
        T0 = serde::Serialize,
        R = serde::de::DeserializeOwned
    )]
    pub async fn create(&self, request: CreateResponse) -> Result<Response, OpenAIError> {
        self.client.post("/responses", request).await
    }

    /// Creates a model response for the given input with streaming.
    ///
    /// Response events will be sent as server-sent events as they become available,
    #[crate::byot(
        T0 = serde::Serialize,
        R = serde::de::DeserializeOwned,
        stream = "true",
        where_clause = "R: std::marker::Send + 'static"
    )]
    #[allow(unused_mut)]
    pub async fn create_stream(
        &self,
        mut request: CreateResponse,
    ) -> Result<ResponseStream, OpenAIError> {
        #[cfg(not(feature = "byot"))]
        {
            if matches!(request.stream, Some(false)) {
                return Err(OpenAIError::InvalidArgument(
                    "When stream is false, use Responses::create".into(),
                ));
            }
            request.stream = Some(true);
        }
        Ok(self.client.post_stream("/responses", request).await)
    }
}
