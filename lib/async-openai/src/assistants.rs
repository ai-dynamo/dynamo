// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use serde::Serialize;

use crate::{
    Client,
    config::Config,
    error::OpenAIError,
    types::{
        AssistantObject, CreateAssistantRequest, DeleteAssistantResponse, ListAssistantsResponse,
        ModifyAssistantRequest,
    },
};

/// Build assistants that can call models and use tools to perform tasks.
///
/// [Get started with the Assistants API](https://platform.openai.com/docs/assistants)
pub struct Assistants<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Assistants<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Create an assistant with a model and instructions.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn create(
        &self,
        request: CreateAssistantRequest,
    ) -> Result<AssistantObject, OpenAIError> {
        self.client.post("/assistants", request).await
    }

    /// Retrieves an assistant.
    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn retrieve(&self, assistant_id: &str) -> Result<AssistantObject, OpenAIError> {
        self.client
            .get(&format!("/assistants/{assistant_id}"))
            .await
    }

    /// Modifies an assistant.
    #[crate::byot(T0 = std::fmt::Display, T1 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn update(
        &self,
        assistant_id: &str,
        request: ModifyAssistantRequest,
    ) -> Result<AssistantObject, OpenAIError> {
        self.client
            .post(&format!("/assistants/{assistant_id}"), request)
            .await
    }

    /// Delete an assistant.
    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn delete(&self, assistant_id: &str) -> Result<DeleteAssistantResponse, OpenAIError> {
        self.client
            .delete(&format!("/assistants/{assistant_id}"))
            .await
    }

    /// Returns a list of assistants.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn list<Q>(&self, query: &Q) -> Result<ListAssistantsResponse, OpenAIError>
    where
        Q: Serialize + ?Sized,
    {
        self.client.get_with_query("/assistants", &query).await
    }
}
