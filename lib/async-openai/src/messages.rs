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
        CreateMessageRequest, DeleteMessageResponse, ListMessagesResponse, MessageObject,
        ModifyMessageRequest,
    },
};

/// Represents a message within a [thread](https://platform.openai.com/docs/api-reference/threads).
pub struct Messages<'c, C: Config> {
    ///  The ID of the [thread](https://platform.openai.com/docs/api-reference/threads) to create a message for.
    pub thread_id: String,
    client: &'c Client<C>,
}

impl<'c, C: Config> Messages<'c, C> {
    pub fn new(client: &'c Client<C>, thread_id: &str) -> Self {
        Self {
            client,
            thread_id: thread_id.into(),
        }
    }

    /// Create a message.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn create(
        &self,
        request: CreateMessageRequest,
    ) -> Result<MessageObject, OpenAIError> {
        self.client
            .post(&format!("/threads/{}/messages", self.thread_id), request)
            .await
    }

    /// Retrieve a message.
    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn retrieve(&self, message_id: &str) -> Result<MessageObject, OpenAIError> {
        self.client
            .get(&format!(
                "/threads/{}/messages/{message_id}",
                self.thread_id
            ))
            .await
    }

    /// Modifies a message.
    #[crate::byot(T0 = std::fmt::Display, T1 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn update(
        &self,
        message_id: &str,
        request: ModifyMessageRequest,
    ) -> Result<MessageObject, OpenAIError> {
        self.client
            .post(
                &format!("/threads/{}/messages/{message_id}", self.thread_id),
                request,
            )
            .await
    }

    /// Returns a list of messages for a given thread.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn list<Q>(&self, query: &Q) -> Result<ListMessagesResponse, OpenAIError>
    where
        Q: Serialize + ?Sized,
    {
        self.client
            .get_with_query(&format!("/threads/{}/messages", self.thread_id), &query)
            .await
    }

    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn delete(&self, message_id: &str) -> Result<DeleteMessageResponse, OpenAIError> {
        self.client
            .delete(&format!(
                "/threads/{}/messages/{message_id}",
                self.thread_id
            ))
            .await
    }
}
