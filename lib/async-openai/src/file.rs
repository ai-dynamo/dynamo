// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Based on https://github.com/64bit/async-openai/ by Himanshu Neema
// Original Copyright (c) 2022 Himanshu Neema
// Licensed under MIT License (see ATTRIBUTIONS-Rust.md)
//
// Modifications Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
// Licensed under Apache 2.0

use bytes::Bytes;
use serde::Serialize;

use crate::{
    Client,
    config::Config,
    error::OpenAIError,
    types::{CreateFileRequest, DeleteFileResponse, ListFilesResponse, OpenAIFile},
};

/// Files are used to upload documents that can be used with features like Assistants and Fine-tuning.
pub struct Files<'c, C: Config> {
    client: &'c Client<C>,
}

impl<'c, C: Config> Files<'c, C> {
    pub fn new(client: &'c Client<C>) -> Self {
        Self { client }
    }

    /// Upload a file that can be used across various endpoints. Individual files can be up to 512 MB, and the size of all files uploaded by one organization can be up to 100 GB.
    ///
    /// The Assistants API supports files up to 2 million tokens and of specific file types. See the [Assistants Tools guide](https://platform.openai.com/docs/assistants/tools) for details.
    ///
    /// The Fine-tuning API only supports `.jsonl` files. The input also has certain required formats for fine-tuning [chat](https://platform.openai.com/docs/api-reference/fine-tuning/chat-input) or [completions](https://platform.openai.com/docs/api-reference/fine-tuning/completions-input) models.
    ///
    ///The Batch API only supports `.jsonl` files up to 100 MB in size. The input also has a specific required [format](https://platform.openai.com/docs/api-reference/batch/request-input).
    ///
    /// Please [contact us](https://help.openai.com/) if you need to increase these storage limits.
    #[crate::byot(
        T0 = Clone,
        R = serde::de::DeserializeOwned,
        where_clause =  "reqwest::multipart::Form: crate::traits::AsyncTryFrom<T0, Error = OpenAIError>",
    )]
    pub async fn create(&self, request: CreateFileRequest) -> Result<OpenAIFile, OpenAIError> {
        self.client.post_form("/files", request).await
    }

    /// Returns a list of files that belong to the user's organization.
    #[crate::byot(T0 = serde::Serialize, R = serde::de::DeserializeOwned)]
    pub async fn list<Q>(&self, query: &Q) -> Result<ListFilesResponse, OpenAIError>
    where
        Q: Serialize + ?Sized,
    {
        self.client.get_with_query("/files", &query).await
    }

    /// Returns information about a specific file.
    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn retrieve(&self, file_id: &str) -> Result<OpenAIFile, OpenAIError> {
        self.client.get(format!("/files/{file_id}").as_str()).await
    }

    /// Delete a file.
    #[crate::byot(T0 = std::fmt::Display, R = serde::de::DeserializeOwned)]
    pub async fn delete(&self, file_id: &str) -> Result<DeleteFileResponse, OpenAIError> {
        self.client
            .delete(format!("/files/{file_id}").as_str())
            .await
    }

    /// Returns the contents of the specified file
    pub async fn content(&self, file_id: &str) -> Result<Bytes, OpenAIError> {
        self.client
            .get_raw(format!("/files/{file_id}/content").as_str())
            .await
    }
}
