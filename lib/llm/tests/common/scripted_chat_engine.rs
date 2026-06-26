// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A deterministic chat-completions engine for HTTP protocol integration tests.

use std::collections::VecDeque;

use anyhow::{Error, Result, anyhow};
use dynamo_llm::protocols::{
    Annotated,
    openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};
use futures::stream;
use tokio::sync::Mutex;

pub type Script = Vec<NvCreateChatCompletionStreamResponse>;

/// Captures translated chat requests and returns one scripted response per request.
pub struct ScriptedChatEngine {
    scripts: Mutex<VecDeque<Script>>,
    requests: Mutex<Vec<NvCreateChatCompletionRequest>>,
}

impl ScriptedChatEngine {
    pub fn new(scripts: impl IntoIterator<Item = Script>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Remove and return all requests observed so far, in arrival order.
    pub async fn take_requests(&self) -> Vec<NvCreateChatCompletionRequest> {
        std::mem::take(&mut *self.requests.lock().await)
    }

    pub async fn remaining_scripts(&self) -> usize {
        self.scripts.lock().await.len()
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for ScriptedChatEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        self.requests.lock().await.push(request);
        let script = self
            .scripts
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| anyhow!("ScriptedChatEngine received an unexpected request"))?;

        let output = stream::iter(script.into_iter().map(Annotated::from_data));
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}
