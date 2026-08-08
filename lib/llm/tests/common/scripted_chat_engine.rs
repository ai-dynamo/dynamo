// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]

//! A deterministic chat-completions engine for HTTP protocol integration tests.

use std::collections::VecDeque;

use anyhow::{Error, Result, anyhow};
use dynamo_llm::protocols::{
    Annotated,
    openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
    openai::completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};
use tokio::sync::{Mutex, Semaphore};

pub type Script = Vec<NvCreateChatCompletionStreamResponse>;
pub type AnnotatedScript = Vec<Annotated<NvCreateChatCompletionStreamResponse>>;
pub type CompletionScript = Vec<Annotated<NvCreateCompletionResponse>>;

enum QueuedScript {
    Immediate(AnnotatedScript),
    PendingAfter(AnnotatedScript),
    Gated {
        chunks: AnnotatedScript,
        split_at: usize,
        release: std::sync::Arc<Semaphore>,
    },
}

pub struct ScriptGate {
    release: std::sync::Arc<Semaphore>,
}

impl ScriptGate {
    pub fn release(self) {
        self.release.add_permits(1);
    }
}

/// Captures translated chat requests and returns one scripted response per request.
pub struct ScriptedChatEngine {
    scripts: Mutex<VecDeque<QueuedScript>>,
    requests: Mutex<Vec<NvCreateChatCompletionRequest>>,
}

impl ScriptedChatEngine {
    pub fn new(scripts: impl IntoIterator<Item = Script>) -> Self {
        Self {
            scripts: Mutex::new(
                scripts
                    .into_iter()
                    .map(|script| {
                        QueuedScript::Immediate(
                            script.into_iter().map(Annotated::from_data).collect(),
                        )
                    })
                    .collect(),
            ),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn new_annotated(scripts: impl IntoIterator<Item = AnnotatedScript>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().map(QueuedScript::Immediate).collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn new_pending_after_annotated(scripts: impl IntoIterator<Item = AnnotatedScript>) -> Self {
        Self {
            scripts: Mutex::new(
                scripts
                    .into_iter()
                    .map(QueuedScript::PendingAfter)
                    .collect(),
            ),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub fn with_gated_tail(script: Script, split_at: usize) -> (Self, ScriptGate) {
        assert!(
            split_at < script.len(),
            "gated script tail must not be empty"
        );
        let release = std::sync::Arc::new(Semaphore::new(0));
        (
            Self {
                scripts: Mutex::new(VecDeque::from([QueuedScript::Gated {
                    chunks: script.into_iter().map(Annotated::from_data).collect(),
                    split_at,
                    release: release.clone(),
                }])),
                requests: Mutex::new(Vec::new()),
            },
            ScriptGate { release },
        )
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

        let output = async_stream::stream! {
            match script {
                QueuedScript::Immediate(chunks) => {
                    for chunk in chunks {
                        yield chunk;
                    }
                }
                QueuedScript::PendingAfter(chunks) => {
                    for chunk in chunks {
                        yield chunk;
                    }
                    std::future::pending::<()>().await;
                }
                QueuedScript::Gated {
                    chunks,
                    split_at,
                    release,
                } => {
                    let mut chunks = chunks.into_iter();
                    for chunk in chunks.by_ref().take(split_at) {
                        yield chunk;
                    }
                    let permit = release
                        .acquire()
                        .await
                        .expect("script gate semaphore was closed");
                    permit.forget();
                    for chunk in chunks {
                        yield chunk;
                    }
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}

/// Captures Completion requests and returns one annotated script per request.
pub struct ScriptedCompletionEngine {
    scripts: Mutex<VecDeque<CompletionScript>>,
    requests: Mutex<Vec<NvCreateCompletionRequest>>,
}

impl ScriptedCompletionEngine {
    pub fn new(scripts: impl IntoIterator<Item = CompletionScript>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    pub async fn take_requests(&self) -> Vec<NvCreateCompletionRequest> {
        std::mem::take(&mut *self.requests.lock().await)
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        Error,
    > for ScriptedCompletionEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();
        self.requests.lock().await.push(request);
        let script =
            self.scripts.lock().await.pop_front().ok_or_else(|| {
                anyhow!("ScriptedCompletionEngine received an unexpected request")
            })?;

        Ok(ResponseStream::new(
            Box::pin(futures::stream::iter(script)),
            ctx,
        ))
    }
}
