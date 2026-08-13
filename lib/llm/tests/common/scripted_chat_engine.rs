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
    AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
    async_trait,
};
use tokio::sync::{Mutex, Semaphore};

pub type Script = Vec<NvCreateChatCompletionStreamResponse>;

enum QueuedScript {
    Immediate(Script),
    Gated {
        chunks: Script,
        split_at: usize,
        release: std::sync::Arc<Semaphore>,
    },
    /// Yields `chunk` forever without ever awaiting, so `next()` is *always*
    /// immediately ready. Models a backend fast enough to keep the frontend's
    /// select loop permanently fed.
    Endless {
        chunk: NvCreateChatCompletionStreamResponse,
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
    contexts: Mutex<Vec<std::sync::Arc<dyn AsyncEngineContext>>>,
}

impl ScriptedChatEngine {
    pub fn new(scripts: impl IntoIterator<Item = Script>) -> Self {
        Self {
            scripts: Mutex::new(scripts.into_iter().map(QueuedScript::Immediate).collect()),
            requests: Mutex::new(Vec::new()),
            contexts: Mutex::new(Vec::new()),
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
                    chunks: script,
                    split_at,
                    release: release.clone(),
                }])),
                requests: Mutex::new(Vec::new()),
                contexts: Mutex::new(Vec::new()),
            },
            ScriptGate { release },
        )
    }

    /// An engine whose stream is *always* immediately ready: it repeats `chunk`
    /// forever and never awaits. Used to prove the frontend cannot be starved out
    /// of enforcing its wall-clock deadline by a backend that never stops
    /// producing.
    // This module is shared by several test binaries; only the Anthropic one
    // exercises deadline starvation.
    #[allow(dead_code)]
    pub fn with_endless_repeat(chunk: NvCreateChatCompletionStreamResponse) -> Self {
        Self {
            scripts: Mutex::new(VecDeque::from([QueuedScript::Endless { chunk }])),
            requests: Mutex::new(Vec::new()),
            contexts: Mutex::new(Vec::new()),
        }
    }

    /// Remove and return all requests observed so far, in arrival order.
    pub async fn take_requests(&self) -> Vec<NvCreateChatCompletionRequest> {
        std::mem::take(&mut *self.requests.lock().await)
    }

    pub async fn remaining_scripts(&self) -> usize {
        self.scripts.lock().await.len()
    }

    /// Whether the most recent request's backend context has been stopped, i.e.
    /// `stop_generating()` (or `kill()`/`stop()`) was called on it. Lets a test
    /// confirm that a frontend-side truncation actually signals the backend to
    /// stop producing tokens, not just that the client-visible SSE looks right.
    // This module is shared by several test binaries; not all of them assert on
    // backend cancellation.
    #[allow(dead_code)]
    pub async fn last_context_stopped(&self) -> bool {
        self.contexts
            .lock()
            .await
            .last()
            .is_some_and(|ctx| ctx.is_stopped())
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
        self.contexts.lock().await.push(ctx.clone());
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
                        yield Annotated::from_data(chunk);
                    }
                }
                QueuedScript::Gated {
                    chunks,
                    split_at,
                    release,
                } => {
                    let mut chunks = chunks.into_iter();
                    for chunk in chunks.by_ref().take(split_at) {
                        yield Annotated::from_data(chunk);
                    }
                    let permit = release
                        .acquire()
                        .await
                        .expect("script gate semaphore was closed");
                    permit.forget();
                    for chunk in chunks {
                        yield Annotated::from_data(chunk);
                    }
                }
                QueuedScript::Endless { chunk } => {
                    // No `.await` anywhere in this arm: every poll resolves
                    // immediately, so a `biased` select that polls this stream
                    // first will never reach a later arm unless the consumer
                    // explicitly checks it.
                    loop {
                        yield Annotated::from_data(chunk.clone());
                    }
                }
            }
        };
        Ok(ResponseStream::new(Box::pin(output), ctx))
    }
}
