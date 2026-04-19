// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ReplayEngine: an `AsyncEngine` impl that replays pre-recorded
//! `chat.completion.chunk` JSONL as a stream. Used by the harness-style
//! integration tests under `lib/llm/tests/` to drive the Anthropic and
//! Responses surfaces with deterministic backend output.
//!
//! The replay format is one `NvCreateChatCompletionStreamResponse` per line
//! — the exact shape that an upstream engine (vLLM, sglang, TRT-LLM) would
//! emit. Test fixtures are handwritten JSONL under `tests/data/replays/harness/`.

#![allow(dead_code)]

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Error, Result};
use async_stream::stream;
use dynamo_llm::protocols::{
    Annotated,
    openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
    },
};
use dynamo_runtime::pipeline::{
    AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn, async_trait,
};
use tokio::sync::Mutex;

/// Engine that replays JSONL chunks to any caller of `/v1/chat/completions`,
/// `/v1/messages`, or `/v1/responses` (all three endpoints ultimately dispatch
/// through the chat-completions engine registered with `ModelManager`).
pub struct ReplayEngine {
    chunks: Vec<NvCreateChatCompletionStreamResponse>,
    interval: Duration,
    observed_cancels: Arc<Mutex<u64>>,
}

impl ReplayEngine {
    /// Load a JSONL file where every non-empty line is a JSON-encoded
    /// `NvCreateChatCompletionStreamResponse`. Blank lines and `//`-prefixed
    /// comment lines are skipped so fixtures stay readable.
    pub fn from_jsonl<P: AsRef<Path>>(path: P, interval: Duration) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read replay file {}", path.display()))?;
        let mut chunks = Vec::new();
        for (idx, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            let chunk: NvCreateChatCompletionStreamResponse = serde_json::from_str(line)
                .with_context(|| {
                    format!(
                        "failed to parse replay chunk at {}:{}: `{}`",
                        path.display(),
                        idx + 1,
                        line
                    )
                })?;
            chunks.push(chunk);
        }
        Ok(Self {
            chunks,
            interval,
            observed_cancels: Arc::new(Mutex::new(0)),
        })
    }

    /// Build an engine from an in-memory slice of chunks (useful when a test
    /// wants to skip the filesystem).
    pub fn from_chunks(
        chunks: Vec<NvCreateChatCompletionStreamResponse>,
        interval: Duration,
    ) -> Self {
        Self {
            chunks,
            interval,
            observed_cancels: Arc::new(Mutex::new(0)),
        }
    }

    /// Number of times a streaming generation observed the client cancelling
    /// mid-flight. Lets tests assert the cancellation token propagated.
    pub async fn observed_cancels(&self) -> u64 {
        *self.observed_cancels.lock().await
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for ReplayEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (_req, context) = request.transfer(());
        let ctx = context.context();

        let chunks = self.chunks.clone();
        let interval = self.interval;

        // Watch the engine context on a detached task so cancellation is
        // observed even when the HTTP layer drops the response stream
        // (which would otherwise cancel our generator mid-await).
        tokio::spawn({
            let ctx = ctx.clone();
            let observed_cancels = self.observed_cancels.clone();
            async move {
                ctx.stopped().await;
                *observed_cancels.lock().await += 1;
            }
        });

        let ctx_clone = ctx.clone();
        let stream = stream! {
            for chunk in chunks {
                if interval > Duration::ZERO {
                    tokio::select! {
                        _ = tokio::time::sleep(interval) => {}
                        _ = ctx_clone.stopped() => break,
                    }
                } else if ctx_clone.is_stopped() {
                    break;
                }
                yield Annotated::from_data(chunk);
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}
