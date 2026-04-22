// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Debug-build stream validator.
//!
//! Wraps the engine's returned stream and panics on contract violations:
//! - a chunk yielded after a terminal chunk (one carrying `finish_reason`)
//! - a terminal chunk missing `completion_usage`
//!
//! The wrapper is compiled out in release — `lib.rs` gates the module
//! with `#[cfg(debug_assertions)]`, so zero cost in release builds.

use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use futures::StreamExt;
use futures::stream::BoxStream;

pub(crate) fn wrap(
    stream: BoxStream<'static, LLMEngineOutput>,
) -> BoxStream<'static, LLMEngineOutput> {
    let mut terminal_seen = false;
    Box::pin(async_stream::stream! {
        let mut inner = stream;
        while let Some(chunk) = inner.next().await {
            assert!(
                !terminal_seen,
                "LLMEngine contract violation: chunk yielded after terminal chunk \
                 (a chunk with finish_reason set must be the last item)"
            );
            if chunk.finish_reason.is_some() {
                assert!(
                    chunk.completion_usage.is_some(),
                    "LLMEngine contract violation: terminal chunk missing completion_usage \
                     (chunks with finish_reason must also set completion_usage)"
                );
                terminal_seen = true;
            }
            yield chunk;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FinishReason, chunk, usage};
    use futures::stream;

    fn to_stream(chunks: Vec<LLMEngineOutput>) -> BoxStream<'static, LLMEngineOutput> {
        Box::pin(stream::iter(chunks))
    }

    #[tokio::test]
    async fn valid_stream_passes_through() {
        let wrapped = wrap(to_stream(vec![
            chunk::token(1),
            chunk::token(2),
            chunk::terminal(vec![3], FinishReason::Length, usage(10, 3)),
        ]));
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 3);
    }

    #[tokio::test]
    #[should_panic(expected = "chunk yielded after terminal chunk")]
    async fn panics_on_chunk_after_terminal() {
        let wrapped = wrap(to_stream(vec![
            chunk::terminal(vec![1], FinishReason::Length, usage(5, 1)),
            chunk::token(2),
        ]));
        let _collected: Vec<_> = wrapped.collect().await;
    }

    #[tokio::test]
    #[should_panic(expected = "terminal chunk missing completion_usage")]
    async fn panics_on_terminal_missing_usage() {
        let bad = LLMEngineOutput {
            token_ids: vec![1],
            finish_reason: Some(FinishReason::Length),
            completion_usage: None,
            ..Default::default()
        };
        let wrapped = wrap(to_stream(vec![bad]));
        let _collected: Vec<_> = wrapped.collect().await;
    }
}
