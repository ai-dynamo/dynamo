// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Debug-build stream validator.
//!
//! Wraps the engine's returned stream and panics on contract violations:
//! - a chunk yielded after a terminal chunk (one carrying `finish_reason`)
//!
//! `completion_usage` on a terminal chunk is optional — the rest of the
//! Dynamo pipeline (frontend, router) treats it as nice-to-have, matching
//! `LLMEngineOutput::cancelled/stop/length/error` which set it to `None`.
//!
//! The wrapper is compiled out in release — `lib.rs` gates the module
//! with `#[cfg(debug_assertions)]`, so zero cost in release builds.

use crate::error::DynamoError;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use futures::StreamExt;
use futures::stream::BoxStream;

pub(crate) fn wrap(
    stream: BoxStream<'static, Result<LLMEngineOutput, DynamoError>>,
) -> BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
    let mut terminal_seen = false;
    Box::pin(async_stream::stream! {
        let mut inner = stream;
        while let Some(item) = inner.next().await {
            assert!(
                !terminal_seen,
                "LLMEngine contract violation: item yielded after terminal item \
                 (a chunk with finish_reason set, or an Err, must be the last item)"
            );
            match &item {
                Ok(chunk) if chunk.finish_reason.is_some() => terminal_seen = true,
                Err(_) => terminal_seen = true,
                _ => {}
            }
            yield item;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FinishReason, chunk};
    use futures::stream;

    fn to_stream(
        chunks: Vec<LLMEngineOutput>,
    ) -> BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
        Box::pin(stream::iter(chunks.into_iter().map(Ok)))
    }

    fn to_stream_with_err(
        chunks: Vec<Result<LLMEngineOutput, DynamoError>>,
    ) -> BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
        Box::pin(stream::iter(chunks))
    }

    #[tokio::test]
    async fn valid_stream_passes_through() {
        let wrapped = wrap(to_stream(vec![
            chunk::token(1),
            chunk::token(2),
            LLMEngineOutput::length(),
        ]));
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 3);
    }

    #[tokio::test]
    async fn valid_terminal_without_usage_passes() {
        let wrapped = wrap(to_stream(vec![
            chunk::token(1),
            LLMEngineOutput::cancelled(),
        ]));
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 2);
        assert!(matches!(
            collected[1].as_ref().unwrap().finish_reason,
            Some(FinishReason::Cancelled)
        ));
    }

    #[tokio::test]
    #[should_panic(expected = "item yielded after terminal item")]
    async fn panics_on_chunk_after_terminal() {
        let wrapped = wrap(to_stream(vec![LLMEngineOutput::length(), chunk::token(2)]));
        let _collected: Vec<_> = wrapped.collect().await;
    }

    #[tokio::test]
    #[should_panic(expected = "item yielded after terminal item")]
    async fn panics_on_chunk_after_err() {
        let wrapped = wrap(to_stream_with_err(vec![
            Err(DynamoError::msg("typed failure")),
            Ok(chunk::token(1)),
        ]));
        let _collected: Vec<_> = wrapped.collect().await;
    }
}
