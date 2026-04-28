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
                terminal_seen = true;
            }
            yield chunk;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{FinishReason, chunk};
    use futures::stream;

    fn to_stream(chunks: Vec<LLMEngineOutput>) -> BoxStream<'static, LLMEngineOutput> {
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
        // LLMEngineOutput::cancelled() sets completion_usage to None — must
        // not trip the validator.
        let wrapped = wrap(to_stream(vec![
            chunk::token(1),
            LLMEngineOutput::cancelled(),
        ]));
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 2);
        assert!(matches!(
            collected[1].finish_reason,
            Some(FinishReason::Cancelled)
        ));
    }

    #[tokio::test]
    #[should_panic(expected = "chunk yielded after terminal chunk")]
    async fn panics_on_chunk_after_terminal() {
        let wrapped = wrap(to_stream(vec![LLMEngineOutput::length(), chunk::token(2)]));
        let _collected: Vec<_> = wrapped.collect().await;
    }
}
