// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Debug-build stream validator.
//!
//! Wraps the engine's returned stream and panics on contract violations:
//! - a chunk yielded after an unindexed/global terminal, or another chunk for
//!   an output index that already emitted its terminal
//! - an Encode-mode non-cancelled terminal that lacks an
//!   `encoder_result: Some(Value::Object(_))` payload (engines that build
//!   the terminal via `LLMEngineOutput::stop()` instead of
//!   `encode_terminal()` would otherwise ship a no-handoff terminal and
//!   the downstream router would silently misroute)
//!
//! `completion_usage` on a terminal chunk is optional — the rest of the
//! Dynamo pipeline (frontend, router) treats it as nice-to-have, matching
//! `LLMEngineOutput::cancelled/stop/length/error` which set it to `None`.
//!
//! The wrapper is compiled out in release — `lib.rs` gates the module
//! with `#[cfg(debug_assertions)]`, so zero cost in release builds.

use crate::disagg::DisaggregationMode;
use crate::error::DynamoError;
use dynamo_llm::protocols::common::FinishReason;
use dynamo_llm::protocols::common::llm_backend::LLMEngineOutput;
use futures::StreamExt;
use futures::stream::BoxStream;
use std::collections::HashSet;

pub(crate) fn wrap(
    stream: BoxStream<'static, Result<LLMEngineOutput, DynamoError>>,
    mode: DisaggregationMode,
) -> BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
    let mut global_terminal_seen = false;
    let mut finished_indexes = HashSet::new();
    Box::pin(async_stream::stream! {
        let mut inner = stream;
        while let Some(item) = inner.next().await {
            assert!(
                !global_terminal_seen,
                "LLMEngine contract violation: item yielded after terminal item \
                 (an unindexed terminal chunk or Err must be the last item)"
            );
            match &item {
                Ok(chunk) => {
                    if let Some(index) = chunk.index {
                        assert!(
                            !finished_indexes.contains(&index),
                            "LLMEngine contract violation: item yielded after terminal item for output index {index}"
                        );
                    }
                    if chunk.finish_reason.is_some() {
                        // Encode-mode terminal rule: successful terminals MUST
                        // carry an object-shaped encoder_result. Cancelled is
                        // exempt because cancellation can land before the encoder
                        // produces a payload; Error terminals are exempt because a
                        // failure path (LLMEngineOutput::error) legitimately has no
                        // encoder_result.
                        if mode.is_encode()
                            && !matches!(
                                chunk.finish_reason,
                                Some(FinishReason::Cancelled | FinishReason::Error(_))
                            )
                        {
                            assert!(
                                matches!(
                                    chunk.encoder_result.as_ref(),
                                    Some(v) if v.is_object()
                                ),
                                "Encode-mode contract violation: non-cancelled terminal \
                                 chunk must carry encoder_result: Some(Value::Object(_)). \
                                 Use LLMEngineOutput::encode_terminal(map) or \
                                 .with_encoder_result(map) instead of LLMEngineOutput::stop(). \
                                 Got finish_reason={:?}, encoder_result={:?}",
                                chunk.finish_reason,
                                chunk.encoder_result,
                            );
                        }
                        if let Some(index) = chunk.index {
                            finished_indexes.insert(index);
                        } else {
                            global_terminal_seen = true;
                        }
                    }
                }
                Err(_) => global_terminal_seen = true,
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
        let wrapped = wrap(
            to_stream(vec![
                chunk::token(1),
                chunk::token(2),
                LLMEngineOutput::length(),
            ]),
            DisaggregationMode::Aggregated,
        );
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 3);
    }

    #[tokio::test]
    async fn valid_terminal_without_usage_passes() {
        let wrapped = wrap(
            to_stream(vec![chunk::token(1), LLMEngineOutput::cancelled()]),
            DisaggregationMode::Aggregated,
        );
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 2);
        assert!(matches!(
            collected[1].as_ref().unwrap().finish_reason,
            Some(FinishReason::Cancelled)
        ));
    }

    #[tokio::test]
    async fn distinct_output_indexes_may_each_emit_a_terminal() {
        let mut first = LLMEngineOutput::length();
        first.index = Some(0);
        let mut second = LLMEngineOutput::stop();
        second.index = Some(1);
        let wrapped = wrap(
            to_stream(vec![first, second]),
            DisaggregationMode::Aggregated,
        );
        assert_eq!(wrapped.collect::<Vec<_>>().await.len(), 2);
    }

    #[tokio::test]
    #[should_panic(expected = "item yielded after terminal item")]
    async fn panics_on_chunk_after_terminal() {
        let wrapped = wrap(
            to_stream(vec![LLMEngineOutput::length(), chunk::token(2)]),
            DisaggregationMode::Aggregated,
        );
        let _collected: Vec<_> = wrapped.collect().await;
    }

    #[tokio::test]
    #[should_panic(expected = "item yielded after terminal item")]
    async fn panics_on_chunk_after_err() {
        let wrapped = wrap(
            to_stream_with_err(vec![
                Err(DynamoError::msg("typed failure")),
                Ok(chunk::token(1)),
            ]),
            DisaggregationMode::Aggregated,
        );
        let _collected: Vec<_> = wrapped.collect().await;
    }

    /// Encode-mode terminal must carry encoder_result: Some(Object). An
    /// engine that builds its terminal via LLMEngineOutput::stop()
    /// (which seeds encoder_result: None) is a producer bug -- catch it
    /// in debug so the failure surfaces where it originates.
    #[tokio::test]
    #[should_panic(expected = "Encode-mode contract violation")]
    async fn panics_on_encode_terminal_without_encoder_result() {
        let wrapped = wrap(
            to_stream(vec![LLMEngineOutput::stop()]),
            DisaggregationMode::Encode,
        );
        let _collected: Vec<_> = wrapped.collect().await;
    }

    /// Encode-mode terminal with a proper encoder_result via
    /// encode_terminal passes the validator.
    #[tokio::test]
    async fn encode_terminal_with_encoder_result_passes() {
        let mut map = serde_json::Map::new();
        map.insert("uri".into(), serde_json::Value::String("nixl://e/0".into()));
        let wrapped = wrap(
            to_stream(vec![LLMEngineOutput::encode_terminal(map)]),
            DisaggregationMode::Encode,
        );
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].as_ref().unwrap().finish_reason,
            Some(FinishReason::Stop)
        ));
    }

    /// Cancelled Encode-mode terminals are exempt -- cancellation can
    /// land before the encoder produces a payload, so requiring a
    /// non-None encoder_result on cancel would force engines to
    /// fabricate one.
    #[tokio::test]
    async fn encode_mode_cancelled_terminal_without_encoder_result_passes() {
        let wrapped = wrap(
            to_stream(vec![LLMEngineOutput::cancelled()]),
            DisaggregationMode::Encode,
        );
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 1);
    }

    /// The Encode rule is mode-gated: non-encode modes (Aggregated,
    /// Prefill, Decode) emit terminals without encoder_result as
    /// normal, and the validator must not fire.
    #[tokio::test]
    async fn aggregated_terminal_without_encoder_result_passes() {
        let wrapped = wrap(
            to_stream(vec![LLMEngineOutput::stop()]),
            DisaggregationMode::Aggregated,
        );
        let collected: Vec<_> = wrapped.collect().await;
        assert_eq!(collected.len(), 1);
    }
}
