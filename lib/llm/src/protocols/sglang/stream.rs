// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `/generate` response rendering.
//!
//! Workers stream SGLang's incremental responses opaquely. The frontend
//! returns them as SSE, or joins them into one unary response. In P/D, the
//! prefill router sends prefill's final response first, and its prompt-side
//! outputs are joined into the decode response by the [`ResponseMerger`].

use std::sync::Arc;

use async_stream::try_stream;
use dynamo_runtime::error::DynamoError;
use futures::{Stream, StreamExt, pin_mut};
use serde_json::Value;

use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::response_merge::{
    PREFILL_OUTPUT_ANNOTATION, ResponseAssembler, ResponseMerger,
};

pub(crate) struct SglangGenerateStream;

/// One item of the worker stream.
enum Piece {
    Prefill(Value),
    Chunk(Value),
}

fn piece(delta: Annotated<LLMEngineOutput>) -> Result<Option<Piece>, DynamoError> {
    if delta.event.as_deref() == Some(PREFILL_OUTPUT_ANNOTATION) {
        // An older prefill worker sends no response: join an empty one so
        // fields that would otherwise be stale are removed.
        let response = delta
            .comment
            .as_ref()
            .and_then(|comment| comment.first())
            .and_then(|comment| serde_json::from_str::<Value>(comment).ok())
            .and_then(|mut engine_data| engine_data.get_mut("sglang_response").map(Value::take))
            .unwrap_or_else(|| Value::Object(Default::default()));
        return Ok(Some(Piece::Prefill(response)));
    }
    let Some(output) = delta.into_data()? else {
        return Ok(None);
    };
    output
        .engine_data
        .and_then(|mut data| data.as_object_mut()?.remove("sglang_response"))
        .map(|response| Some(Piece::Chunk(response)))
        .ok_or_else(|| DynamoError::msg("missing opaque SGLang response"))
}

fn merge_error(error: impl std::fmt::Display) -> DynamoError {
    DynamoError::msg(format!("cannot join SGLang response: {error}"))
}

impl SglangGenerateStream {
    /// Streaming: forward each SGLang response, with prefill's prompt-side
    /// outputs joined in. The HTTP layer supplies SSE framing and `[DONE]`.
    pub(crate) fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        merger: Arc<dyn ResponseMerger>,
    ) -> impl Stream<Item = Result<Value, DynamoError>> {
        try_stream! {
            pin_mut!(stream);
            let mut assembler = ResponseAssembler::new(merger);
            while let Some(delta) = stream.next().await {
                match piece(delta)? {
                    Some(Piece::Prefill(prefill)) => assembler.set_prefill(prefill),
                    Some(Piece::Chunk(chunk)) => {
                        yield assembler.stream_chunk(chunk).map_err(merge_error)?;
                    }
                    None => {}
                }
            }
        }
    }

    /// Unary: join every SGLang response into the one SGLang would return for
    /// `stream=false`.
    pub(crate) async fn fold(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        merger: Arc<dyn ResponseMerger>,
    ) -> Result<Value, DynamoError> {
        pin_mut!(stream);
        let mut assembler = ResponseAssembler::new(merger);
        while let Some(delta) = stream.next().await {
            match piece(delta)? {
                Some(Piece::Prefill(prefill)) => assembler.set_prefill(prefill),
                Some(Piece::Chunk(chunk)) => assembler.push(chunk).map_err(merge_error)?,
                None => {}
            }
        }
        assembler
            .finish()
            .map_err(merge_error)?
            .ok_or_else(|| DynamoError::msg("SGLang stream ended without a response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::sglang::merge::SglangMerger;

    fn chunk(response: Value) -> Annotated<LLMEngineOutput> {
        Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({ "sglang_response": response })),
            ..Default::default()
        })
    }

    fn prefill(engine_data: Value) -> Annotated<LLMEngineOutput> {
        Annotated::from_annotation(PREFILL_OUTPUT_ANNOTATION, &engine_data).unwrap()
    }

    #[tokio::test]
    async fn preserves_complete_native_sglang_response() {
        let native_response = serde_json::json!({
            "text": "a",
            "output_ids": [101],
            "meta_info": {
                "id": "req-stream",
                "finish_reason": {"type": "length", "length": 1},
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "output_token_logprobs": [[-0.1, 101, "native-token-text"]]
            },
            "future_sglang_field": {"opaque": true}
        });
        let stream = futures::stream::iter([chunk(native_response.clone())]);

        let values: Vec<_> =
            SglangGenerateStream::from_annotated_stream(stream, SglangMerger::shared())
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

        assert_eq!(values, [native_response]);
    }

    #[tokio::test]
    async fn streamed_decode_chunks_receive_prefill_prompt_logprobs() {
        let stream = futures::stream::iter([
            prefill(serde_json::json!({
                "kv_cache_hit": {"prompt_tokens": 2, "reused_tokens": 0},
                "sglang_response": {"meta_info": {"input_token_logprobs": [[null, 1, null]]}},
            })),
            chunk(
                serde_json::json!({"output_ids": [5], "meta_info": {"input_token_logprobs": []}}),
            ),
        ]);
        let values: Vec<_> =
            SglangGenerateStream::from_annotated_stream(stream, SglangMerger::shared())
                .collect::<Vec<_>>()
                .await;
        assert_eq!(
            values[0].as_ref().unwrap()["meta_info"]["input_token_logprobs"],
            serde_json::json!([[null, 1, null]])
        );
    }

    #[tokio::test]
    async fn unary_joins_chunks_and_prefill_output() {
        let stream = futures::stream::iter([
            prefill(serde_json::json!({"sglang_response": {"meta_info": {
                "input_token_logprobs": [[null, 1, null]],
            }}})),
            chunk(
                serde_json::json!({"text": "a", "output_ids": [5], "meta_info": {
                    "completion_tokens": 1, "input_token_logprobs": [],
                    "output_token_logprobs": [[-0.1, 5, null]],
                }}),
            ),
            chunk(
                serde_json::json!({"text": "b", "output_ids": [6], "meta_info": {
                    "completion_tokens": 2, "finish_reason": {"type": "length"},
                    "input_token_logprobs": [], "output_token_logprobs": [[-0.2, 6, null]],
                }}),
            ),
        ]);
        assert_eq!(
            SglangGenerateStream::fold(stream, SglangMerger::shared())
                .await
                .unwrap(),
            serde_json::json!({"text": "ab", "output_ids": [5, 6], "meta_info": {
                "completion_tokens": 2, "finish_reason": {"type": "length"},
                "input_token_logprobs": [[null, 1, null]],
                "output_token_logprobs": [[-0.1, 5, null], [-0.2, 6, null]],
            }})
        );
    }

    #[tokio::test]
    async fn prefill_without_a_response_removes_stale_routed_experts() {
        let stream = futures::stream::iter([
            prefill(serde_json::json!({"kv_cache_hit": {}})),
            chunk(serde_json::json!({"output_ids": [5], "meta_info": {
                "completion_tokens": 1, "routed_experts": "AAAA",
            }})),
        ]);
        let response = SglangGenerateStream::fold(stream, SglangMerger::shared())
            .await
            .unwrap();
        assert!(response["meta_info"].get("routed_experts").is_none());
    }

    #[tokio::test]
    async fn rejects_chunks_without_native_response() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput::default())]);
        let result = SglangGenerateStream::from_annotated_stream(stream, SglangMerger::shared())
            .collect::<Vec<_>>()
            .await;

        assert_eq!(result.len(), 1);
        assert!(
            result[0]
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("missing opaque SGLang response")
        );
        let stream = futures::stream::iter(Vec::<Annotated<LLMEngineOutput>>::new());
        assert!(
            SglangGenerateStream::fold(stream, SglangMerger::shared())
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn preserves_typed_stream_errors() {
        use dynamo_runtime::error::ErrorType;

        let error = DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message("invalid sampling parameters")
            .build();
        let errored = || {
            futures::stream::iter([Annotated::<LLMEngineOutput> {
                data: None,
                id: None,
                event: Some("error".to_string()),
                comment: None,
                error: Some(error.clone()),
            }])
        };

        let output = SglangGenerateStream::from_annotated_stream(errored(), SglangMerger::shared());
        pin_mut!(output);
        let streamed = output.next().await.unwrap().unwrap_err();
        assert_eq!(streamed.error_type(), ErrorType::InvalidArgument);
        assert_eq!(streamed.message(), "invalid sampling parameters");

        let unary = SglangGenerateStream::fold(errored(), SglangMerger::shared())
            .await
            .unwrap_err();
        assert_eq!(unary.error_type(), ErrorType::InvalidArgument);
    }
}
