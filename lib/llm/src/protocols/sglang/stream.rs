// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental native SGLang `/generate` response rendering.
//!
//! Aggregate workers pass native input-logprob metadata through unchanged. In
//! disaggregated mode, prefill-produced input logprobs are not forwarded yet;
//! future support should carry that opaque metadata to decode and merge it into
//! the native stream without teaching this frontend the SGLang schema.

use async_stream::try_stream;
use dynamo_runtime::error::DynamoError;
use futures::{Stream, StreamExt, pin_mut};
use serde_json::Value;

use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::LLMEngineOutput;

pub(crate) struct SglangGenerateStream;

impl SglangGenerateStream {
    /// Forward SGLang incremental-mode response objects opaquely. The HTTP
    /// layer supplies SSE framing and `[DONE]`.
    pub(crate) fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
    ) -> impl Stream<Item = Result<Value, DynamoError>> {
        try_stream! {
            pin_mut!(stream);
            while let Some(delta) = stream.next().await {
                let Some(output) = delta.into_data()? else {
                    continue;
                };
                let response = output
                    .engine_data
                    .and_then(|mut data| data.as_object_mut()?.remove("sglang_response"))
                    .ok_or_else(|| DynamoError::msg("missing opaque SGLang response"))?;
                yield response;
            }
        }
    }

    /// Return the sole native response produced by SGLang's non-streaming
    /// request mode.
    pub(crate) async fn unary_from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
    ) -> Result<Value, DynamoError> {
        let responses = Self::from_annotated_stream(stream);
        pin_mut!(responses);

        let response = responses
            .next()
            .await
            .ok_or_else(|| DynamoError::msg("SGLang generate returned no response"))??;
        if responses.next().await.is_some() {
            return Err(DynamoError::msg(
                "non-streaming SGLang generate returned multiple responses",
            ));
        }
        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![101],
            text: Some("a".to_string()),
            index: Some(0),
            engine_data: Some(serde_json::json!({
                "sglang_response": native_response
            })),
            ..Default::default()
        })]);

        let values: Vec<_> = SglangGenerateStream::from_annotated_stream(stream)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(values, [native_response]);
    }

    #[tokio::test]
    async fn rejects_chunks_without_native_response() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput::default())]);
        let result = SglangGenerateStream::from_annotated_stream(stream)
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
    }

    #[tokio::test]
    async fn preserves_one_non_streaming_native_response() {
        let native_response = serde_json::json!({
            "text": "complete",
            "output_ids": [101],
            "meta_info": {"finish_reason": {"type": "stop"}}
        });
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({
                "sglang_response": native_response
            })),
            ..Default::default()
        })]);

        let response = SglangGenerateStream::unary_from_annotated_stream(stream)
            .await
            .unwrap();

        assert_eq!(response, native_response);
    }

    #[tokio::test]
    async fn rejects_multiple_non_streaming_native_responses() {
        let output = || {
            Annotated::from_data(LLMEngineOutput {
                engine_data: Some(serde_json::json!({
                    "sglang_response": {"text": "chunk"}
                })),
                ..Default::default()
            })
        };
        let error = SglangGenerateStream::unary_from_annotated_stream(futures::stream::iter([
            output(),
            output(),
        ]))
        .await
        .unwrap_err();

        assert!(error.to_string().contains("multiple responses"));
    }

    #[tokio::test]
    async fn preserves_typed_stream_errors() {
        use dynamo_runtime::error::ErrorType;

        let error = DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message("invalid sampling parameters")
            .build();
        let stream = futures::stream::iter([Annotated::<LLMEngineOutput> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: None,
            error: Some(error),
        }]);

        let output = SglangGenerateStream::from_annotated_stream(stream);
        pin_mut!(output);

        let error = output.next().await.unwrap().unwrap_err();

        assert_eq!(error.error_type(), ErrorType::InvalidArgument);
        assert_eq!(error.message(), "invalid sampling parameters");
    }
}
