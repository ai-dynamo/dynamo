// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `/generate` response rendering.
//!
//! Workers stream SGLang's incremental responses opaquely. In P/D the prefill
//! router sends prefill's final response first; the SGLang rules join it and
//! the chunks into what SGLang returns for `stream=true` or `stream=false`.

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

/// The SGLang response in `delta`, and whether it is prefill's.
fn response(delta: Annotated<LLMEngineOutput>) -> Result<Option<(bool, Value)>, DynamoError> {
    let prefill = delta.event.as_deref() == Some(PREFILL_OUTPUT_ANNOTATION);
    let Some(output) = delta.into_data()? else {
        return Ok(None);
    };
    match output
        .engine_data
        .and_then(|mut data| data.as_object_mut()?.remove("sglang_response"))
    {
        Some(response) => Ok(Some((prefill, response))),
        // An older prefill worker sends none; joining nothing drops stale fields.
        None if prefill => Ok(Some((true, Value::Object(Default::default())))),
        None => Err(DynamoError::msg("missing opaque SGLang response")),
    }
}

impl SglangGenerateStream {
    /// Streaming: `render` serializes each SGLang response. The HTTP layer
    /// supplies SSE framing and `[DONE]`.
    pub(crate) fn from_annotated_stream<T>(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        merger: Arc<dyn ResponseMerger>,
        mut render: impl FnMut(&Value) -> T,
    ) -> impl Stream<Item = Result<T, DynamoError>> {
        try_stream! {
            pin_mut!(stream);
            let mut assembler = ResponseAssembler::new(merger);
            while let Some(delta) = stream.next().await {
                match response(delta)? {
                    Some((true, prefill)) => assembler.set_prefill(prefill),
                    Some((false, chunk)) => yield assembler.render_chunk(chunk, &mut render),
                    None => {}
                }
            }
        }
    }

    /// Unary: the response SGLang returns for `stream=false`.
    pub(crate) async fn fold(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        merger: Arc<dyn ResponseMerger>,
    ) -> Result<Value, DynamoError> {
        pin_mut!(stream);
        let mut assembler = ResponseAssembler::new(merger);
        while let Some(delta) = stream.next().await {
            match response(delta)? {
                Some((true, prefill)) => assembler.set_prefill(prefill),
                Some((false, chunk)) => assembler.push(chunk),
                None => {}
            }
        }
        assembler
            .finish()
            .ok_or_else(|| DynamoError::msg("SGLang stream ended without a response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::sglang::merge::merger;

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

        let values: Vec<_> =
            SglangGenerateStream::from_annotated_stream(stream, merger(), Value::clone)
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();

        assert_eq!(values, [native_response]);
    }

    #[tokio::test]
    async fn typed_error_terminal_still_renders_native_response() {
        // Workers type an error-bearing SGLang abort for the router; the client
        // still receives SGLang's own response.
        let native_response = serde_json::json!({
            "output_ids": [],
            "meta_info": {"finish_reason": {"type": "abort", "status_code": 400}}
        });
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            engine_data: Some(serde_json::json!({"sglang_response": native_response})),
            ..LLMEngineOutput::error("bad request".to_string())
        })]);
        assert_eq!(
            SglangGenerateStream::fold(stream, merger()).await.unwrap(),
            native_response
        );
    }

    #[tokio::test]
    async fn rejects_chunks_without_native_response() {
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput::default())]);
        let result = SglangGenerateStream::from_annotated_stream(stream, merger(), Value::clone)
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

        let output = SglangGenerateStream::from_annotated_stream(stream, merger(), Value::clone);
        pin_mut!(output);

        let error = output.next().await.unwrap().unwrap_err();

        assert_eq!(error.error_type(), ErrorType::InvalidArgument);
        assert_eq!(error.message(), "invalid sampling parameters");
    }

    #[tokio::test]
    async fn unary_joins_prefill_output_and_chunks() {
        let item = |event: Option<&str>, response: Value| Annotated {
            data: Some(LLMEngineOutput {
                engine_data: Some(serde_json::json!({ "sglang_response": response })),
                ..Default::default()
            }),
            event: event.map(str::to_string),
            ..Annotated::from_data(LLMEngineOutput::default())
        };
        let stream = futures::stream::iter([
            item(
                Some(PREFILL_OUTPUT_ANNOTATION),
                serde_json::json!({"meta_info": {"input_token_logprobs": [[null, 1, null]]}}),
            ),
            item(
                None,
                serde_json::json!({"output_ids": [5], "meta_info": {"input_token_logprobs": []}}),
            ),
            item(
                None,
                serde_json::json!({"output_ids": [6], "meta_info": {"input_token_logprobs": []}}),
            ),
        ]);
        assert_eq!(
            SglangGenerateStream::fold(stream, merger()).await.unwrap(),
            serde_json::json!({"output_ids": [5, 6], "meta_info": {"input_token_logprobs": [[null, 1, null]]}})
        );
        let empty = futures::stream::iter(Vec::<Annotated<LLMEngineOutput>>::new());
        assert!(SglangGenerateStream::fold(empty, merger()).await.is_err());
    }
}
