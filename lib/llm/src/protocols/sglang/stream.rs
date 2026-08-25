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
use serde_json::{Map, Value};

use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::LLMEngineOutput;

pub(crate) struct SglangGenerateStream;

#[derive(Default)]
struct UnaryAccumulator {
    fields: Map<String, Value>,
    meta_info: Map<String, Value>,
    text: Option<String>,
    output_ids: Vec<Value>,
    output_token_logprobs: Vec<Value>,
    output_top_logprobs: Vec<Value>,
    saw_output_token_logprobs: bool,
    saw_output_top_logprobs: bool,
    chunks: usize,
}

impl UnaryAccumulator {
    fn push(&mut self, response: Value) -> Result<(), DynamoError> {
        let mut response = match response {
            Value::Object(response) => response,
            _ => {
                return Err(DynamoError::msg(
                    "SGLang generate response must be an object",
                ));
            }
        };
        let chunk_output_ids = take_array(&mut response, "output_ids")?;
        let has_output = !chunk_output_ids.is_empty();
        self.output_ids.extend(chunk_output_ids);

        match response.remove("text") {
            Some(Value::String(text)) => self.text.get_or_insert_with(String::new).push_str(&text),
            Some(Value::Null) | None if has_output => self.text = None,
            Some(Value::Null) | None => {}
            Some(_) => {
                return Err(DynamoError::msg(
                    "SGLang generate text must be a string or null",
                ));
            }
        }

        let mut meta_info = match response.remove("meta_info") {
            Some(Value::Object(meta_info)) => meta_info,
            _ => {
                return Err(DynamoError::msg(
                    "SGLang generate response is missing meta_info",
                ));
            }
        };
        append_optional_array(
            &mut meta_info,
            "output_token_logprobs",
            &mut self.output_token_logprobs,
            &mut self.saw_output_token_logprobs,
        )?;
        append_optional_array(
            &mut meta_info,
            "output_top_logprobs",
            &mut self.output_top_logprobs,
            &mut self.saw_output_top_logprobs,
        )?;

        if let Some(reported) = meta_info
            .get("output_token_logprobs_length")
            .and_then(Value::as_u64)
            && self.saw_output_token_logprobs
            && reported as usize != self.output_token_logprobs.len()
        {
            return Err(DynamoError::msg(format!(
                "SGLang output_token_logprobs_length is inconsistent: reported={reported}, collected={}",
                self.output_token_logprobs.len()
            )));
        }
        if let Some(reported) = meta_info.get("completion_tokens").and_then(Value::as_u64)
            && reported as usize != self.output_ids.len()
        {
            return Err(DynamoError::msg(format!(
                "SGLang completion_tokens is inconsistent: reported={reported}, collected={}",
                self.output_ids.len()
            )));
        }

        self.fields.extend(response);
        self.meta_info.extend(meta_info);
        self.chunks += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<Value, DynamoError> {
        if self.chunks == 0 {
            return Err(DynamoError::msg("SGLang generate returned no response"));
        }
        if self
            .meta_info
            .get("finish_reason")
            .is_none_or(Value::is_null)
        {
            return Err(DynamoError::msg(
                "SGLang generate ended without a terminal finish_reason",
            ));
        }

        self.fields.insert(
            "text".to_string(),
            self.text.map(Value::String).unwrap_or(Value::Null),
        );
        let completion_tokens = self.output_ids.len();
        self.fields
            .insert("output_ids".to_string(), Value::Array(self.output_ids));
        if self.saw_output_token_logprobs {
            self.meta_info.insert(
                "output_token_logprobs_length".to_string(),
                Value::from(self.output_token_logprobs.len()),
            );
            self.meta_info.insert(
                "output_token_logprobs".to_string(),
                Value::Array(self.output_token_logprobs),
            );
        }
        if self.saw_output_top_logprobs {
            self.meta_info.insert(
                "output_top_logprobs".to_string(),
                Value::Array(self.output_top_logprobs),
            );
        }
        self.meta_info.insert(
            "completion_tokens".to_string(),
            Value::from(completion_tokens),
        );
        self.fields
            .insert("meta_info".to_string(), Value::Object(self.meta_info));
        Ok(Value::Object(self.fields))
    }
}

fn take_array(response: &mut Map<String, Value>, field: &str) -> Result<Vec<Value>, DynamoError> {
    match response.remove(field) {
        Some(Value::Array(values)) => Ok(values),
        Some(Value::Null) | None => Ok(Vec::new()),
        Some(_) => Err(DynamoError::msg(format!(
            "SGLang generate {field} must be an array when present"
        ))),
    }
}

fn append_optional_array(
    response: &mut Map<String, Value>,
    field: &str,
    output: &mut Vec<Value>,
    seen: &mut bool,
) -> Result<(), DynamoError> {
    match response.remove(field) {
        Some(Value::Array(values)) => {
            *seen = true;
            output.extend(values);
            Ok(())
        }
        Some(Value::Null) | None => Ok(()),
        Some(_) => Err(DynamoError::msg(format!(
            "SGLang generate meta_info.{field} must be an array when present"
        ))),
    }
}
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

    /// Fold SGLang's disjoint incremental chunks into its aggregate JSON
    /// response shape for a non-streaming client.
    pub(crate) async fn unary_from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
    ) -> Result<Value, DynamoError> {
        let responses = Self::from_annotated_stream(stream);
        pin_mut!(responses);

        let mut accumulator = UnaryAccumulator::default();
        while let Some(response) = responses.next().await {
            accumulator.push(response?)?;
        }
        accumulator.finish()
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
            "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 1}
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
    async fn folds_disjoint_non_streaming_native_responses() {
        let output = |response| {
            Annotated::from_data(LLMEngineOutput {
                engine_data: Some(serde_json::json!({"sglang_response": response})),
                ..Default::default()
            })
        };
        let stream = futures::stream::iter([
            output(serde_json::json!({
                "text": "h",
                "output_ids": [101],
                "meta_info": {
                    "finish_reason": null,
                    "completion_tokens": 1,
                    "output_token_logprobs_length": 1,
                    "output_token_logprobs": [[-0.1, 101, "h"]],
                    "output_top_logprobs": [[[-0.1, 101, "h"]]],
                    "response_sent_to_client_ts": 1.0
                }
            })),
            output(serde_json::json!({
                "text": "i",
                "output_ids": [102],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "completion_tokens": 2,
                    "output_token_logprobs_length": 2,
                    "output_token_logprobs": [[-0.2, 102, "i"]],
                    "output_top_logprobs": [[[-0.2, 102, "i"]]],
                    "routed_experts": "opaque"
                }
            })),
        ]);

        let response = SglangGenerateStream::unary_from_annotated_stream(stream)
            .await
            .unwrap();

        assert_eq!(response["text"], "hi");
        assert_eq!(response["output_ids"], serde_json::json!([101, 102]));
        assert_eq!(response["meta_info"]["completion_tokens"], 2);
        assert_eq!(response["meta_info"]["output_token_logprobs_length"], 2);
        assert_eq!(
            response["meta_info"]["output_token_logprobs"],
            serde_json::json!([[-0.1, 101, "h"], [-0.2, 102, "i"]])
        );
        assert_eq!(
            response["meta_info"]["output_top_logprobs"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(response["meta_info"]["response_sent_to_client_ts"], 1.0);
        assert_eq!(response["meta_info"]["routed_experts"], "opaque");
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
