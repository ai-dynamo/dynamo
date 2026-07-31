// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental native SGLang `/generate` response rendering.
//!
//! Aggregate workers pass native input-logprob metadata through unchanged. In
//! disaggregated mode, prefill-produced input logprobs are not forwarded yet;
//! future support should carry that opaque metadata to decode and merge it into
//! the native stream without teaching this frontend the SGLang schema.

use anyhow::Result;
use async_stream::try_stream;
use futures::{Stream, StreamExt, pin_mut};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::protocols::Annotated;
use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::LLMEngineOutput;

#[derive(Debug, Serialize)]
struct SglangGenerateStreamResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    output_ids: Vec<u32>,
    meta_info: Map<String, Value>,
}

pub(crate) struct SglangGenerateStream;

impl SglangGenerateStream {
    /// Convert Dynamo's disjoint engine chunks into SGLang incremental-mode
    /// response objects. The HTTP layer supplies SSE framing and `[DONE]`.
    pub(crate) fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
    ) -> impl Stream<Item = Result<Value>> {
        try_stream! {
            pin_mut!(stream);
            while let Some(delta) = stream.next().await {
                let delta = delta.ok().map_err(anyhow::Error::msg)?;
                let Some(output) = delta.data else {
                    continue;
                };
                if output.index.unwrap_or(0) != 0 {
                    Err(anyhow::anyhow!(
                        "SGLang returned a non-zero choice index for n=1"
                    ))?;
                }
                if output.token_ids.is_empty() && output.finish_reason.is_none() {
                    continue;
                }
                yield serde_json::to_value(render_incremental_response(
                    output,
                    &request_id,
                )?)?;
            }
        }
    }
}

fn render_incremental_response(
    output: LLMEngineOutput,
    request_id: &str,
) -> Result<SglangGenerateStreamResponse> {
    let mut meta_info = output
        .engine_data
        .as_ref()
        .and_then(|data| data.get("sglang_meta_info"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    meta_info.insert("id".to_string(), Value::String(request_id.to_string()));
    let native_finish_reason = meta_info.remove("finish_reason");
    let fallback_finish_reason = finish_reason_from_output(&output)?;
    meta_info.insert(
        "finish_reason".to_string(),
        native_finish_reason
            .filter(|reason| !reason.is_null())
            .unwrap_or(fallback_finish_reason),
    );

    Ok(SglangGenerateStreamResponse {
        text: output.text,
        output_ids: output.token_ids,
        meta_info,
    })
}

fn finish_reason_from_output(output: &LLMEngineOutput) -> Result<Value> {
    let Some(reason) = output.finish_reason.as_ref() else {
        return Ok(Value::Null);
    };
    match reason {
        FinishReason::Error(message) => anyhow::bail!(message.clone()),
        FinishReason::Cancelled => anyhow::bail!("backend cancelled generation"),
        reason => {
            let mut finish_reason = Map::new();
            finish_reason.insert("type".to_string(), Value::String(reason.to_string()));
            if let Some(stop_reason) = output.stop_reason.as_ref() {
                finish_reason.insert("matched".to_string(), serde_json::to_value(stop_reason)?);
            }
            Ok(Value::Object(finish_reason))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn preserves_native_sglang_stream_metadata() {
        let native_logprobs = serde_json::json!([[-0.1, 101, "native-token-text"]]);
        let stream = futures::stream::iter([Annotated::from_data(LLMEngineOutput {
            token_ids: vec![101],
            text: Some("a".to_string()),
            finish_reason: Some(FinishReason::Length),
            index: Some(0),
            engine_data: Some(serde_json::json!({
                "sglang_meta_info": {
                    "finish_reason": {"type": "length", "length": 1},
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "output_token_logprobs": native_logprobs,
                    "future_sglang_field": {"opaque": true}
                }
            })),
            ..Default::default()
        })]);

        let values: Vec<_> =
            SglangGenerateStream::from_annotated_stream(stream, "req-stream".to_string())
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<_>>()
                .unwrap();

        assert_eq!(values.len(), 1);
        assert_eq!(values[0]["text"], "a");
        assert_eq!(values[0]["output_ids"], serde_json::json!([101]));
        assert_eq!(values[0]["meta_info"]["id"], "req-stream");
        assert_eq!(
            values[0]["meta_info"]["output_token_logprobs"],
            native_logprobs
        );
        assert_eq!(
            values[0]["meta_info"]["future_sglang_field"]["opaque"],
            true
        );
        assert_eq!(values[0]["meta_info"]["finish_reason"]["length"], 1);
    }
}
