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
use serde_json::Value;

use crate::protocols::Annotated;
use crate::protocols::common::llm_backend::LLMEngineOutput;

pub(crate) struct SglangGenerateStream;

impl SglangGenerateStream {
    /// Forward SGLang incremental-mode response objects opaquely. The HTTP
    /// layer supplies SSE framing and `[DONE]`.
    pub(crate) fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
    ) -> impl Stream<Item = Result<Value>> {
        try_stream! {
            pin_mut!(stream);
            while let Some(delta) = stream.next().await {
                let delta = delta.ok().map_err(anyhow::Error::msg)?;
                let Some(output) = delta.data else {
                    continue;
                };
                let response = output
                    .engine_data
                    .and_then(|data| data.get("sglang_response").cloned())
                    .ok_or_else(|| anyhow::anyhow!("missing opaque SGLang response"))?;
                yield response;
            }
        }
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
            .collect::<Result<_>>()
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
}
