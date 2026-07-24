// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental native SGLang `/generate` response rendering.

use anyhow::Result;
use async_stream::try_stream;
use futures::{Stream, StreamExt, pin_mut};
use serde::Serialize;
use serde_json::{Map, Value};

use super::generate::SglangResponseOptions;
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
        options: SglangResponseOptions,
    ) -> impl Stream<Item = Result<Value>> {
        try_stream! {
            let mut output_token_count = 0usize;
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
                output_token_count += output.token_ids.len();
                yield serde_json::to_value(render_incremental_response(
                    output,
                    &request_id,
                    options,
                    output_token_count,
                )?)?;
            }
        }
    }
}

fn render_incremental_response(
    output: LLMEngineOutput,
    request_id: &str,
    options: SglangResponseOptions,
    output_token_count: usize,
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
    apply_usage(&mut meta_info, &output)?;
    apply_logprobs(&mut meta_info, &output, options, output_token_count)?;

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

fn apply_usage(meta_info: &mut Map<String, Value>, output: &LLMEngineOutput) -> Result<()> {
    let Some(usage) = output.completion_usage.as_ref() else {
        return Ok(());
    };
    let Value::Object(usage) = serde_json::to_value(usage)? else {
        return Ok(());
    };
    for key in ["prompt_tokens", "completion_tokens"] {
        if let Some(value) = usage.get(key) {
            meta_info.insert(key.to_string(), value.clone());
        }
    }
    if let Some(cached_tokens) = usage
        .get("prompt_tokens_details")
        .and_then(Value::as_object)
        .and_then(|details| details.get("cached_tokens"))
    {
        meta_info.insert("cached_tokens".to_string(), cached_tokens.clone());
    }
    Ok(())
}

fn apply_logprobs(
    meta_info: &mut Map<String, Value>,
    output: &LLMEngineOutput,
    options: SglangResponseOptions,
    output_token_count: usize,
) -> Result<()> {
    if !options.return_logprob {
        for key in [
            "input_token_logprobs",
            "input_top_logprobs",
            "output_token_logprobs",
            "output_top_logprobs",
            "output_token_logprobs_length",
        ] {
            meta_info.remove(key);
        }
        return Ok(());
    }

    let output_token_logprobs = if output.token_ids.is_empty() {
        Vec::new()
    } else {
        let logprobs = output.log_probs.as_ref().ok_or_else(|| {
            anyhow::anyhow!("SGLang response requested logprobs but returned none")
        })?;
        anyhow::ensure!(
            logprobs.len() == output.token_ids.len(),
            "SGLang returned {} selected-token logprobs for {} output IDs",
            logprobs.len(),
            output.token_ids.len()
        );
        logprobs
            .iter()
            .zip(&output.token_ids)
            .map(|(logprob, token_id)| serde_json::json!([logprob, token_id, null]))
            .collect()
    };
    meta_info.insert(
        "output_token_logprobs".to_string(),
        Value::Array(output_token_logprobs),
    );
    meta_info.insert(
        "output_token_logprobs_length".to_string(),
        Value::from(output_token_count),
    );

    if options.top_logprobs_num > 0 {
        let output_top_logprobs = if output.token_ids.is_empty() {
            Vec::new()
        } else {
            let top_logprobs = output.top_logprobs.as_ref().ok_or_else(|| {
                anyhow::anyhow!("SGLang response requested top logprobs but returned none")
            })?;
            anyhow::ensure!(
                top_logprobs.len() == output.token_ids.len(),
                "SGLang returned {} top-logprob positions for {} output IDs",
                top_logprobs.len(),
                output.token_ids.len()
            );
            top_logprobs
                .iter()
                .map(|position| {
                    Value::Array(
                        position
                            .iter()
                            .take(options.top_logprobs_num as usize)
                            .map(|entry| serde_json::json!([entry.logprob, entry.token_id, null]))
                            .collect(),
                    )
                })
                .collect()
        };
        meta_info.insert(
            "output_top_logprobs".to_string(),
            Value::Array(output_top_logprobs),
        );
    } else {
        meta_info.remove("output_top_logprobs");
    }

    if !options.include_input_logprobs {
        meta_info.insert("input_token_logprobs".to_string(), Value::Array(Vec::new()));
        if options.top_logprobs_num > 0 {
            meta_info.insert("input_top_logprobs".to_string(), Value::Array(Vec::new()));
        } else {
            meta_info.remove("input_top_logprobs");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::llm_backend::TopLogprob;

    #[tokio::test]
    async fn emits_disjoint_logprobs_with_cumulative_length() {
        let options = SglangResponseOptions {
            return_logprob: true,
            include_input_logprobs: false,
            top_logprobs_num: 1,
        };
        let top = |token_id| {
            Some(vec![vec![TopLogprob {
                rank: 0,
                token_id,
                token: None,
                logprob: -0.5,
                bytes: None,
            }]])
        };
        let stream = futures::stream::iter([
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![101],
                text: Some("a".to_string()),
                log_probs: Some(vec![-0.1]),
                top_logprobs: top(101),
                index: Some(0),
                engine_data: Some(serde_json::json!({
                    "sglang_meta_info": {"finish_reason": null, "prompt_tokens": 2}
                })),
                ..Default::default()
            }),
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![102],
                text: Some("b".to_string()),
                log_probs: Some(vec![-0.2]),
                top_logprobs: top(102),
                finish_reason: Some(FinishReason::Length),
                index: Some(0),
                engine_data: Some(serde_json::json!({
                    "sglang_meta_info": {
                        "finish_reason": {"type": "length", "length": 2},
                        "prompt_tokens": 2,
                        "completion_tokens": 2
                    }
                })),
                ..Default::default()
            }),
        ]);

        let values: Vec<_> =
            SglangGenerateStream::from_annotated_stream(stream, "req-stream".to_string(), options)
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect::<Result<_>>()
                .unwrap();

        assert_eq!(values[0]["text"], "a");
        assert_eq!(values[0]["output_ids"], serde_json::json!([101]));
        assert_eq!(
            values[0]["meta_info"]["output_token_logprobs"],
            serde_json::json!([[-0.1, 101, null]])
        );
        assert_eq!(values[0]["meta_info"]["output_token_logprobs_length"], 1);
        assert_eq!(values[0]["meta_info"]["finish_reason"], Value::Null);
        assert_eq!(values[1]["output_ids"], serde_json::json!([102]));
        assert_eq!(
            values[1]["meta_info"]["output_token_logprobs"],
            serde_json::json!([[-0.2, 102, null]])
        );
        assert_eq!(values[1]["meta_info"]["output_token_logprobs_length"], 2);
        assert_eq!(values[1]["meta_info"]["finish_reason"]["length"], 2);
    }
}
