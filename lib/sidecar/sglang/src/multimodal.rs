// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adapt Dynamo multimodal token requests to native SGLang HTTP generation.
//! SGLang's language-only server owns encoder dispatch and embedding transfer.

use std::collections::HashMap;

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, LLMEngineOutput, LLMEngineOutputExt, MultimodalData,
    PreprocessedRequest, usage,
};
use serde_json::{Map, Value, json};

use crate::{client, protocol};

pub(crate) fn request_body(
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<Option<Map<String, Value>>, DynamoError> {
    let Some(media) = request.multi_modal_data.as_ref() else {
        return Ok(None);
    };
    if !media.values().any(|items| !items.is_empty()) {
        return Ok(None);
    }
    if request.mm_processor_kwargs.is_some()
        || request.media_io_kwargs.is_some()
        || request.multi_modal_uuids.is_some()
    {
        return Err(client::invalid_arg(
            "SGLang multimodal HTTP generation does not support mm_processor_kwargs, media_io_kwargs, or multi_modal_uuids",
        ));
    }
    if request.encoder_result.is_some() {
        return Err(client::invalid_arg(
            "SGLang manages encoder dispatch through --language-only --encoder-urls; encoder_result is not supported",
        ));
    }
    let fields =
        protocol::build_generate_fields(request, request_id, mode, bootstrap_host, bootstrap_port)?;
    let sampling = fields
        .sampling_params
        .expect("sampling fields are always set");
    let mut sampling = json!({
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "min_p": sampling.min_p,
        "frequency_penalty": sampling.frequency_penalty,
        "presence_penalty": sampling.presence_penalty,
        "repetition_penalty": sampling.repetition_penalty,
        "max_new_tokens": sampling.max_new_tokens,
        "min_new_tokens": sampling.min_new_tokens,
        "stop": sampling.stop,
        "stop_token_ids": sampling.stop_token_ids,
        "ignore_eos": sampling.ignore_eos,
        "n": sampling.n,
        "json_schema": sampling.json_schema,
        "regex": sampling.regex,
    });
    // Omitted options must use SGLang's defaults, not explicit JSON nulls.
    sampling
        .as_object_mut()
        .unwrap()
        .retain(|_, value| !value.is_null());
    let mut body = Map::from_iter([
        ("sampling_params".into(), sampling),
        ("return_logprob".into(), json!(fields.return_logprob)),
        ("top_logprobs_num".into(), json!(fields.top_logprobs_num)),
        ("logprob_start_len".into(), json!(fields.logprob_start_len)),
        (
            "return_text_in_logprobs".into(),
            json!(
                !request
                    .output_options
                    .return_tokens_as_token_ids
                    .unwrap_or(false)
            ),
        ),
        ("require_reasoning".into(), json!(request.require_reasoning)),
    ]);
    for (modality, items) in media {
        if items.is_empty() {
            continue;
        }
        let field = match modality.as_str() {
            "image_url" => "image_data",
            "video_url" => "video_data",
            "audio_url" => "audio_data",
            _ => {
                return Err(client::invalid_arg(format!(
                    "unsupported SGLang media modality `{modality}`"
                )));
            }
        };
        let sources = items.iter().map(|item| match item {
            MultimodalData::Url(url) => Ok(Value::String(url.to_string())),
            MultimodalData::RawUrl(url) => Ok(Value::String(url.clone())),
            MultimodalData::Decoded(_) => Err(client::invalid_arg(
                "SGLang sidecar requires URL passthrough; pre-decoded RDMA media is unsupported",
            )),
            MultimodalData::UuidOnly(_) => Err(client::invalid_arg(
                "SGLang sidecar requires a media source; UUID-only media is unsupported",
            )),
        }).collect::<Result<Vec<_>, _>>()?;
        body.insert(field.into(), Value::Array(sources));
    }
    Ok(Some(body))
}

/// SGLang emits delta IDs when incremental_streaming_output is enabled.
/// Reuse the gRPC metadata conversion so finish reasons and logprobs agree.
pub(crate) fn output(
    response: Value,
    is_prefill: bool,
    return_tokens_as_ids: bool,
) -> Result<(LLMEngineOutput, bool), DynamoError> {
    if let Some(error) = response.get("error") {
        return Err(client::protocol_error(format!(
            "SGLang generation failed: {error}"
        )));
    }
    let meta = response
        .get("meta_info")
        .and_then(Value::as_object)
        .ok_or_else(|| client::protocol_error("SGLang response is missing meta_info"))?;
    let finished = meta
        .get("finish_reason")
        .is_some_and(|reason| !reason.is_null());
    let prompt_tokens = meta
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .unwrap_or(0);
    let generated = if is_prefill {
        0
    } else {
        meta.get("completion_tokens")
            .and_then(Value::as_u64)
            .and_then(|n| u32::try_from(n).ok())
            .unwrap_or(0)
    };
    let meta: HashMap<String, String> = meta
        .iter()
        .map(|(key, value)| (key.clone(), value.to_string()))
        .collect();
    let mut output = if finished {
        protocol::terminal_from_meta(&meta, prompt_tokens, generated)?
    } else {
        LLMEngineOutput::default().with_usage(usage(prompt_tokens, generated))
    };
    if !is_prefill {
        let ids = response
            .get("output_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| client::protocol_error("SGLang response is missing output_ids"))?;
        output.token_ids = ids
            .iter()
            .map(|id| {
                id.as_u64()
                    .and_then(|id| u32::try_from(id).ok())
                    .ok_or_else(|| {
                        client::protocol_error(format!("invalid SGLang output token id: {id}"))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        (output.log_probs, output.top_logprobs) =
            protocol::extract_logprobs(&meta, return_tokens_as_ids)?;
        output.engine_data = protocol::engine_data_from_meta(&meta, finished)?;
    }
    Ok((output, finished))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_backend_common::{FinishReason, OutputOptions, SamplingOptions, StopConditions};

    fn request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions::default())
            .build()
            .unwrap()
    }

    #[test]
    fn lowers_media_in_order_and_preserves_sampling() {
        let mut req = request();
        req.multi_modal_data = Some(HashMap::from([
            (
                "image_url".into(),
                vec![
                    MultimodalData::Url("https://example.com/a.png".parse().unwrap()),
                    MultimodalData::RawUrl("data:image/png;base64,Yg==".into()),
                ],
            ),
            (
                "video_url".into(),
                vec![MultimodalData::RawUrl("https://example.com/a.mp4".into())],
            ),
        ]));
        req.sampling_options.temperature = Some(0.0);
        req.stop_conditions.max_tokens = Some(16);
        req.stop_conditions.min_tokens = Some(2);
        req.output_options.logprobs = Some(3);
        req.require_reasoning = true;
        let body = request_body(&req, "req", DisaggregationMode::Aggregated, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            body["image_data"],
            json!(["https://example.com/a.png", "data:image/png;base64,Yg=="])
        );
        assert_eq!(body["video_data"], json!(["https://example.com/a.mp4"]));
        assert_eq!(body["sampling_params"]["temperature"], 0.0);
        assert_eq!(body["sampling_params"]["max_new_tokens"], 16);
        assert!(body["sampling_params"].get("top_p").is_none());
        assert_eq!(body["top_logprobs_num"], 3);
        assert_eq!(body["require_reasoning"], true);
        let body = request_body(
            &req,
            "req",
            DisaggregationMode::Prefill,
            Some("p"),
            Some(8998),
        )
        .unwrap()
        .unwrap();
        assert_eq!(body["sampling_params"]["max_new_tokens"], 1);
        assert!(body["sampling_params"].get("min_new_tokens").is_none());
        assert_eq!(body["return_logprob"], false);
    }

    #[test]
    fn rejects_payloads_that_cannot_be_forwarded_losslessly() {
        let mut req = request();
        req.multi_modal_data = Some(HashMap::from([(
            "image_url".into(),
            vec![MultimodalData::UuidOnly("cached".into())],
        )]));
        let lower = |req: &PreprocessedRequest| {
            request_body(req, "req", DisaggregationMode::Aggregated, None, None)
        };
        assert!(lower(&req).unwrap_err().to_string().contains("UUID-only"));
        req.multi_modal_data = Some(HashMap::from([(
            "image_url".into(),
            vec![MultimodalData::RawUrl("https://example.com/a.png".into())],
        )]));
        req.mm_processor_kwargs = Some(json!({"max_pixels": 1024}));
        assert!(
            lower(&req)
                .unwrap_err()
                .to_string()
                .contains("mm_processor_kwargs")
        );
        req.mm_processor_kwargs = None;
        req.encoder_result = Some(json!({}));
        assert!(
            lower(&req)
                .unwrap_err()
                .to_string()
                .contains("encoder_result")
        );
    }

    #[test]
    fn token_response_preserves_finish_usage_and_logprobs() {
        let (out, finished) = output(
            json!({
                "output_ids": [42],
                "meta_info": {"prompt_tokens": 300, "completion_tokens": 2,
                    "finish_reason": {"type": "length"},
                    "output_token_logprobs": [[-0.5, 42, "answer"]]}
            }),
            false,
            false,
        )
        .unwrap();
        assert!(finished);
        assert_eq!(out.token_ids, [42]);
        assert_eq!(out.finish_reason, Some(FinishReason::Length));
        assert_eq!(out.log_probs, Some(vec![-0.5]));
        assert_eq!(out.completion_usage.unwrap(), usage(300, 2));
        assert!(output(json!({"output_ids": [-1], "meta_info": {}}), false, false).is_err());
        assert!(output(json!({"meta_info": {"finish_reason": {"type": "abort", "message": "encoder unavailable"}}}), true, false).unwrap_err().to_string().contains("encoder unavailable"));
    }
}
