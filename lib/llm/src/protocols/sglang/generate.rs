// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `/generate` request and unary response types.
//!
//! Dynamo exposes the token-input subset of SGLang's endpoint. The public
//! request keeps SGLang's field names and preserves `sampling_params`
//! opaquely for the version-matched worker. Text, batched, multimodal, and
//! streaming requests remain outside this token-in/token-out frontend.

use anyhow::Result;
use futures::{Stream, StreamExt, pin_mut};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

use crate::protocols::Annotated;
use crate::protocols::common::FinishReason;
use crate::protocols::common::llm_backend::LLMEngineOutput;

/// SGLang sampling parameters backed by their original JSON object.
#[derive(Debug, Clone, Default)]
pub struct SglangSamplingParams {
    raw: Map<String, Value>,
    max_new_tokens: Option<u32>,
    min_new_tokens: Option<u32>,
    n: Option<u32>,
    ignore_eos: Option<bool>,
}

impl SglangSamplingParams {
    pub fn as_map(&self) -> &Map<String, Value> {
        &self.raw
    }

    pub fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens.unwrap_or(128)
    }

    pub fn min_new_tokens(&self) -> u32 {
        self.min_new_tokens.unwrap_or(0)
    }

    pub fn n(&self) -> u32 {
        self.n.unwrap_or(1)
    }

    pub fn ignore_eos(&self) -> bool {
        self.ignore_eos.unwrap_or(false)
    }
}

impl Serialize for SglangSamplingParams {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.raw.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SglangSamplingParams {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Map::<String, Value>::deserialize(deserializer)?;
        Ok(Self {
            max_new_tokens: sampling_field(&raw, "max_new_tokens")
                .map_err(serde::de::Error::custom)?,
            min_new_tokens: sampling_field(&raw, "min_new_tokens")
                .map_err(serde::de::Error::custom)?,
            n: sampling_field(&raw, "n").map_err(serde::de::Error::custom)?,
            ignore_eos: sampling_field(&raw, "ignore_eos").map_err(serde::de::Error::custom)?,
            raw,
        })
    }
}

fn sampling_field<T>(object: &Map<String, Value>, name: &str) -> Result<Option<T>, String>
where
    T: serde::de::DeserializeOwned,
{
    object
        .get(name)
        .map(|value| serde_json::from_value::<Option<T>>(value.clone()))
        .transpose()
        .map(Option::flatten)
        .map_err(|error| format!("sampling_params.{name}: {error}"))
}

/// Native SGLang token-input request.
#[derive(Debug, Clone, Deserialize)]
pub struct SglangGenerateRequest {
    #[serde(default)]
    pub rid: Option<String>,
    pub input_ids: Vec<u32>,
    #[serde(default)]
    pub sampling_params: Option<SglangSamplingParams>,
    #[serde(default)]
    pub return_logprob: Option<bool>,
    #[serde(default)]
    pub logprob_start_len: Option<i32>,
    #[serde(default)]
    pub top_logprobs_num: Option<u32>,
    #[serde(default)]
    pub token_ids_logprob: Option<Vec<u32>>,
    #[serde(default)]
    pub return_text_in_logprobs: bool,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub return_routed_experts: bool,
    #[serde(default)]
    pub routed_experts_start_len: u32,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(flatten)]
    passthrough: Map<String, Value>,
}

impl SglangGenerateRequest {
    fn sampling(&self) -> SglangSamplingParams {
        self.sampling_params.clone().unwrap_or_default()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.input_ids.is_empty() {
            return Err("input_ids cannot be empty.".to_string());
        }
        let sampling = self.sampling();
        if sampling.n() != 1 {
            return Err(
                "sampling_params.n must be 1; parallel sampling is not supported.".to_string(),
            );
        }
        if sampling.min_new_tokens() > sampling.max_new_tokens() {
            return Err(format!(
                "min_new_tokens must be in [0, max_new_tokens({})], got {}.",
                sampling.max_new_tokens(),
                sampling.min_new_tokens()
            ));
        }
        if self.return_text_in_logprobs {
            return Err(
                "return_text_in_logprobs=true requires a tokenizer and is not supported by Dynamo's token-input SGLang endpoint."
                    .to_string(),
            );
        }
        if self.token_ids_logprob.is_some() {
            return Err(
                "token_ids_logprob is not supported by Dynamo's SGLang response adapter yet."
                    .to_string(),
            );
        }
        for field in [
            "bootstrap_info",
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "bootstrap_pair_key",
            "decode_tp_size",
            "disaggregated_params",
            "disagg_prefill_dp_rank",
            "routed_dp_rank",
            "external_trace_header",
            "received_time",
        ] {
            if self.passthrough.contains_key(field) {
                return Err(format!(
                    "`{field}` is internal Dynamo routing state and cannot be set by clients"
                ));
            }
        }
        if !self.passthrough.is_empty() {
            let mut unsupported: Vec<_> = self.passthrough.keys().cloned().collect();
            unsupported.sort();
            return Err(format!(
                "unsupported top-level SGLang generate field(s): {}",
                unsupported.join(", ")
            ));
        }
        Ok(())
    }

    pub fn max_new_tokens(&self) -> u32 {
        self.sampling().max_new_tokens()
    }

    pub fn min_new_tokens(&self) -> u32 {
        self.sampling().min_new_tokens()
    }

    pub fn ignore_eos(&self) -> bool {
        self.sampling().ignore_eos()
    }

    pub fn response_options(&self) -> SglangResponseOptions {
        SglangResponseOptions {
            return_logprob: self.return_logprob.unwrap_or(false),
            include_input_logprobs: self.return_logprob.unwrap_or(false)
                && self.logprob_start_len.unwrap_or(-1) >= 0,
            top_logprobs_num: self.top_logprobs_num.unwrap_or(0),
        }
    }

    /// Build the engine-owned envelope without canonical token IDs, request ID,
    /// or Dynamo-owned transport state.
    pub fn worker_envelope(&self) -> Value {
        let mut envelope = Map::new();
        envelope.insert(
            "sampling_params".to_string(),
            Value::Object(
                self.sampling_params
                    .as_ref()
                    .map(|sampling| sampling.as_map().clone())
                    .unwrap_or_default(),
            ),
        );
        envelope.insert(
            "return_logprob".to_string(),
            Value::Bool(self.return_logprob.unwrap_or(false)),
        );
        envelope.insert(
            "logprob_start_len".to_string(),
            Value::from(self.logprob_start_len.unwrap_or(-1)),
        );
        envelope.insert(
            "top_logprobs_num".to_string(),
            Value::from(self.top_logprobs_num.unwrap_or(0)),
        );
        envelope.insert(
            "return_routed_experts".to_string(),
            Value::Bool(self.return_routed_experts),
        );
        envelope.insert(
            "routed_experts_start_len".to_string(),
            Value::from(self.routed_experts_start_len),
        );
        Value::Object(envelope)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SglangResponseOptions {
    pub(super) return_logprob: bool,
    pub(super) include_input_logprobs: bool,
    pub(super) top_logprobs_num: u32,
}

#[derive(Debug, Serialize)]
pub struct SglangGenerateResponse {
    pub output_ids: Vec<u32>,
    pub meta_info: Map<String, Value>,
}

#[derive(Default)]
struct SglangChoiceAccumulator {
    output_ids: Vec<u32>,
    output_token_logprobs: Vec<Value>,
    output_top_logprobs: Vec<Value>,
    finish_reason: Option<Value>,
    completion_usage: Option<Value>,
    native_meta_info: Map<String, Value>,
}

impl SglangChoiceAccumulator {
    fn apply(&mut self, output: &LLMEngineOutput, options: SglangResponseOptions) -> Result<()> {
        if options.return_logprob && !output.token_ids.is_empty() {
            let logprobs = output.log_probs.as_ref().ok_or_else(|| {
                anyhow::anyhow!("SGLang response requested logprobs but returned none")
            })?;
            anyhow::ensure!(
                logprobs.len() == output.token_ids.len(),
                "SGLang returned {} selected-token logprobs for {} output IDs",
                logprobs.len(),
                output.token_ids.len()
            );
            for (logprob, token_id) in logprobs.iter().zip(&output.token_ids) {
                self.output_token_logprobs
                    .push(serde_json::json!([logprob, token_id, null]));
            }
            if options.top_logprobs_num > 0 {
                let top_logprobs = output.top_logprobs.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("SGLang response requested top logprobs but returned none")
                })?;
                anyhow::ensure!(
                    top_logprobs.len() == output.token_ids.len(),
                    "SGLang returned {} top-logprob positions for {} output IDs",
                    top_logprobs.len(),
                    output.token_ids.len()
                );
                self.output_top_logprobs
                    .extend(top_logprobs.iter().map(|position| {
                        Value::Array(
                            position
                                .iter()
                                .take(options.top_logprobs_num as usize)
                                .map(|entry| {
                                    serde_json::json!([entry.logprob, entry.token_id, null])
                                })
                                .collect(),
                        )
                    }));
            }
        }
        self.output_ids.extend_from_slice(&output.token_ids);
        if let Some(usage) = output.completion_usage.as_ref() {
            self.completion_usage = Some(serde_json::to_value(usage)?);
        }
        if let Some(engine_data) = output.engine_data.as_ref()
            && let Some(meta_info) = engine_data
                .get("sglang_meta_info")
                .and_then(Value::as_object)
        {
            self.native_meta_info = meta_info.clone();
        }
        if let Some(reason) = output.finish_reason.as_ref() {
            match reason {
                FinishReason::Error(message) => anyhow::bail!("{message}"),
                FinishReason::Cancelled => anyhow::bail!("backend cancelled generation"),
                reason => {
                    let mut finish_reason = Map::new();
                    finish_reason.insert("type".to_string(), Value::String(reason.to_string()));
                    if let Some(stop_reason) = output.stop_reason.as_ref() {
                        finish_reason
                            .insert("matched".to_string(), serde_json::to_value(stop_reason)?);
                    }
                    self.finish_reason = Some(Value::Object(finish_reason));
                }
            }
        }
        Ok(())
    }

    fn into_response(
        mut self,
        request_id: String,
        options: SglangResponseOptions,
    ) -> Result<SglangGenerateResponse> {
        let finish_reason = self
            .finish_reason
            .ok_or_else(|| anyhow::anyhow!("SGLang stream ended without a finish reason"))?;
        let finish_reason = self
            .native_meta_info
            .remove("finish_reason")
            .unwrap_or(finish_reason);
        self.native_meta_info
            .insert("id".to_string(), Value::String(request_id));
        self.native_meta_info
            .insert("finish_reason".to_string(), finish_reason);
        if let Some(Value::Object(usage)) = self.completion_usage {
            for key in ["prompt_tokens", "completion_tokens"] {
                if let Some(value) = usage.get(key) {
                    self.native_meta_info.insert(key.to_string(), value.clone());
                }
            }
            if let Some(cached_tokens) = usage
                .get("prompt_tokens_details")
                .and_then(Value::as_object)
                .and_then(|details| details.get("cached_tokens"))
            {
                self.native_meta_info
                    .insert("cached_tokens".to_string(), cached_tokens.clone());
            }
        }
        if options.return_logprob {
            self.native_meta_info.insert(
                "output_token_logprobs_length".to_string(),
                Value::from(self.output_token_logprobs.len()),
            );
            self.native_meta_info.insert(
                "output_token_logprobs".to_string(),
                Value::Array(self.output_token_logprobs),
            );
            if options.top_logprobs_num > 0 {
                self.native_meta_info.insert(
                    "output_top_logprobs".to_string(),
                    Value::Array(self.output_top_logprobs),
                );
            }
            if !options.include_input_logprobs {
                self.native_meta_info
                    .insert("input_token_logprobs".to_string(), Value::Array(Vec::new()));
                if options.top_logprobs_num > 0 {
                    self.native_meta_info
                        .insert("input_top_logprobs".to_string(), Value::Array(Vec::new()));
                } else {
                    self.native_meta_info.remove("input_top_logprobs");
                }
            }
        } else {
            for key in [
                "input_token_logprobs",
                "input_top_logprobs",
                "output_token_logprobs",
                "output_top_logprobs",
                "output_token_logprobs_length",
            ] {
                self.native_meta_info.remove(key);
            }
        }
        Ok(SglangGenerateResponse {
            output_ids: self.output_ids,
            meta_info: self.native_meta_info,
        })
    }
}

impl SglangGenerateResponse {
    pub async fn from_annotated_stream(
        stream: impl Stream<Item = Annotated<LLMEngineOutput>>,
        request_id: String,
        options: SglangResponseOptions,
    ) -> Result<Value> {
        let mut choice = SglangChoiceAccumulator::default();
        pin_mut!(stream);
        while let Some(delta) = stream.next().await {
            let delta = delta.ok().map_err(anyhow::Error::msg)?;
            if let Some(output) = delta.data {
                anyhow::ensure!(
                    output.index.unwrap_or(0) == 0,
                    "SGLang returned a non-zero choice index for n=1"
                );
                choice.apply(&output, options)?;
            }
        }
        Ok(serde_json::to_value(
            choice.into_response(request_id, options)?,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_preserves_native_sampling_params() {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "rid": "req-1",
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "max_new_tokens": 7,
                "sampling_seed": 42,
                "future_sglang_field": {"opaque": true}
            },
            "return_logprob": true,
            "top_logprobs_num": 2,
            "return_routed_experts": true,
            "priority": 9
        }))
        .unwrap();
        assert_eq!(request.max_new_tokens(), 7);
        assert!(request.validate().is_ok());
        assert_eq!(
            request.worker_envelope()["sampling_params"]["future_sglang_field"]["opaque"],
            true
        );
        assert_eq!(request.worker_envelope()["return_routed_experts"], true);
    }

    #[test]
    fn request_rejects_transport_owned_fields() {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [1],
            "bootstrap_host": "client-controlled"
        }))
        .unwrap();
        assert!(request.validate().unwrap_err().contains("internal Dynamo"));
    }

    #[test]
    fn request_rejects_parallel_sampling() {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [1],
            "sampling_params": {"n": 2}
        }))
        .unwrap();
        assert!(request.validate().unwrap_err().contains("must be 1"));
    }

    #[tokio::test]
    async fn response_uses_native_output_ids_meta_info_and_logprobs() {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [1, 2],
            "return_logprob": true,
            "logprob_start_len": -1
        }))
        .unwrap();
        let stream = futures::stream::iter([
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![101],
                log_probs: Some(vec![-0.1]),
                index: Some(0),
                ..Default::default()
            }),
            Annotated::from_data(LLMEngineOutput {
                token_ids: vec![102],
                log_probs: Some(vec![-0.2]),
                finish_reason: Some(FinishReason::Stop),
                index: Some(0),
                engine_data: Some(serde_json::json!({
                    "sglang_meta_info": {
                        "id": "worker-id",
                        "finish_reason": {"type": "stop", "matched": 151645},
                        "prompt_tokens": 2,
                        "completion_tokens": 2,
                        "cached_tokens": 1,
                        "weight_version": "v1",
                        "num_retractions": 0,
                        "input_token_logprobs": [[null, 1, null], [-0.3, 2, null]],
                        "routed_experts": "base64-experts"
                    }
                })),
                ..Default::default()
            }),
        ]);

        let response = SglangGenerateResponse::from_annotated_stream(
            stream,
            "req-native".to_string(),
            request.response_options(),
        )
        .await
        .unwrap();

        assert_eq!(response["output_ids"], serde_json::json!([101, 102]));
        assert_eq!(response["meta_info"]["id"], "req-native");
        assert_eq!(response["meta_info"]["finish_reason"]["type"], "stop");
        assert_eq!(response["meta_info"]["finish_reason"]["matched"], 151645);
        assert_eq!(response["meta_info"]["weight_version"], "v1");
        assert_eq!(response["meta_info"]["routed_experts"], "base64-experts");
        assert_eq!(
            response["meta_info"]["output_token_logprobs"],
            serde_json::json!([[-0.1, 101, null], [-0.2, 102, null]])
        );
        assert_eq!(
            response["meta_info"]["input_token_logprobs"],
            serde_json::json!([])
        );
        assert!(response.get("choices").is_none());
        assert!(response.get("request_id").is_none());
    }
}
