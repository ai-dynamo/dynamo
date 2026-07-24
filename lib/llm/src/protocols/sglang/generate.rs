// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `/generate` request types.
//!
//! Dynamo exposes the token-input subset of SGLang's endpoint. The public
//! request keeps SGLang's field names and preserves `sampling_params`
//! opaquely for the version-matched worker. Text, batched, multimodal, and
//! non-streaming requests remain outside this token-in/token-out frontend.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

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
}
