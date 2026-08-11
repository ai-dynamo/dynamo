// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native SGLang `/generate` request types.
//!
//! Dynamo exposes the token-input subset of SGLang's endpoint. The public
//! request keeps SGLang's field names and preserves `sampling_params`
//! opaquely for the version-matched worker. Text, batched, multimodal, and
//! non-streaming requests remain outside this token-in/token-out frontend.

use serde::Deserialize;
use serde_json::{Map, Value};

fn sampling_field<T>(object: Option<&Map<String, Value>>, name: &str) -> Result<Option<T>, String>
where
    T: serde::de::DeserializeOwned,
{
    object
        .and_then(|object| object.get(name))
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
    pub sampling_params: Option<Map<String, Value>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub priority: Option<i32>,
    #[serde(flatten)]
    passthrough: Map<String, Value>,
}

impl SglangGenerateRequest {
    pub fn validate(&self) -> Result<(), String> {
        if self.input_ids.is_empty() {
            return Err("input_ids cannot be empty.".to_string());
        }
        if sampling_field::<u32>(self.sampling_params.as_ref(), "n")?.unwrap_or(1) != 1 {
            return Err(
                "sampling_params.n must be 1; parallel sampling is not supported.".to_string(),
            );
        }
        for field in [
            "bootstrap_info",
            "bootstrap_host",
            "bootstrap_port",
            "bootstrap_room",
            "bootstrap_pair_key",
            "data_parallel_rank",
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
        Ok(())
    }

    pub fn max_new_tokens(&self) -> Result<Option<u32>, String> {
        sampling_field(self.sampling_params.as_ref(), "max_new_tokens")
    }

    pub fn min_new_tokens(&self) -> Result<Option<u32>, String> {
        sampling_field(self.sampling_params.as_ref(), "min_new_tokens")
    }

    pub fn ignore_eos(&self) -> Result<Option<bool>, String> {
        sampling_field(self.sampling_params.as_ref(), "ignore_eos")
    }

    /// Move the native request into its routed input and opaque worker envelope.
    pub fn into_worker_envelope(self, request_id: &str) -> (Vec<u32>, Value) {
        let mut envelope = self.passthrough;
        envelope.insert(
            "sampling_params".to_string(),
            self.sampling_params
                .map(Value::Object)
                .unwrap_or(Value::Null),
        );
        envelope.insert("rid".to_string(), Value::String(request_id.to_string()));
        envelope.insert("stream".to_string(), Value::Bool(true));
        (self.input_ids, Value::Object(envelope))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_preserves_native_body_opaquely() {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "rid": "client-request",
            "input_ids": [1, 2, 3],
            "sampling_params": {
                "max_new_tokens": 7,
                "sampling_seed": 42,
                "future_sglang_field": {"opaque": true}
            },
            "return_logprob": true,
            "return_text_in_logprobs": true,
            "token_ids_logprob": [17],
            "session_id": "session-1",
            "future_top_level_field": {"opaque": true},
            "priority": 9
        }))
        .unwrap();

        assert_eq!(request.max_new_tokens().unwrap(), Some(7));
        assert!(request.validate().is_ok());
        let (input_ids, envelope) = request.into_worker_envelope("resolved-request");
        assert_eq!(input_ids, [1, 2, 3]);
        assert_eq!(envelope["rid"], "resolved-request");
        assert_eq!(envelope["stream"], true);
        assert_eq!(envelope["session_id"], "session-1");
        assert_eq!(envelope["return_text_in_logprobs"], true);
        assert_eq!(envelope["token_ids_logprob"], serde_json::json!([17]));
        assert_eq!(
            envelope["sampling_params"]["future_sglang_field"]["opaque"],
            true
        );
        assert_eq!(envelope["future_top_level_field"]["opaque"], true);
        assert!(envelope.get("input_ids").is_none());
        assert!(envelope.get("priority").is_none());
    }

    #[test]
    fn omitted_sampling_defaults_stay_omitted() {
        let request: SglangGenerateRequest = serde_json::from_value(serde_json::json!({
            "input_ids": [1],
            "sampling_params": {}
        }))
        .unwrap();

        assert_eq!(request.max_new_tokens().unwrap(), None);
        assert_eq!(request.min_new_tokens().unwrap(), None);
        assert_eq!(request.ignore_eos().unwrap(), None);
    }

    #[test]
    fn request_rejects_transport_owned_fields() {
        for field in ["bootstrap_host", "data_parallel_rank", "routed_dp_rank"] {
            let mut body = serde_json::json!({"input_ids": [1]});
            body.as_object_mut()
                .unwrap()
                .insert(field.to_string(), serde_json::json!(1));
            let request: SglangGenerateRequest = serde_json::from_value(body).unwrap();
            assert!(request.validate().unwrap_err().contains("internal Dynamo"));
        }
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
