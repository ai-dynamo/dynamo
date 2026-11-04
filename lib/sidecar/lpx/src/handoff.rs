// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The vLLM prefill handoff, carried into the decode engine's `kv.session`.
//!
//! A vLLM prefill worker hands decode its `kv_transfer_params` through
//! `PrefillResult.disaggregated_params`, in one of two shapes:
//!
//! - wrapped, `{"kv_transfer_params": {...}}`, from the Python `dynamo.vllm`
//!   worker;
//! - flat, the object itself, from the Rust vLLM sidecar.
//!
//! The sidecar unwraps it and sends the object unchanged as
//! `KvSessionRef.attributes_struct`. It checks only the envelope: the engine
//! validates the producer contract, as it does for its own request plane.

use dynamo_backend_common::{DynamoError, PrefillResult};
use dynamo_sidecar_common::json_to_struct;
use serde_json::{Map, Value};

use crate::client::invalid_argument;
use crate::pb;

/// The transfer backend a decode-role engine pulls vLLM's KV cache with.
const NIXL: &str = "nixl";
/// The key the Python vLLM worker wraps its parameters in.
const WRAPPER_KEY: &str = "kv_transfer_params";
/// A `Struct` number is an IEEE-754 double, exact only up to 2^53.
const MAX_EXACT_INTEGER: u64 = 1 << 53;

/// Builds the decode request's `kv.session` from the prefill handoff.
pub(crate) fn decode_session(prefill: &PrefillResult) -> Result<pb::KvSessionRef, DynamoError> {
    let Value::Object(top) = &prefill.disaggregated_params else {
        return Err(invalid_argument(
            "prefill_result.disaggregated_params must be a JSON object",
        ));
    };
    let params = match top.get(WRAPPER_KEY) {
        Some(Value::Object(inner)) => inner.clone(),
        _ => top.clone(),
    };
    let session_id = params
        .get("remote_request_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| {
            invalid_argument(
                "prefill handoff is not a vLLM NIXL handoff: remote_request_id is missing or empty",
            )
        })?
        .to_owned();
    // kv.proto: carry integers a double cannot hold exactly as decimal strings.
    let params = stringify_inexact_integers(Value::Object(params));
    Ok(pb::KvSessionRef {
        session_id,
        transfer_backend: NIXL.to_string(),
        // The engine finds the producer through `remote_host` and `remote_port`.
        endpoints: Vec::new(),
        dp_rank: 0,
        attributes_struct: Some(json_to_struct(params, "vLLM kv_transfer_params")?),
    })
}

/// Replaces every integer outside +/-2^53 with its decimal string, at any
/// depth. Floats are left as they are.
fn stringify_inexact_integers(value: Value) -> Value {
    match value {
        Value::Number(number) if is_inexact_integer(&number) => {
            let text = number.to_string();
            tracing::debug!(
                value = %text,
                "carrying a vLLM handoff integer above 2^53 as a string"
            );
            Value::String(text)
        }
        Value::Array(values) => {
            Value::Array(values.into_iter().map(stringify_inexact_integers).collect())
        }
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .map(|(key, value)| (key, stringify_inexact_integers(value)))
                .collect::<Map<_, _>>(),
        ),
        other => other,
    }
}

fn is_inexact_integer(number: &serde_json::Number) -> bool {
    number
        .as_u64()
        .is_some_and(|value| value > MAX_EXACT_INTEGER)
        || number
            .as_i64()
            .is_some_and(|value| value.unsigned_abs() > MAX_EXACT_INTEGER)
}
