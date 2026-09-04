// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Disaggregated prefill/decode handoff between Dynamo and OpenEngine.
//!
//! OpenEngine has no request-type field. A context (prefill) request is marked
//! by `extra.request_type = "context_only"`, and the server answers it with a
//! terminal `PrefillReady` event carrying a [`pb::KvSessionRef`]. A generation
//! (decode) request replays that same session in `kv.session`, which the server
//! decodes back into TensorRT-LLM's context handoff.
//!
//! Dynamo carries the handoff as opaque JSON (`PrefillResult.disaggregated_params`),
//! so this module is the codec between the two. The JSON mirrors `KvSessionRef`
//! field-for-field; it is written by the prefill worker and read by the decode
//! worker, and never interpreted in between.

use dynamo_backend_common::DynamoError;
use serde_json::{Map, Value, json};

use dynamo_sidecar_common::{json_to_struct, struct_to_json};

use crate::client;
use crate::proto as pb;

/// `extra` key the OpenEngine servicer reads to select the disaggregation phase.
pub const REQUEST_TYPE_KEY: &str = "request_type";
/// `extra.request_type` value marking a prefill-only request.
pub const CONTEXT_ONLY: &str = "context_only";

const ATTRIBUTES: &str = "prefill handoff attributes";

/// Encodes the prefill worker's `KvSessionRef` as the opaque JSON Dynamo
/// forwards to the decode worker.
pub(crate) fn session_to_json(session: pb::KvSessionRef) -> Result<Value, DynamoError> {
    let pb::KvSessionRef {
        session_id,
        transfer_backend,
        endpoints,
        dp_rank,
        attributes_struct,
    } = session;

    if session_id.is_empty() {
        return Err(client::protocol_error(
            "prefill_ready carried no kv_session.session_id",
        ));
    }

    let endpoints: Vec<Value> = endpoints
        .into_iter()
        .map(|endpoint| {
            json!({
                "host": endpoint.host,
                "port": endpoint.port,
                "protocol": endpoint.protocol,
            })
        })
        .collect();

    let mut fields = Map::new();
    fields.insert("session_id".to_string(), Value::String(session_id));
    fields.insert(
        "transfer_backend".to_string(),
        Value::String(transfer_backend),
    );
    fields.insert("endpoints".to_string(), Value::Array(endpoints));
    fields.insert("dp_rank".to_string(), json!(dp_rank));
    if let Some(attributes) = attributes_struct {
        fields.insert(
            "attributes".to_string(),
            struct_to_json(attributes, "TensorRT-LLM", ATTRIBUTES)?,
        );
    }
    Ok(Value::Object(fields))
}

/// Decodes the handoff JSON produced by [`session_to_json`] back into the
/// `KvSessionRef` the decode request replays.
pub(crate) fn session_from_json(value: &Value) -> Result<pb::KvSessionRef, DynamoError> {
    let Value::Object(fields) = value else {
        return Err(client::invalid_argument(
            "decode request prefill_result.disaggregated_params must be a JSON object",
        ));
    };

    let session_id = string_field(fields, "session_id")?.ok_or_else(|| {
        client::invalid_argument("decode request prefill handoff is missing session_id")
    })?;
    let transfer_backend = string_field(fields, "transfer_backend")?.unwrap_or_default();
    let dp_rank = match fields.get("dp_rank") {
        None | Some(Value::Null) => 0,
        Some(value) => u32::try_from(value.as_u64().ok_or_else(|| {
            client::invalid_argument("decode request prefill handoff dp_rank must be an integer")
        })?)
        .map_err(|_| {
            client::invalid_argument("decode request prefill handoff dp_rank does not fit in u32")
        })?,
    };

    let endpoints = match fields.get("endpoints") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(values)) => values
            .iter()
            .map(endpoint_from_json)
            .collect::<Result<_, _>>()?,
        Some(_) => {
            return Err(client::invalid_argument(
                "decode request prefill handoff endpoints must be an array",
            ));
        }
    };

    let attributes_struct = match fields.get("attributes") {
        None | Some(Value::Null) => None,
        Some(value) => Some(json_to_struct(value.clone(), ATTRIBUTES)?),
    };

    Ok(pb::KvSessionRef {
        session_id,
        transfer_backend,
        endpoints,
        dp_rank,
        attributes_struct,
    })
}

fn endpoint_from_json(value: &Value) -> Result<pb::KvEndpoint, DynamoError> {
    let Value::Object(fields) = value else {
        return Err(client::invalid_argument(
            "decode request prefill handoff endpoint must be a JSON object",
        ));
    };
    let port = fields
        .get("port")
        .and_then(Value::as_u64)
        .and_then(|port| u32::try_from(port).ok())
        .ok_or_else(|| {
            client::invalid_argument("decode request prefill handoff endpoint port is invalid")
        })?;
    Ok(pb::KvEndpoint {
        host: string_field(fields, "host")?.unwrap_or_default(),
        port,
        protocol: string_field(fields, "protocol")?.unwrap_or_default(),
    })
}

fn string_field(fields: &Map<String, Value>, key: &str) -> Result<Option<String>, DynamoError> {
    match fields.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(client::invalid_argument(format!(
            "decode request prefill handoff {key} must be a string"
        ))),
    }
}

/// `extra` payload marking a request as prefill-only.
pub fn context_only_extra() -> prost_types::Struct {
    prost_types::Struct {
        fields: [(
            REQUEST_TYPE_KEY.to_string(),
            prost_types::Value {
                kind: Some(prost_types::value::Kind::StringValue(
                    CONTEXT_ONLY.to_string(),
                )),
            },
        )]
        .into_iter()
        .collect(),
    }
}
