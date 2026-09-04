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
use serde::{Deserialize, Serialize};
use serde_json::Value;

use dynamo_sidecar_common::{json_to_struct, struct_to_json};

use crate::client;
use crate::proto as pb;

/// `extra` key the OpenEngine servicer reads to select the disaggregation phase.
pub const REQUEST_TYPE_KEY: &str = "request_type";
/// `extra.request_type` value marking a prefill-only request.
pub const CONTEXT_ONLY: &str = "context_only";

const ATTRIBUTES: &str = "prefill handoff attributes";

/// The handoff's JSON shape, mirroring [`pb::KvSessionRef`] field for field.
///
/// Both directions go through this one type, so the decode worker requires
/// exactly what the prefill worker writes: a handoff that lost a field in
/// transit fails here by name instead of decoding into a plausible-but-wrong
/// session (a dropped `dp_rank` would otherwise read as rank 0 and pull KV from
/// the wrong shard).
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Handoff {
    session_id: String,
    transfer_backend: String,
    endpoints: Vec<Endpoint>,
    dp_rank: u32,
    /// TensorRT-LLM's own opaque state, absent only if the server sent none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    attributes: Option<Value>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Endpoint {
    host: String,
    port: u32,
    protocol: String,
}

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

    let handoff = Handoff {
        session_id,
        transfer_backend,
        endpoints: endpoints
            .into_iter()
            .map(|endpoint| Endpoint {
                host: endpoint.host,
                port: endpoint.port,
                protocol: endpoint.protocol,
            })
            .collect(),
        dp_rank,
        attributes: attributes_struct
            .map(|attributes| struct_to_json(attributes, "TensorRT-LLM", ATTRIBUTES))
            .transpose()?,
    };
    serde_json::to_value(handoff).map_err(|error| {
        client::protocol_error(format!("prefill handoff could not be encoded: {error}"))
    })
}

/// Decodes the handoff JSON produced by [`session_to_json`] back into the
/// `KvSessionRef` the decode request replays.
pub(crate) fn session_from_json(value: &Value) -> Result<pb::KvSessionRef, DynamoError> {
    let handoff: Handoff = serde_json::from_value(value.clone()).map_err(|error| {
        client::invalid_argument(format!(
            "decode request prefill_result.disaggregated_params is not a TensorRT-LLM handoff: \
             {error}"
        ))
    })?;
    if handoff.session_id.is_empty() {
        return Err(client::invalid_argument(
            "decode request prefill handoff has an empty session_id",
        ));
    }
    Ok(pb::KvSessionRef {
        session_id: handoff.session_id,
        transfer_backend: handoff.transfer_backend,
        endpoints: handoff
            .endpoints
            .into_iter()
            .map(|endpoint| pb::KvEndpoint {
                host: endpoint.host,
                port: endpoint.port,
                protocol: endpoint.protocol,
            })
            .collect(),
        dp_rank: handoff.dp_rank,
        attributes_struct: handoff
            .attributes
            .map(|attributes| json_to_struct(attributes, ATTRIBUTES))
            .transpose()?,
    })
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
