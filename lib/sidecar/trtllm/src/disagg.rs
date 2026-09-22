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
/// Every field is required, `attributes` most of all: the server reads the
/// session's location and rank out of it (`opaque_state`, `ctx_info_endpoint`,
/// `ctx_dp_rank`), treats `endpoints` only as a fallback, and never reads
/// `transfer_backend`. A handoff that lost `attributes` in transit still passes
/// the server's own guard, which accepts an endpoint in place of an
/// `opaque_state`, and then resumes a session with no opaque state and a rank
/// defaulted to 0. Requiring it here turns that into a named failure.
///
/// Unknown fields are accepted on purpose. The required fields already reject
/// another engine's handoff, so rejecting unknown ones would only reject a
/// *newer* peer's -- during a rolling upgrade, every new-prefill/old-decode
/// request, as a non-migratable 400.
#[derive(Serialize, Deserialize)]
struct Handoff {
    session_id: String,
    transfer_backend: String,
    endpoints: Vec<Endpoint>,
    dp_rank: u32,
    attributes: Value,
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
    // The decode leg cannot locate the session without these, so refuse to emit
    // a handoff that would fail there instead of here.
    let Some(attributes_struct) = attributes_struct else {
        return Err(client::protocol_error(
            "prefill_ready carried no kv_session.attributes_struct",
        ));
    };

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
        attributes: struct_to_json(attributes_struct, "TensorRT-LLM", ATTRIBUTES)?,
    };
    serde_json::to_value(handoff).map_err(|error| {
        client::protocol_error(format!("prefill handoff could not be encoded: {error}"))
    })
}

/// Decodes the handoff JSON produced by [`session_to_json`] back into the
/// `KvSessionRef` the decode request replays.
pub(crate) fn session_from_json(value: &Value) -> Result<pb::KvSessionRef, DynamoError> {
    // Deserialize from the borrowed tree: `from_value` would deep-clone the whole
    // handoff, including `opaque_state` and `first_gen_log_probs`, only to drop
    // the copy again.
    let handoff = Handoff::deserialize(value).map_err(|error| {
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
        attributes_struct: Some(json_to_struct(handoff.attributes, ATTRIBUTES)?),
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
