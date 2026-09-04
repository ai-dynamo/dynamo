// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Disaggregated handoff: what the prefill role stamps into `KvSessionRef` and
//! what the decode role demands back.
//!
//! The decode checks are stricter than a real engine's on purpose. The sidecar
//! relays this payload through a JSON codec (`session_to_json` /
//! `session_from_json`), whose failure modes are dropped fields defaulting to
//! empty, coerced number types, and flattened lists — each of which still yields
//! a *plausible* session. The four attributes below exist so that each of those
//! is caught: a string, a whole-valued double, a fractional double, and a list.

use dynamo_trtllm_sidecar::proto as pb;
use prost_types::{ListValue, Struct, Value, value::Kind};
use tonic::Status;

use super::{BoxedStatusResult, DP_RANK, MockerServerConfig};

pub(super) const TRANSFER_BACKEND: &str = "MOCKER";
pub(super) const KV_PROTOCOL: &str = "mocker";
pub(super) const SESSION_PREFIX: &str = "mocker-prefill-";

/// Opaque attributes the prefill role stamps into every handoff. The decode role
/// cannot reconstruct any of them, so requiring them proves the sidecar
/// forwarded `attributes_struct` verbatim rather than rebuilding it from the
/// fields it happens to understand.
pub(super) const ATTR_REQUEST_ID: &str = "mocker_request_id";
pub(super) const ATTR_PROMPT_TOKENS: &str = "mocker_prompt_tokens";
pub(super) const ATTR_TTFT_MS: &str = "mocker_ttft_ms";
pub(super) const ATTR_FIRST_GEN_TOKENS: &str = "mocker_first_gen_tokens";

/// Deliberately fractional: a codec that rounded Struct numbers to integers
/// would round-trip every other numeric attribute unnoticed.
const TTFT_MS: f64 = 12.5;

fn invalid<T>(message: impl Into<String>) -> BoxedStatusResult<T> {
    Err(Box::new(Status::invalid_argument(message)))
}

pub(super) fn session_id(uuid: uuid::Uuid) -> String {
    format!("{SESSION_PREFIX}{uuid}")
}

pub(super) fn build_session(
    config: &MockerServerConfig,
    session_id: String,
    request_id: &str,
    prompt_tokens: usize,
    first_gen_token: u32,
) -> pb::KvSessionRef {
    let attributes = [
        (ATTR_REQUEST_ID, string_value(request_id)),
        (ATTR_PROMPT_TOKENS, number_value(prompt_tokens as f64)),
        (ATTR_TTFT_MS, number_value(TTFT_MS)),
        (
            ATTR_FIRST_GEN_TOKENS,
            Value {
                kind: Some(Kind::ListValue(ListValue {
                    values: vec![number_value(f64::from(first_gen_token))],
                })),
            },
        ),
    ];

    pb::KvSessionRef {
        session_id,
        transfer_backend: TRANSFER_BACKEND.to_string(),
        endpoints: vec![pb::KvEndpoint {
            host: config.kv_host.clone(),
            port: u32::from(config.kv_port),
            protocol: KV_PROTOCOL.to_string(),
        }],
        dp_rank: DP_RANK,
        attributes_struct: Some(Struct {
            fields: attributes
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        }),
    }
}

/// The first token the context phase produced, which a real generation worker
/// replays as the decode leg's first output.
pub(super) fn first_gen_token(session: &pb::KvSessionRef) -> Option<u32> {
    let attributes = session.attributes_struct.as_ref()?;
    let Some(Kind::ListValue(list)) = attributes.fields.get(ATTR_FIRST_GEN_TOKENS)?.kind.as_ref()
    else {
        return None;
    };
    match list.values.first()?.kind.as_ref()? {
        Kind::NumberValue(value) if value.is_finite() && *value >= 0.0 => Some(*value as u32),
        _ => None,
    }
}

pub(super) fn validate_session(session: &pb::KvSessionRef) -> BoxedStatusResult<()> {
    if !session.session_id.starts_with(SESSION_PREFIX) {
        return invalid(format!(
            "kv_session.session_id must start with '{SESSION_PREFIX}', got '{}'",
            session.session_id
        ));
    }
    if session.transfer_backend != TRANSFER_BACKEND {
        return invalid(format!(
            "kv_session.transfer_backend must be '{TRANSFER_BACKEND}', got '{}'",
            session.transfer_backend
        ));
    }
    let [endpoint] = session.endpoints.as_slice() else {
        return invalid(format!(
            "kv_session.endpoints must carry exactly one endpoint, got {}",
            session.endpoints.len()
        ));
    };
    if endpoint.protocol != KV_PROTOCOL {
        return invalid(format!(
            "kv_session.endpoints[0].protocol must be '{KV_PROTOCOL}', got '{}'",
            endpoint.protocol
        ));
    }
    // The prefill worker's endpoint is its own; only its survival is checkable
    // here, not its value.
    if endpoint.host.is_empty() || endpoint.port == 0 {
        return invalid(format!(
            "kv_session.endpoints[0] lost its address, got '{}':{}",
            endpoint.host, endpoint.port
        ));
    }
    if session.dp_rank != DP_RANK {
        return invalid(format!(
            "kv_session.dp_rank must be {DP_RANK}, got {}",
            session.dp_rank
        ));
    }

    let attributes = session
        .attributes_struct
        .as_ref()
        .ok_or_else(|| Box::new(Status::invalid_argument("kv_session carries no attributes")))?;

    attribute_string(attributes, ATTR_REQUEST_ID)?;
    let prompt_tokens = attribute_number(attributes, ATTR_PROMPT_TOKENS)?;
    if prompt_tokens.fract() != 0.0 || prompt_tokens < 0.0 {
        return invalid(format!(
            "kv_session attribute '{ATTR_PROMPT_TOKENS}' must be a whole count, got {prompt_tokens}"
        ));
    }
    let ttft_ms = attribute_number(attributes, ATTR_TTFT_MS)?;
    if ttft_ms != TTFT_MS {
        return invalid(format!(
            "kv_session attribute '{ATTR_TTFT_MS}' must survive as {TTFT_MS}, got {ttft_ms}"
        ));
    }
    if first_gen_token(session).is_none() {
        return invalid(format!(
            "kv_session attribute '{ATTR_FIRST_GEN_TOKENS}' must be a non-empty list of tokens"
        ));
    }
    Ok(())
}

fn attribute_string(attributes: &Struct, key: &str) -> BoxedStatusResult<String> {
    match attributes.fields.get(key).map(|value| &value.kind) {
        Some(Some(Kind::StringValue(value))) => Ok(value.clone()),
        Some(_) => invalid(format!("kv_session attribute '{key}' must be a string")),
        None => invalid(format!("kv_session is missing attribute '{key}'")),
    }
}

fn attribute_number(attributes: &Struct, key: &str) -> BoxedStatusResult<f64> {
    match attributes.fields.get(key).map(|value| &value.kind) {
        Some(Some(Kind::NumberValue(value))) => Ok(*value),
        Some(_) => invalid(format!("kv_session attribute '{key}' must be a number")),
        None => invalid(format!("kv_session is missing attribute '{key}'")),
    }
}

fn string_value(value: &str) -> Value {
    Value {
        kind: Some(Kind::StringValue(value.to_string())),
    }
}

fn number_value(value: f64) -> Value {
    Value {
        kind: Some(Kind::NumberValue(value)),
    }
}
