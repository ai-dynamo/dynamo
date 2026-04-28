// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy ext_proc protocol helpers: header extraction, metadata handling,
//! body chunking, and response construction.

use std::collections::HashMap;

use crate::proto::envoy::config::core::v3::{
    header_value_option::HeaderAppendAction, HeaderMap, HeaderValue, HeaderValueOption,
};
use crate::proto::envoy::service::ext_proc::v3::{
    BodyMutation, BodyResponse, CommonResponse, HeaderMutation, HeadersResponse,
    ImmediateResponse, ProcessingRequest, ProcessingResponse, StreamedBodyResponse,
    TrailersResponse,
};
use crate::proto::envoy::r#type::v3::{HttpStatus, StatusCode};

/// EPP protocol constants from proposal 004-endpoint-picker-protocol.
/// These match both the full EPP and the LW-EPP from GAIE (issue #2834).
pub mod metadata {
    pub const SUBSET_FILTER_NAMESPACE: &str = "envoy.lb.subset_hint";
    pub const SUBSET_FILTER_KEY: &str = "x-gateway-destination-endpoint-subset";
    pub const DESTINATION_ENDPOINT_NAMESPACE: &str = "envoy.lb";
    pub const DESTINATION_ENDPOINT_KEY: &str = "x-gateway-destination-endpoint";
    pub const DESTINATION_ENDPOINT_SERVED_KEY: &str = "x-gateway-destination-endpoint-served";
    pub const REQUEST_ID_HEADER_KEY: &str = "x-request-id";
}

/// Max body chunk size (62 KB). Envoy implementations cap at 64 KB;
/// we use a safe margin below that.
const BODY_BYTE_LIMIT: usize = 62_000;

/// System-owned headers that must not leak to the backend.
const SYSTEM_OWNED_HEADERS: &[&str] = &[
    "x-gateway-inference-fairness-id",
    "x-gateway-inference-objective",
    "x-gateway-model-name-rewrite",
    "x-gateway-destination-endpoint-subset",
    "x-gateway-destination-endpoint",
    "x-gateway-destination-endpoint-served",
    "content-length",
];

pub fn is_system_owned_header(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    SYSTEM_OWNED_HEADERS.iter().any(|h| *h == lower)
}

/// Build a `HeaderValueOption` that **replaces** any existing value for the key.
fn header_overwrite(key: &str, raw_value: &[u8]) -> HeaderValueOption {
    HeaderValueOption {
        header: Some(HeaderValue {
            key: key.to_string(),
            raw_value: raw_value.to_vec(),
            ..Default::default()
        }),
        append_action: HeaderAppendAction::OverwriteIfExistsOrAdd.into(),
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Header helpers
// ---------------------------------------------------------------------------

/// Get the string value from a HeaderValue, preferring `raw_value` over `value`.
pub fn get_header_value(header: &HeaderValue) -> String {
    if !header.raw_value.is_empty() {
        String::from_utf8_lossy(&header.raw_value).into_owned()
    } else {
        header.value.clone()
    }
}

/// Case-insensitive header lookup in a HeaderMap.
pub fn extract_header_value(headers: &HeaderMap, key: &str) -> Option<String> {
    let key_lower = key.to_ascii_lowercase();
    headers
        .headers
        .iter()
        .find(|h| h.key.to_ascii_lowercase() == key_lower)
        .map(get_header_value)
}

/// Extract filter metadata from a ProcessingRequest.
pub fn extract_metadata_values(req: &ProcessingRequest) -> HashMap<String, prost_types::Struct> {
    req.metadata_context
        .as_ref()
        .map(|m| m.filter_metadata.clone())
        .unwrap_or_default()
}

/// Preserve the full header list from the incoming request, retaining
/// duplicate keys. Returns a `Vec` of `(key, value)` pairs rather than
/// collapsing into a `HashMap` which would discard repeated headers.
pub fn collect_headers(header_map: &HeaderMap) -> Vec<(String, String)> {
    header_map
        .headers
        .iter()
        .map(|h| (h.key.clone(), get_header_value(h)))
        .collect()
}

/// Lookup a header value from a collected header list (case-insensitive).
pub fn find_header<'a>(headers: &'a [(String, String)], key: &str) -> Option<&'a str> {
    let key_lower = key.to_ascii_lowercase();
    headers
        .iter()
        .find(|(k, _)| k.to_ascii_lowercase() == key_lower)
        .map(|(_, v)| v.as_str())
}

// ---------------------------------------------------------------------------
// Response construction
// ---------------------------------------------------------------------------

/// Build the request-header response that tells Envoy where to route.
pub fn build_request_header_response(
    target_endpoint: &str,
    content_length: Option<usize>,
    extra_headers: &[(String, String)],
) -> ProcessingResponse {
    let mut set_headers: Vec<HeaderValueOption> = vec![
        header_overwrite(metadata::DESTINATION_ENDPOINT_KEY, target_endpoint.as_bytes()),
    ];

    if let Some(len) = content_length {
        set_headers.push(header_overwrite("Content-Length", len.to_string().as_bytes()));
    }

    for (key, value) in extra_headers {
        if is_system_owned_header(key) {
            continue;
        }
        set_headers.push(HeaderValueOption {
            header: Some(HeaderValue {
                key: key.clone(),
                raw_value: value.as_bytes().to_vec(),
                ..Default::default()
            }),
            ..Default::default()
        });
    }

    let dynamic_metadata = build_endpoint_metadata(target_endpoint);

    ProcessingResponse {
        response: Some(
            crate::proto::envoy::service::ext_proc::v3::processing_response::Response::RequestHeaders(
                HeadersResponse {
                    response: Some(CommonResponse {
                        clear_route_cache: true,
                        header_mutation: Some(HeaderMutation {
                            set_headers,
                            remove_headers: vec![],
                        }),
                        ..Default::default()
                    }),
                },
            ),
        ),
        dynamic_metadata: Some(dynamic_metadata),
        ..Default::default()
    }
}

/// Build the response-header response (pass-through, no mutations).
pub fn build_response_header_response() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(
            crate::proto::envoy::service::ext_proc::v3::processing_response::Response::ResponseHeaders(
                HeadersResponse {
                    response: Some(CommonResponse::default()),
                },
            ),
        ),
        ..Default::default()
    }
}

/// Build chunked body responses for request path.
pub fn build_request_body_responses(body: &[u8]) -> Vec<ProcessingResponse> {
    build_chunked_body_responses(body, true)
        .into_iter()
        .map(|common_resp| ProcessingResponse {
            response: Some(
                crate::proto::envoy::service::ext_proc::v3::processing_response::Response::RequestBody(
                    BodyResponse {
                        response: Some(common_resp),
                    },
                ),
            ),
            ..Default::default()
        })
        .collect()
}

/// Build chunked body responses for response path.
pub fn build_response_body_responses(
    body: &[u8],
    set_eos: bool,
    dynamic_metadata: Option<prost_types::Struct>,
) -> Vec<ProcessingResponse> {
    let mut responses: Vec<ProcessingResponse> = build_chunked_body_responses(body, set_eos)
        .into_iter()
        .map(|common_resp| ProcessingResponse {
            response: Some(
                crate::proto::envoy::service::ext_proc::v3::processing_response::Response::ResponseBody(
                    BodyResponse {
                        response: Some(common_resp),
                    },
                ),
            ),
            ..Default::default()
        })
        .collect();

    if let (Some(last), Some(dm)) = (responses.last_mut(), dynamic_metadata) {
        last.dynamic_metadata = Some(dm);
    }

    responses
}

/// Build a response-trailer response.
pub fn build_response_trailer_response() -> ProcessingResponse {
    ProcessingResponse {
        response: Some(
            crate::proto::envoy::service::ext_proc::v3::processing_response::Response::ResponseTrailers(
                TrailersResponse {
                    header_mutation: None,
                },
            ),
        ),
        ..Default::default()
    }
}

/// Build an ImmediateResponse for error cases.
pub fn build_error_response(status_code: StatusCode, body: Option<&str>) -> ProcessingResponse {
    ProcessingResponse {
        response: Some(
            crate::proto::envoy::service::ext_proc::v3::processing_response::Response::ImmediateResponse(
                ImmediateResponse {
                    status: Some(HttpStatus {
                        code: status_code.into(),
                    }),
                    body: body.map(|b| b.as_bytes().to_vec()).unwrap_or_default(),
                    ..Default::default()
                },
            ),
        ),
        ..Default::default()
    }
}

/// Build the eviction response (HTTP 429).
pub fn build_eviction_response() -> ProcessingResponse {
    build_error_response(
        StatusCode::TooManyRequests,
        Some("request evicted by flow control"),
    )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Split body bytes into chunks of BODY_BYTE_LIMIT, wrapping each in a
/// CommonResponse with StreamedBodyResponse. Mirrors the Go `BuildChunkedBodyResponses`.
fn build_chunked_body_responses(body: &[u8], set_eos: bool) -> Vec<CommonResponse> {
    if body.is_empty() {
        return vec![CommonResponse {
            body_mutation: Some(BodyMutation {
                mutation: Some(
                    crate::proto::envoy::service::ext_proc::v3::body_mutation::Mutation::StreamedResponse(
                        StreamedBodyResponse {
                            body: vec![],
                            end_of_stream: set_eos,
                            ..Default::default()
                        },
                    ),
                ),
            }),
            ..Default::default()
        }];
    }

    let mut responses = Vec::new();
    let mut offset = 0;

    while offset < body.len() {
        let end = std::cmp::min(offset + BODY_BYTE_LIMIT, body.len());
        let chunk = &body[offset..end];
        let eos = set_eos && end >= body.len();

        responses.push(CommonResponse {
            body_mutation: Some(BodyMutation {
                mutation: Some(
                    crate::proto::envoy::service::ext_proc::v3::body_mutation::Mutation::StreamedResponse(
                        StreamedBodyResponse {
                            body: chunk.to_vec(),
                            end_of_stream: eos,
                            ..Default::default()
                        },
                    ),
                ),
            }),
            ..Default::default()
        });
        offset = end;
    }

    responses
}

/// Build the `dynamic_metadata` Struct that tells Envoy the target endpoint.
/// Layout: `{"envoy.lb": {"x-gateway-destination-endpoint": "<endpoint>"}}`
fn build_endpoint_metadata(endpoint: &str) -> prost_types::Struct {
    use prost_types::{value::Kind, Struct, Value};

    let inner = Struct {
        fields: [(
            metadata::DESTINATION_ENDPOINT_KEY.to_string(),
            Value {
                kind: Some(Kind::StringValue(endpoint.to_string())),
            },
        )]
        .into(),
    };

    Struct {
        fields: [(
            metadata::DESTINATION_ENDPOINT_NAMESPACE.to_string(),
            Value {
                kind: Some(Kind::StructValue(inner)),
            },
        )]
        .into(),
    }
}

/// Replace occurrences of the target (internal) model name with the incoming
/// (client-facing) model name in the response body bytes. No-op when names
/// match or either is empty.
pub fn rewrite_model_name(body: &[u8], target_model: &str, incoming_model: &str) -> Vec<u8> {
    if target_model.is_empty()
        || incoming_model.is_empty()
        || target_model == incoming_model
    {
        return body.to_vec();
    }

    let old = format!("\"model\":\"{target_model}\"");
    let new = format!("\"model\":\"{incoming_model}\"");

    let body_str = String::from_utf8_lossy(body);
    let replaced = body_str.replace(&old, &new);

    if replaced.as_bytes() != body {
        return replaced.into_bytes();
    }

    let old_spaced = format!("\"model\": \"{target_model}\"");
    let new_spaced = format!("\"model\": \"{incoming_model}\"");
    body_str.replace(&old_spaced, &new_spaced).into_bytes()
}
