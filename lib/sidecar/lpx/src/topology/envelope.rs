// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The JSON envelope that carries `GetSnapshot` over the Dynamo request plane
//! between the conductor-side sidecar and the publisher.
//!
//! Both ends are this crate, but the protobuf messages stay opaque bytes so
//! the request plane never reinterprets the contract.

// The protocol's errors are gRPC statuses, which tonic does not box.
#![allow(clippy::result_large_err)]

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use prost::Message;
use serde::{Deserialize, Serialize};
use tonic::{Code, Status};

use super::proto::{GetSnapshotRequest, GetSnapshotResponse};

/// `{"request": "<base64 GetSnapshotRequest>"}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RelayRequest {
    pub request: String,
}

/// `{"code": <gRPC code>, "message": "...", "response": "<base64
/// GetSnapshotResponse>"}`, with `response` only when `code` is 0.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RelayResponse {
    pub code: i32,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<String>,
}

fn decode<M: Message + Default>(encoded: &str) -> Result<M, String> {
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|error| format!("invalid base64: {error}"))?;
    M::decode(bytes.as_slice()).map_err(|error| format!("invalid protobuf: {error}"))
}

impl RelayRequest {
    pub(crate) fn encode(request: &GetSnapshotRequest) -> Self {
        Self {
            request: STANDARD.encode(request.encode_to_vec()),
        }
    }

    pub(crate) fn decode(&self) -> Result<GetSnapshotRequest, String> {
        decode(&self.request)
    }
}

impl RelayResponse {
    pub(crate) fn encode(result: Result<GetSnapshotResponse, Status>) -> Self {
        match result {
            Ok(response) => Self {
                code: Code::Ok as i32,
                message: String::new(),
                response: Some(STANDARD.encode(response.encode_to_vec())),
            },
            Err(status) => Self {
                code: status.code() as i32,
                message: status.message().to_string(),
                response: None,
            },
        }
    }

    /// The answer the envelope carries; a malformed envelope is INTERNAL.
    pub(crate) fn decode(self) -> Result<GetSnapshotResponse, Status> {
        let malformed = |problem: String| {
            Status::internal(format!("malformed topology relay response: {problem}"))
        };
        match (Code::from_i32(self.code), self.response) {
            (Code::Ok, Some(response)) => decode(&response).map_err(malformed),
            (Code::Ok, None) => Err(malformed("code 0 without a response".to_string())),
            (_, Some(_)) => Err(malformed(format!("code {} with a response", self.code))),
            (code, None) => Err(Status::new(code, self.message)),
        }
    }
}
