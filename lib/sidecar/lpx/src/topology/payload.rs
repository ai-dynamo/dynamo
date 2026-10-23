// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Required LPX PoC exception to OpenEngine's optional `extra` metadata.
#![allow(clippy::result_large_err)]

use super::{
    check_snapshot,
    proto::{Snapshot, TopologyPayload},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use prost::Message;
use tonic::Status;

pub(crate) fn decode_payload(encoded: &str) -> Result<Snapshot, Status> {
    let invalid = |reason| Status::failed_precondition(format!("invalid lpx_topology: {reason}"));
    let bytes = STANDARD
        .decode(encoded)
        .map_err(|error| invalid(format!("base64: {error}")))?;
    let payload = TopologyPayload::decode(bytes.as_slice())
        .map_err(|error| invalid(format!("protobuf: {error}")))?;
    if payload.schema_version != 1 {
        return Err(invalid(format!(
            "unsupported schema version {}",
            payload.schema_version
        )));
    }
    let snapshot = payload
        .snapshot
        .ok_or_else(|| invalid("missing snapshot".into()))?;
    check_snapshot(&snapshot).map_err(invalid)?;
    Ok(snapshot)
}
