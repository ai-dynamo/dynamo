// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! vLLM's side of the shared `Struct` <-> JSON codec.
//!
//! The conversion itself lives in `dynamo_sidecar_common::json`, which emits it
//! once per protobuf version; vLLM is generated against prost 0.14. Only the
//! labels that name the payload and the peer in error messages are supplied
//! here.

use dynamo_backend_common::DynamoError;

/// Both of vLLM's opaque payloads (`kv_transfer_params` and
/// `ec_transfer_params`) share this label, as they did before the codec moved.
const WHAT: &str = "kv_transfer_params";

pub(crate) fn json_to_struct(
    value: serde_json::Value,
) -> Result<prost_types_v14::Struct, DynamoError> {
    dynamo_sidecar_common::json_to_struct_v14(value, WHAT)
}

pub(crate) fn struct_to_json(
    value: prost_types_v14::Struct,
) -> Result<serde_json::Value, DynamoError> {
    dynamo_sidecar_common::struct_to_json_v14(value, "vLLM", WHAT)
}
