// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An external catalog changing only the client encoding of routed-expert bytes.
//!
//! Point `dynamo-generate-format-catalog` in the frontend binding's Cargo.toml
//! at this directory, with `package = "custom-generate-format-catalog"`.
//! Build the frontend and select `DYN_GENERATE_FORMAT=sglang-zstd`.
//! Workers still send native Base64. Joining and migration run before this codec.
//! Clients Base64-decode `meta_info.routed_experts`, then decompress the Zstandard
//! frame to recover the original bytes. All other fields keep their native format.
//!
//! Run independently with `cargo test --manifest-path <this directory>/Cargo.toml`.
//! No Dynamo source changes or GPU are needed to exercise the codec and HTTP route.

use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_llm::generate_format::Registry;
use dynamo_sglang_generate::{ClientCodec, Encoded, SglangFormat};
use serde_json::Value;

pub fn register(registry: &mut Registry) -> anyhow::Result<()> {
    builtin_catalog::register(registry)?;
    registry.register(Arc::new(SglangFormat::with_codec("sglang-zstd", Zstd)))
}

pub struct Zstd;

impl ClientCodec for Zstd {
    fn encode_response(
        &self,
        response: &Value,
    ) -> Result<Encoded, dynamo_runtime::error::DynamoError> {
        use dynamo_runtime::error::DynamoError;
        let Some(value) = response
            .pointer("/meta_info/routed_experts")
            .filter(|v| !v.is_null())
        else {
            return Ok(Encoded::Native);
        };
        let encoded = value
            .as_str()
            .ok_or_else(|| DynamoError::msg("routed_experts must be a string"))?;
        let bytes = STANDARD
            .decode(encoded)
            .map_err(|_| DynamoError::msg("invalid routed_experts Base64"))?;
        let compressed = zstd::stream::encode_all(bytes.as_slice(), 3)
            .map_err(|_| DynamoError::msg("routed_experts compression failed"))?;
        let mut response = response.clone();
        response["meta_info"]["routed_experts"] = STANDARD.encode(compressed).into();
        Ok(Encoded::Replace(response))
    }
}
