// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An external catalog changing only the client encoding of routed-expert bytes.
//!
//! Point `dynamo-generate-format-catalog` in the frontend binding's Cargo.toml
//! at this directory, with `package = "custom-generate-format-catalog"`.
//! Build the frontend and select `DYN_GENERATE_FORMAT=sglang-base85`.
//! Workers still send native Base64. Joining and migration run before this codec.
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
    registry.register(Arc::new(SglangFormat::with_codec("sglang-base85", Base85)))
}

pub struct Base85;

impl ClientCodec for Base85 {
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
        let mut response = response.clone();
        response["meta_info"]["routed_experts"] = encode(&bytes).into();
        Ok(Encoded::Replace(response))
    }
}

/// Python `base64.b85encode` alphabet and unpadded final block convention.
fn encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 85] =
        b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!#$%&()*+-;<=>?@^_`{|}~";
    let mut output = String::new();
    for chunk in bytes.chunks(4) {
        let mut word = [0; 4];
        word[..chunk.len()].copy_from_slice(chunk);
        let mut value = u32::from_be_bytes(word);
        let mut digits = [0; 5];
        for digit in digits.iter_mut().rev() {
            *digit = ALPHABET[(value % 85) as usize];
            value /= 85;
        }
        output.extend(digits[..chunk.len() + 1].iter().map(|&c| c as char));
    }
    output
}
