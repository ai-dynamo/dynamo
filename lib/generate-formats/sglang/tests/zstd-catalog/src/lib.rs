// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An external catalog carrying compressed routed-expert bytes end to end.
//!
//! Point `dynamo-generate-format-catalog` in the frontend binding's Cargo.toml
//! at this directory, with `package = "custom-generate-format-catalog"`.
//! Build the frontend and select `DYN_GENERATE_FORMAT=sglang-zstd`.
//! Workers must emit `Base64(zstd(bytes))` in `meta_info.routed_experts`,
//! compressing before Base64 encoding. Stock SGLang needs a producer change.
//! Aggregated responses pass through unchanged. Only a P/D join decompresses
//! the two values, overlays prefill's prefix, and recompresses for the client.
//! Clients Base64-decode the field, then decompress Zstandard.
//!
//! Run independently with `cargo test --manifest-path <this directory>/Cargo.toml`.
//! No Dynamo source changes or GPU are needed to exercise the codec and HTTP route.

use std::sync::Arc;

use base64::{Engine as _, engine::general_purpose::STANDARD};
use dynamo_llm::generate_format::Registry;
use dynamo_sglang_generate::{ClientCodec, Native, SglangFormat};
use serde_json::Value;

pub fn register(registry: &mut Registry) -> anyhow::Result<()> {
    builtin_catalog::register(registry)?;
    registry.register(Arc::new(SglangFormat::with_codec("sglang-zstd", Zstd)))
}

pub struct Zstd;

impl ClientCodec for Zstd {
    fn overlay_sequence(&self, path: &str, prefix: &Value, target: &mut Value) -> bool {
        if path != "/meta_info/routed_experts" {
            return Native.overlay_sequence(path, prefix, target);
        }
        let (Value::String(prefix), Value::String(target)) = (prefix, target) else {
            return false;
        };
        let decode = |value: &str| {
            let frame = STANDARD.decode(value).ok()?;
            zstd::stream::decode_all(frame.as_slice()).ok()
        };
        let (Some(mut joined), Some(tail)) = (decode(prefix), decode(target)) else {
            return false;
        };
        joined.extend_from_slice(tail.get(joined.len()..).unwrap_or_default());
        let Ok(frame) = zstd::stream::encode_all(joined.as_slice(), 3) else {
            return false;
        };
        *target = STANDARD.encode(frame);
        true
    }
}
