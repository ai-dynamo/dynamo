// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared preprocessed-multimodal wire limits and cache identity.

use sha2::{Digest, Sha256};

pub const MAX_PREPROCESSED_MM_FEATURES: usize = 64;
pub const MAX_PREPROCESSED_MM_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_PREPROCESSED_MM_MODALITY_BYTES: usize = 64;
pub const MAX_PREPROCESSED_MM_ROUTING_HASH_BYTES: usize = 512;

pub fn preprocessed_mm_cache_identifier(modality: &str, raw: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"vllm.grpc.preprocessed-mm.v1");
    hasher.update((modality.len() as u64).to_be_bytes());
    hasher.update(modality.as_bytes());
    hasher.update((raw.len() as u64).to_be_bytes());
    hasher.update(raw);
    format!("grpc-mm:{:x}", hasher.finalize())
}
