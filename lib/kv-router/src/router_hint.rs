// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-generated hints that are attached to selected backend requests.

use serde::{Deserialize, Serialize};

use crate::protocols::ExternalSequenceBlockHash;

/// Extra-args key for router-generated backend hints.
pub const ROUTER_HINT_EXTRA_ARGS_KEY: &str = "router_hint";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RouterHint {
    pub request_id: String,
    pub source_control_endpoint: String,
    pub kv_block_hashes: Vec<ExternalSequenceBlockHash>,
    /// Position in the request's prefix where `kv_block_hashes[0]` lives.
    pub start_block_index: u32,
}
