// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV transfer hints attached to backend requests, and the runtime-config keys that advertise them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::block_hash::ExternalSequenceBlockHash;

/// The selected worker can consume a `TRANSFER` hint with the v1 payload.
// TODO: Rename these constants and wire values with the matching KVCC names.
pub const KV_HINT_TRANSFER_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime-data keys used to build transfer hints.
pub const KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";
pub const KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

const KV_HINT_PROTOCOL_VERSION: &str = "0.1";
const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";
const KV_FETCH_ACTION_VERSION: &str = "1.0";

/// Typed payload for the `kv.fetch@1.0` point-to-point action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvSourceLocationsPayload {
    pub source_control_endpoint: String,
    /// Root-aligned source-side KV block hashes. `block_hashes[i]`
    /// corresponds to request block `i`; the target decides which suffix to fetch.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

/// One versioned action in a [`KvHint`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHintAction {
    pub action_id: String,
    pub action_type: String,
    pub action_version: String,
    pub payload: BTreeMap<String, serde_json::Value>,
}

impl KvHintAction {
    pub fn new(
        action_id: impl Into<String>,
        action_type: impl Into<String>,
        action_version: impl Into<String>,
        payload: BTreeMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            action_id: action_id.into(),
            action_type: action_type.into(),
            action_version: action_version.into(),
            payload,
        }
    }

    pub fn fetch(action_id: impl Into<String>, payload: KvSourceLocationsPayload) -> Self {
        let KvSourceLocationsPayload {
            source_control_endpoint,
            block_hashes,
        } = payload;
        Self::new(
            action_id,
            KV_FETCH_ACTION_TYPE,
            KV_FETCH_ACTION_VERSION,
            BTreeMap::from([
                (
                    "source_control_endpoint".to_string(),
                    serde_json::json!(source_control_endpoint),
                ),
                ("block_hashes".to_string(), serde_json::json!(block_hashes)),
            ]),
        )
    }
}

/// One versioned KV hint message for the selected backend request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHint {
    pub protocol_version: String,
    pub message_id: String,
    pub actions: Vec<KvHintAction>,
}

impl KvHint {
    pub fn new(message_id: impl Into<String>, actions: Vec<KvHintAction>) -> Self {
        Self {
            protocol_version: KV_HINT_PROTOCOL_VERSION.to_string(),
            message_id: message_id.into(),
            actions,
        }
    }
}
