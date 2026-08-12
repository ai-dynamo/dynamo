// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned KV-cache actions attached to selected backend requests.

use serde::{Deserialize, Serialize};

use crate::protocols::ExternalSequenceBlockHash;

/// The selected worker can consume a `MIGRATE` hint with the v1 payload.
pub const KV_HINT_MIGRATE_CAPABILITY_KEY: &str = "kv_hint.migrate.v1";

/// Worker runtime-data keys used to build migration transfer plans.
pub const KV_HINT_MIGRATE_WORKER_TYPE_RUNTIME_KEY: &str = "kv_hint_migrate_worker_type";
pub const KV_HINT_MIGRATE_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "kv_hint_migrate_source_control_endpoints";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvTransferPlan {
    pub source_control_endpoint: String,
    /// Root-aligned source-side KV block hashes. `block_hashes[i]`
    /// corresponds to request block `i`; the target decides which suffix to fetch.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MigrateHint {
    pub transfer_plan: KvTransferPlan,
}

/// Typed KV-cache actions for the selected backend request.
///
/// Each action has its own capability and payload version. New actions extend
/// this envelope without changing worker selection contracts.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHints {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migrate: Option<MigrateHint>,
}

impl KvHints {
    pub fn is_empty(&self) -> bool {
        self.migrate.is_none()
    }
}
