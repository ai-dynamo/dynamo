// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-client bookkeeping held by the registry.

use std::time::SystemTime;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque server-assigned identifier for a single registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RegistrationId(pub Uuid);

impl RegistrationId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for RegistrationId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RegistrationId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Bookkeeping for a single registered client.
#[derive(Debug, Clone)]
pub struct ClientEntry {
    pub id: RegistrationId,
    pub client_id: String,
    pub reserved_slots: u32,
    pub registered_at: SystemTime,
}

impl ClientEntry {
    pub fn new(client_id: String, reserved_slots: u32) -> Self {
        Self {
            id: RegistrationId::new(),
            client_id,
            reserved_slots,
            registered_at: SystemTime::now(),
        }
    }
}
