// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Stable identifier for one prefill-to-decode handoff attempt.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HandoffId(Uuid);

impl HandoffId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for HandoffId {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Uuid> for HandoffId {
    fn from(value: Uuid) -> Self {
        Self(value)
    }
}

impl From<HandoffId> for Uuid {
    fn from(value: HandoffId) -> Self {
        value.0
    }
}
