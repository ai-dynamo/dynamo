// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use dynamo_kv_router::config::{KvRouterConfig, RouterConfigOverride};

use serde::{Deserialize, Serialize};

/// How the router discovers its worker set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkerDiscoveryMode {
    Dynamo,
    External,
}

impl Default for WorkerDiscoveryMode {
    fn default() -> Self {
        Self::Dynamo
    }
}
