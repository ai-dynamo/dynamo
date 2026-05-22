// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-owned wire protocol for the disagg (conditional-disagg) feature.
//!
//! Mirrors [`features/indexer/protocol.rs`](crate::features::indexer::protocol):
//! all paths are **relative** — the server nests them under
//! `/v1/features/{ROUTE_PREFIX}` (see
//! [`FeatureManager::route_prefix`](crate::features::FeatureManager::route_prefix)).
//! The feature owns its whole namespace; nothing here lives in the central
//! [`crate::protocol::paths`].

use serde::{Deserialize, Serialize};
use velo_ext::InstanceId;

/// URL segment the server nests this feature's routers under
/// (`/v1/features/disagg/...`).
pub const ROUTE_PREFIX: &str = "disagg";

/// Relative route paths (mounted under `/v1/features/disagg`).
pub mod paths {
    /// `GET /instances` — registered instances split by P/D role.
    pub const INSTANCES: &str = "/instances";
}

/// Response body for `GET /v1/features/disagg/instances` — the registered
/// instances split by conditional-disagg role.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ConditionalDisaggInstancesResponse {
    /// Instance ids currently registered in the Prefill role.
    pub prefill: Vec<InstanceId>,
    /// Instance ids currently registered in the Decode role.
    pub decode: Vec<InstanceId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cd_instances_response_serde_round_trip() {
        let a = InstanceId::new_v4();
        let b = InstanceId::new_v4();
        let orig = ConditionalDisaggInstancesResponse {
            prefill: vec![a],
            decode: vec![b],
        };
        let json = serde_json::to_string(&orig).unwrap();
        let back: ConditionalDisaggInstancesResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back, orig);
    }
}
