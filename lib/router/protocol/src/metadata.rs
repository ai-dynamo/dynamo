// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker and cache-owner metadata published in runtime configuration.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Persistent source metadata used to resolve a cache owner into a router hint.
///
/// This is advisory discovery metadata. Endpoint health and ownership takeover
/// are guaranteed by the persistent cache implementation, not probed by Dynamo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterHintSourceMetadata {
    pub source_control_endpoint: String,
    pub worker_type: String,
    /// Topology labels of the cache, like a worker's `topology_domains`. Hint
    /// targets with a `kv_transfer_domain` only fetch from the same island.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub topology_domains: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_hint_source_topology_is_optional_on_the_wire() {
        let without_topology = serde_json::json!({
            "source_control_endpoint": "tcp://owner:23280",
            "worker_type": "prefill",
        });
        let metadata: RouterHintSourceMetadata =
            serde_json::from_value(without_topology.clone()).unwrap();
        assert!(metadata.topology_domains.is_empty());
        assert_eq!(serde_json::to_value(&metadata).unwrap(), without_topology);
    }
}
