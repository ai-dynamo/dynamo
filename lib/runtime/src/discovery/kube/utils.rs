// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Hash a pod name to get a consistent instance ID
pub fn hash_pod_name(pod_name: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    pod_name.hash(&mut hasher);
    hasher.finish()
}

/// Extract endpoint information from an EndpointSlice
/// Returns (instance_id, pod_name) tuples for ready endpoints
pub(super) fn extract_endpoint_info(slice: &EndpointSlice) -> Vec<(u64, String)> {
    let mut result = Vec::new();

    for endpoint in &slice.endpoints {
        let is_ready = endpoint
            .conditions
            .as_ref()
            .and_then(|c| c.ready)
            .unwrap_or(false);

        if !is_ready {
            continue;
        }

        let pod_name = match endpoint.target_ref.as_ref() {
            Some(target_ref) => target_ref.name.as_deref().unwrap_or(""),
            None => continue,
        };

        if pod_name.is_empty() {
            continue;
        }

        let instance_id = hash_pod_name(pod_name);

        result.push((instance_id, pod_name.to_string()));
    }

    result
}

/// Pod information extracted from environment
#[derive(Debug, Clone)]
pub(super) struct PodInfo {
    pub pod_name: String,
    pub pod_namespace: String,
    pub pod_uid: String,
    pub system_port: u16,
}

impl PodInfo {
    /// Discover pod information from environment variables
    ///
    /// Required environment variables:
    /// - `POD_NAME`: Name of the pod (required)
    /// - `POD_UID`: UID of the pod (required for CR owner reference)
    /// - `POD_NAMESPACE`: Namespace of the pod (defaults to "default")
    pub fn from_env() -> Result<Self> {
        let pod_name = std::env::var("POD_NAME")
            .map_err(|_| anyhow::anyhow!("POD_NAME environment variable not set"))?;

        let pod_uid = std::env::var("POD_UID")
            .map_err(|_| anyhow::anyhow!("POD_UID environment variable not set"))?;

        let pod_namespace = std::env::var("POD_NAMESPACE").unwrap_or_else(|_| {
            tracing::warn!("POD_NAMESPACE not set, defaulting to 'default'");
            "default".to_string()
        });

        // Read system server port from config
        let config = crate::config::RuntimeConfig::from_settings().unwrap_or_default();
        let system_port = config.system_port as u16;

        Ok(Self {
            pod_name,
            pod_namespace,
            pod_uid,
            system_port,
        })
    }
}
