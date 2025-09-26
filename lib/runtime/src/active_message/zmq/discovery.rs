// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use uuid::Uuid;

use crate::active_message::client::{Endpoint, PeerInfo};

pub fn detect_local_host() -> Result<String> {
    local_ip_address::local_ip()
        .map(|ip| ip.to_string())
        .map_err(|e| anyhow::anyhow!("Failed to detect local IP: {}", e))
}

pub fn create_tcp_endpoint(host: &str, port: u16) -> Endpoint {
    format!("tcp://{}:{}", host, port)
}

pub fn create_ipc_endpoint(instance_id: Uuid) -> Endpoint {
    format!("ipc:///tmp/dynamo-am-{}.sock", instance_id)
}

pub fn detect_transport_preference(
    my_endpoint: &str,
    peer_endpoint: &str,
    my_instance: Uuid,
) -> Endpoint {
    if is_same_host(my_endpoint, peer_endpoint) {
        create_ipc_endpoint(my_instance)
    } else {
        peer_endpoint.to_string()
    }
}

fn is_same_host(endpoint1: &str, endpoint2: &str) -> bool {
    match (extract_host(endpoint1), extract_host(endpoint2)) {
        (Some(host1), Some(host2)) => host1 == host2,
        _ => false,
    }
}

fn extract_host(endpoint: &str) -> Option<String> {
    if endpoint.starts_with("tcp://") {
        endpoint
            .strip_prefix("tcp://")
            .and_then(|s| s.split(':').next())
            .map(|s| s.to_string())
    } else if endpoint.starts_with("ipc://") {
        Some("localhost".to_string())
    } else {
        None
    }
}
