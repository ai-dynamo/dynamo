// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Discovery configuration for Nova peer discovery.
//!
//! Supports three discovery backends:
//! - **Etcd**: Centralized discovery using etcd key-value store
//! - **P2P**: Decentralized discovery using libp2p DHT
//! - **Filesystem**: File-based discovery for development/testing

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Discovery configuration - only one type can be active at a time.
///
/// # JSON Configuration Examples
///
/// ## Etcd Discovery
/// ```json
/// {
///   "type": "etcd",
///   "cluster_id": "my-cluster",
///   "endpoints": ["http://etcd1:2379", "http://etcd2:2379"],
///   "ttl_secs": 60
/// }
/// ```
///
/// ## P2P Discovery
/// ```json
/// {
///   "type": "p2p",
///   "cluster_id": "my-cluster",
///   "listen_port": 0,
///   "bootstrap_peers": ["192.168.1.10:4001"],
///   "enable_mdns": true
/// }
/// ```
///
/// ## Filesystem Discovery
/// ```json
/// {
///   "type": "filesystem",
///   "path": "/tmp/discovery.json"
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum DiscoveryConfig {
    /// Etcd-based discovery (centralized).
    Etcd(EtcdDiscoveryConfig),
    /// P2P discovery using libp2p DHT (decentralized).
    P2p(P2pDiscoveryConfig),
    /// Filesystem-based discovery (for dev/testing).
    Filesystem(FilesystemDiscoveryConfig),
}

/// Etcd discovery configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct EtcdDiscoveryConfig {
    /// Cluster ID / key prefix for discovery (required).
    pub cluster_id: String,

    /// Etcd endpoints (default: ["http://localhost:2379"]).
    #[serde(default = "default_etcd_endpoints")]
    pub endpoints: Vec<String>,

    /// Lease TTL in seconds (default: 60, range: 10-600).
    #[serde(default = "default_etcd_ttl")]
    #[validate(range(min = 10, max = 600))]
    pub ttl_secs: u64,

    /// Operation timeout in seconds (default: 30).
    #[serde(default = "default_operation_timeout")]
    pub operation_timeout_secs: u64,

    /// Max retries for operations (default: 3).
    #[serde(default = "default_max_retries")]
    #[validate(range(min = 0, max = 10))]
    pub max_retries: u32,
}

/// P2P discovery configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct P2pDiscoveryConfig {
    /// Cluster ID / swarm key (required).
    pub cluster_id: String,

    /// Listen port (default: 0 = OS-assigned).
    #[serde(default)]
    pub listen_port: u16,

    /// Bootstrap peer addresses.
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,

    /// DHT replication factor (default: 3).
    #[serde(default = "default_replication_factor")]
    pub replication_factor: usize,

    /// Enable mDNS for local network discovery (default: false).
    #[serde(default)]
    pub enable_mdns: bool,

    /// Record TTL in seconds (default: 600).
    #[serde(default = "default_record_ttl")]
    pub record_ttl_secs: u64,
}

/// Filesystem discovery configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilesystemDiscoveryConfig {
    /// Path to the discovery JSON file.
    pub path: PathBuf,
}

fn default_etcd_endpoints() -> Vec<String> {
    vec!["http://localhost:2379".to_string()]
}

fn default_etcd_ttl() -> u64 {
    60
}

fn default_operation_timeout() -> u64 {
    30
}

fn default_max_retries() -> u32 {
    3
}

fn default_replication_factor() -> usize {
    3
}

fn default_record_ttl() -> u64 {
    600
}

impl Default for EtcdDiscoveryConfig {
    fn default() -> Self {
        Self {
            cluster_id: String::new(),
            endpoints: default_etcd_endpoints(),
            ttl_secs: default_etcd_ttl(),
            operation_timeout_secs: default_operation_timeout(),
            max_retries: default_max_retries(),
        }
    }
}

impl Default for P2pDiscoveryConfig {
    fn default() -> Self {
        Self {
            cluster_id: String::new(),
            listen_port: 0,
            bootstrap_peers: Vec::new(),
            replication_factor: default_replication_factor(),
            enable_mdns: false,
            record_ttl_secs: default_record_ttl(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_etcd_config() {
        let json = r#"{
            "type": "etcd",
            "cluster_id": "test-cluster",
            "endpoints": ["http://etcd1:2379"],
            "ttl_secs": 120
        }"#;

        let config: DiscoveryConfig = serde_json::from_str(json).unwrap();
        match config {
            DiscoveryConfig::Etcd(etcd) => {
                assert_eq!(etcd.cluster_id, "test-cluster");
                assert_eq!(etcd.endpoints, vec!["http://etcd1:2379"]);
                assert_eq!(etcd.ttl_secs, 120);
                assert_eq!(etcd.operation_timeout_secs, 30); // default
                assert_eq!(etcd.max_retries, 3); // default
            }
            _ => panic!("Expected Etcd config"),
        }
    }

    #[test]
    fn test_deserialize_p2p_config() {
        let json = r#"{
            "type": "p2p",
            "cluster_id": "test-cluster",
            "listen_port": 4001,
            "bootstrap_peers": ["192.168.1.10:4001"],
            "enable_mdns": true
        }"#;

        let config: DiscoveryConfig = serde_json::from_str(json).unwrap();
        match config {
            DiscoveryConfig::P2p(p2p) => {
                assert_eq!(p2p.cluster_id, "test-cluster");
                assert_eq!(p2p.listen_port, 4001);
                assert_eq!(p2p.bootstrap_peers, vec!["192.168.1.10:4001"]);
                assert!(p2p.enable_mdns);
                assert_eq!(p2p.replication_factor, 3); // default
                assert_eq!(p2p.record_ttl_secs, 600); // default
            }
            _ => panic!("Expected P2p config"),
        }
    }

    #[test]
    fn test_deserialize_filesystem_config() {
        let json = r#"{
            "type": "filesystem",
            "path": "/tmp/discovery.json"
        }"#;

        let config: DiscoveryConfig = serde_json::from_str(json).unwrap();
        match config {
            DiscoveryConfig::Filesystem(fs) => {
                assert_eq!(fs.path, PathBuf::from("/tmp/discovery.json"));
            }
            _ => panic!("Expected Filesystem config"),
        }
    }

    #[test]
    fn test_serialize_etcd_config() {
        let config = DiscoveryConfig::Etcd(EtcdDiscoveryConfig {
            cluster_id: "my-cluster".to_string(),
            endpoints: vec!["http://localhost:2379".to_string()],
            ttl_secs: 60,
            operation_timeout_secs: 30,
            max_retries: 3,
        });

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains(r#""type":"etcd""#));
        assert!(json.contains(r#""cluster_id":"my-cluster""#));
    }

    #[test]
    fn test_etcd_default() {
        let config = EtcdDiscoveryConfig::default();
        assert!(config.cluster_id.is_empty());
        assert_eq!(config.endpoints, vec!["http://localhost:2379"]);
        assert_eq!(config.ttl_secs, 60);
        assert_eq!(config.operation_timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_p2p_default() {
        let config = P2pDiscoveryConfig::default();
        assert!(config.cluster_id.is_empty());
        assert_eq!(config.listen_port, 0);
        assert!(config.bootstrap_peers.is_empty());
        assert_eq!(config.replication_factor, 3);
        assert!(!config.enable_mdns);
        assert_eq!(config.record_ttl_secs, 600);
    }
}
