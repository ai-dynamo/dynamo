// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage configuration for KVBM.
//!
//! Defines configuration for object storage backends (S3, NIXL) used for
//! the G4 tier (object storage) in the cache hierarchy.

use serde::{Deserialize, Serialize};
use validator::Validate;

/// Top-level object storage configuration.
///
/// When present, enables object storage operations on workers.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ObjectConfig {
    /// Which object client implementation to use.
    pub client: ObjectClientConfig,
}

/// Object client implementation selector.
///
/// Determines whether to use direct S3 client or NIXL agent for object storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ObjectClientConfig {
    /// Direct S3/MinIO client using AWS SDK.
    S3(S3ObjectConfig),
    /// NIXL agent with object storage backend.
    Nixl(NixlObjectConfig),
}

/// S3-compatible object storage configuration.
///
/// Used for both direct S3 access and as a backend for NIXL.
/// Compatible with AWS S3 and S3-compatible services like MinIO.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct S3ObjectConfig {
    /// Custom endpoint URL for S3-compatible services (e.g., MinIO).
    /// If None, uses the default AWS S3 endpoint.
    #[serde(default)]
    pub endpoint_url: Option<String>,

    /// S3 bucket name for storing blocks.
    pub bucket: String,

    /// AWS region.
    #[serde(default = "default_region")]
    pub region: String,

    /// Use path-style URLs instead of virtual-hosted-style.
    /// Required for MinIO and some S3-compatible services.
    #[serde(default)]
    pub force_path_style: bool,

    /// Maximum number of concurrent S3 requests.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,
}

fn default_region() -> String {
    "us-east-1".to_string()
}

fn default_max_concurrent() -> usize {
    16
}

impl Default for S3ObjectConfig {
    fn default() -> Self {
        Self {
            endpoint_url: None,
            bucket: "kvbm-blocks".to_string(),
            region: default_region(),
            force_path_style: false,
            max_concurrent_requests: default_max_concurrent(),
        }
    }
}

impl S3ObjectConfig {
    /// Create configuration for AWS S3.
    pub fn aws(bucket: String, region: String) -> Self {
        Self {
            endpoint_url: None,
            bucket,
            region,
            force_path_style: false,
            max_concurrent_requests: default_max_concurrent(),
        }
    }

    /// Create configuration for MinIO or other S3-compatible services.
    pub fn minio(endpoint_url: String, bucket: String) -> Self {
        Self {
            endpoint_url: Some(endpoint_url),
            bucket,
            region: default_region(),
            force_path_style: true,
            max_concurrent_requests: default_max_concurrent(),
        }
    }
}

/// NIXL object storage backend configuration.
///
/// NIXL can use various object storage backends. Each variant
/// specifies the backend type and its configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "lowercase")]
pub enum NixlObjectConfig {
    /// S3-compatible backend via NIXL.
    S3(S3ObjectConfig),
    // Future backends can be added here:
    // Gcs(GcsObjectConfig),
    // Azure(AzureObjectConfig),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_default() {
        let config = S3ObjectConfig::default();
        assert!(config.endpoint_url.is_none());
        assert_eq!(config.bucket, "kvbm-blocks");
        assert_eq!(config.region, "us-east-1");
        assert!(!config.force_path_style);
        assert_eq!(config.max_concurrent_requests, 16);
    }

    #[test]
    fn test_s3_config_aws() {
        let config = S3ObjectConfig::aws("my-bucket".into(), "us-west-2".into());
        assert!(config.endpoint_url.is_none());
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.region, "us-west-2");
        assert!(!config.force_path_style);
    }

    #[test]
    fn test_s3_config_minio() {
        let config = S3ObjectConfig::minio("http://localhost:9000".into(), "test".into());
        assert_eq!(config.endpoint_url, Some("http://localhost:9000".into()));
        assert_eq!(config.bucket, "test");
        assert!(config.force_path_style);
    }

    #[test]
    fn test_object_config_serde_s3() {
        let json = r#"{
            "client": {
                "type": "s3",
                "bucket": "my-bucket",
                "region": "us-west-2"
            }
        }"#;
        let config: ObjectConfig = serde_json::from_str(json).unwrap();
        match config.client {
            ObjectClientConfig::S3(s3) => {
                assert_eq!(s3.bucket, "my-bucket");
                assert_eq!(s3.region, "us-west-2");
            }
            _ => panic!("Expected S3 config"),
        }
    }

    #[test]
    fn test_object_config_serde_nixl_s3() {
        let json = r#"{
            "client": {
                "type": "nixl",
                "backend": "s3",
                "bucket": "nixl-bucket",
                "endpoint_url": "http://minio:9000",
                "force_path_style": true
            }
        }"#;
        let config: ObjectConfig = serde_json::from_str(json).unwrap();
        match config.client {
            ObjectClientConfig::Nixl(NixlObjectConfig::S3(s3)) => {
                assert_eq!(s3.bucket, "nixl-bucket");
                assert_eq!(s3.endpoint_url, Some("http://minio:9000".into()));
                assert!(s3.force_path_style);
            }
            _ => panic!("Expected Nixl S3 config"),
        }
    }
}

