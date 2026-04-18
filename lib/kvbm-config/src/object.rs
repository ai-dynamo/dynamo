// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/// Selects the NIXL OBJ plugin backend. Parameters are forwarded to
/// `nixl_sys::Agent::create_backend("OBJ", params)` at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "backend", rename_all = "lowercase")]
pub enum NixlObjectConfig {
    /// S3-compatible backend via NIXL's OBJ plugin.
    ///
    /// Uses [`NixlS3Config`] rather than [`S3ObjectConfig`] because the NIXL
    /// plugin exposes a different (and more complete) parameter surface than
    /// the AWS SDK path.
    S3(NixlS3Config),
}

/// S3-compatible configuration for NIXL's OBJ backend plugin.
///
/// Parameter names match those accepted by NIXL's OBJ plugin
/// (`src/plugins/obj` in the nixl repository).  All fields are optional;
/// unset fields fall back to the plugin's own defaults (which in turn fall
/// back to standard AWS environment variables: `AWS_ACCESS_KEY_ID`,
/// `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_OVERRIDE`, `AWS_DEFAULT_BUCKET`, …).
#[derive(Debug, Clone, Default, Serialize, Deserialize, Validate)]
pub struct NixlS3Config {
    // ── Authentication ────────────────────────────────────────────────────────
    /// AWS access key ID.  Falls back to `AWS_ACCESS_KEY_ID` env var.
    #[serde(default)]
    pub access_key: Option<String>,

    /// AWS secret access key.  Falls back to `AWS_SECRET_ACCESS_KEY` env var.
    #[serde(default)]
    pub secret_key: Option<String>,

    /// AWS session token for temporary/federated credentials.
    /// Falls back to `AWS_SESSION_TOKEN` env var.
    #[serde(default)]
    pub session_token: Option<String>,

    // ── Bucket & endpoint ─────────────────────────────────────────────────────
    /// S3 bucket name.  Falls back to `AWS_DEFAULT_BUCKET` env var.
    #[serde(default)]
    pub bucket: Option<String>,

    /// Custom S3-compatible endpoint URL (e.g. `http://minio:9000`).
    /// Maps to NIXL param `endpoint_override`.
    /// Falls back to `AWS_ENDPOINT_OVERRIDE` env var.
    #[serde(default)]
    pub endpoint_override: Option<String>,

    /// HTTP scheme: `"http"` or `"https"` (default `"https"`).
    #[serde(default)]
    pub scheme: Option<String>,

    /// AWS region.  Falls back to `AWS_REGION` / `AWS_DEFAULT_REGION` env var.
    /// NIXL default: `us-east-1`.
    #[serde(default)]
    pub region: Option<String>,

    // ── URL style ─────────────────────────────────────────────────────────────
    /// Use virtual-hosted-style URLs (`bucket.host/key`) when `true`.
    /// Set to `false` for path-style (`host/bucket/key`) required by MinIO.
    /// NIXL default: `false` (path-style).
    #[serde(default)]
    pub use_virtual_addressing: Option<bool>,

    // ── Security & checksums ──────────────────────────────────────────────────
    /// Request checksum mode: `"required"` or `"supported"`.
    #[serde(default)]
    pub req_checksum: Option<String>,

    /// Path to a custom CA certificate bundle for TLS verification.
    #[serde(default)]
    pub ca_bundle: Option<String>,

    // ── Performance ───────────────────────────────────────────────────────────
    /// Minimum object size in bytes to route through the S3 CRT client.
    /// Recommended: 10 MiB (`10485760`).  Must be ≥ 5 MiB if set.
    /// Omit (or set to 0) to disable CRT acceleration.
    #[serde(default)]
    pub crt_min_limit_bytes: Option<u64>,

    // ── Transfer behaviour ────────────────────────────────────────────────────
    /// Maximum time in seconds to wait for a single NIXL OBJ transfer to
    /// complete.  Returns an error after this deadline.
    /// Default: 60 s.  Set to 0 to wait indefinitely (not recommended).
    #[serde(default)]
    pub transfer_timeout_secs: Option<u64>,
}

/// Default transfer timeout when `transfer_timeout_secs` is not set.
pub const DEFAULT_TRANSFER_TIMEOUT_SECS: u64 = 60;

impl NixlS3Config {
    /// Create a configuration for S3-compatible services accessed at a custom
    /// endpoint (e.g. `http://storage-host:9000`).
    pub fn with_endpoint(endpoint: impl Into<String>, bucket: impl Into<String>) -> Self {
        Self {
            endpoint_override: Some(endpoint.into()),
            bucket: Some(bucket.into()),
            use_virtual_addressing: Some(false),
            ..Default::default()
        }
    }

    /// Create an AWS S3 configuration.
    pub fn aws(bucket: impl Into<String>, region: impl Into<String>) -> Self {
        Self {
            bucket: Some(bucket.into()),
            region: Some(region.into()),
            use_virtual_addressing: Some(true),
            ..Default::default()
        }
    }

    /// Convert to the flat `HashMap<String, String>` accepted by NIXL's
    /// `create_backend("OBJ", params)`.  Only set fields are emitted; absent
    /// fields let NIXL fall back to its own defaults / env vars.
    pub fn to_nixl_params(&self) -> std::collections::HashMap<String, String> {
        let mut m = std::collections::HashMap::new();
        if let Some(v) = &self.access_key {
            m.insert("access_key".to_string(), v.clone());
        }
        if let Some(v) = &self.secret_key {
            m.insert("secret_key".to_string(), v.clone());
        }
        if let Some(v) = &self.session_token {
            m.insert("session_token".to_string(), v.clone());
        }
        if let Some(v) = &self.bucket {
            m.insert("bucket".to_string(), v.clone());
        }
        if let Some(v) = &self.endpoint_override {
            m.insert("endpoint_override".to_string(), v.clone());
        }
        if let Some(v) = &self.scheme {
            m.insert("scheme".to_string(), v.clone());
        }
        if let Some(v) = &self.region {
            m.insert("region".to_string(), v.clone());
        }
        if let Some(v) = self.use_virtual_addressing {
            m.insert("use_virtual_addressing".to_string(), v.to_string());
        }
        if let Some(v) = &self.req_checksum {
            m.insert("req_checksum".to_string(), v.clone());
        }
        if let Some(v) = &self.ca_bundle {
            m.insert("ca_bundle".to_string(), v.clone());
        }
        if let Some(v) = self.crt_min_limit_bytes {
            if v > 0 {
                m.insert("crtMinLimit".to_string(), v.to_string());
            }
        }
        m
    }

    /// Maximum time to wait for a single NIXL OBJ transfer.
    ///
    /// Returns `None` when `transfer_timeout_secs` is `Some(0)`, meaning no
    /// timeout (wait indefinitely).  Returns `Some(duration)` otherwise.
    /// The default (no config) is [`DEFAULT_TRANSFER_TIMEOUT_SECS`] seconds.
    pub fn transfer_timeout(&self) -> Option<std::time::Duration> {
        match self.transfer_timeout_secs {
            Some(0) => None,
            Some(s) => Some(std::time::Duration::from_secs(s)),
            None => Some(std::time::Duration::from_secs(DEFAULT_TRANSFER_TIMEOUT_SECS)),
        }
    }

    /// Derive the bucket name (for use in NIXL Object descriptors and S3 HEAD
    /// requests).  Falls back to the `AWS_DEFAULT_BUCKET` env var.
    pub fn bucket_name(&self) -> Option<String> {
        self.bucket
            .clone()
            .or_else(|| std::env::var("AWS_DEFAULT_BUCKET").ok())
    }
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
                "endpoint_override": "http://storage-host:9000",
                "use_virtual_addressing": false
            }
        }"#;
        let config: ObjectConfig = serde_json::from_str(json).unwrap();
        match config.client {
            ObjectClientConfig::Nixl(NixlObjectConfig::S3(s3)) => {
                assert_eq!(s3.bucket, Some("nixl-bucket".into()));
                assert_eq!(
                    s3.endpoint_override,
                    Some("http://storage-host:9000".into())
                );
                assert_eq!(s3.use_virtual_addressing, Some(false));
            }
            _ => panic!("Expected Nixl S3 config"),
        }
    }

    #[test]
    fn test_nixl_s3_to_nixl_params_full() {
        let cfg = NixlS3Config {
            access_key: Some("AKID".into()),
            secret_key: Some("SECRET".into()),
            session_token: Some("TOKEN".into()),
            bucket: Some("my-bucket".into()),
            endpoint_override: Some("http://host:9000".into()),
            scheme: Some("http".into()),
            region: Some("us-west-2".into()),
            use_virtual_addressing: Some(false),
            req_checksum: Some("required".into()),
            ca_bundle: Some("/etc/ssl/ca.pem".into()),
            crt_min_limit_bytes: Some(10_485_760),
            transfer_timeout_secs: None,
        };
        let params = cfg.to_nixl_params();
        assert_eq!(params["access_key"], "AKID");
        assert_eq!(params["secret_key"], "SECRET");
        assert_eq!(params["session_token"], "TOKEN");
        assert_eq!(params["bucket"], "my-bucket");
        assert_eq!(params["endpoint_override"], "http://host:9000");
        assert_eq!(params["scheme"], "http");
        assert_eq!(params["region"], "us-west-2");
        assert_eq!(params["use_virtual_addressing"], "false");
        assert_eq!(params["req_checksum"], "required");
        assert_eq!(params["ca_bundle"], "/etc/ssl/ca.pem");
        assert_eq!(params["crtMinLimit"], "10485760");
    }

    #[test]
    fn test_nixl_s3_to_nixl_params_sparse() {
        // Only bucket set — all other keys should be absent (NIXL uses env var fallbacks)
        let cfg = NixlS3Config {
            bucket: Some("sparse-bucket".into()),
            ..Default::default()
        };
        let params = cfg.to_nixl_params();
        assert_eq!(params.len(), 1);
        assert_eq!(params["bucket"], "sparse-bucket");
    }

    #[test]
    fn test_nixl_s3_transfer_timeout_default() {
        let cfg = NixlS3Config::default();
        assert_eq!(
            cfg.transfer_timeout(),
            Some(std::time::Duration::from_secs(DEFAULT_TRANSFER_TIMEOUT_SECS))
        );
    }

    #[test]
    fn test_nixl_s3_transfer_timeout_custom() {
        let cfg = NixlS3Config {
            transfer_timeout_secs: Some(120),
            ..Default::default()
        };
        assert_eq!(
            cfg.transfer_timeout(),
            Some(std::time::Duration::from_secs(120))
        );
    }

    #[test]
    fn test_nixl_s3_transfer_timeout_zero_means_no_timeout() {
        let cfg = NixlS3Config {
            transfer_timeout_secs: Some(0),
            ..Default::default()
        };
        assert_eq!(cfg.transfer_timeout(), None);
    }

    #[test]
    fn test_nixl_s3_transfer_timeout_not_emitted_in_nixl_params() {
        // transfer_timeout_secs is a kvbm-engine-side control and must NOT
        // appear in the NIXL plugin params map.
        let cfg = NixlS3Config {
            transfer_timeout_secs: Some(30),
            ..Default::default()
        };
        let params = cfg.to_nixl_params();
        assert!(!params.contains_key("transfer_timeout_secs"));
    }

    #[test]
    fn test_nixl_s3_crt_zero_omitted() {
        let cfg = NixlS3Config {
            crt_min_limit_bytes: Some(0),
            ..Default::default()
        };
        let params = cfg.to_nixl_params();
        assert!(!params.contains_key("crtMinLimit"));
    }
}
