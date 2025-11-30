// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use object_store::{ObjectStore, aws::AmazonS3Builder, path::Path as ObjectPath};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};
use url::Url;

/// Minimal trait for LoRA sources
/// Users can implement this in Rust for custom sources
#[async_trait]
pub trait LoRASource: Send + Sync {
    /// Download LoRA from source to destination path
    /// Returns the actual path where files were written
    async fn download(&self, lora_uri: &str, dest_path: &Path) -> Result<PathBuf>;

    /// Check if LoRA exists in this source
    async fn exists(&self, lora_uri: &str) -> Result<bool>;
}

/// Local filesystem LoRA source
/// For file:// URIs, just validates the path exists
pub struct LocalLoRASource;

impl Default for LocalLoRASource {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalLoRASource {
    pub fn new() -> Self {
        Self
    }

    /// Parse file:// URI to extract local path
    /// Format: file:///absolute/path/to/lora
    fn parse_file_uri(uri: &str) -> Result<PathBuf> {
        if !uri.starts_with("file://") {
            anyhow::bail!("Invalid file URI scheme: expected file://");
        }

        let path_str = uri.strip_prefix("file://").unwrap();
        Ok(PathBuf::from(path_str))
    }
}

#[async_trait]
impl LoRASource for LocalLoRASource {
    async fn download(&self, file_uri: &str, _dest_path: &Path) -> Result<PathBuf> {
        let source_path = Self::parse_file_uri(file_uri)?;

        if !source_path.exists() {
            anyhow::bail!("LoRA path does not exist: {}", source_path.display());
        }

        if !source_path.is_dir() {
            anyhow::bail!("LoRA path is not a directory: {}", source_path.display());
        }

        // For local files, we don't copy - just return the source path
        // This avoids unnecessary disk I/O
        tracing::info!("Using local LoRA at: {:?}", source_path);

        Ok(source_path)
    }

    async fn exists(&self, file_uri: &str) -> Result<bool> {
        let source_path = Self::parse_file_uri(file_uri)?;
        Ok(source_path.exists() && source_path.is_dir())
    }
}

/// S3-based LoRA source using object_store crate
/// Reads credentials from environment variables
pub struct S3LoRASource {
    access_key_id: String,
    secret_access_key: String,
    region: String,
    endpoint: Option<String>,
}

impl S3LoRASource {
    /// Create S3 source from environment variables:
    /// - AWS_ACCESS_KEY_ID
    /// - AWS_SECRET_ACCESS_KEY
    /// - AWS_REGION (optional, defaults to us-east-1)
    /// - AWS_ENDPOINT (optional, for custom S3-compatible endpoints)
    pub fn from_env() -> Result<Self> {
        let access_key_id =
            std::env::var("AWS_ACCESS_KEY_ID").context("AWS_ACCESS_KEY_ID not set")?;
        let secret_access_key =
            std::env::var("AWS_SECRET_ACCESS_KEY").context("AWS_SECRET_ACCESS_KEY not set")?;
        let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        let endpoint = std::env::var("AWS_ENDPOINT").ok();

        Ok(Self {
            access_key_id,
            secret_access_key,
            region,
            endpoint,
        })
    }

    /// Build an ObjectStore for a specific bucket
    fn build_store(&self, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        let mut builder = AmazonS3Builder::new()
            .with_access_key_id(&self.access_key_id)
            .with_secret_access_key(&self.secret_access_key)
            .with_region(&self.region)
            .with_bucket_name(bucket);

        if let Some(ref endpoint) = self.endpoint {
            builder = builder
                .with_endpoint(endpoint)
                // Use path-style URLs for custom endpoints (e.g., MinIO)
                .with_virtual_hosted_style_request(false);

            // Only allow HTTP when explicitly enabled via environment variable
            // HTTPS is the default for security
            if std::env::var("AWS_ALLOW_HTTP")
                .map(|v| v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            {
                builder = builder.with_allow_http(true);
            }
        }

        let store = builder.build()?;
        Ok(Arc::new(store))
    }

    /// Parse S3 URI to extract bucket and key
    /// Format: s3://bucket-name/path/to/lora
    fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
        let url = Url::parse(uri)?;

        if url.scheme() != "s3" {
            anyhow::bail!("Invalid S3 URI scheme: {}", url.scheme());
        }

        let bucket = url
            .host_str()
            .ok_or_else(|| anyhow::anyhow!("No bucket in S3 URI"))?
            .to_string();

        let key = url.path().trim_start_matches('/').to_string();

        Ok((bucket, key))
    }
}

#[async_trait]
impl LoRASource for S3LoRASource {
    async fn download(&self, s3_uri: &str, dest_path: &Path) -> Result<PathBuf> {
        let (bucket, prefix) = Self::parse_s3_uri(s3_uri)?;

        tracing::info!(
            "Downloading LoRA from S3: bucket={}, prefix={}",
            bucket,
            prefix
        );

        // Build store for this specific bucket
        let bucket_store = self.build_store(&bucket)?;

        // List all objects under the prefix
        let object_prefix = ObjectPath::from(prefix.clone());
        let mut list_stream = bucket_store.list(Some(&object_prefix));

        // Create destination directory
        tokio::fs::create_dir_all(dest_path).await?;

        let mut file_count = 0;
        while let Some(meta_result) = list_stream.next().await {
            let meta = meta_result?;

            // Get relative path (remove prefix)
            let rel_path = meta
                .location
                .as_ref()
                .strip_prefix(prefix.as_str())
                .unwrap_or(meta.location.as_ref())
                .trim_start_matches('/');

            if rel_path.is_empty() {
                continue; // Skip the prefix itself
            }

            let file_path = dest_path.join(rel_path);

            // Create parent directories
            if let Some(parent) = file_path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }

            // Download file
            let bytes = bucket_store.get(&meta.location).await?.bytes().await?;
            tokio::fs::write(&file_path, &bytes).await?;

            file_count += 1;
            tracing::debug!("Downloaded: {} ({} bytes)", rel_path, bytes.len());
        }

        if file_count == 0 {
            anyhow::bail!("No files found at S3 URI: {}", s3_uri);
        }

        tracing::info!("Downloaded {} files from S3 to {:?}", file_count, dest_path);

        Ok(dest_path.to_path_buf())
    }

    async fn exists(&self, s3_uri: &str) -> Result<bool> {
        let (bucket, prefix) = Self::parse_s3_uri(s3_uri)?;

        let bucket_store = self.build_store(&bucket)?;

        let object_prefix = ObjectPath::from(prefix);
        let mut list_stream = bucket_store.list(Some(&object_prefix));

        // Check if at least one object exists, propagating errors
        match list_stream.next().await {
            Some(Ok(_)) => Ok(true),
            Some(Err(e)) => Err(e.into()),
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_file_uri() {
        let uri = "file:///path/to/lora";
        let path = LocalLoRASource::parse_file_uri(uri).unwrap();
        assert_eq!(path, PathBuf::from("/path/to/lora"));
    }

    #[test]
    fn test_parse_file_uri_invalid() {
        let uri = "http://example.com/lora";
        assert!(LocalLoRASource::parse_file_uri(uri).is_err());
    }

    #[test]
    fn test_parse_s3_uri() {
        let uri = "s3://my-bucket/path/to/lora";
        let (bucket, key) = S3LoRASource::parse_s3_uri(uri).unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/lora");
    }

    #[test]
    fn test_parse_s3_uri_invalid() {
        let uri = "file:///path/to/lora";
        assert!(S3LoRASource::parse_s3_uri(uri).is_err());
    }
}
