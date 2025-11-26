// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use object_store::{ObjectStore, aws::AmazonS3Builder, path::Path as ObjectPath};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
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

/// Retry configuration for S3 operations
impl S3LoRASource {
    /// Maximum number of retry attempts for S3 operations
    const MAX_RETRIES: u32 = 3;
    /// Initial backoff duration in milliseconds
    const INITIAL_BACKOFF_MS: u64 = 1000;
    /// Maximum backoff duration in milliseconds
    const MAX_BACKOFF_MS: u64 = 30000;

    /// Download a single file with retry logic and exponential backoff
    async fn download_file_with_retry(
        store: &Arc<dyn ObjectStore>,
        location: &ObjectPath,
    ) -> Result<Bytes> {
        for attempt in 1..=Self::MAX_RETRIES {
            let result = store.get(location).await;
            let error = match result {
                Ok(get_result) => match get_result.bytes().await {
                    Ok(bytes) => return Ok(bytes),
                    Err(e) => anyhow::anyhow!("Failed to read bytes: {}", e),
                },
                Err(e) => anyhow::anyhow!("Failed to get object: {}", e),
            };

            if attempt >= Self::MAX_RETRIES {
                return Err(error);
            }

            // Calculate backoff with exponential increase, capped at MAX_BACKOFF_MS
            let backoff_ms = std::cmp::min(
                Self::INITIAL_BACKOFF_MS * 2u64.pow(attempt - 1),
                Self::MAX_BACKOFF_MS,
            );
            tracing::warn!(
                "S3 download failed (attempt {}/{}), retrying in {}ms: {}",
                attempt,
                Self::MAX_RETRIES,
                backoff_ms,
                error
            );
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
        }

        // This should be unreachable, but provide a fallback
        Err(anyhow::anyhow!(
            "S3 download failed after {} retries",
            Self::MAX_RETRIES
        ))
    }
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

        // Create a temporary directory in the same parent as dest_path for atomic download
        // This prevents data loss if dest_path already exists
        let parent = dest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Destination path has no parent directory"))?;
        let dest_name = dest_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("Destination path has no file name"))?;

        // Generate unique temp directory name
        let temp_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir_name = format!("{}.tmp.{}", dest_name, temp_suffix);
        let temp_path = parent.join(&temp_dir_name);

        // Create temporary directory
        tokio::fs::create_dir_all(&temp_path)
            .await
            .context("Failed to create temporary directory")?;

        // Cleanup closure that only removes the temp directory on error
        let cleanup_on_error = async |err: anyhow::Error| -> anyhow::Error {
            tracing::warn!(
                "S3 download failed, cleaning up temporary directory at {:?}",
                temp_path
            );
            if let Err(cleanup_err) = tokio::fs::remove_dir_all(&temp_path).await {
                tracing::warn!("Failed to cleanup temporary directory: {}", cleanup_err);
            }
            err
        };

        let mut file_count = 0;
        while let Some(meta_result) = list_stream.next().await {
            let meta = match meta_result {
                Ok(m) => m,
                Err(e) => return Err(cleanup_on_error(e.into()).await),
            };

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

            let file_path = temp_path.join(rel_path);

            // Create parent directories
            #[allow(clippy::collapsible_if)]
            if let Some(parent) = file_path.parent() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return Err(cleanup_on_error(e.into()).await);
                }
            }

            // Download file with retry logic
            let bytes = match Self::download_file_with_retry(&bucket_store, &meta.location).await {
                Ok(b) => b,
                Err(e) => return Err(cleanup_on_error(e).await),
            };

            if let Err(e) = tokio::fs::write(&file_path, &bytes).await {
                return Err(cleanup_on_error(e.into()).await);
            }

            file_count += 1;
            tracing::debug!("Downloaded: {} ({} bytes)", rel_path, bytes.len());
        }

        if file_count == 0 {
            return Err(
                cleanup_on_error(anyhow::anyhow!("No files found at S3 URI: {}", s3_uri)).await,
            );
        }

        // Atomically rename temp directory to final destination
        // Remove dest_path if it exists (only after successful download to avoid data loss)
        if dest_path.exists() {
            tokio::fs::remove_dir_all(dest_path)
                .await
                .context("Failed to remove existing destination directory")?;
        }
        // Rename is atomic on most filesystems
        tokio::fs::rename(&temp_path, dest_path)
            .await
            .context("Failed to atomically move temporary directory to destination")?;

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
