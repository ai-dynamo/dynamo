// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::cache::LoRACache;
use crate::hub::{self, HfRepoSpec};
use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use hf_hub::Cache;
use object_store::{ClientOptions, ObjectStore, aws::AmazonS3Builder, path::Path as ObjectPath};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};
use tokio::io::AsyncWriteExt;
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

/// Hugging Face Hub LoRA source.
///
/// Downloads repositories into the standard Hugging Face cache and returns the
/// immutable snapshot directory, avoiding a second copy under `DYN_LORA_PATH`.
pub struct HuggingFaceLoRASource {
    cache: Cache,
}

impl Default for HuggingFaceLoRASource {
    fn default() -> Self {
        Self::from_env()
    }
}

impl HuggingFaceLoRASource {
    pub fn from_env() -> Self {
        Self {
            cache: hub::huggingface_cache(),
        }
    }

    #[cfg(test)]
    fn with_cache(cache: Cache) -> Self {
        Self { cache }
    }

    fn cached_snapshot(&self, spec: &HfRepoSpec) -> Result<Option<PathBuf>> {
        let Some(snapshot) = hub::cached_hf_snapshot(&self.cache, spec, "adapter_config.json")
        else {
            return Ok(None);
        };
        Ok(LoRACache::validate_path(&snapshot)?.then_some(snapshot))
    }
}

#[async_trait]
impl LoRASource for HuggingFaceLoRASource {
    async fn download(&self, hf_uri: &str, _dest_path: &Path) -> Result<PathBuf> {
        let spec = HfRepoSpec::from_uri(hf_uri)?;
        if let Some(snapshot) = self.cached_snapshot(&spec)? {
            tracing::debug!(uri = hf_uri, path = %snapshot.display(), "using cached Hugging Face LoRA");
            return Ok(snapshot);
        }

        tracing::info!(uri = hf_uri, "downloading LoRA from Hugging Face Hub");
        let snapshot = hub::download_hf_snapshot(&self.cache, &spec).await?;
        if !LoRACache::validate_path(&snapshot)? {
            anyhow::bail!(
                "Hugging Face repository {hf_uri} is not a valid LoRA: expected adapter_config.json and adapter weights"
            );
        }
        Ok(snapshot)
    }

    async fn exists(&self, hf_uri: &str) -> Result<bool> {
        HfRepoSpec::from_uri(hf_uri)?;
        Ok(true)
    }
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
    const MAX_RETRIES: u32 = 3;
    const INITIAL_BACKOFF_MS: u64 = 1000;
    const MAX_BACKOFF_MS: u64 = 30000;

    async fn stream_to_file(
        store: &Arc<dyn ObjectStore>,
        location: &ObjectPath,
        dest: &std::path::Path,
    ) -> Result<u64> {
        let get_result = store
            .get(location)
            .await
            .with_context(|| format!("Failed to GET {}", location))?;

        let mut stream = get_result.into_stream();
        let mut file = tokio::fs::File::create(dest)
            .await
            .with_context(|| format!("Failed to create file {:?}", dest))?;

        let mut total_bytes: u64 = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.with_context(|| format!("Error reading stream for {}", location))?;
            file.write_all(&chunk)
                .await
                .with_context(|| format!("Failed to write chunk to {:?}", dest))?;
            total_bytes += chunk.len() as u64;
        }
        file.flush().await?;

        Ok(total_bytes)
    }

    async fn download_file_with_retry(
        store: &Arc<dyn ObjectStore>,
        location: &ObjectPath,
        dest: &std::path::Path,
    ) -> Result<u64> {
        for attempt in 1..=Self::MAX_RETRIES {
            match Self::stream_to_file(store, location, dest).await {
                Ok(bytes_written) => return Ok(bytes_written),
                Err(error) => {
                    if attempt >= Self::MAX_RETRIES {
                        return Err(error);
                    }

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
            }
        }

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

    fn build_store(&self, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        let timeout_secs: u64 = std::env::var("LORA_DOWNLOAD_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3600);

        let client_opts = ClientOptions::new().with_timeout(Duration::from_secs(timeout_secs));

        let mut builder = AmazonS3Builder::new()
            .with_access_key_id(&self.access_key_id)
            .with_secret_access_key(&self.secret_access_key)
            .with_region(&self.region)
            .with_bucket_name(bucket)
            .with_client_options(client_opts);

        if let Some(ref endpoint) = self.endpoint {
            builder = builder
                .with_endpoint(endpoint)
                .with_virtual_hosted_style_request(false);

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

        let bucket_store = self.build_store(&bucket)?;
        let object_prefix = ObjectPath::from(prefix.clone());
        let mut list_stream = bucket_store.list(Some(&object_prefix));

        let parent = dest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Destination path has no parent directory"))?;
        let dest_name = dest_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("Destination path has no file name"))?;

        let temp_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir_name = format!("{}.tmp.{}", dest_name, temp_suffix);
        let temp_path = parent.join(&temp_dir_name);

        tokio::fs::create_dir_all(&temp_path)
            .await
            .context("Failed to create temporary directory")?;

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

            let rel_path = meta
                .location
                .as_ref()
                .strip_prefix(prefix.as_str())
                .unwrap_or(meta.location.as_ref())
                .trim_start_matches('/');

            if rel_path.is_empty() {
                continue;
            }

            let file_path = temp_path.join(rel_path);

            #[allow(clippy::collapsible_if)]
            if let Some(parent) = file_path.parent() {
                if let Err(e) = tokio::fs::create_dir_all(parent).await {
                    return Err(cleanup_on_error(e.into()).await);
                }
            }

            let bytes_written =
                match Self::download_file_with_retry(&bucket_store, &meta.location, &file_path)
                    .await
                {
                    Ok(n) => n,
                    Err(e) => return Err(cleanup_on_error(e).await),
                };

            file_count += 1;
            tracing::debug!("Downloaded: {} ({} bytes)", rel_path, bytes_written);
        }

        if file_count == 0 {
            return Err(
                cleanup_on_error(anyhow::anyhow!("No files found at S3 URI: {}", s3_uri)).await,
            );
        }

        if dest_path.exists() {
            tokio::fs::remove_dir_all(dest_path)
                .await
                .context("Failed to remove existing destination directory")?;
        }
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
    use hf_hub::Cache;
    use mockito::Matcher;
    use std::fs;
    use tempfile::TempDir;

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

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_reuses_valid_native_snapshot_in_offline_mode() {
        let temp = TempDir::new().unwrap();
        let repo_dir = temp.path().join("models--org--adapter");
        let snapshot = repo_dir.join("snapshots/abc123");
        fs::create_dir_all(repo_dir.join("refs")).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(repo_dir.join("refs/main"), "abc123").unwrap();
        fs::write(snapshot.join("adapter_config.json"), "{}").unwrap();
        fs::write(snapshot.join("adapter_model.safetensors"), "weights").unwrap();

        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        let result = temp_env::async_with_vars(
            [("HF_HUB_OFFLINE", Some("1"))],
            source.download("hf://org/adapter", Path::new("unused")),
        )
        .await
        .unwrap();

        assert_eq!(result, snapshot);
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_reuses_commit_snapshot_without_ref_file() {
        let temp = TempDir::new().unwrap();
        let revision = "0123456789abcdef0123456789abcdef01234567";
        let snapshot = temp
            .path()
            .join("models--org--adapter/snapshots")
            .join(revision);
        fs::create_dir_all(&snapshot).unwrap();
        fs::write(snapshot.join("adapter_config.json"), "{}").unwrap();
        fs::write(snapshot.join("adapter_model.safetensors"), "weights").unwrap();

        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        let result = temp_env::async_with_vars(
            [("HF_HUB_OFFLINE", Some("1"))],
            source.download(
                format!("hf://org/adapter@{revision}").as_str(),
                Path::new("unused"),
            ),
        )
        .await
        .unwrap();

        assert_eq!(result, snapshot);
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_downloads_complete_revision_pinned_snapshot() {
        let mut server = mockito::Server::new_async().await;
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let token = "test-token";
        let info = server
            .mock("GET", "/api/models/org/adapter/revision/main")
            .match_header("authorization", format!("Bearer {token}").as_str())
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"siblings":[{{"rfilename":"adapter_config.json"}},{{"rfilename":"adapter_model.safetensors"}}],"sha":"{commit}"}}"#
            ))
            .create_async()
            .await;
        let config = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_config.json").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(200)
            .with_header("x-repo-commit", commit)
            .with_header("etag", "config-etag")
            .with_header("content-range", "bytes 0-1/2")
            .with_body("{}")
            .expect(2)
            .create_async()
            .await;
        let weights = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_model.safetensors").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(200)
            .with_header("x-repo-commit", commit)
            .with_header("etag", "weights-etag")
            .with_header("content-range", "bytes 0-6/7")
            .with_body("weights")
            .expect(2)
            .create_async()
            .await;

        let temp = TempDir::new().unwrap();
        let token_path = temp.path().join("token");
        fs::write(&token_path, token).unwrap();
        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        let snapshot = temp_env::async_with_vars(
            [
                ("HF_ENDPOINT", Some(server.url().as_str())),
                ("HF_TOKEN", None),
                ("HF_TOKEN_PATH", token_path.to_str()),
                ("HF_HUB_OFFLINE", None),
            ],
            source.download("hf://org/adapter", Path::new("unused")),
        )
        .await
        .unwrap();

        assert_eq!(
            snapshot,
            temp.path()
                .join("models--org--adapter/snapshots")
                .join(commit)
        );
        assert_eq!(
            fs::read(snapshot.join("adapter_config.json")).unwrap(),
            b"{}"
        );
        assert_eq!(
            fs::read(snapshot.join("adapter_model.safetensors")).unwrap(),
            b"weights"
        );
        assert_eq!(
            fs::read_to_string(temp.path().join("models--org--adapter/refs/main")).unwrap(),
            commit
        );

        info.assert_async().await;
        config.assert_async().await;
        weights.assert_async().await;
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_does_not_publish_requested_ref_after_partial_download() {
        let mut server = mockito::Server::new_async().await;
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let info = server
            .mock("GET", "/api/models/org/adapter/revision/main")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"siblings":[{{"rfilename":"adapter_config.json"}},{{"rfilename":"adapter_model.safetensors"}}],"sha":"{commit}"}}"#
            ))
            .create_async()
            .await;
        let config = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_config.json").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(200)
            .with_header("x-repo-commit", commit)
            .with_header("etag", "config-etag")
            .with_header("content-range", "bytes 0-1/2")
            .with_body("{}")
            .expect(2)
            .create_async()
            .await;
        let weights = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/adapter_model.safetensors").as_str(),
            )
            .match_header("range", Matcher::Regex("bytes=0-.*".to_string()))
            .with_status(500)
            .expect(1)
            .create_async()
            .await;

        let temp = TempDir::new().unwrap();
        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        let result = temp_env::async_with_vars(
            [
                ("HF_ENDPOINT", Some(server.url().as_str())),
                ("HF_TOKEN", None),
                ("HF_TOKEN_PATH", None),
                ("HF_HUB_OFFLINE", None),
            ],
            source.download("hf://org/adapter", Path::new("unused")),
        )
        .await;

        assert!(result.is_err());
        assert!(!temp.path().join("models--org--adapter/refs/main").exists());
        info.assert_async().await;
        config.assert_async().await;
        weights.assert_async().await;
    }
}
