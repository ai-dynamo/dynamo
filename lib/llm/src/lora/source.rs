// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use object_store::{
    BackoffConfig, ClientOptions, ObjectMeta, ObjectStore, RetryConfig, aws::AmazonS3Builder,
    path::Path as ObjectPath,
};
#[cfg(unix)]
use std::os::unix::io::AsRawFd;
use std::{
    io::ErrorKind,
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

#[derive(Clone, Debug)]
struct S3DownloadObject {
    location: ObjectPath,
    relative_object_path: ObjectPath,
    relative_fs_path: PathBuf,
    size: u64,
    e_tag: Option<String>,
    last_modified: String,
}

impl S3DownloadObject {
    fn marker_contents(&self) -> String {
        format!(
            "location={}\nsize={}\netag={}\nlast_modified={}\n",
            self.location,
            self.size,
            self.e_tag.as_deref().unwrap_or(""),
            self.last_modified
        )
    }
}

#[cfg(unix)]
struct DownloadLock {
    path: PathBuf,
    file: std::fs::File,
}

#[cfg(unix)]
impl Drop for DownloadLock {
    fn drop(&mut self) {
        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(not(unix))]
struct DownloadLock {
    path: PathBuf,
}

#[cfg(not(unix))]
impl Drop for DownloadLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

impl S3LoRASource {
    const DEFAULT_DOWNLOAD_CONCURRENCY: usize = 8;
    const MAX_DOWNLOAD_CONCURRENCY: usize = 64;
    const RESUME_MANIFEST_DIR: &'static str = ".dynamo-download-manifest";

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
        let retry_config = RetryConfig {
            max_retries: 5,
            retry_timeout: Duration::from_secs(600),
            backoff: BackoffConfig {
                init_backoff: Duration::from_secs(1),
                max_backoff: Duration::from_secs(30),
                base: 2.0,
            },
        };

        let mut builder = AmazonS3Builder::new()
            .with_access_key_id(&self.access_key_id)
            .with_secret_access_key(&self.secret_access_key)
            .with_region(&self.region)
            .with_bucket_name(bucket)
            .with_client_options(client_opts)
            .with_retry(retry_config);

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

    fn download_concurrency() -> usize {
        std::env::var("LORA_DOWNLOAD_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .map(|v| v.min(Self::MAX_DOWNLOAD_CONCURRENCY))
            .unwrap_or(Self::DEFAULT_DOWNLOAD_CONCURRENCY)
    }

    fn download_object_from_meta(
        meta: ObjectMeta,
        prefix: &ObjectPath,
    ) -> Result<Option<S3DownloadObject>> {
        let relative_object_path: ObjectPath = meta
            .location
            .prefix_match(prefix)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Listed object {} did not match requested prefix {}",
                    meta.location,
                    prefix
                )
            })?
            .collect();

        if relative_object_path.as_ref().is_empty() {
            return Ok(None);
        }

        let mut relative_fs_path = PathBuf::new();
        for part in relative_object_path.parts() {
            relative_fs_path.push(part.as_ref());
        }

        Ok(Some(S3DownloadObject {
            location: meta.location,
            relative_object_path,
            relative_fs_path,
            size: meta.size,
            e_tag: meta.e_tag,
            last_modified: meta.last_modified.to_rfc3339(),
        }))
    }

    async fn list_download_objects(
        store: &Arc<dyn ObjectStore>,
        prefix: &ObjectPath,
    ) -> Result<Vec<S3DownloadObject>> {
        let metas = store
            .list(Some(prefix))
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to list S3 prefix")?;

        let mut objects = Vec::with_capacity(metas.len());
        for meta in metas {
            if let Some(object) = Self::download_object_from_meta(meta, prefix)? {
                objects.push(object);
            }
        }
        objects.sort_by(|a, b| a.location.cmp(&b.location));

        Ok(objects)
    }

    fn object_set_marker_path(staging_path: &Path) -> PathBuf {
        staging_path
            .join(Self::RESUME_MANIFEST_DIR)
            .join("objects.metadata")
    }

    fn object_set_marker_contents(objects: &[S3DownloadObject]) -> String {
        objects
            .iter()
            .map(S3DownloadObject::marker_contents)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn marker_path(staging_path: &Path, object: &S3DownloadObject) -> PathBuf {
        let mut marker_path = staging_path
            .join(Self::RESUME_MANIFEST_DIR)
            .join(&object.relative_fs_path);

        if let Some(file_name) = marker_path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
        {
            marker_path.set_file_name(format!("{}.metadata", file_name));
        }

        marker_path
    }

    async fn staging_file_is_complete(
        staging_path: &Path,
        object: &S3DownloadObject,
    ) -> Result<bool> {
        let file_path = staging_path.join(&object.relative_fs_path);
        let local_meta = match tokio::fs::metadata(&file_path).await {
            Ok(meta) => meta,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(false),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("Failed to stat staged file {:?}", file_path));
            }
        };

        if !local_meta.is_file() || local_meta.len() != object.size {
            return Ok(false);
        }

        let marker_path = Self::marker_path(staging_path, object);
        let marker_contents = match tokio::fs::read_to_string(&marker_path).await {
            Ok(contents) => contents,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(false),
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("Failed to read resume marker {:?}", marker_path));
            }
        };

        Ok(marker_contents == object.marker_contents())
    }

    async fn write_resume_marker(staging_path: &Path, object: &S3DownloadObject) -> Result<()> {
        let marker_path = Self::marker_path(staging_path, object);
        if let Some(parent) = marker_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create marker directory {:?}", parent))?;
        }

        tokio::fs::write(&marker_path, object.marker_contents())
            .await
            .with_context(|| format!("Failed to write resume marker {:?}", marker_path))
    }

    async fn download_one_object(
        store: Arc<dyn ObjectStore>,
        staging_path: PathBuf,
        object: S3DownloadObject,
    ) -> Result<(String, u64, bool)> {
        let rel_path = object.relative_object_path.to_string();
        let file_path = staging_path.join(&object.relative_fs_path);

        if Self::staging_file_is_complete(&staging_path, &object).await? {
            tracing::debug!("Resume: skipping completed shard {}", rel_path);
            return Ok((rel_path, object.size, true));
        }

        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create parent dir for {:?}", file_path))?;
        }

        let bytes_written = Self::stream_to_file(&store, &object.location, &file_path).await?;
        if bytes_written != object.size {
            anyhow::bail!(
                "Downloaded {} bytes for {}, expected {} bytes",
                bytes_written,
                object.location,
                object.size
            );
        }

        Self::write_resume_marker(&staging_path, &object).await?;
        Ok((rel_path, bytes_written, false))
    }

    async fn directory_has_objects(directory: &Path, objects: &[S3DownloadObject]) -> Result<bool> {
        for object in objects {
            let file_path = directory.join(&object.relative_fs_path);
            let local_meta = match tokio::fs::metadata(&file_path).await {
                Ok(meta) => meta,
                Err(e) if e.kind() == ErrorKind::NotFound => return Ok(false),
                Err(e) => {
                    return Err(e).with_context(|| {
                        format!("Failed to stat downloaded file {:?}", file_path)
                    });
                }
            };

            if !local_meta.is_file() || local_meta.len() != object.size {
                return Ok(false);
            }
        }

        Ok(true)
    }

    async fn prepare_staging_directory(
        staging_path: &Path,
        objects: &[S3DownloadObject],
    ) -> Result<()> {
        let expected_marker = Self::object_set_marker_contents(objects);

        match tokio::fs::read_to_string(Self::object_set_marker_path(staging_path)).await {
            Ok(existing_marker) if existing_marker == expected_marker => {}
            Ok(_) => {
                tracing::info!(
                    "S3 LoRA staging metadata changed, resetting {:?}",
                    staging_path
                );
                Self::remove_path_if_exists(staging_path).await?;
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {
                Self::remove_path_if_exists(staging_path).await?;
            }
            Err(e) => {
                return Err(e).with_context(|| {
                    format!(
                        "Failed to read object set marker {:?}",
                        Self::object_set_marker_path(staging_path)
                    )
                });
            }
        }

        tokio::fs::create_dir_all(staging_path)
            .await
            .context("Failed to create staging directory")?;

        let marker_path = Self::object_set_marker_path(staging_path);
        if let Some(parent) = marker_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("Failed to create marker directory {:?}", parent))?;
        }

        tokio::fs::write(&marker_path, expected_marker)
            .await
            .with_context(|| format!("Failed to write object set marker {:?}", marker_path))
    }

    async fn remove_path_if_exists(path: &Path) -> Result<()> {
        let meta = match tokio::fs::symlink_metadata(path).await {
            Ok(meta) => meta,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).with_context(|| format!("Failed to stat {:?}", path)),
        };

        if meta.is_dir() {
            tokio::fs::remove_dir_all(path)
                .await
                .with_context(|| format!("Failed to remove directory {:?}", path))?;
        } else {
            tokio::fs::remove_file(path)
                .await
                .with_context(|| format!("Failed to remove file {:?}", path))?;
        }

        Ok(())
    }

    #[cfg(unix)]
    async fn acquire_download_lock(lock_path: &Path) -> Result<DownloadLock> {
        let lock_path = lock_path.to_path_buf();
        tokio::task::spawn_blocking(move || -> Result<DownloadLock> {
            if let Some(parent) = lock_path.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create lock directory {:?}", parent))?;
            }

            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .open(&lock_path)
                .with_context(|| format!("Failed to open S3 download lock {:?}", lock_path))?;

            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if rc != 0 {
                return Err(std::io::Error::last_os_error())
                    .with_context(|| format!("Failed to lock {:?}", lock_path));
            }

            Ok(DownloadLock {
                path: lock_path,
                file,
            })
        })
        .await
        .context("S3 download lock task failed")?
    }

    #[cfg(not(unix))]
    async fn acquire_download_lock(lock_path: &Path) -> Result<DownloadLock> {
        loop {
            match tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(lock_path)
                .await
            {
                Ok(_) => {
                    return Ok(DownloadLock {
                        path: lock_path.to_path_buf(),
                    });
                }
                Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Err(e) => {
                    return Err(e)
                        .with_context(|| format!("Failed to create lock {:?}", lock_path));
                }
            }
        }
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
        let download_objects = Self::list_download_objects(&bucket_store, &object_prefix).await?;

        if download_objects.is_empty() {
            anyhow::bail!("No files found at S3 URI: {}", s3_uri);
        }

        let parent = dest_path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Destination path has no parent directory"))?;
        let dest_name = dest_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("Destination path has no file name"))?;
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create cache parent directory")?;

        let staging_path = parent.join(format!("{}.partial", dest_name));
        let lock_path = parent.join(format!("{}.download.lock", dest_name));
        let _download_lock = Self::acquire_download_lock(&lock_path).await?;

        if Self::directory_has_objects(dest_path, &download_objects).await? {
            tracing::debug!("LoRA already downloaded to {:?}", dest_path);
            return Ok(dest_path.to_path_buf());
        }

        Self::prepare_staging_directory(&staging_path, &download_objects).await?;

        let concurrency = Self::download_concurrency();
        let download_futs = download_objects.iter().cloned().map(|object| {
            Self::download_one_object(bucket_store.clone(), staging_path.clone(), object)
        });

        let mut stream = futures::stream::iter(download_futs).buffer_unordered(concurrency);
        let mut downloaded_count = 0usize;
        let mut skipped_count = 0usize;

        while let Some(result) = stream.next().await {
            let (rel_path, bytes_written, skipped) = result?;
            if skipped {
                skipped_count += 1;
                tracing::debug!("Reused: {} ({} bytes)", rel_path, bytes_written);
            } else {
                downloaded_count += 1;
                tracing::debug!("Downloaded: {} ({} bytes)", rel_path, bytes_written);
            }
        }

        if !Self::directory_has_objects(&staging_path, &download_objects).await? {
            anyhow::bail!(
                "Downloaded files at {:?} did not match expected S3 objects for {}",
                staging_path,
                s3_uri
            );
        }

        if dest_path.exists() {
            Self::remove_path_if_exists(dest_path).await?;
        }

        Self::remove_path_if_exists(&staging_path.join(Self::RESUME_MANIFEST_DIR)).await?;
        tokio::fs::rename(&staging_path, dest_path)
            .await
            .context("Failed to atomically move staging directory to destination")?;

        tracing::info!(
            "Downloaded {} files from S3 to {:?} ({} reused from staging)",
            downloaded_count,
            dest_path,
            skipped_count
        );

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
