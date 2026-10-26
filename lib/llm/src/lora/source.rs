// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::cache::LoRACache;
use crate::hub::{self, HfRepoSpec};
use anyhow::{Context, Result};
use async_trait::async_trait;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use aws_types::service_config::ServiceConfigKey;
use futures::StreamExt;
use hf_hub::Cache;
use object_store::{
    CredentialProvider, Error as ObjectStoreError, ObjectStore,
    aws::{AmazonS3Builder, AwsCredential, AwsCredentialProvider},
    client::ClientConfigKey,
    path::Path as ObjectPath,
};
use parking_lot::RwLock;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::{
    io::AsyncWriteExt,
    sync::{Mutex, OnceCell},
};
use url::Url;

/// Minimal trait for LoRA sources
/// Users can implement this in Rust for custom sources
#[async_trait]
pub trait LoRASource: Send + Sync {
    /// Returns whether this source handles the URI.
    fn supports(&self, _lora_uri: &str) -> bool {
        true
    }

    /// Download LoRA from source to destination path
    /// Returns the actual path where files were written
    async fn download(&self, lora_uri: &str, dest_path: &Path) -> Result<PathBuf>;

    /// Check if LoRA exists in this source
    async fn exists(&self, lora_uri: &str) -> Result<bool>;

    /// Return a complete source-owned cache path without network access.
    fn cached_path(&self, _lora_uri: &str) -> Result<Option<PathBuf>> {
        Ok(None)
    }
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
    fn supports(&self, lora_uri: &str) -> bool {
        lora_uri.starts_with("hf://")
    }

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
        hub::finalize_hf_snapshot(&self.cache, &spec, &snapshot)?;
        Ok(snapshot)
    }

    async fn exists(&self, hf_uri: &str) -> Result<bool> {
        HfRepoSpec::from_uri(hf_uri)?;
        Ok(true)
    }

    fn cached_path(&self, hf_uri: &str) -> Result<Option<PathBuf>> {
        if !hf_uri.starts_with("hf://") {
            return Ok(None);
        }
        self.cached_snapshot(&HfRepoSpec::from_uri(hf_uri)?)
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
    fn supports(&self, lora_uri: &str) -> bool {
        lora_uri.starts_with("file://")
    }

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

/// Refresh credentials before their expiration to leave time for in-flight requests.
const CREDENTIAL_REFRESH_BUFFER: Duration = Duration::from_secs(60);
/// Re-resolve credentials periodically when the provider does not supply an expiration.
const CREDENTIAL_MAX_CACHE_AGE: Duration = Duration::from_secs(15 * 60);

/// A cached object-store credential and the expiration of its AWS source credential.
#[derive(Debug)]
struct CachedAwsCredential {
    cached_at: SystemTime,
    credential: Arc<AwsCredential>,
    expires_at: Option<SystemTime>,
}

impl CachedAwsCredential {
    /// Returns whether the credential is safe to use without refreshing it.
    fn is_current(&self) -> bool {
        let now = SystemTime::now();
        let within_max_age = self
            .cached_at
            .checked_add(CREDENTIAL_MAX_CACHE_AGE)
            .is_some_and(|refresh_at| now < refresh_at);
        within_max_age
            && self.expires_at.is_none_or(|expires_at| {
                now.checked_add(CREDENTIAL_REFRESH_BUFFER)
                    .is_some_and(|refresh_at| refresh_at < expires_at)
            })
    }
}

/// Adapts AWS SDK credentials for the object-store S3 client with expiry-aware caching.
#[derive(Debug)]
struct AwsSdkCredentialProvider {
    credentials: SharedCredentialsProvider,
    cached_credentials: RwLock<Option<CachedAwsCredential>>,
    refresh_lock: Mutex<()>,
}

impl AwsSdkCredentialProvider {
    /// Creates an object-store credential provider backed by an AWS SDK provider chain.
    fn new(credentials: SharedCredentialsProvider) -> Self {
        Self {
            credentials,
            cached_credentials: RwLock::new(None),
            refresh_lock: Mutex::new(()),
        }
    }
}

#[async_trait]
impl CredentialProvider for AwsSdkCredentialProvider {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> object_store::Result<Arc<Self::Credential>> {
        if let Some(cached) = self
            .cached_credentials
            .read()
            .as_ref()
            .filter(|cached| cached.is_current())
        {
            return Ok(Arc::clone(&cached.credential));
        }

        let _refresh_guard = self.refresh_lock.lock().await;
        if let Some(cached) = self
            .cached_credentials
            .read()
            .as_ref()
            .filter(|cached| cached.is_current())
        {
            return Ok(Arc::clone(&cached.credential));
        }

        let credentials = self
            .credentials
            .provide_credentials()
            .await
            .map_err(|source| ObjectStoreError::Generic {
                store: "S3",
                source: Box::new(source),
            })?;

        let credential = Arc::new(AwsCredential {
            key_id: credentials.access_key_id().to_string(),
            secret_key: credentials.secret_access_key().to_string(),
            token: credentials.session_token().map(ToString::to_string),
        });
        *self.cached_credentials.write() = Some(CachedAwsCredential {
            cached_at: SystemTime::now(),
            credential: Arc::clone(&credential),
            expires_at: credentials.expiry(),
        });

        Ok(credential)
    }
}

/// AWS S3 settings shared by all requests for a LoRA source.
#[derive(Debug)]
struct AwsS3Configuration {
    credentials: AwsCredentialProvider,
    endpoint: Option<String>,
    region: String,
}

/// S3-based LoRA source using object_store with the AWS SDK credential provider chain.
pub struct S3LoRASource {
    configuration: OnceCell<AwsS3Configuration>,
    stores: RwLock<HashMap<String, Arc<dyn ObjectStore>>>,
}

impl S3LoRASource {
    /// Creates an S3 source using standard AWS configuration at request time.
    ///
    /// Credentials can come from environment variables, shared AWS configuration,
    /// workload identity, container credentials, or instance metadata.
    pub fn from_env() -> Self {
        Self {
            configuration: OnceCell::new(),
            stores: RwLock::new(HashMap::new()),
        }
    }
}

impl S3LoRASource {
    /// Returns the explicitly configured S3 endpoint with AWS service-specific precedence.
    fn endpoint_from_env() -> Option<String> {
        ["AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL", "AWS_ENDPOINT"]
            .into_iter()
            .find_map(|name| {
                std::env::var(name)
                    .ok()
                    .filter(|value| !value.trim().is_empty())
            })
    }

    /// Resolves the S3 endpoint using the AWS SDK's service and profile precedence.
    fn endpoint_from_aws_config(aws_config: &aws_config::SdkConfig) -> Result<Option<String>> {
        let key = ServiceConfigKey::builder()
            .service_id("S3")
            .env("AWS_ENDPOINT_URL")
            .profile("endpoint_url")
            .build()
            .context("Failed to construct the S3 endpoint configuration key")?;
        let service_endpoint = aws_config
            .service_config()
            .and_then(|config| config.load_config(key))
            .filter(|value| !value.trim().is_empty());
        let generic_endpoint = aws_config
            .endpoint_url()
            .map(ToString::to_string)
            .filter(|value| !value.trim().is_empty());

        Ok(service_endpoint
            .or(generic_endpoint)
            .or_else(Self::endpoint_from_env))
    }

    /// Adds the bucket to a custom endpoint when virtual-hosted addressing is enabled.
    fn bucket_qualified_endpoint(endpoint: &str, bucket: &str) -> Result<String> {
        let mut endpoint =
            Url::parse(endpoint).with_context(|| format!("Invalid S3 endpoint URL: {endpoint}"))?;
        let host = endpoint
            .host_str()
            .context("S3 endpoint URL must include a host")?;
        endpoint
            .set_host(Some(&format!("{bucket}.{host}")))
            .map_err(|_| anyhow::anyhow!("Invalid S3 bucket name for virtual-hosted endpoint"))?;
        Ok(String::from(endpoint).trim_end_matches('/').to_string())
    }

    /// Reads one explicitly supported boolean setting from the environment.
    fn config_bool_from_env(name: &str) -> Result<Option<bool>> {
        let Ok(value) = std::env::var(name) else {
            return Ok(None);
        };
        dynamo_runtime::config::parse_bool(&value)
            .map(Some)
            .with_context(|| format!("{name} must be a boolean value"))
    }

    /// Builds an S3 object-store client builder from the resolved AWS configuration.
    fn build_s3_builder(
        bucket: &str,
        region: &str,
        credentials: AwsCredentialProvider,
        endpoint: Option<&str>,
        timeout_secs: u64,
    ) -> Result<AmazonS3Builder> {
        let allow_http = Self::config_bool_from_env("AWS_ALLOW_HTTP")?.unwrap_or_default();
        let virtual_hosted_style =
            Self::config_bool_from_env("AWS_VIRTUAL_HOSTED_STYLE_REQUEST")?.unwrap_or_default();
        let mut builder = AmazonS3Builder::new()
            .with_region(region)
            .with_bucket_name(bucket)
            .with_allow_http(allow_http)
            .with_virtual_hosted_style_request(virtual_hosted_style)
            .with_config(
                object_store::aws::AmazonS3ConfigKey::Client(ClientConfigKey::Timeout),
                format!("{timeout_secs}s"),
            )
            .with_credentials(credentials);

        if let Some(endpoint) = endpoint {
            let endpoint = if virtual_hosted_style {
                Self::bucket_qualified_endpoint(endpoint, bucket)?
            } else {
                endpoint.to_string()
            };
            builder = builder.with_endpoint(endpoint);
        }

        Ok(builder)
    }

    /// Loads the AWS SDK configuration once for this LoRA source.
    async fn aws_s3_configuration(&self) -> Result<&AwsS3Configuration> {
        self.configuration
            .get_or_try_init(|| async {
                // Preserve the legacy us-east-1 fallback without querying IMDS for a region.
                let region_provider = aws_config::meta::region::RegionProviderChain::first_try(
                    aws_config::environment::region::EnvironmentVariableRegionProvider::new(),
                )
                .or_else(aws_config::profile::region::ProfileFileRegionProvider::new())
                .or_else(aws_types::region::Region::new("us-east-1"));
                let aws_config = aws_config::defaults(aws_config::BehaviorVersion::v2026_01_12())
                    .region(region_provider)
                    .load()
                    .await;
                let credentials = aws_config
                    .credentials_provider()
                    .ok_or_else(|| anyhow::anyhow!("AWS credential provider is not configured"))?;
                let region = aws_config
                    .region()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "us-east-1".to_string());
                let endpoint = Self::endpoint_from_aws_config(&aws_config)?;

                Ok(AwsS3Configuration {
                    credentials: Arc::new(AwsSdkCredentialProvider::new(credentials)),
                    endpoint,
                    region,
                })
            })
            .await
    }

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
    /// Builds an S3 object store configured for the requested bucket.
    async fn build_store(&self, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        if let Some(store) = self.stores.read().get(bucket) {
            return Ok(Arc::clone(store));
        }

        let timeout_secs: u64 = std::env::var("LORA_DOWNLOAD_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3600);
        let configuration = self.aws_s3_configuration().await?;
        let store = Self::build_s3_builder(
            bucket,
            &configuration.region,
            Arc::clone(&configuration.credentials),
            configuration.endpoint.as_deref(),
            timeout_secs,
        )?
        .build()?;
        let store: Arc<dyn ObjectStore> = Arc::new(store);
        Ok(Arc::clone(
            self.stores
                .write()
                .entry(bucket.to_string())
                .or_insert(store),
        ))
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
    fn supports(&self, lora_uri: &str) -> bool {
        lora_uri.starts_with("s3://")
    }

    async fn download(&self, s3_uri: &str, dest_path: &Path) -> Result<PathBuf> {
        let (bucket, prefix) = Self::parse_s3_uri(s3_uri)?;

        tracing::info!(
            "Downloading LoRA from S3: bucket={}, prefix={}",
            bucket,
            prefix
        );

        let bucket_store = self.build_store(&bucket).await?;
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

        let bucket_store = self.build_store(&bucket).await?;

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
    use aws_credential_types::{
        Credentials,
        provider::{ProvideCredentials, SharedCredentialsProvider, future},
    };
    use hf_hub::Cache;
    use mockito::Matcher;
    use object_store::{
        StaticCredentialProvider,
        aws::{AmazonS3ConfigKey, AwsCredentialProvider},
    };
    use std::{
        fs,
        sync::atomic::{AtomicUsize, Ordering},
    };
    use tempfile::TempDir;

    #[derive(Debug)]
    struct CountingCredentialProvider {
        credentials: Credentials,
        calls: Arc<AtomicUsize>,
    }

    impl ProvideCredentials for CountingCredentialProvider {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            self.calls.fetch_add(1, Ordering::SeqCst);
            future::ProvideCredentials::ready(Ok(self.credentials.clone()))
        }
    }

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
    #[test]
    fn s3_builder_prioritizes_service_endpoint_and_preserves_addressing_style() {
        let credentials: AwsCredentialProvider =
            Arc::new(StaticCredentialProvider::new(AwsCredential {
                key_id: "test-access-key".to_string(),
                secret_key: "test-secret-key".to_string(),
                token: None,
            }));
        let builder = temp_env::with_vars(
            [
                ("AWS_ENDPOINT_URL_S3", Some("https://s3-specific.example")),
                ("AWS_ENDPOINT_URL", Some("https://generic.example")),
                ("AWS_ENDPOINT", Some("https://legacy.example")),
                ("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", Some("1")),
            ],
            || {
                let endpoint = S3LoRASource::endpoint_from_env();
                S3LoRASource::build_s3_builder(
                    "bucket",
                    "us-east-1",
                    credentials,
                    endpoint.as_deref(),
                    60,
                )
                .unwrap()
            },
        );

        assert_eq!(
            builder.get_config_value(&AmazonS3ConfigKey::Endpoint),
            Some("https://bucket.s3-specific.example".to_string())
        );
        assert_eq!(
            builder.get_config_value(&AmazonS3ConfigKey::VirtualHostedStyleRequest),
            Some("true".to_string())
        );
    }

    #[serial_test::serial]
    #[test]
    fn s3_endpoint_ignores_empty_values_before_legacy_fallback() {
        let endpoint = temp_env::with_vars(
            [
                ("AWS_ENDPOINT_URL_S3", Some("")),
                ("AWS_ENDPOINT_URL", Some("  ")),
                ("AWS_ENDPOINT", Some("https://legacy.example")),
            ],
            S3LoRASource::endpoint_from_env,
        );

        assert_eq!(endpoint.as_deref(), Some("https://legacy.example"));
    }

    #[tokio::test]
    async fn s3_credential_provider_caches_valid_credentials() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = AwsSdkCredentialProvider::new(SharedCredentialsProvider::new(
            CountingCredentialProvider {
                credentials: Credentials::new(
                    "test-access-key",
                    "test-secret-key",
                    None,
                    None,
                    "test",
                ),
                calls: Arc::clone(&calls),
            },
        ));

        let first = provider.get_credential().await.unwrap();
        let second = provider.get_credential().await.unwrap();

        assert_eq!(first.key_id, "test-access-key");
        assert_eq!(second.key_id, "test-access-key");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn s3_credential_provider_refreshes_expiring_credentials() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = AwsSdkCredentialProvider::new(SharedCredentialsProvider::new(
            CountingCredentialProvider {
                credentials: Credentials::new(
                    "test-access-key",
                    "test-secret-key",
                    None,
                    Some(SystemTime::now() + Duration::from_secs(30)),
                    "test",
                ),
                calls: Arc::clone(&calls),
            },
        ));

        provider.get_credential().await.unwrap();
        provider.get_credential().await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn s3_credential_provider_refreshes_non_expiring_credentials_after_max_age() {
        let calls = Arc::new(AtomicUsize::new(0));
        let provider = AwsSdkCredentialProvider::new(SharedCredentialsProvider::new(
            CountingCredentialProvider {
                credentials: Credentials::new(
                    "test-access-key",
                    "test-secret-key",
                    None,
                    None,
                    "test",
                ),
                calls: Arc::clone(&calls),
            },
        ));

        provider.get_credential().await.unwrap();
        provider
            .cached_credentials
            .write()
            .as_mut()
            .unwrap()
            .cached_at = SystemTime::now() - CREDENTIAL_MAX_CACHE_AGE;
        provider.get_credential().await.unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn s3_source_ignores_skip_signature_with_shared_credentials_and_profile_endpoint() {
        let mut server = mockito::Server::new_async().await;
        let list = server
            .mock("GET", "/bucket")
            .match_query(Matcher::AllOf(vec![
                Matcher::UrlEncoded("list-type".into(), "2".into()),
                Matcher::UrlEncoded("prefix".into(), "adapter/".into()),
            ]))
            .match_header("host", server.host_with_port().as_str())
            .match_header("authorization", Matcher::Regex("Credential=profile-access-key/".into()))
            .with_status(200)
            .with_header("content-type", "application/xml")
            .with_body(
                r#"<ListBucketResult><Contents><Key>adapter/adapter_config.json</Key><Size>2</Size><LastModified>2026-01-01T00:00:00Z</LastModified><ETag>"etag"</ETag></Contents></ListBucketResult>"#,
            )
            .create_async()
            .await;
        let get = server
            .mock("GET", "/bucket/adapter/adapter_config.json")
            .match_header("host", server.host_with_port().as_str())
            .match_header(
                "authorization",
                Matcher::Regex("Credential=profile-access-key/".into()),
            )
            .with_status(200)
            .with_body("{}")
            .create_async()
            .await;

        let temp = TempDir::new().unwrap();
        let credentials_path = temp.path().join("credentials");
        let config_path = temp.path().join("config");
        fs::write(
            &credentials_path,
            "[profile]\naws_access_key_id = profile-access-key\naws_secret_access_key = profile-secret-key\n",
        )
        .unwrap();
        fs::write(
            &config_path,
            format!(
                "[profile profile]\nregion = us-east-1\nendpoint_url = {}\n",
                server.url()
            ),
        )
        .unwrap();
        let destination = temp.path().join("adapter");

        let downloaded = temp_env::async_with_vars(
            [
                ("AWS_ACCESS_KEY_ID", None),
                ("AWS_SECRET_ACCESS_KEY", None),
                ("AWS_SESSION_TOKEN", None),
                ("AWS_REGION", None),
                ("AWS_DEFAULT_REGION", None),
                ("AWS_ENDPOINT", None),
                ("AWS_ENDPOINT_URL_S3", None),
                ("AWS_SHARED_CREDENTIALS_FILE", credentials_path.to_str()),
                ("AWS_CONFIG_FILE", config_path.to_str()),
                ("AWS_PROFILE", Some("profile")),
                ("AWS_ENDPOINT_URL", None),
                ("AWS_ALLOW_HTTP", Some("true")),
                ("AWS_VIRTUAL_HOSTED_STYLE_REQUEST", None),
                ("AWS_SKIP_SIGNATURE", Some("true")),
                ("AWS_PROXY_URL", None),
                ("AWS_PROXY_EXCLUDES", None),
                ("AWS_EC2_METADATA_DISABLED", Some("true")),
            ],
            async {
                let source = S3LoRASource::from_env();
                source.download("s3://bucket/adapter", &destination).await
            },
        )
        .await
        .unwrap();

        assert_eq!(downloaded, destination);
        assert_eq!(
            fs::read_to_string(downloaded.join("adapter_config.json")).unwrap(),
            "{}"
        );
        list.assert_async().await;
        get.assert_async().await;
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn s3_configuration_defaults_region_without_querying_imds() {
        let mut imds = mockito::Server::new_async().await;
        let token = imds
            .mock("PUT", "/latest/api/token")
            .expect(0)
            .with_status(200)
            .with_header("x-aws-ec2-metadata-token-ttl-seconds", "21600")
            .with_body("token")
            .create_async()
            .await;
        let region = imds
            .mock("GET", "/latest/meta-data/placement/region")
            .expect(0)
            .with_status(200)
            .with_body("us-west-2")
            .create_async()
            .await;

        let temp = TempDir::new().unwrap();
        let config_path = temp.path().join("config");
        let credentials_path = temp.path().join("credentials");
        fs::write(&config_path, "").unwrap();
        fs::write(&credentials_path, "").unwrap();
        let imds_endpoint = imds.url();

        let resolved_region = temp_env::async_with_vars(
            [
                ("AWS_ACCESS_KEY_ID", Some("test-access-key")),
                ("AWS_SECRET_ACCESS_KEY", Some("test-secret-key")),
                ("AWS_REGION", None),
                ("AWS_DEFAULT_REGION", None),
                ("AWS_CONFIG_FILE", config_path.to_str()),
                ("AWS_SHARED_CREDENTIALS_FILE", credentials_path.to_str()),
                ("AWS_PROFILE", None),
                ("AWS_ENDPOINT_URL", Some("https://s3.example")),
                ("AWS_EC2_METADATA_DISABLED", Some("false")),
                (
                    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
                    Some(imds_endpoint.as_str()),
                ),
            ],
            async {
                let source = S3LoRASource::from_env();
                Ok::<_, anyhow::Error>(source.aws_s3_configuration().await?.region.clone())
            },
        )
        .await
        .unwrap();

        assert_eq!(resolved_region, "us-east-1");
        token.assert_async().await;
        region.assert_async().await;
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
        fs::write(snapshot.join(".dynamo_lora_complete"), "1\n").unwrap();

        let source = HuggingFaceLoRASource::with_cache(Cache::new(temp.path().to_path_buf()));
        assert_eq!(
            source.cached_path("hf://org/adapter").unwrap(),
            Some(snapshot.clone())
        );
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
        fs::write(snapshot.join(".dynamo_lora_complete"), "1\n").unwrap();

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
                ("HUGGING_FACE_HUB_TOKEN", None),
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
        assert_eq!(
            fs::read_to_string(snapshot.join(".dynamo_lora_complete")).unwrap(),
            "1\n"
        );

        info.assert_async().await;
        config.assert_async().await;
        weights.assert_async().await;
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn hf_source_does_not_mark_partial_commit_snapshot_complete() {
        let mut server = mockito::Server::new_async().await;
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let info = server
            .mock(
                "GET",
                format!("/api/models/org/adapter/revision/{commit}").as_str(),
            )
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(format!(
                r#"{{"siblings":[{{"rfilename":"adapter_config.json"}},{{"rfilename":"adapter_model.safetensors"}},{{"rfilename":"README.md"}}],"sha":"{commit}"}}"#
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
        let readme = server
            .mock(
                "GET",
                format!("/org/adapter/resolve/{commit}/README.md").as_str(),
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
                ("HUGGING_FACE_HUB_TOKEN", None),
                ("HF_TOKEN_PATH", None),
                ("HF_HUB_OFFLINE", None),
            ],
            source.download(
                format!("hf://org/adapter@{commit}").as_str(),
                Path::new("unused"),
            ),
        )
        .await;

        assert!(result.is_err());
        let snapshot = temp
            .path()
            .join("models--org--adapter/snapshots")
            .join(commit);
        assert!(snapshot.join("adapter_config.json").is_file());
        assert!(snapshot.join("adapter_model.safetensors").is_file());
        assert!(!snapshot.join(".dynamo_lora_complete").exists());

        let offline_result = temp_env::async_with_vars(
            [("HF_HUB_OFFLINE", Some("1"))],
            source.download(
                format!("hf://org/adapter@{commit}").as_str(),
                Path::new("unused"),
            ),
        )
        .await;
        assert!(offline_result.is_err());

        info.assert_async().await;
        config.assert_async().await;
        weights.assert_async().await;
        readme.assert_async().await;
    }
}
