// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3-compatible object storage client for block management.
//!
//! This module provides an implementation of [`ObjectBlockOps`] using the AWS S3 SDK.
//! It supports S3-compatible storage services including MinIO.

use anyhow::{Result, anyhow};
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::StreamExt;

use crate::v2::distributed::object::{
    DefaultKeyFormatter, KeyFormatter, LayoutConfigExt, ObjectBlockOps,
};
use crate::v2::physical::transfer::PhysicalLayout;
use crate::{BlockId, SequenceHash};
use std::sync::Arc;

/// Configuration for S3 object storage client.
#[derive(Debug, Clone)]
pub struct S3Config {
    /// Custom endpoint URL for S3-compatible services (e.g., MinIO).
    /// If None, uses the default AWS S3 endpoint.
    pub endpoint_url: Option<String>,

    /// S3 bucket name for storing blocks.
    pub bucket: String,

    /// AWS region.
    pub region: String,

    /// Use path-style URLs instead of virtual-hosted-style.
    /// Required for MinIO and some S3-compatible services.
    pub force_path_style: bool,

    /// Maximum number of concurrent S3 requests.
    pub max_concurrent_requests: usize,
}

impl Default for S3Config {
    /// Returns a default configuration suitable for local MinIO testing.
    fn default() -> Self {
        Self {
            endpoint_url: Some("http://localhost:9000".into()),
            bucket: "kvbm-blocks".into(),
            region: "us-east-1".into(),
            force_path_style: true,
            max_concurrent_requests: 16,
        }
    }
}

impl S3Config {
    /// Create a new S3Config for AWS S3 (not MinIO).
    pub fn aws(bucket: String, region: String) -> Self {
        Self {
            endpoint_url: None,
            bucket,
            region,
            force_path_style: false,
            max_concurrent_requests: 16,
        }
    }

    /// Create a new S3Config for MinIO.
    pub fn minio(endpoint_url: String, bucket: String) -> Self {
        Self {
            endpoint_url: Some(endpoint_url),
            bucket,
            region: "us-east-1".into(),
            force_path_style: true,
            max_concurrent_requests: 16,
        }
    }

    /// Create from kvbm-config's S3ObjectConfig.
    pub fn from_object_config(config: &dynamo_kvbm_config::S3ObjectConfig) -> Self {
        Self {
            endpoint_url: config.endpoint_url.clone(),
            bucket: config.bucket.clone(),
            region: config.region.clone(),
            force_path_style: config.force_path_style,
            max_concurrent_requests: config.max_concurrent_requests,
        }
    }

    /// Set the maximum number of concurrent requests.
    pub fn with_max_concurrent_requests(mut self, max: usize) -> Self {
        self.max_concurrent_requests = max;
        self
    }
}

/// S3-compatible object storage client for block operations.
///
/// This client implements [`ObjectBlockOps`] using the AWS S3 SDK.
/// It supports parallel block operations and uses rayon for CPU-bound memory copies.
///
/// # Key Formatting
///
/// Uses a [`KeyFormatter`] to convert `SequenceHash` to object keys. The formatter
/// can embed rank, namespace, or other prefixes for key uniqueness across workers.
pub struct S3ObjectBlockClient {
    /// AWS S3 client
    client: Client,

    /// S3 configuration
    config: S3Config,

    /// Key formatter for converting SequenceHash to object keys.
    key_formatter: Arc<dyn KeyFormatter>,
}

impl S3ObjectBlockClient {
    /// Create a new S3ObjectBlockClient with default key formatting.
    ///
    /// # Arguments
    /// * `config` - S3 configuration
    ///
    /// # Errors
    /// Returns an error if the S3 client cannot be initialized.
    pub async fn new(config: S3Config) -> Result<Self> {
        let client = build_s3_client(&config).await?;
        Ok(Self {
            client,
            config,
            key_formatter: Arc::new(DefaultKeyFormatter),
        })
    }

    /// Create a new S3ObjectBlockClient with a custom key formatter.
    ///
    /// # Arguments
    /// * `config` - S3 configuration
    /// * `key_formatter` - Custom key formatter for SequenceHash → String conversion
    ///
    /// # Errors
    /// Returns an error if the S3 client cannot be initialized.
    pub async fn with_key_formatter(
        config: S3Config,
        key_formatter: Arc<dyn KeyFormatter>,
    ) -> Result<Self> {
        let client = build_s3_client(&config).await?;
        Ok(Self {
            client,
            config,
            key_formatter,
        })
    }

    /// Create from an existing AWS S3 client with default key formatting.
    pub fn from_client(client: Client, config: S3Config) -> Self {
        Self {
            client,
            config,
            key_formatter: Arc::new(DefaultKeyFormatter),
        }
    }

    /// Create from an existing AWS S3 client with a custom key formatter.
    pub fn from_client_with_formatter(
        client: Client,
        config: S3Config,
        key_formatter: Arc<dyn KeyFormatter>,
    ) -> Self {
        Self {
            client,
            config,
            key_formatter,
        }
    }

    /// Get a reference to the S3 client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &S3Config {
        &self.config
    }

    /// Get a reference to the key formatter.
    pub fn key_formatter(&self) -> &Arc<dyn KeyFormatter> {
        &self.key_formatter
    }

    /// Get a reference to the bucket name.
    pub fn bucket(&self) -> &str {
        &self.config.bucket
    }

    /// Ensure the bucket exists, creating it if necessary.
    pub async fn ensure_bucket_exists(&self) -> Result<()> {
        match self
            .client
            .head_bucket()
            .bucket(&self.config.bucket)
            .send()
            .await
        {
            Ok(_) => Ok(()),
            Err(_) => {
                // Bucket doesn't exist, create it
                self.client
                    .create_bucket()
                    .bucket(&self.config.bucket)
                    .send()
                    .await
                    .map_err(|e| {
                        anyhow!("failed to create bucket '{}': {}", self.config.bucket, e)
                    })?;
                Ok(())
            }
        }
    }

    /// Put an object with a conditional check (If-None-Match: *).
    ///
    /// This performs an atomic write that only succeeds if the object does not
    /// already exist. Returns:
    /// - `Ok(true)` if the object was created successfully
    /// - `Ok(false)` if the object already exists (PreconditionFailed)
    /// - `Err(...)` for other errors
    ///
    /// # Arguments
    /// * `key` - Object key
    /// * `data` - Object data to write
    pub async fn put_if_not_exists(&self, key: &str, data: Bytes) -> Result<bool> {
        match self
            .client
            .put_object()
            .bucket(&self.config.bucket)
            .key(key)
            .if_none_match("*")
            .body(ByteStream::from(data))
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                // Check if this is a precondition failed error
                let service_error = e.into_service_error();
                if service_error.is_precondition_failed() {
                    Ok(false)
                } else {
                    Err(anyhow!(
                        "S3 conditional put failed for key '{}': {}",
                        key,
                        service_error
                    ))
                }
            }
        }
    }

    /// Get an object's raw bytes.
    ///
    /// # Arguments
    /// * `key` - Object key
    ///
    /// # Returns
    /// - `Ok(Some(bytes))` if the object exists
    /// - `Ok(None)` if the object does not exist
    /// - `Err(...)` for other errors
    pub async fn get_object(&self, key: &str) -> Result<Option<Bytes>> {
        match self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(resp) => {
                let data = resp
                    .body
                    .collect()
                    .await
                    .map_err(|e| anyhow!("failed to collect S3 response body: {}", e))?
                    .into_bytes();
                Ok(Some(data))
            }
            Err(e) => {
                let service_error = e.into_service_error();
                if service_error.is_no_such_key() {
                    Ok(None)
                } else {
                    Err(anyhow!(
                        "S3 get_object failed for key '{}': {}",
                        key,
                        service_error
                    ))
                }
            }
        }
    }

    /// Delete an object.
    ///
    /// # Arguments
    /// * `key` - Object key
    ///
    /// # Returns
    /// - `Ok(true)` if the object was deleted
    /// - `Ok(false)` if the object did not exist
    /// - `Err(...)` for other errors
    pub async fn delete_object(&self, key: &str) -> Result<bool> {
        match self
            .client
            .delete_object()
            .bucket(&self.config.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                let service_error = e.into_service_error();
                if service_error.is_no_such_key() {
                    Ok(false)
                } else {
                    Err(anyhow!(
                        "S3 delete_object failed for key '{}': {}",
                        key,
                        service_error
                    ))
                }
            }
        }
    }

    /// Check if an object exists (HEAD request).
    ///
    /// # Arguments
    /// * `key` - Object key
    ///
    /// # Returns
    /// - `Ok(true)` if the object exists
    /// - `Ok(false)` if the object does not exist
    /// - `Err(...)` for other errors
    pub async fn has_object(&self, key: &str) -> Result<bool> {
        match self
            .client
            .head_object()
            .bucket(&self.config.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                let service_error = e.into_service_error();
                if service_error.is_not_found() {
                    Ok(false)
                } else {
                    Err(anyhow!(
                        "S3 head_object failed for key '{}': {}",
                        key,
                        service_error
                    ))
                }
            }
        }
    }

    /// Put an object unconditionally (overwrite if exists).
    ///
    /// # Arguments
    /// * `key` - Object key
    /// * `data` - Object data to write
    pub async fn put_object(&self, key: &str, data: Bytes) -> Result<()> {
        self.client
            .put_object()
            .bucket(&self.config.bucket)
            .key(key)
            .body(ByteStream::from(data))
            .send()
            .await
            .map_err(|e| anyhow!("S3 put_object failed for key '{}': {}", key, e))?;
        Ok(())
    }
}

/// Build an S3 client from configuration.
async fn build_s3_client(config: &S3Config) -> Result<Client> {
    let sdk_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_sdk_s3::config::Region::new(config.region.clone()))
        .load()
        .await;

    let mut s3_config_builder = aws_sdk_s3::config::Builder::from(&sdk_config);

    if let Some(endpoint) = &config.endpoint_url {
        s3_config_builder = s3_config_builder.endpoint_url(endpoint);
    }

    if config.force_path_style {
        s3_config_builder = s3_config_builder.force_path_style(true);
    }

    let s3_config = s3_config_builder.build();
    Ok(Client::from_conf(s3_config))
}

/// Copy block data from a layout to a Bytes buffer.
///
/// For fully contiguous layouts, this is a single memcpy.
/// For layer-separate layouts, this iterates over all regions.
fn copy_block_to_bytes(
    layout: &PhysicalLayout,
    block_id: BlockId,
    block_size: usize,
    region_size: usize,
    is_contiguous: bool,
) -> Result<Bytes> {
    if is_contiguous {
        // Fast path: single contiguous region
        let region = layout.memory_region(block_id, 0, 0)?;
        let slice = unsafe { std::slice::from_raw_parts(region.addr() as *const u8, block_size) };
        Ok(Bytes::copy_from_slice(slice))
    } else {
        // Slow path: iterate over all regions
        let mut buf = Vec::with_capacity(block_size);
        let inner_layout = layout.layout();
        for layer_id in 0..inner_layout.num_layers() {
            for outer_id in 0..inner_layout.outer_dim() {
                let region = layout.memory_region(block_id, layer_id, outer_id)?;
                let slice =
                    unsafe { std::slice::from_raw_parts(region.addr() as *const u8, region_size) };
                buf.extend_from_slice(slice);
            }
        }
        Ok(Bytes::from(buf))
    }
}

/// Copy data from a Bytes buffer to a layout.
///
/// For fully contiguous layouts, this is a single memcpy.
/// For layer-separate layouts, this iterates over all regions.
fn copy_bytes_to_block(
    data: &[u8],
    layout: &PhysicalLayout,
    block_id: BlockId,
    block_size: usize,
    region_size: usize,
    is_contiguous: bool,
) -> Result<()> {
    if is_contiguous {
        // Fast path: single contiguous region
        let region = layout.memory_region(block_id, 0, 0)?;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), region.addr() as *mut u8, block_size);
        }
    } else {
        // Slow path: iterate over all regions
        let mut offset = 0;
        let inner_layout = layout.layout();
        for layer_id in 0..inner_layout.num_layers() {
            for outer_id in 0..inner_layout.outer_dim() {
                let region = layout.memory_region(block_id, layer_id, outer_id)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data[offset..].as_ptr(),
                        region.addr() as *mut u8,
                        region_size,
                    );
                }
                offset += region_size;
            }
        }
    }
    Ok(())
}

impl ObjectBlockOps for S3ObjectBlockClient {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        let max_concurrent = self.config.max_concurrent_requests;
        let client = self.client.clone();
        let bucket = self.config.bucket.clone();
        let formatter = self.key_formatter.clone();

        Box::pin(async move {
            let tasks = keys.into_iter().map(|key| {
                let client = client.clone();
                let bucket = bucket.clone();
                let key_str = formatter.format_key(&key);

                async move {
                    match client
                        .head_object()
                        .bucket(&bucket)
                        .key(&key_str)
                        .send()
                        .await
                    {
                        Ok(resp) => (key, resp.content_length().map(|l| l as usize)),
                        Err(_) => (key, None), // Not found or error
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        let is_contiguous = layout.layout().is_fully_contiguous();
        let max_concurrent = self.config.max_concurrent_requests;
        let client = self.client.clone();
        let bucket = self.config.bucket.clone();
        let formatter = self.key_formatter.clone();

        Box::pin(async move {
            let work_items: Vec<_> = keys.into_iter().zip(block_ids.into_iter()).collect();

            let tasks = work_items.into_iter().map(|(key, block_id)| {
                let client = client.clone();
                let bucket = bucket.clone();
                let key_str = formatter.format_key(&key);
                let layout = layout.clone();

                async move {
                    let result: Result<(), anyhow::Error> = async {
                        // Copy block data to bytes on rayon thread pool
                        let data = tokio_rayon::spawn(move || {
                            copy_block_to_bytes(
                                &layout,
                                block_id,
                                block_size,
                                region_size,
                                is_contiguous,
                            )
                        })
                        .await?;

                        // Upload to S3
                        client
                            .put_object()
                            .bucket(&bucket)
                            .key(&key_str)
                            .body(ByteStream::from(data))
                            .send()
                            .await
                            .map_err(|e| anyhow!("S3 put_object failed: {}", e))?;

                        Ok(())
                    }
                    .await;

                    match result {
                        Ok(()) => Ok(key),
                        Err(_) => Err(key),
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        let is_contiguous = layout.layout().is_fully_contiguous();
        let max_concurrent = self.config.max_concurrent_requests;
        let client = self.client.clone();
        let bucket = self.config.bucket.clone();
        let formatter = self.key_formatter.clone();

        Box::pin(async move {
            let work_items: Vec<_> = keys.into_iter().zip(block_ids.into_iter()).collect();

            let tasks = work_items.into_iter().map(|(key, block_id)| {
                let client = client.clone();
                let bucket = bucket.clone();
                let key_str = formatter.format_key(&key);
                let layout = layout.clone();

                async move {
                    let result: Result<(), anyhow::Error> = async {
                        // Download from S3
                        let resp = client
                            .get_object()
                            .bucket(&bucket)
                            .key(&key_str)
                            .send()
                            .await
                            .map_err(|e| anyhow!("S3 get_object failed: {}", e))?;

                        let data = resp
                            .body
                            .collect()
                            .await
                            .map_err(|e| anyhow!("failed to collect S3 response body: {}", e))?
                            .into_bytes();

                        // Copy bytes to block on rayon thread pool
                        tokio_rayon::spawn(move || {
                            copy_bytes_to_block(
                                &data,
                                &layout,
                                block_id,
                                block_size,
                                region_size,
                                is_contiguous,
                            )
                        })
                        .await?;

                        Ok(())
                    }
                    .await;

                    match result {
                        Ok(()) => Ok(key),
                        Err(_) => Err(key),
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s3_config_default() {
        let config = S3Config::default();
        assert_eq!(config.endpoint_url, Some("http://localhost:9000".into()));
        assert_eq!(config.bucket, "kvbm-blocks");
        assert_eq!(config.region, "us-east-1");
        assert!(config.force_path_style);
        assert_eq!(config.max_concurrent_requests, 16);
    }

    #[test]
    fn test_s3_config_aws() {
        let config = S3Config::aws("my-bucket".into(), "us-west-2".into());
        assert_eq!(config.endpoint_url, None);
        assert_eq!(config.bucket, "my-bucket");
        assert_eq!(config.region, "us-west-2");
        assert!(!config.force_path_style);
    }

    #[test]
    fn test_s3_config_minio() {
        let config = S3Config::minio("http://minio:9000".into(), "test-bucket".into());
        assert_eq!(config.endpoint_url, Some("http://minio:9000".into()));
        assert_eq!(config.bucket, "test-bucket");
        assert!(config.force_path_style);
    }
}


