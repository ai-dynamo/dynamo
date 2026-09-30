// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3-compatible object storage client for block management.
//!
//! This module provides an implementation of [`ObjectBlockOps`] using the AWS S3 SDK.
//! It supports S3-compatible storage services including MinIO.

use anyhow::{Result, anyhow};
use aws_sdk_s3::Client;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::primitives::ByteStream;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::StreamExt;

use crate::object::{DefaultKeyFormatter, KeyFormatter, LayoutConfigExt, ObjectBlockOps};
use crate::{BlockId, SequenceHash};
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::transfer::PhysicalLayout;
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
    pub fn from_object_config(config: &kvbm_config::S3ObjectConfig) -> Self {
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
                // Check if this is a precondition failed error (HTTP 412)
                let service_error = e.into_service_error();
                if service_error.code() == Some("PreconditionFailed") {
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
                if service_error.code() == Some("NoSuchKey") {
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
                if service_error.code() == Some("NoSuchKey") {
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
                // HeadObject returns "NotFound" when object doesn't exist
                if service_error.code() == Some("NotFound") {
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

    /// Put blocks to object storage using a physical layout.
    ///
    /// This is the internal implementation that workers call after resolving
    /// the logical layout handle to a physical layout.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `layout` - Physical layout containing the block data
    /// * `block_ids` - Block IDs within the layout to upload
    ///
    /// Returns a vector of results for each block:
    /// - Ok(hash) indicates the block was successfully stored
    /// - Err(hash) indicates the block failed to store
    pub fn put_blocks_with_layout(
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
            let work_items: Vec<_> = keys.into_iter().zip(block_ids).collect();

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
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "put block transfer failed");
                            Err(key)
                        }
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }

    /// Get blocks from object storage into a physical layout.
    ///
    /// This is the internal implementation that workers call after resolving
    /// the logical layout handle to a physical layout.
    ///
    /// # Arguments
    /// * `keys` - Sequence hashes identifying each block
    /// * `layout` - Physical layout to write the block data into
    /// * `block_ids` - Block IDs within the layout to download into
    ///
    /// Returns a vector of results for each block:
    /// - Ok(hash) indicates the block was successfully retrieved
    /// - Err(hash) indicates the block failed to retrieve
    pub fn get_blocks_with_layout(
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
            let work_items: Vec<_> = keys.into_iter().zip(block_ids).collect();

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
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "get block transfer failed");
                            Err(key)
                        }
                    }
                }
            });

            futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await
        })
    }

    /// Get an object's raw bytes along with its ETag.
    ///
    /// Used for conditional updates (CAS-style operations) where the caller
    /// needs the current ETag to perform a conditional PUT.
    ///
    /// # Returns
    /// - `Ok(Some((bytes, etag)))` if the object exists
    /// - `Ok(None)` if the object does not exist
    /// - `Err(...)` for other errors
    pub async fn get_object_with_etag(&self, key: &str) -> Result<Option<(Bytes, Option<String>)>> {
        match self
            .client
            .get_object()
            .bucket(&self.config.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(resp) => {
                let etag = resp.e_tag().map(|s| s.to_string());
                let data = resp
                    .body
                    .collect()
                    .await
                    .map_err(|e| anyhow!("failed to collect S3 response body: {}", e))?
                    .into_bytes();
                Ok(Some((data, etag)))
            }
            Err(e) => {
                let service_error = e.into_service_error();
                if service_error.code() == Some("NoSuchKey") {
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

    /// Put an object with an ETag precondition (If-Match).
    ///
    /// This performs a conditional write that only succeeds if the object's current
    /// ETag matches the provided value. Used for CAS-style atomic updates.
    ///
    /// # Returns
    /// - `Ok(true)` if the write succeeded (ETag matched)
    /// - `Ok(false)` if the ETag did not match (412 PreconditionFailed — lost the race)
    /// - `Err(...)` for other errors
    pub async fn put_object_if_match(&self, key: &str, data: Bytes, etag: &str) -> Result<bool> {
        match self
            .client
            .put_object()
            .bucket(&self.config.bucket)
            .key(key)
            .if_match(etag)
            .body(ByteStream::from(data))
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                let service_error = e.into_service_error();
                if service_error.code() == Some("PreconditionFailed") {
                    Ok(false)
                } else {
                    Err(anyhow!(
                        "S3 conditional put (if-match) failed for key '{}': {}",
                        key,
                        service_error
                    ))
                }
            }
        }
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
        // Fast path: single contiguous region — the layout guarantees that
        // block_size bytes are contiguous from region(block_id, 0, 0).addr().
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
                if region.size() < region_size {
                    return Err(anyhow!(
                        "memory region too small: got {} bytes, need {}",
                        region.size(),
                        region_size
                    ));
                }
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
        // Fast path: single contiguous region — the layout guarantees that
        // block_size bytes are contiguous from region(block_id, 0, 0).addr().
        if data.len() < block_size {
            return Err(anyhow!(
                "S3 data too short: got {} bytes, expected {}",
                data.len(),
                block_size
            ));
        }
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
                if offset + region_size > data.len() {
                    return Err(anyhow!(
                        "S3 data too short at offset {}: need {} more bytes, only {} remain",
                        offset,
                        region_size,
                        data.len().saturating_sub(offset)
                    ));
                }
                let region = layout.memory_region(block_id, layer_id, outer_id)?;
                if region.size() < region_size {
                    return Err(anyhow!(
                        "memory region too small: got {} bytes, need {}",
                        region.size(),
                        region_size
                    ));
                }
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
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "head_object failed, treating as missing");
                            (key, None)
                        }
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
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // S3ObjectBlockClient cannot resolve LogicalLayoutHandle to PhysicalLayout.
        // Workers should use put_blocks_with_layout() instead after resolving the handle.
        tracing::error!(
            "S3ObjectBlockClient::put_blocks called with LogicalLayoutHandle - \
             use put_blocks_with_layout() via DirectWorker instead"
        );
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // S3ObjectBlockClient cannot resolve LogicalLayoutHandle to PhysicalLayout.
        // Workers should use get_blocks_with_layout() instead after resolving the handle.
        tracing::error!(
            "S3ObjectBlockClient::get_blocks called with LogicalLayoutHandle - \
             use get_blocks_with_layout() via DirectWorker instead"
        );
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn put_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Delegate to the inherent method
        S3ObjectBlockClient::put_blocks_with_layout(self, keys, layout, block_ids)
    }

    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Delegate to the inherent method
        S3ObjectBlockClient::get_blocks_with_layout(self, keys, layout, block_ids)
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

#[cfg(all(test, feature = "testing"))]
mod bounds_check_tests {
    use super::*;
    use crate::object::LayoutConfigExt;
    use kvbm_physical::testing::{create_fc_layout, create_lw_layout, create_test_agent};
    use kvbm_physical::transfer::StorageKind;

    #[test]
    fn test_copy_bytes_to_block_rejects_short_data_contiguous() {
        let agent = create_test_agent("test_short_data_fc");
        let layout = create_fc_layout(agent, StorageKind::System, 2);
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();

        // Data is one byte short
        let short_data = vec![0u8; block_size - 1];
        let err = copy_bytes_to_block(&short_data, &layout, 0, block_size, region_size, true)
            .expect_err("should reject short data");
        assert!(
            err.to_string().contains("S3 data too short"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_copy_bytes_to_block_rejects_short_data_non_contiguous() {
        let agent = create_test_agent("test_short_data_lw");
        let layout = create_lw_layout(agent, StorageKind::System, 2);
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();

        // Data is one region short
        let short_data = vec![0u8; block_size - region_size];
        let err = copy_bytes_to_block(&short_data, &layout, 0, block_size, region_size, false)
            .expect_err("should reject short data in non-contiguous path");
        assert!(
            err.to_string().contains("S3 data too short"),
            "unexpected error: {}",
            err
        );
    }

    #[test]
    fn test_copy_bytes_to_block_accepts_exact_size() {
        let agent = create_test_agent("test_exact_fc");
        let layout = create_fc_layout(agent, StorageKind::System, 2);
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();

        let data = vec![42u8; block_size];
        copy_bytes_to_block(&data, &layout, 0, block_size, region_size, true)
            .expect("exact-size data should succeed");
    }

    #[test]
    fn test_copy_block_to_bytes_roundtrip_contiguous() {
        let agent = create_test_agent("test_roundtrip_fc");
        let layout = create_fc_layout(agent, StorageKind::System, 2);
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();

        // Write known data
        let data = vec![0xAB_u8; block_size];
        copy_bytes_to_block(&data, &layout, 0, block_size, region_size, true).unwrap();

        // Read it back
        let out = copy_block_to_bytes(&layout, 0, block_size, region_size, true).unwrap();
        assert_eq!(out.as_ref(), &data[..]);
    }

    #[test]
    fn test_copy_block_to_bytes_roundtrip_non_contiguous() {
        let agent = create_test_agent("test_roundtrip_lw");
        let layout = create_lw_layout(agent, StorageKind::System, 2);
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();

        let data = vec![0xCD_u8; block_size];
        copy_bytes_to_block(&data, &layout, 0, block_size, region_size, false).unwrap();

        let out = copy_block_to_bytes(&layout, 0, block_size, region_size, false).unwrap();
        assert_eq!(out.as_ref(), &data[..]);
    }
}

#[cfg(all(test, feature = "testing-s3"))]
pub mod s3_integration {
    use super::*;

    /// Create an S3ObjectBlockClient connected to the test MinIO instance.
    ///
    /// Reads `S3_TEST_ENDPOINT` from the environment (set by `test-s3.sh`).
    /// Falls back to `http://localhost:9876`.
    pub async fn create_test_client(bucket: &str) -> S3ObjectBlockClient {
        let endpoint =
            std::env::var("S3_TEST_ENDPOINT").unwrap_or_else(|_| "http://localhost:9876".into());
        let config = S3Config::minio(endpoint, bucket.to_string());
        let client = S3ObjectBlockClient::new(config).await.unwrap();
        client.ensure_bucket_exists().await.unwrap();
        client
    }

    #[tokio::test]
    async fn test_put_get_roundtrip() {
        let client = create_test_client("test-roundtrip").await;
        let key = format!("roundtrip-{}", uuid::Uuid::new_v4());
        let payload = Bytes::from("hello world");

        client.put_object(&key, payload.clone()).await.unwrap();

        let result = client.get_object(&key).await.unwrap();
        assert_eq!(result, Some(payload));

        // Cleanup
        client.delete_object(&key).await.unwrap();
    }

    #[tokio::test]
    async fn test_put_object_if_match_rejects_stale_etag() {
        let client = create_test_client("test-if-match").await;
        let key = format!("if-match-{}", uuid::Uuid::new_v4());

        // Write initial object
        client
            .put_object(&key, Bytes::from("version1"))
            .await
            .unwrap();

        // Get with ETag
        let (_, etag) = client
            .get_object_with_etag(&key)
            .await
            .unwrap()
            .expect("object should exist");
        let etag = etag.expect("should have etag");

        // Overwrite the object to change its ETag
        client
            .put_object(&key, Bytes::from("version2"))
            .await
            .unwrap();

        // Conditional put with stale ETag should fail
        let won = client
            .put_object_if_match(&key, Bytes::from("version3"), &etag)
            .await
            .unwrap();
        assert!(!won, "conditional put with stale ETag should return false");

        // Verify the object still has version2
        let data = client.get_object(&key).await.unwrap().unwrap();
        assert_eq!(data, Bytes::from("version2"));

        // Cleanup
        client.delete_object(&key).await.unwrap();
    }

    #[tokio::test]
    async fn test_put_object_if_match_accepts_current_etag() {
        let client = create_test_client("test-if-match-ok").await;
        let key = format!("if-match-ok-{}", uuid::Uuid::new_v4());

        client
            .put_object(&key, Bytes::from("version1"))
            .await
            .unwrap();

        let (_, etag) = client
            .get_object_with_etag(&key)
            .await
            .unwrap()
            .expect("object should exist");
        let etag = etag.expect("should have etag");

        // Conditional put with current ETag should succeed
        let won = client
            .put_object_if_match(&key, Bytes::from("version2"), &etag)
            .await
            .unwrap();
        assert!(won, "conditional put with current ETag should succeed");

        let data = client.get_object(&key).await.unwrap().unwrap();
        assert_eq!(data, Bytes::from("version2"));

        // Cleanup
        client.delete_object(&key).await.unwrap();
    }
}
