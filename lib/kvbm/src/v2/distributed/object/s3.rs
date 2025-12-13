// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3-compatible object storage client for block management.
//!
//! This module provides an implementation of [`ObjectBlockClient`] using the AWS S3 SDK.
//! It supports S3-compatible storage services including MinIO.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use bytes::Bytes;
use futures::stream::StreamExt;

use super::{LayoutConfigExt, ObjectBlockClient};
use crate::v2::physical::transfer::PhysicalLayout;
use crate::{BlockId, SequenceHash};

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

    /// Set the maximum number of concurrent requests.
    pub fn with_max_concurrent_requests(mut self, max: usize) -> Self {
        self.max_concurrent_requests = max;
        self
    }
}

/// S3-compatible object storage client for block operations.
///
/// This client implements [`ObjectBlockClient`] using the AWS S3 SDK.
/// It supports parallel block operations and uses rayon for CPU-bound memory copies.
pub struct S3ObjectBlockClient {
    /// AWS S3 client
    client: Client,

    /// S3 configuration
    config: S3Config,
}

impl S3ObjectBlockClient {
    /// Create a new S3ObjectBlockClient.
    ///
    /// # Arguments
    /// * `config` - S3 configuration
    ///
    /// # Errors
    /// Returns an error if the S3 client cannot be initialized.
    pub async fn new(config: S3Config) -> Result<Self> {
        let client = build_s3_client(&config).await?;
        Ok(Self { client, config })
    }

    /// Create from an existing AWS S3 client.
    pub fn from_client(client: Client, config: S3Config) -> Self {
        Self { client, config }
    }

    /// Get a reference to the S3 client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &S3Config {
        &self.config
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

/// Convert a SequenceHash to an S3 object key.
fn seq_hash_to_key(hash: &SequenceHash) -> String {
    // Use debug representation for now - could be optimized to hex encoding
    format!("{:?}", hash)
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

#[async_trait]
impl ObjectBlockClient for S3ObjectBlockClient {
    async fn has_blocks(&self, keys: &[SequenceHash]) -> Vec<(SequenceHash, Option<usize>)> {
        // Copy keys to owned vec to avoid lifetime issues with async closures
        let keys_owned: Vec<_> = keys.to_vec();
        let max_concurrent = self.config.max_concurrent_requests;
        let client = self.client.clone();
        let bucket = self.config.bucket.clone();

        let tasks = keys_owned.into_iter().map(move |key| {
            let client = client.clone();
            let bucket = bucket.clone();
            let key_str = seq_hash_to_key(&key);

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
    }

    async fn put_blocks(
        &self,
        keys: &[SequenceHash],
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
    ) -> Vec<Result<SequenceHash, SequenceHash>> {
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        let is_contiguous = layout.layout().is_fully_contiguous();
        let max_concurrent = self.config.max_concurrent_requests;
        let client = self.client.clone();
        let bucket = self.config.bucket.clone();

        // Zip and collect to owned values to avoid lifetime issues
        let work_items: Vec<_> = keys
            .iter()
            .copied()
            .zip(block_ids.iter().copied())
            .collect();

        let tasks = work_items.into_iter().map(move |(key, block_id)| {
            let client = client.clone();
            let bucket = bucket.clone();
            let key_str = seq_hash_to_key(&key);
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
    }

    async fn get_blocks(
        &self,
        keys: &[SequenceHash],
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
    ) -> Vec<Result<SequenceHash, SequenceHash>> {
        let config = layout.layout().config();
        let block_size = config.block_size_bytes();
        let region_size = config.region_size();
        let is_contiguous = layout.layout().is_fully_contiguous();
        let max_concurrent = self.config.max_concurrent_requests;
        let client = self.client.clone();
        let bucket = self.config.bucket.clone();

        // Zip and collect to owned values to avoid lifetime issues
        let work_items: Vec<_> = keys
            .iter()
            .copied()
            .zip(block_ids.iter().copied())
            .collect();

        let tasks = work_items.into_iter().map(move |(key, block_id)| {
            let client = client.clone();
            let bucket = bucket.clone();
            let key_str = seq_hash_to_key(&key);
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
