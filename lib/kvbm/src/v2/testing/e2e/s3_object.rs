// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3 Object Storage Integration Tests
//!
//! These tests verify the S3ObjectBlockClient implementation against a real
//! S3-compatible storage backend (MinIO).
//!
//! # Prerequisites
//!
//! Start MinIO locally before running these tests:
//! ```bash
//! docker-compose -f lib/kvbm/docker-compose.minio.yml up -d
//! ```
//!
//! # Running Tests
//!
//! ```bash
//! AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
//!     cargo test -p dynamo-kvbm --features s3 -- s3_object --test-threads=1
//! ```

use std::collections::HashMap;

use anyhow::Result;
use rstest::rstest;

use crate::v2::distributed::object::LayoutConfigExt;
use crate::v2::distributed::object::s3::{S3Config, S3ObjectBlockClient};
use crate::v2::physical::layout::{BlockDimension, PhysicalLayout};
use crate::v2::physical::transfer::{
    BlockChecksum, FillPattern, NixlAgent, compute_block_checksums, fill_blocks,
};
use crate::v2::testing::physical::{LayoutKind, standard_config};
use crate::{BlockId, KvbmSequenceHashProvider, SequenceHash};
use dynamo_tokens::TokenBlockSequence;

/// Generate unique test bucket name to avoid collisions.
fn test_bucket_name() -> String {
    format!("kvbm-test-{}", uuid::Uuid::new_v4())
}

/// Create a test NIXL agent.
fn create_test_agent(name: &str) -> NixlAgent {
    NixlAgent::new(name).expect("Failed to create NIXL agent")
}

/// Create a fully contiguous layout with System memory.
fn create_fc_system_layout(agent: NixlAgent, num_blocks: usize) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .fully_contiguous()
        .allocate_system()
        .build()
        .expect("Failed to create FC layout")
}

/// Create a layer-separate layout with System memory.
fn create_lw_system_layout(agent: NixlAgent, num_blocks: usize) -> PhysicalLayout {
    let config = standard_config(num_blocks);
    PhysicalLayout::builder(agent)
        .with_config(config)
        .layer_separate(BlockDimension::BlockIsFirstDim)
        .allocate_system()
        .build()
        .expect("Failed to create LW layout")
}

/// Create a layout based on kind.
fn create_layout(agent: NixlAgent, kind: LayoutKind, num_blocks: usize) -> PhysicalLayout {
    match kind {
        LayoutKind::FC => create_fc_system_layout(agent, num_blocks),
        LayoutKind::LW => create_lw_system_layout(agent, num_blocks),
    }
}

/// Generate test sequence hashes for the given number of blocks.
fn generate_test_hashes(count: usize, seed: usize) -> Vec<SequenceHash> {
    let tokens_per_block = 16;
    let total_tokens = count * tokens_per_block;
    let tokens: Vec<u32> = (0..total_tokens as u32)
        .map(|t| t.wrapping_add(seed as u32 * 1000))
        .collect();
    let seq = TokenBlockSequence::from_slice(&tokens, tokens_per_block as u32, Some(seed as u64));
    seq.blocks()
        .iter()
        .take(count)
        .map(|b| b.kvbm_sequence_hash())
        .collect()
}

/// Fill blocks and compute checksums.
fn fill_and_checksum(
    layout: &PhysicalLayout,
    block_ids: &[BlockId],
    pattern: FillPattern,
) -> Result<HashMap<BlockId, BlockChecksum>> {
    fill_blocks(layout, block_ids, pattern)?;
    compute_block_checksums(layout, block_ids)
}

/// Check if MinIO is available by attempting to connect.
async fn is_minio_available() -> bool {
    // Try to connect to MinIO
    let config = S3Config::default();
    match S3ObjectBlockClient::new(config).await {
        Ok(client) => client.client().list_buckets().send().await.is_ok(),
        Err(_) => false,
    }
}

/// RAII wrapper for S3 test client that cleans up the bucket on drop.
///
/// This ensures test buckets are deleted after each test, preventing
/// accumulation of test data in MinIO.
pub struct TestS3Client {
    inner: S3ObjectBlockClient,
    bucket: String,
    /// Runtime handle for async cleanup in Drop
    runtime: tokio::runtime::Handle,
}

#[allow(dead_code)]
impl TestS3Client {
    /// Create a new test client with a unique bucket name.
    ///
    /// The bucket is created automatically. It will be deleted when this
    /// client is dropped.
    pub async fn new() -> Result<Self> {
        let bucket = test_bucket_name();
        Self::with_bucket(bucket).await
    }

    /// Create a new test client with a specific bucket name.
    pub async fn with_bucket(bucket: String) -> Result<Self> {
        let config = S3Config {
            bucket: bucket.clone(),
            ..S3Config::default()
        };

        let inner = S3ObjectBlockClient::new(config).await?;
        inner.ensure_bucket_exists().await?;

        let runtime = tokio::runtime::Handle::current();

        Ok(Self {
            inner,
            bucket,
            runtime,
        })
    }

    /// Create a new test client with custom config.
    pub async fn with_config(mut config: S3Config) -> Result<Self> {
        let bucket = config.bucket.clone();
        if bucket == "kvbm-blocks" {
            // Replace default bucket with unique test bucket
            config.bucket = test_bucket_name();
        }

        let inner = S3ObjectBlockClient::new(config.clone()).await?;
        inner.ensure_bucket_exists().await?;

        let runtime = tokio::runtime::Handle::current();

        Ok(Self {
            inner,
            bucket: config.bucket,
            runtime,
        })
    }

    /// Get reference to the underlying S3 client.
    pub fn client(&self) -> &S3ObjectBlockClient {
        &self.inner
    }

    /// Get the bucket name.
    pub fn bucket(&self) -> &str {
        &self.bucket
    }

    /// Check if blocks exist in S3.
    pub async fn has_blocks(&self, keys: &[SequenceHash]) -> Vec<(SequenceHash, Option<usize>)> {
        use crate::v2::distributed::object::ObjectBlockOps;
        self.inner.has_blocks(keys.to_vec()).await
    }

    /// Put blocks to S3 using physical layout.
    pub async fn put_blocks(
        &self,
        keys: &[SequenceHash],
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
    ) -> Vec<Result<SequenceHash, SequenceHash>> {
        self.inner
            .put_blocks_with_layout(keys.to_vec(), layout.clone(), block_ids.to_vec())
            .await
    }

    /// Get blocks from S3 into physical layout.
    pub async fn get_blocks(
        &self,
        keys: &[SequenceHash],
        layout: &PhysicalLayout,
        block_ids: &[BlockId],
    ) -> Vec<Result<SequenceHash, SequenceHash>> {
        self.inner
            .get_blocks_with_layout(keys.to_vec(), layout.clone(), block_ids.to_vec())
            .await
    }

    /// Delete all objects in the bucket.
    async fn delete_all_objects(&self) -> Result<()> {
        let s3_client = self.inner.client();

        // List and delete all objects
        let mut continuation_token: Option<String> = None;

        loop {
            let mut list_req = s3_client.list_objects_v2().bucket(&self.bucket);

            if let Some(token) = continuation_token.take() {
                list_req = list_req.continuation_token(token);
            }

            let list_result = list_req.send().await?;

            for object in list_result.contents() {
                if let Some(key) = object.key() {
                    s3_client
                        .delete_object()
                        .bucket(&self.bucket)
                        .key(key)
                        .send()
                        .await
                        .ok(); // Ignore individual delete errors
                }
            }

            if list_result.is_truncated() == Some(true) {
                continuation_token = list_result.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Delete the bucket and all its contents.
    async fn cleanup(&self) -> Result<()> {
        // First delete all objects
        self.delete_all_objects().await.ok();

        // Then delete the bucket
        self.inner
            .client()
            .delete_bucket()
            .bucket(&self.bucket)
            .send()
            .await
            .ok(); // Ignore errors (bucket might not exist)

        Ok(())
    }
}

impl Drop for TestS3Client {
    fn drop(&mut self) {
        let bucket = self.bucket.clone();
        let client = self.inner.client().clone();

        // Use spawn_blocking to run async cleanup in drop
        self.runtime.spawn(async move {
            // Delete all objects first
            let mut continuation_token: Option<String> = None;

            loop {
                let mut list_req = client.list_objects_v2().bucket(&bucket);

                if let Some(token) = continuation_token.take() {
                    list_req = list_req.continuation_token(token);
                }

                match list_req.send().await {
                    Ok(list_result) => {
                        for object in list_result.contents() {
                            if let Some(key) = object.key() {
                                client
                                    .delete_object()
                                    .bucket(&bucket)
                                    .key(key)
                                    .send()
                                    .await
                                    .ok();
                            }
                        }

                        if list_result.is_truncated() == Some(true) {
                            continuation_token =
                                list_result.next_continuation_token().map(|s| s.to_string());
                        } else {
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }

            // Delete the bucket
            client.delete_bucket().bucket(&bucket).send().await.ok();
        });
    }
}

impl std::ops::Deref for TestS3Client {
    type Target = S3ObjectBlockClient;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Skip test if MinIO is not available.
macro_rules! skip_if_no_minio {
    () => {
        if !is_minio_available().await {
            eprintln!(
                "Skipping test '{}': MinIO not available. \
                 Start with: docker-compose -f lib/kvbm/docker-compose.minio.yml up -d",
                module_path!()
            );
            return Ok(());
        }
    };
}

// =============================================================================
// Basic S3 Client Tests
// =============================================================================

/// Test S3 client creation and bucket operations.
#[tokio::test]
async fn test_s3_client_creation() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;
    println!(
        "✓ S3 client created and bucket '{}' exists",
        test_client.bucket()
    );
    // Bucket will be cleaned up on drop
    Ok(())
}

/// Test has_blocks returns None for non-existent blocks.
#[tokio::test]
async fn test_has_blocks_not_found() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    let hashes = generate_test_hashes(3, 12345);
    let results = test_client.has_blocks(&hashes).await;

    assert_eq!(results.len(), 3);
    for (hash, size) in &results {
        assert!(size.is_none(), "Block {:?} should not exist", hash);
    }

    println!("✓ has_blocks correctly returns None for non-existent blocks");
    Ok(())
}

// =============================================================================
// Put/Get Round-Trip Tests
// =============================================================================

/// Test put_blocks followed by has_blocks.
#[rstest]
#[case(LayoutKind::FC, "fc")]
#[case(LayoutKind::LW, "lw")]
#[tokio::test]
async fn test_put_blocks_then_has_blocks(
    #[case] layout_kind: LayoutKind,
    #[case] suffix: &str,
) -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    // Create layout and fill with data
    let agent = create_test_agent(&format!("test_put_has_{}", suffix));
    let layout = create_layout(agent, layout_kind, 4);
    let block_ids: Vec<BlockId> = vec![0, 1];
    let hashes = generate_test_hashes(block_ids.len(), 100);

    fill_blocks(&layout, &block_ids, FillPattern::Sequential)?;

    // Put blocks to S3
    let put_results = test_client.put_blocks(&hashes, &layout, &block_ids).await;
    assert!(
        put_results.iter().all(|r| r.is_ok()),
        "All puts should succeed"
    );
    println!("✓ Put {} {:?} blocks to S3", block_ids.len(), layout_kind);

    // Verify blocks exist with correct size
    let has_results = test_client.has_blocks(&hashes).await;
    let expected_size = layout.layout().config().block_size_bytes();

    for (hash, size) in &has_results {
        assert!(size.is_some(), "Block {:?} should exist", hash);
        assert_eq!(
            size.unwrap(),
            expected_size,
            "Block size should match layout"
        );
    }

    println!(
        "✓ has_blocks returns correct size ({} bytes) for {:?} layout",
        expected_size, layout_kind
    );
    Ok(())
}

/// Test full round-trip: put blocks, get blocks, verify checksums.
#[rstest]
#[case(LayoutKind::FC, "fc")]
#[case(LayoutKind::LW, "lw")]
#[tokio::test]
async fn test_put_get_roundtrip(
    #[case] layout_kind: LayoutKind,
    #[case] suffix: &str,
) -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    // Create source layout and fill with sequential data
    let agent_src = create_test_agent(&format!("test_roundtrip_src_{}", suffix));
    let src_layout = create_layout(agent_src, layout_kind, 4);
    let src_block_ids: Vec<BlockId> = vec![0, 1, 2];
    let hashes = generate_test_hashes(src_block_ids.len(), 200);

    let src_checksums = fill_and_checksum(&src_layout, &src_block_ids, FillPattern::Sequential)?;
    println!(
        "✓ Filled {} source blocks with sequential pattern",
        src_block_ids.len()
    );

    // Put blocks to S3
    let put_results = test_client
        .put_blocks(&hashes, &src_layout, &src_block_ids)
        .await;
    assert!(
        put_results.iter().all(|r| r.is_ok()),
        "All puts should succeed"
    );
    println!("✓ Put blocks to S3");

    // Create destination layout (same config but different memory)
    let agent_dst = create_test_agent(&format!("test_roundtrip_dst_{}", suffix));
    let dst_layout = create_layout(agent_dst, layout_kind, 4);
    let dst_block_ids: Vec<BlockId> = vec![1, 2, 3]; // Different block IDs

    // Get blocks from S3
    let get_results = test_client
        .get_blocks(&hashes, &dst_layout, &dst_block_ids)
        .await;
    assert!(
        get_results.iter().all(|r| r.is_ok()),
        "All gets should succeed"
    );
    println!("✓ Got blocks from S3");

    // Verify checksums match
    let dst_checksums = compute_block_checksums(&dst_layout, &dst_block_ids)?;

    for ((&src_id, &dst_id), hash) in src_block_ids
        .iter()
        .zip(dst_block_ids.iter())
        .zip(hashes.iter())
    {
        let src_checksum = src_checksums.get(&src_id).expect("src checksum");
        let dst_checksum = dst_checksums.get(&dst_id).expect("dst checksum");
        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch for hash {:?}: src[{}]={} != dst[{}]={}",
            hash, src_id, src_checksum, dst_id, dst_checksum
        );
    }

    println!(
        "✓ Round-trip verified: {} {:?} blocks match",
        src_block_ids.len(),
        layout_kind
    );
    Ok(())
}

/// Test cross-layout round-trip: FC source -> S3 -> LW destination.
#[tokio::test]
async fn test_cross_layout_roundtrip() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    // Create FC source layout
    let agent_src = create_test_agent("test_cross_src");
    let src_layout = create_fc_system_layout(agent_src, 4);
    let src_block_ids: Vec<BlockId> = vec![0, 1];
    let hashes = generate_test_hashes(src_block_ids.len(), 300);

    let src_checksums = fill_and_checksum(&src_layout, &src_block_ids, FillPattern::Sequential)?;

    // Put FC blocks to S3
    let put_results = test_client
        .put_blocks(&hashes, &src_layout, &src_block_ids)
        .await;
    assert!(
        put_results.iter().all(|r| r.is_ok()),
        "All puts should succeed"
    );
    println!("✓ Put FC blocks to S3");

    // Create LW destination layout
    let agent_dst = create_test_agent("test_cross_dst");
    let dst_layout = create_lw_system_layout(agent_dst, 4);
    let dst_block_ids: Vec<BlockId> = vec![2, 3];

    // Get blocks from S3 into LW layout
    let get_results = test_client
        .get_blocks(&hashes, &dst_layout, &dst_block_ids)
        .await;
    assert!(
        get_results.iter().all(|r| r.is_ok()),
        "All gets should succeed"
    );
    println!("✓ Got blocks into LW layout");

    // Verify checksums match
    let dst_checksums = compute_block_checksums(&dst_layout, &dst_block_ids)?;

    for (&src_id, &dst_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_checksum = src_checksums.get(&src_id).expect("src checksum");
        let dst_checksum = dst_checksums.get(&dst_id).expect("dst checksum");
        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch: FC[{}] != LW[{}]",
            src_id, dst_id
        );
    }

    println!("✓ Cross-layout round-trip verified: FC -> S3 -> LW");
    Ok(())
}

// =============================================================================
// Concurrency and Parallelism Tests
// =============================================================================

/// Test parallel put of many blocks.
#[rstest]
#[case(LayoutKind::FC, 16, "fc_16")]
#[case(LayoutKind::LW, 16, "lw_16")]
#[case(LayoutKind::FC, 64, "fc_64")]
#[tokio::test]
async fn test_parallel_put_blocks(
    #[case] layout_kind: LayoutKind,
    #[case] num_blocks: usize,
    #[case] suffix: &str,
) -> Result<()> {
    skip_if_no_minio!();

    let config = S3Config::default().with_max_concurrent_requests(8);
    let test_client = TestS3Client::with_config(config).await?;

    let agent = create_test_agent(&format!("test_parallel_{}", suffix));
    let layout = create_layout(agent, layout_kind, num_blocks);
    let block_ids: Vec<BlockId> = (0..num_blocks).collect();
    let hashes = generate_test_hashes(num_blocks, 400);

    fill_blocks(&layout, &block_ids, FillPattern::Sequential)?;

    let start = std::time::Instant::now();
    let put_results = test_client.put_blocks(&hashes, &layout, &block_ids).await;
    let elapsed = start.elapsed();

    let success_count = put_results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(success_count, num_blocks, "All puts should succeed");

    println!(
        "✓ Parallel put {} {:?} blocks in {:?} ({:.1} blocks/sec)",
        num_blocks,
        layout_kind,
        elapsed,
        num_blocks as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

/// Test parallel get of many blocks.
#[rstest]
#[case(LayoutKind::FC, 16, "fc_16")]
#[case(LayoutKind::LW, 16, "lw_16")]
#[tokio::test]
async fn test_parallel_get_blocks(
    #[case] layout_kind: LayoutKind,
    #[case] num_blocks: usize,
    #[case] suffix: &str,
) -> Result<()> {
    skip_if_no_minio!();

    let config = S3Config::default().with_max_concurrent_requests(8);
    let test_client = TestS3Client::with_config(config).await?;

    // First, put blocks
    let agent_src = create_test_agent(&format!("test_par_get_src_{}", suffix));
    let src_layout = create_layout(agent_src, layout_kind, num_blocks);
    let block_ids: Vec<BlockId> = (0..num_blocks).collect();
    let hashes = generate_test_hashes(num_blocks, 500);

    let src_checksums = fill_and_checksum(&src_layout, &block_ids, FillPattern::Sequential)?;
    let _ = test_client
        .put_blocks(&hashes, &src_layout, &block_ids)
        .await;

    // Now test parallel get
    let agent_dst = create_test_agent(&format!("test_par_get_dst_{}", suffix));
    let dst_layout = create_layout(agent_dst, layout_kind, num_blocks);

    let start = std::time::Instant::now();
    let get_results = test_client
        .get_blocks(&hashes, &dst_layout, &block_ids)
        .await;
    let elapsed = start.elapsed();

    let success_count = get_results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(success_count, num_blocks, "All gets should succeed");

    // Verify all checksums
    let dst_checksums = compute_block_checksums(&dst_layout, &block_ids)?;
    for &block_id in &block_ids {
        assert_eq!(
            src_checksums.get(&block_id),
            dst_checksums.get(&block_id),
            "Checksum mismatch for block {}",
            block_id
        );
    }

    println!(
        "✓ Parallel get {} {:?} blocks in {:?} ({:.1} blocks/sec), checksums verified",
        num_blocks,
        layout_kind,
        elapsed,
        num_blocks as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/// Test get_blocks returns Err for non-existent blocks.
#[tokio::test]
async fn test_get_nonexistent_blocks() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    let agent = create_test_agent("test_get_nonexistent");
    let layout = create_fc_system_layout(agent, 4);
    let block_ids: Vec<BlockId> = vec![0, 1, 2];
    let hashes = generate_test_hashes(block_ids.len(), 999);

    let get_results = test_client.get_blocks(&hashes, &layout, &block_ids).await;

    // All should fail (return Err variant with hash)
    for result in &get_results {
        assert!(result.is_err(), "Get of non-existent block should fail");
    }

    println!(
        "✓ get_blocks correctly returns Err for {} non-existent blocks",
        block_ids.len()
    );
    Ok(())
}

/// Test partial success: some blocks exist, some don't.
#[tokio::test]
async fn test_partial_get_success() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    let agent = create_test_agent("test_partial");
    let layout = create_fc_system_layout(agent, 4);

    // Put only blocks 0 and 1
    let put_block_ids: Vec<BlockId> = vec![0, 1];
    let put_hashes = generate_test_hashes(2, 600);
    fill_blocks(&layout, &put_block_ids, FillPattern::Sequential)?;
    let _ = test_client
        .put_blocks(&put_hashes, &layout, &put_block_ids)
        .await;

    // Try to get blocks 0, 1, 2, 3 (2 and 3 don't exist)
    let all_block_ids: Vec<BlockId> = vec![0, 1, 2, 3];
    let mut all_hashes = put_hashes.clone();
    all_hashes.extend(generate_test_hashes(2, 601)); // Non-existent hashes

    let get_results = test_client
        .get_blocks(&all_hashes, &layout, &all_block_ids)
        .await;

    let success_count = get_results.iter().filter(|r| r.is_ok()).count();
    let failure_count = get_results.iter().filter(|r| r.is_err()).count();

    assert_eq!(success_count, 2, "2 blocks should succeed");
    assert_eq!(failure_count, 2, "2 blocks should fail");

    println!(
        "✓ Partial get: {} succeeded, {} failed as expected",
        success_count, failure_count
    );
    Ok(())
}

// =============================================================================
// G4 Search Integration Tests
// =============================================================================

/// Test G4 search: blocks pre-uploaded to S3 can be discovered via has_blocks.
///
/// This simulates the G4 search flow:
/// 1. Upload blocks to S3 (simulating offloaded G4 data)
/// 2. has_blocks discovers them
/// 3. Verify size metadata is correct
#[tokio::test]
async fn test_g4_search_finds_offloaded_blocks() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    // Step 1: Upload blocks to S3 (simulating G4 offload)
    let agent = create_test_agent("g4_search_src");
    let layout = create_fc_system_layout(agent, 8);
    let block_ids: Vec<BlockId> = (0..4).collect();
    let hashes = generate_test_hashes(4, 700);

    fill_blocks(&layout, &block_ids, FillPattern::Sequential)?;
    let put_results = test_client.put_blocks(&hashes, &layout, &block_ids).await;
    assert!(
        put_results.iter().all(|r| r.is_ok()),
        "Offload should succeed"
    );
    println!(
        "✓ Pre-uploaded {} blocks to S3 (simulating G4)",
        block_ids.len()
    );

    // Step 2: G4 search via has_blocks
    let search_results = test_client.has_blocks(&hashes).await;

    // Step 3: Verify all blocks found
    let expected_size = layout.layout().config().block_size_bytes();
    let found_count = search_results.iter().filter(|(_, s)| s.is_some()).count();
    assert_eq!(found_count, 4, "G4 search should find all 4 blocks");

    for (hash, size_opt) in &search_results {
        assert!(size_opt.is_some(), "Block {:?} should exist in G4", hash);
        assert_eq!(size_opt.unwrap(), expected_size, "Block size should match");
    }

    println!(
        "✓ G4 search found all {} blocks with correct size ({} bytes)",
        found_count, expected_size
    );
    Ok(())
}

/// Test G4 search with mixed results: some blocks in S3, some not.
///
/// This tests the race scenario where G4 might only have some blocks.
#[tokio::test]
async fn test_g4_search_partial_results() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    // Upload only blocks 0-2 to S3
    let agent = create_test_agent("g4_partial_src");
    let layout = create_fc_system_layout(agent, 8);
    let uploaded_block_ids: Vec<BlockId> = (0..3).collect();
    let uploaded_hashes = generate_test_hashes(3, 710);

    fill_blocks(&layout, &uploaded_block_ids, FillPattern::Sequential)?;
    let _ = test_client
        .put_blocks(&uploaded_hashes, &layout, &uploaded_block_ids)
        .await;
    println!(
        "✓ Uploaded {} blocks (simulating partial G4)",
        uploaded_block_ids.len()
    );

    // Search for blocks 0-5 (0-2 exist, 3-5 don't)
    let mut search_hashes = uploaded_hashes.clone();
    search_hashes.extend(generate_test_hashes(3, 711)); // Non-existent blocks

    let search_results = test_client.has_blocks(&search_hashes).await;

    let found_count = search_results.iter().filter(|(_, s)| s.is_some()).count();
    let missing_count = search_results.iter().filter(|(_, s)| s.is_none()).count();

    assert_eq!(found_count, 3, "Should find 3 blocks that exist");
    assert_eq!(
        missing_count, 3,
        "Should not find 3 blocks that don't exist"
    );

    println!(
        "✓ G4 search correctly identified {} found, {} missing",
        found_count, missing_count
    );
    Ok(())
}

/// Test G4 load: get_blocks retrieves data correctly.
///
/// This simulates the G4 load flow after search:
/// 1. Upload blocks to S3
/// 2. Allocate destination blocks (different layout)
/// 3. get_blocks downloads into destination
/// 4. Verify data integrity via checksums
#[tokio::test]
async fn test_g4_load_downloads_blocks() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    // Step 1: Upload blocks to S3
    let agent_src = create_test_agent("g4_load_src");
    let src_layout = create_fc_system_layout(agent_src, 8);
    let src_block_ids: Vec<BlockId> = (0..4).collect();
    let hashes = generate_test_hashes(4, 720);

    let src_checksums = fill_and_checksum(&src_layout, &src_block_ids, FillPattern::Sequential)?;
    let _ = test_client
        .put_blocks(&hashes, &src_layout, &src_block_ids)
        .await;
    println!("✓ Uploaded {} blocks to G4", src_block_ids.len());

    // Step 2: Allocate destination blocks (simulating G2 allocation)
    let agent_dst = create_test_agent("g4_load_dst");
    let dst_layout = create_fc_system_layout(agent_dst, 8);
    let dst_block_ids: Vec<BlockId> = (4..8).collect(); // Different block IDs

    // Step 3: Download via get_blocks
    let get_results = test_client
        .get_blocks(&hashes, &dst_layout, &dst_block_ids)
        .await;

    let success_count = get_results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(success_count, 4, "All G4 loads should succeed");
    println!("✓ Downloaded {} blocks from G4", success_count);

    // Step 4: Verify checksums
    let dst_checksums = compute_block_checksums(&dst_layout, &dst_block_ids)?;

    for ((&src_id, &dst_id), _hash) in src_block_ids
        .iter()
        .zip(dst_block_ids.iter())
        .zip(hashes.iter())
    {
        let src_checksum = src_checksums.get(&src_id).expect("src checksum");
        let dst_checksum = dst_checksums.get(&dst_id).expect("dst checksum");
        assert_eq!(
            src_checksum, dst_checksum,
            "Checksum mismatch: src[{}] != dst[{}]",
            src_id, dst_id
        );
    }

    println!(
        "✓ G4 load verified: all {} blocks have matching checksums",
        success_count
    );
    Ok(())
}

/// Test G4 load with per-block failures.
///
/// This tests error handling when some blocks fail to load from G4.
#[tokio::test]
async fn test_g4_load_partial_failure() -> Result<()> {
    skip_if_no_minio!();

    let test_client = TestS3Client::new().await?;

    // Upload only blocks 0, 2 (skip 1, 3)
    let agent_src = create_test_agent("g4_fail_src");
    let src_layout = create_fc_system_layout(agent_src, 8);

    // Upload block 0 and 2 only
    let uploaded_ids: Vec<BlockId> = vec![0, 2];
    let uploaded_hashes: Vec<SequenceHash> = generate_test_hashes(2, 730);

    fill_blocks(&src_layout, &uploaded_ids, FillPattern::Sequential)?;
    let _ = test_client
        .put_blocks(&uploaded_hashes, &src_layout, &uploaded_ids)
        .await;
    println!("✓ Uploaded 2 blocks (0, 2) - blocks 1, 3 don't exist");

    // Try to load all 4 blocks (0, 1, 2, 3 - but 1 and 3 don't exist)
    let mut all_hashes = vec![uploaded_hashes[0]]; // Block 0 exists
    all_hashes.push(generate_test_hashes(1, 731)[0]); // Block 1 doesn't exist
    all_hashes.push(uploaded_hashes[1]); // Block 2 exists
    all_hashes.push(generate_test_hashes(1, 732)[0]); // Block 3 doesn't exist

    let agent_dst = create_test_agent("g4_fail_dst");
    let dst_layout = create_fc_system_layout(agent_dst, 8);
    let dst_block_ids: Vec<BlockId> = vec![4, 5, 6, 7];

    let get_results = test_client
        .get_blocks(&all_hashes, &dst_layout, &dst_block_ids)
        .await;

    let success_count = get_results.iter().filter(|r| r.is_ok()).count();
    let failure_count = get_results.iter().filter(|r| r.is_err()).count();

    assert_eq!(success_count, 2, "2 blocks should succeed");
    assert_eq!(failure_count, 2, "2 blocks should fail");

    // Verify the failures are for the correct blocks (indices 1 and 3)
    assert!(get_results[0].is_ok(), "Block 0 should succeed");
    assert!(get_results[1].is_err(), "Block 1 should fail");
    assert!(get_results[2].is_ok(), "Block 2 should succeed");
    assert!(get_results[3].is_err(), "Block 3 should fail");

    println!(
        "✓ G4 load partial failure: {} succeeded, {} failed as expected",
        success_count, failure_count
    );
    Ok(())
}
