// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G4 (Object Storage) integration testing utilities.
//!
//! This module provides test infrastructure for G4 transfers using DirectWorker
//! with `RemoteDescriptor::Object` for object storage operations.
//!
//! # Usage Pattern
//!
//! ```ignore
//! // Create worker with G4 support
//! let worker = create_direct_worker_with_g4(
//!     "test-worker",
//!     &layout_config,
//!     TEST_BUCKET,
//!     get_endpoint(),
//! )?;
//!
//! // Offload using RemoteDescriptor::Object
//! let keys = generate_object_keys(2);
//! worker.execute_remote_offload(
//!     LogicalLayoutHandle::G2,
//!     block_ids.into(),
//!     RemoteDescriptor::Object { keys: keys.clone() },
//!     TransferOptions::default(),
//! )?;
//!
//! // Onboard back
//! worker.execute_remote_onboard(
//!     RemoteDescriptor::Object { keys },
//!     LogicalLayoutHandle::G2,
//!     dst_block_ids.into(),
//!     TransferOptions::default(),
//! )?;
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;

use crate::{
    physical::manager::{LayoutHandle, TransferManager},
    v2::{
        distributed::worker::{DirectWorker, Worker},
        physical::layout::{LayoutConfig, PhysicalLayout},
    },
};
use dynamo_memory::{MemoryDescriptor, StorageKind, nixl::NixlAgent};
use dynamo_tokens::PositionalSequenceHash;
use super::physical;

// ============================================================================
// Configuration
// ============================================================================

/// Default bucket name for G4 tests.
pub const DEFAULT_G4_BUCKET: &str = "demo";

/// Get the Object endpoint from environment or use default.
pub fn get_endpoint() -> String {
    std::env::var("DYN_KVBM_OBJ_ENDPOINT").unwrap_or_else(|_| "http://localhost:9000".to_string())
}

/// Get the Object access key from environment or use default.
pub fn get_access_key() -> String {
    std::env::var("DYN_KVBM_OBJ_ACCESS_KEY").expect("DYN_KVBM_OBJ_ACCESS_KEY must be set")
}

/// Get the Object secret key from environment or use default.
pub fn get_secret_key() -> String {
    std::env::var("DYN_KVBM_OBJ_SECRET_KEY").expect("DYN_KVBM_OBJ_SECRET_KEY must be set")
}

// ============================================================================
// Test Helpers
// ============================================================================

/// Container for a test worker with G4 support.
pub struct TestWorkerWithG4 {
    pub worker: Arc<DirectWorker>,
    pub manager: Arc<TransferManager>,
    pub g2_handle: LayoutHandle,
    pub g4_handle: LayoutHandle,
    pub agent: NixlAgent,
}

/// Create a NIXL agent with OBJ backend configured for Object.
pub fn create_obj_agent(name: &str, bucket: &str, endpoint: &str) -> Result<NixlAgent> {
    let mut agent = NixlAgent::new(name)?;

    // Configure OBJ backend with Object parameters
    let mut obj_params = HashMap::new();
    obj_params.insert("endpoint_override".to_string(), endpoint.to_string());
    obj_params.insert("bucket".to_string(), bucket.to_string());
    obj_params.insert("access_key".to_string(), get_access_key());
    obj_params.insert("secret_key".to_string(), get_secret_key());
    obj_params.insert("region".to_string(), "us-east-1".to_string());
    obj_params.insert("scheme".to_string(), "http".to_string());
    obj_params.insert("use_virtual_addressing".to_string(), "false".to_string());

    agent.add_backend_with_params("OBJ", Some(&obj_params))?;

    tracing::info!(
        "Created OBJ agent '{}' with endpoint={}, bucket={}",
        name,
        endpoint,
        bucket
    );

    Ok(agent)
}

/// Create a DirectWorker with both G2 and G4 layouts registered.
///
/// This follows the same pattern as `create_direct_worker` in `distributed.rs`
/// but adds G4 (object storage) support.
///
/// # Arguments
/// * `agent_name` - Name for the NIXL agent
/// * `layout_config` - Configuration for G2 layout (block size, layers, etc.)
/// * `bucket` - Object bucket name for G4
/// * `endpoint` - Object endpoint URL
/// * `num_g4_blocks` - Number of blocks to pre-allocate in G4 layout
///
/// # Returns
/// TestWorkerWithG4 containing the worker, manager, and handles
pub fn create_direct_worker_with_g4(
    agent_name: &str,
    layout_config: &LayoutConfig,
    bucket: &str,
    endpoint: &str,
    num_g4_blocks: usize,
) -> Result<TestWorkerWithG4> {
    // Create NixlAgent with OBJ backend
    let agent = create_obj_agent(agent_name, bucket, endpoint)?;

    // Create event system (use 0 for test worker_id)
    let event_system = dynamo_nova::events::LocalEventSystem::new(0);

    // Create TransferManager
    let manager = TransferManager::builder()
        .event_system(event_system)
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;

    // Create and register G2 (pinned memory) layout
    let g2_layout =
        physical::create_fc_layout_with_config(agent.clone(), StorageKind::Pinned, layout_config.clone());
    let g2_handle = manager.register_layout(g2_layout)?;

    // Create and register G4 (object storage) layout
    // Pre-allocate blocks with placeholder keys - real keys come from RemoteDescriptor::Object
    let placeholder_keys: Vec<u128> = (0..num_g4_blocks as u128).collect();
    let g4_layout = PhysicalLayout::builder(agent.clone())
        .with_config(layout_config.clone())
        .object_layout()
        .allocate_objects(bucket.to_string(), placeholder_keys)?
        .build()?;
    let g4_handle = manager.register_layout(g4_layout)?;

    // Create DirectWorker and set handles
    let direct_worker = DirectWorker::new(manager.clone());
    direct_worker.set_g2_handle(g2_handle)?;
    direct_worker.set_g4_handle(g4_handle)?;

    Ok(TestWorkerWithG4 {
        worker: Arc::new(direct_worker),
        manager: Arc::new(manager),
        g2_handle,
        g4_handle,
        agent,
    })
}

/// Generate unique object keys as PositionalSequenceHash for testing.
///
/// Uses timestamp + index to ensure uniqueness across test runs.
/// Creates valid PositionalSequenceHash values using the constructor.
pub fn generate_object_keys(count: usize) -> Vec<PositionalSequenceHash> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    (0..count)
        .map(|i| {
            // Use timestamp as sequence_hash, index as position, and 0 as local_block_hash
            PositionalSequenceHash::new(now.wrapping_add(i as u64), i as u64, 0)
        })
        .collect()
}

/// Generate object keys as raw u128 values.
pub fn generate_raw_object_keys(count: usize) -> Vec<u128> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u128;

    (0..count).map(|i| now + i as u128).collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::{
        distributed::worker::{LogicalLayoutHandle, RemoteDescriptor, WorkerTransfers},
        physical::transfer::{FillPattern, TransferOptions},
        physical::transfer::options::{MultipartOp, ObjectOptArgs, PartEtag},
    };
    use rstest::rstest;

    // =========================================================================
    // Test Configuration Helpers
    // =========================================================================

    /// Standard layout config for tests: specified blocks, 2 layers
    fn standard_config(num_blocks: usize) -> LayoutConfig {
        physical::standard_config(num_blocks)
    }

    /// Layout config for multipart tests: 5MB+ blocks (Object minimum part size)
    ///
    /// Block size calculation:
    /// - page_size * inner_dim * dtype_width = bytes per layer
    /// - 1024 * 640 * 2 = 1,310,720 bytes per layer
    /// - Total per block = 1,310,720 * num_layers * outer_dim = 5,242,880 bytes = 5MB
    fn multipart_config(num_blocks: usize) -> LayoutConfig {
        LayoutConfig::builder()
            .num_blocks(num_blocks)
            .num_layers(2)
            .outer_dim(2)
            .page_size(1024)
            .inner_dim(640)
            .dtype_width_bytes(2)
            .build()
            .expect("multipart config should build")
    }

    // =========================================================================
    // G4 Test Context - Reduces boilerplate across all tests
    // =========================================================================

    /// Unified test context for G4 operations.
    struct G4TestContext {
        worker: TestWorkerWithG4,
        config: LayoutConfig,
    }

    impl G4TestContext {
        /// Create a new context with standard config.
        fn new(name: &str, num_blocks: usize) -> Result<Self> {
            let config = standard_config(num_blocks);
            let worker = create_direct_worker_with_g4(
                name,
                &config,
                DEFAULT_G4_BUCKET,
                &get_endpoint(),
                config.num_blocks,
            )?;
            Ok(Self { worker, config })
        }

        /// Create a new context with multipart-compatible config (5MB+ blocks).
        fn new_multipart(name: &str, num_blocks: usize) -> Result<Self> {
            let config = multipart_config(num_blocks);
            let worker = create_direct_worker_with_g4(
                name,
                &config,
                DEFAULT_G4_BUCKET,
                &get_endpoint(),
                config.num_blocks,
            )?;
            Ok(Self { worker, config })
        }

        /// Fill blocks with test data and return checksums.
        fn fill_blocks(
            &self,
            blocks: &[usize],
            pattern: FillPattern,
        ) -> Result<std::collections::HashMap<usize, crate::v2::physical::transfer::BlockChecksum>> {
            physical::fill_and_checksum_manager(
                &self.worker.manager,
                self.worker.g2_handle,
                blocks,
                pattern,
            )
        }

        /// Execute a full roundtrip: G2 -> G4 -> G2 and verify checksums.
        async fn roundtrip_and_verify(
            &self,
            src_blocks: Vec<usize>,
            dst_blocks: Vec<usize>,
        ) -> Result<()> {
            let checksums = self.fill_blocks(&src_blocks, FillPattern::Sequential)?;
            let keys = generate_object_keys(src_blocks.len());

            // Offload: G2 -> G4
            let notification = self.worker.worker.execute_remote_offload(
                LogicalLayoutHandle::G2,
                src_blocks.clone().into_iter().collect(),
                RemoteDescriptor::Object { keys: keys.clone() },
                TransferOptions::default(),
            )?;
            notification.await?;

            // Onboard: G4 -> G2
            let notification = self.worker.worker.execute_remote_onboard(
                RemoteDescriptor::Object { keys },
                LogicalLayoutHandle::G2,
                dst_blocks.clone().into_iter().collect(),
                TransferOptions::default(),
            )?;
            notification.await?;

            // Verify
            self.verify_checksums(&checksums, &src_blocks, &dst_blocks)
        }

        /// Verify checksums match between source and destination blocks.
        fn verify_checksums(
            &self,
            src_checksums: &std::collections::HashMap<usize, crate::v2::physical::transfer::BlockChecksum>,
            src_blocks: &[usize],
            dst_blocks: &[usize],
        ) -> Result<()> {
            let registry = self.worker.manager.registry().read().unwrap();
            let layout = registry
                .get_layout(self.worker.g2_handle)
                .ok_or_else(|| anyhow::anyhow!("Layout not found"))?;
            physical::verify_checksums_by_position(src_checksums, src_blocks, layout, dst_blocks)
        }

        /// Get a multipart helper for this context.
        fn multipart_helper(&self, query_block_id: usize) -> MultipartHelper<'_> {
            MultipartHelper::new(
                &self.worker.manager,
                self.worker.g2_handle,
                self.worker.g4_handle,
                query_block_id,
            )
        }

        /// Calculate block/part size in bytes.
        fn part_size(&self) -> usize {
            self.config.num_layers
                * self.config.outer_dim
                * self.config.page_size
                * self.config.inner_dim
                * self.config.dtype_width_bytes
        }
    }

    // =========================================================================
    // Multipart Upload Helper
    // =========================================================================

    /// Helper for multipart upload operations - reduces test boilerplate.
    struct MultipartHelper<'a> {
        manager: &'a TransferManager,
        g2_handle: LayoutHandle,
        g4_handle: LayoutHandle,
        query_block_id: usize,
    }

    impl<'a> MultipartHelper<'a> {
        fn new(
            manager: &'a TransferManager,
            g2_handle: LayoutHandle,
            g4_handle: LayoutHandle,
            query_block_id: usize,
        ) -> Self {
            Self { manager, g2_handle, g4_handle, query_block_id }
        }

        /// Execute any ObjectOptArgs-based transfer.
        async fn execute(&self, opts: ObjectOptArgs) -> Result<()> {
            let dummy: Arc<[usize]> = vec![0usize].into();
            let notification = self.manager.execute_transfer(
                self.g2_handle,
                &dummy,
                self.g4_handle,
                &dummy,
                TransferOptions {
                    backend_opts: Some(Box::new(opts)),
                    ..Default::default()
                },
            )?;
            notification.await?;
            Ok(())
        }

        /// Execute transfer with custom source blocks.
        async fn execute_from_blocks(&self, src_blocks: &[usize], opts: ObjectOptArgs) -> Result<()> {
            let src: Arc<[usize]> = src_blocks.to_vec().into();
            let dummy: Arc<[usize]> = vec![0usize].into();
            let notification = self.manager.execute_transfer(
                self.g2_handle,
                &src,
                self.g4_handle,
                &dummy,
                TransferOptions {
                    backend_opts: Some(Box::new(opts)),
                    ..Default::default()
                },
            )?;
            notification.await?;
            Ok(())
        }

        /// Execute query operation and read result string.
        async fn query(&self, opts: ObjectOptArgs) -> Result<String> {
            let query_block: Arc<[usize]> = vec![self.query_block_id].into();
            let dummy: Arc<[usize]> = vec![0usize].into();
            let notification = self.manager.execute_transfer(
                self.g2_handle,
                &query_block,
                self.g4_handle,
                &dummy,
                TransferOptions {
                    backend_opts: Some(Box::new(opts)),
                    ..Default::default()
                },
            )?;
            notification.await?;
            read_string_from_block(self.manager, self.g2_handle, self.query_block_id)
        }

        /// Create multipart upload and return upload_id.
        async fn create_multipart(&self, object_key: u128) -> Result<String> {
            self.execute(
                ObjectOptArgs::new()
                    .with_keys(vec![object_key])
                    .with_multipart(MultipartOp::Create, "")
            ).await?;

            self.query(
                ObjectOptArgs::new()
                    .with_keys(vec![object_key])
                    .with_multipart(MultipartOp::QueryUploadId, "")
            ).await
        }

        /// Upload a single part.
        async fn upload_part(
            &self,
            object_key: u128,
            upload_id: &str,
            src_block: usize,
            part_number: u32,
        ) -> Result<()> {
            self.execute_from_blocks(
                &[src_block],
                ObjectOptArgs::new()
                    .with_keys(vec![object_key])
                    .with_multipart(MultipartOp::UploadPart, upload_id)
                    .with_part_numbers(vec![part_number])
            ).await
        }

        /// Query ETags for an upload.
        async fn query_etags(&self, upload_id: &str) -> Result<Vec<PartEtag>> {
            let etags_str = self.query(
                ObjectOptArgs::new()
                    .with_multipart(MultipartOp::QueryEtags, upload_id)
            ).await?;
            Ok(parse_etags_string(&etags_str))
        }

        /// Complete multipart upload with ETags.
        async fn complete_multipart(
            &self,
            object_key: u128,
            upload_id: &str,
            etags: Vec<PartEtag>,
        ) -> Result<()> {
            self.execute(
                ObjectOptArgs::new()
                    .with_keys(vec![object_key])
                    .with_multipart(MultipartOp::Complete, upload_id)
                    .with_etags(etags)
            ).await
        }

        /// Full multipart upload flow: create -> upload parts -> complete.
        async fn upload_multipart(
            &self,
            object_key: u128,
            src_blocks: &[usize],
        ) -> Result<()> {
            tracing::info!("Creating multipart upload for key {}...", object_key);
            let upload_id = self.create_multipart(object_key).await?;
            tracing::info!("Got upload_id: {}...", &upload_id[..40.min(upload_id.len())]);

            for (i, &block) in src_blocks.iter().enumerate() {
                let part_num = (i + 1) as u32;
                tracing::info!("Uploading part {} from block {}...", part_num, block);
                self.upload_part(object_key, &upload_id, block, part_num).await?;
            }

            tracing::info!("Querying ETags...");
            let etags = self.query_etags(&upload_id).await?;
            tracing::info!("Got {} ETags, completing...", etags.len());

            self.complete_multipart(object_key, &upload_id, etags).await?;
            tracing::info!("Multipart upload complete!");
            Ok(())
        }
    }

    /// Read a null-terminated string from a block's memory.
    fn read_string_from_block(
        manager: &TransferManager,
        handle: LayoutHandle,
        block_id: usize,
    ) -> Result<String> {
        let registry = manager.registry().read().unwrap();
        let layout = registry
            .get_layout(handle)
            .ok_or_else(|| anyhow::anyhow!("Layout not found"))?;

        let region = layout.layout().memory_region(block_id, 0, 0)?;
        let ptr = region.addr() as *const i8;
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
        Ok(cstr.to_string_lossy().into_owned())
    }

    /// Parse ETags from format "1:etag1,2:etag2,..."
    fn parse_etags_string(s: &str) -> Vec<PartEtag> {
        s.split(',')
            .filter_map(|part| {
                let mut iter = part.splitn(2, ':');
                let num = iter.next()?.parse().ok()?;
                let etag = iter.next()?.to_string();
                Some(PartEtag { part_number: num, etag })
            })
            .collect()
    }

    // =========================================================================
    // Parameterized Roundtrip Tests
    // =========================================================================

    /// Test G2 <-> G4 roundtrip with various block configurations.
    #[rstest]
    #[case::single_block(vec![0], vec![1])]
    #[case::two_blocks(vec![0, 1], vec![2, 3])]
    #[case::four_blocks(vec![0, 1, 2, 3], vec![4, 5, 6, 7])]
    #[tokio::test]
    #[ignore = "requires object storage"]
    async fn test_g4_object_roundtrip(
        #[case] src_blocks: Vec<usize>,
        #[case] dst_blocks: Vec<usize>,
    ) -> Result<()> {
        let num_blocks = dst_blocks.iter().max().unwrap() + 1;
        let ctx = G4TestContext::new("g4_roundtrip", num_blocks)?;
        ctx.roundtrip_and_verify(src_blocks, dst_blocks).await?;
        tracing::info!("G4 roundtrip verified!");
        Ok(())
    }

    // =========================================================================
    // Multipart Upload Tests
    // =========================================================================

    /// Test multipart upload with various part counts.
    #[rstest]
    #[case::two_parts(2)]
    #[case::three_parts(3)]
    #[tokio::test]
    #[ignore = "requires object storage"]
    async fn test_g4_object_multipart_upload(#[case] num_parts: usize) -> Result<()> {
        let ctx = G4TestContext::new_multipart("multipart_test", num_parts * 2 + 1)?;
        let query_block_id = num_parts * 2;

        // Fill source blocks
        let src_blocks: Vec<usize> = (0..num_parts).collect();
        ctx.fill_blocks(&src_blocks, FillPattern::Sequential)?;

        let object_key = generate_raw_object_keys(1)[0];
        tracing::info!("Multipart upload for key {} with {} parts", object_key, num_parts);

        let helper = ctx.multipart_helper(query_block_id);
        helper.upload_multipart(object_key, &src_blocks).await?;

        tracing::info!("Multipart upload test with {} parts completed!", num_parts);
        Ok(())
    }

    /// Test multipart upload with full query-based flow.
    #[tokio::test]
    #[ignore = "requires object storage"]
    async fn test_g4_object_multipart_with_query_flow() -> Result<()> {
        const NUM_PARTS: usize = 2;
        let ctx = G4TestContext::new_multipart("multipart_query", NUM_PARTS * 2 + 1)?;
        let query_block_id = NUM_PARTS * 2;

        let src_blocks: Vec<usize> = (0..NUM_PARTS).collect();
        ctx.fill_blocks(&src_blocks, FillPattern::Sequential)?;

        let object_key = generate_raw_object_keys(1)[0];
        tracing::info!("Multipart with query for key: {}", object_key);

        let helper = ctx.multipart_helper(query_block_id);

        tracing::info!("Creating multipart upload...");
        helper.execute(
            ObjectOptArgs::new()
                .with_keys(vec![object_key])
                .with_multipart(MultipartOp::Create, "")
        ).await?;

        tracing::info!("Querying upload_id...");
        let upload_id = helper.query(
            ObjectOptArgs::new()
                .with_keys(vec![object_key])
                .with_multipart(MultipartOp::QueryUploadId, "")
        ).await?;
        tracing::info!("Retrieved upload_id: {}", upload_id);

        tracing::info!("Uploading {} parts...", NUM_PARTS);
        for (i, &block_id) in src_blocks.iter().enumerate() {
            let part_number = (i + 1) as u32;
            helper.upload_part(object_key, &upload_id, block_id, part_number).await?;
            tracing::info!("  Uploaded part {}", part_number);
        }

        tracing::info!("Querying ETags...");
        let etags = helper.query_etags(&upload_id).await?;
        tracing::info!("Retrieved {} ETags: {:?}", etags.len(), etags);

        tracing::info!("Completing multipart upload with {} ETags...", etags.len());
        helper.complete_multipart(object_key, &upload_id, etags).await?;

        tracing::info!("Multipart upload with query completed!");
        Ok(())
    }

    // =========================================================================
    // Multi-Agent Multipart with Byte-Range Downloads
    // =========================================================================

    /// Test complete multi-agent multipart flow with byte-range downloads.
    ///
    /// This test demonstrates the full distributed pattern:
    /// 1. Leader creates multipart upload -> queries upload_id
    /// 2. Workers upload their parts in parallel -> store ETags
    /// 3. Leader queries ETags -> completes multipart
    /// 4. Workers download their portions using byte-range reads
    /// 5. Verify data integrity
    #[tokio::test]
    #[ignore = "requires object storage"]
    async fn test_g4_object_multiagent_multipart_with_byterange_download() -> Result<()> {
        use dynamo_memory::ObjectStorage;

        const NUM_WORKERS: usize = 3;
        const MB: usize = 1024 * 1024;

        // Layout: NUM_WORKERS * 2 + 1 = 7 blocks
        //   [0, 1, 2] = source blocks (filled with test data)
        //   [3, 4, 5] = destination blocks (for byte-range downloads)
        //   [6] = query block (for upload_id and ETags responses)
        let ctx = G4TestContext::new_multipart("multiagent_worker", NUM_WORKERS * 2 + 1)?;
        let query_block_id = NUM_WORKERS * 2;
        let part_size = ctx.part_size();

        tracing::info!(
            "Multi-agent test: {} workers, part_size={}MB",
            NUM_WORKERS,
            part_size / MB
        );

        // Fill source blocks
        let src_blocks: Vec<usize> = (0..NUM_WORKERS).collect();
        let checksums = ctx.fill_blocks(&src_blocks, FillPattern::Sequential)?;

        let object_key = generate_raw_object_keys(1)[0];
        tracing::info!("Multipart upload to key: {}", object_key);

        let helper = ctx.multipart_helper(query_block_id);
        helper.upload_multipart(object_key, &src_blocks).await?;

        tracing::info!("Workers downloading their portions via byte-range...");

        let dst_blocks: Vec<usize> = (NUM_WORKERS..NUM_WORKERS * 2).collect();
        let nixl_agent = ctx.worker.manager.nixl_agent();
        let single_block_config = multipart_config(1);

        for worker_id in 0..NUM_WORKERS {
            let byte_offset = worker_id * part_size;
            tracing::info!(
                "Worker {} downloading bytes {}-{} ({}MB offset)...",
                worker_id,
                byte_offset,
                byte_offset + part_size,
                byte_offset / MB
            );

            // Create ObjectStorage with the correct byte offset for this worker
            let obj_storage = ObjectStorage::new(
                DEFAULT_G4_BUCKET,
                object_key,
                part_size,
            )?.with_offset(byte_offset);

            // Build a temporary G4 layout with this offset-specific ObjectStorage
            let temp_g4 = crate::v2::physical::layout::PhysicalLayoutBuilder::new(nixl_agent.clone())
                .with_config(single_block_config.clone())
                .object_layout()
                .with_memory_regions(vec![obj_storage])?
                .build()?;

            let temp_handle = ctx.worker.manager.register_layout(temp_g4)?;

            let get_opts = ObjectOptArgs::new().with_keys(vec![object_key]);

            let src_block: Arc<[usize]> = vec![0usize].into();
            let dst_block: Arc<[usize]> = vec![dst_blocks[worker_id]].into();

            let notification = ctx.worker.manager.execute_transfer(
                temp_handle,
                &src_block,
                ctx.worker.g2_handle,
                &dst_block,
                TransferOptions {
                    backend_opts: Some(Box::new(get_opts)),
                    ..Default::default()
                },
            )?;
            notification.await?;

            tracing::info!("Worker {} download complete", worker_id);
        }

        tracing::info!("Verifying data integrity...");
        ctx.verify_checksums(&checksums, &src_blocks, &dst_blocks)?;

        tracing::info!("Multi-agent multipart with byte-range download completed!");
        Ok(())
    }
}
