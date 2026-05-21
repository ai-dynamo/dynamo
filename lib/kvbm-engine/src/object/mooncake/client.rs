// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mooncake Store object storage client for block management.

use std::sync::Arc;

use anyhow::{Result, anyhow};
use futures::future::BoxFuture;
use futures::stream::StreamExt;

use crate::object::{DefaultKeyFormatter, KeyFormatter, LayoutConfigExt, ObjectBlockOps};
use crate::{BlockId, SequenceHash};
use kvbm_common::LogicalLayoutHandle;
use kvbm_physical::transfer::PhysicalLayout;

/// Format a storage key with optional namespace prefix.
///
/// Centralizes the key formatting logic used across put, get, and has
/// operations so the namespace-prefixing pattern isn't duplicated.
pub(crate) fn format_storage_key(
    hash: &SequenceHash,
    namespace: &Option<String>,
    formatter: &dyn KeyFormatter,
) -> String {
    match namespace {
        Some(ns) => format!("{}/{}", ns, formatter.format_key(hash)),
        None => formatter.format_key(hash),
    }
}

/// Mooncake distributed KV store client for block operations.
pub struct MooncakeObjectBlockClient {
    pub(crate) store: Arc<mooncake_store::MooncakeStore>,

    key_formatter: Arc<dyn KeyFormatter>,
    max_concurrent: usize,
    namespace: Option<String>,
    use_zero_copy: bool,
}

impl MooncakeObjectBlockClient {
    /// Create a new Mooncake object block client.
    pub fn new(config: &kvbm_config::MooncakeObjectConfig) -> Result<Self> {
        let store = Arc::new(
            mooncake_store::MooncakeStore::new()
                .map_err(|e| anyhow!("MooncakeStore::new failed: {}", e))?,
        );

        store
            .setup(
                &config.local_hostname,
                &config.metadata_server,
                config.global_segment_size,
                config.local_buffer_size,
                &config.protocol.to_mooncake_str(),
                &config.device_name,
                &config.master_server_addr,
            )
            .map_err(|e| anyhow!("MooncakeStore::setup failed: {}", e))?;

        Ok(Self {
            store,

            key_formatter: Arc::new(DefaultKeyFormatter),
            max_concurrent: config.max_concurrent_requests,
            namespace: config.namespace.clone(),
            use_zero_copy: false,
        })
    }

    /// Set a custom key formatter.
    pub fn with_key_formatter(mut self, formatter: Arc<dyn KeyFormatter>) -> Self {
        self.key_formatter = formatter;
        self
    }

    /// Enable or disable zero-copy RDMA transfers.
    ///
    /// When enabled, contiguous layouts use `put_from`/`get_into` directly
    /// instead of intermediate copies. Callers must register the layout
    /// buffer via [`register_layout`] before use.
    pub fn with_zero_copy(mut self, enabled: bool) -> Self {
        self.use_zero_copy = enabled;
        self
    }

    /// Format a key with optional namespace prefix.
    pub(crate) fn format_key(&self, hash: &SequenceHash) -> String {
        format_storage_key(hash, &self.namespace, self.key_formatter.as_ref())
    }

    /// Register a physical layout's buffers with Mooncake for RDMA zero-copy.
    ///
    /// This must be called before using zero-copy transfers on the layout.
    /// For fully-contiguous layouts, registers the single backing buffer.
    /// For layer-separate layouts, registers each layer's buffer individually.
    pub fn register_layout(&self, layout: &PhysicalLayout) -> Result<()> {
        for buffer in layout.layout().memory_regions() {
            let addr = buffer.addr();
            let size = buffer.size();
            unsafe {
                self.store
                    .register_buffer(addr as *mut std::ffi::c_void, size)
                    .map_err(|e| anyhow!("register_buffer failed: {}", e))?;
            }
        }
        Ok(())
    }

    /// Unregister a physical layout's buffers from Mooncake.
    ///
    /// Call this when the layout is no longer used for zero-copy transfers
    /// to free RDMA registration resources.
    pub fn unregister_layout(&self, layout: &PhysicalLayout) -> Result<()> {
        for buffer in layout.layout().memory_regions() {
            let addr = buffer.addr();
            unsafe {
                self.store
                    .unregister_buffer(addr as *mut std::ffi::c_void)
                    .map_err(|e| anyhow!("unregister_buffer failed: {}", e))?;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Block serialization / deserialization
    // ------------------------------------------------------------------

    /// Copy a single block from layout to a Vec.
    fn copy_block_to_vec(
        layout: &PhysicalLayout,
        block_id: BlockId,
        block_size: usize,
        region_size: usize,
        is_contiguous: bool,
    ) -> Result<Vec<u8>> {
        if is_contiguous {
            let region = layout.memory_region(block_id, 0, 0)?;
            let slice = unsafe { std::slice::from_raw_parts(region.addr as *const u8, block_size) };
            Ok(slice.to_vec())
        } else {
            let mut buf = Vec::with_capacity(block_size);
            let inner = layout.layout();
            for layer_id in 0..inner.num_layers() {
                for outer_id in 0..inner.outer_dim() {
                    let region = layout.memory_region(block_id, layer_id, outer_id)?;
                    if region.size < region_size {
                        return Err(anyhow!(
                            "memory region too small: got {} bytes, need {}",
                            region.size,
                            region_size
                        ));
                    }
                    let slice = unsafe {
                        std::slice::from_raw_parts(region.addr as *const u8, region_size)
                    };
                    buf.extend_from_slice(slice);
                }
            }
            Ok(buf)
        }
    }

    /// Copy data from a slice to a layout block.
    fn copy_slice_to_block(
        data: &[u8],
        layout: &PhysicalLayout,
        block_id: BlockId,
        block_size: usize,
        region_size: usize,
        is_contiguous: bool,
    ) -> Result<()> {
        if is_contiguous {
            if data.len() < block_size {
                return Err(anyhow!(
                    "Mooncake data too short: got {} bytes, expected {}",
                    data.len(),
                    block_size
                ));
            }
            let region = layout.memory_region(block_id, 0, 0)?;
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), region.addr as *mut u8, block_size);
            }
        } else {
            let mut offset = 0;
            let inner = layout.layout();
            for layer_id in 0..inner.num_layers() {
                for outer_id in 0..inner.outer_dim() {
                    if offset + region_size > data.len() {
                        return Err(anyhow!("Mooncake data too short at offset {}", offset));
                    }
                    let region = layout.memory_region(block_id, layer_id, outer_id)?;
                    if region.size < region_size {
                        return Err(anyhow!(
                            "memory region too small: got {} bytes, need {}",
                            region.size,
                            region_size
                        ));
                    }
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data[offset..].as_ptr(),
                            region.addr as *mut u8,
                            region_size,
                        );
                    }
                    offset += region_size;
                }
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Physical layout methods (inherent, used by trait impl)
    // ------------------------------------------------------------------

    /// Put blocks to Mooncake from a physical layout.
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
        let max_concurrent = self.max_concurrent;
        let store = self.store.clone();
        let formatter = self.key_formatter.clone();
        let namespace = self.namespace.clone();
        let use_zero_copy = self.use_zero_copy;

        Box::pin(async move {
            if keys.len() != block_ids.len() {
                tracing::warn!(
                    keys_len = keys.len(),
                    block_ids_len = block_ids.len(),
                    "put_blocks_with_layout: mismatched lengths, returning all errors"
                );
                return keys.into_iter().map(Err).collect();
            }
            let work_items: Vec<_> = keys.into_iter().zip(block_ids).collect();

            // Fast batch path for contiguous zero-copy layouts
            if is_contiguous && use_zero_copy && !work_items.is_empty() {
                let keys_str: Vec<String> = work_items
                    .iter()
                    .map(|(hash, _)| format_storage_key(hash, &namespace, formatter.as_ref()))
                    .collect();
                let key_refs: Vec<&str> = keys_str.iter().map(|s| s.as_str()).collect();

                // Collect buffers; if any memory_region fails, fall through to
                // the single-block path instead of passing null pointers to the
                // C library.
                let buffers: Result<Vec<*mut std::ffi::c_void>> = work_items
                    .iter()
                    .map(|(_, block_id)| {
                        layout
                            .memory_region(*block_id, 0, 0)
                            .map(|r| r.addr as *mut std::ffi::c_void)
                    })
                    .collect();

                if let Ok(buffers) = buffers {
                    let sizes: Vec<usize> = std::iter::repeat(block_size)
                        .take(work_items.len())
                        .collect();

                    let batch_result = unsafe {
                        store
                            .batch_put_from(&key_refs, &buffers, &sizes, None)
                            .map_err(|e| anyhow!("Mooncake batch_put_from failed: {}", e))
                    };

                    match batch_result {
                        Ok(rcodes) => {
                            return work_items.into_iter().zip(rcodes.into_iter())
                                .map(|((hash, _), rc)| {
                                    if rc == 0 {
                                        Ok(hash)
                                    } else {
                                        tracing::warn!(hash = %hash, rc = rc, "batch_put_from block failed");
                                        Err(hash)
                                    }
                                })
                                .collect();
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "batch_put_from failed, falling back to single-block");
                            // Fall through to single-block path
                        }
                    }
                } else {
                    tracing::warn!(
                        "batch put: memory_region failed for one or more blocks, falling back to single-block"
                    );
                    // Fall through to single-block path
                }
            }

            let tasks = work_items.into_iter().enumerate().map(|(idx, (hash, block_id))| {
                let store = store.clone();
                let key = format_storage_key(&hash, &namespace, formatter.as_ref());
                let layout = layout.clone();

                async move {
                    let result: Result<(), anyhow::Error> = async {
                        if is_contiguous && use_zero_copy {
                            // Zero-copy path: upload directly from layout memory
                            let region = layout.memory_region(block_id, 0, 0)?;
                            unsafe {
                                store.put_from(
                                    &key,
                                    region.addr as *mut std::ffi::c_void,
                                    block_size,
                                    None,
                                ).map_err(|e| anyhow!("Mooncake put_from failed: {}", e))?;
                            }
                        } else {
                            // Copy path: serialize to Vec, then upload
                            let data = tokio_rayon::spawn(move || {
                                Self::copy_block_to_vec(
                                    &layout, block_id, block_size, region_size, is_contiguous,
                                )
                            }).await.map_err(|e| anyhow!("rayon task failed: {}", e))?;

                            store.put(&key, &data, None)
                                .map_err(|e| anyhow!("Mooncake put failed: {}", e))?;
                        }
                        Ok(())
                    }.await;

                    let res = match result {
                        Ok(()) => Ok(hash),
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "put block to Mooncake failed");
                            Err(hash)
                        }
                    };
                    (idx, res)
                }
            });

            let mut results: Vec<_> = futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await;
            // Sort by original input index to preserve deterministic ordering.
            // SpmdParallelWorkers indexes results by position, so out-of-order
            // returns would misattribute success/failure.
            results.sort_by_key(|(idx, _)| *idx);
            results.into_iter().map(|(_, res)| res).collect()
        })
    }

    /// Get blocks from Mooncake into a physical layout.
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
        let max_concurrent = self.max_concurrent;
        let store = self.store.clone();
        let formatter = self.key_formatter.clone();
        let namespace = self.namespace.clone();
        let use_zero_copy = self.use_zero_copy;

        Box::pin(async move {
            if keys.len() != block_ids.len() {
                tracing::warn!(
                    keys_len = keys.len(),
                    block_ids_len = block_ids.len(),
                    "get_blocks_with_layout: mismatched lengths, returning all errors"
                );
                return keys.into_iter().map(Err).collect();
            }
            let work_items: Vec<_> = keys.into_iter().zip(block_ids).collect();

            // Fast batch path for contiguous zero-copy layouts
            if is_contiguous && use_zero_copy && !work_items.is_empty() {
                let keys_str: Vec<String> = work_items
                    .iter()
                    .map(|(hash, _)| format_storage_key(hash, &namespace, formatter.as_ref()))
                    .collect();
                let key_refs: Vec<&str> = keys_str.iter().map(|s| s.as_str()).collect();

                // Collect buffers; if any memory_region fails, fall through to
                // the single-block path instead of passing null pointers to the
                // C library.
                let buffers: Result<Vec<*mut std::ffi::c_void>> = work_items
                    .iter()
                    .map(|(_, block_id)| {
                        layout
                            .memory_region(*block_id, 0, 0)
                            .map(|r| r.addr as *mut std::ffi::c_void)
                    })
                    .collect();

                if let Ok(buffers) = buffers {
                    let sizes: Vec<usize> = std::iter::repeat(block_size)
                        .take(work_items.len())
                        .collect();

                    let batch_result = unsafe {
                        store
                            .batch_get_into(&key_refs, &buffers, &sizes)
                            .map_err(|e| anyhow!("Mooncake batch_get_into failed: {}", e))
                    };

                    match batch_result {
                        Ok(written_bytes) => {
                            return work_items
                                .into_iter()
                                .zip(written_bytes.into_iter())
                                .map(|((hash, _), written)| {
                                    if written == block_size as i64 {
                                        Ok(hash)
                                    } else {
                                        tracing::warn!(
                                            hash = %hash,
                                            expected = block_size,
                                            got = written,
                                            "batch_get_into short read"
                                        );
                                        Err(hash)
                                    }
                                })
                                .collect();
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "batch_get_into failed, falling back to single-block");
                            // Fall through to single-block path
                        }
                    }
                } else {
                    tracing::warn!(
                        "batch get: memory_region failed for one or more blocks, falling back to single-block"
                    );
                    // Fall through to single-block path
                }
            }

            let tasks = work_items.into_iter().enumerate().map(|(idx, (hash, block_id))| {
                let store = store.clone();
                let key = format_storage_key(&hash, &namespace, formatter.as_ref());
                let layout = layout.clone();

                async move {
                    let result: Result<(), anyhow::Error> = async {
                        if is_contiguous && use_zero_copy {
                            // Zero-copy path: download directly into layout memory
                            let region = layout.memory_region(block_id, 0, 0)?;
                            let written = unsafe {
                                store.get_into(
                                    &key,
                                    region.addr as *mut std::ffi::c_void,
                                    block_size,
                                ).map_err(|e| anyhow!("Mooncake get_into failed: {}", e))?
                            };
                            if written != block_size as i64 {
                                return Err(anyhow!(
                                    "Mooncake get_into short read: expected {} bytes, got {}",
                                    block_size, written
                                ));
                            }
                        } else {
                            // Copy path: download to Vec, then deserialize
                            let data = store.get(&key)
                                .map_err(|e| anyhow!("Mooncake get failed: {}", e))?;

                            tokio_rayon::spawn(move || {
                                Self::copy_slice_to_block(
                                    &data, &layout, block_id, block_size, region_size, is_contiguous,
                                )
                            }).await.map_err(|e| anyhow!("rayon task failed: {}", e))?;
                        }
                        Ok(())
                    }.await;

                    let res = match result {
                        Ok(()) => Ok(hash),
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "get block from Mooncake failed");
                            Err(hash)
                        }
                    };
                    (idx, res)
                }
            });

            let mut results: Vec<_> = futures::stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await;
            // Sort by original input index to preserve deterministic ordering.
            // SpmdParallelWorkers indexes results by position, so out-of-order
            // returns would misattribute success/failure.
            results.sort_by_key(|(idx, _)| *idx);
            results.into_iter().map(|(_, res)| res).collect()
        })
    }
}

impl ObjectBlockOps for MooncakeObjectBlockClient {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        let store = self.store.clone();
        let formatter = self.key_formatter.clone();
        let namespace = self.namespace.clone();
        let max_concurrent = self.max_concurrent;

        Box::pin(async move {
            // Format all keys upfront
            let formatted: Vec<String> = keys
                .iter()
                .map(|hash| format_storage_key(hash, &namespace, formatter.as_ref()))
                .collect();

            let key_refs: Vec<&str> = formatted.iter().map(|s| s.as_str()).collect();

            // Batch existence check (single RPC round-trip)
            let exist_results = match store.batch_is_exist(&key_refs) {
                Ok(results) => results,
                Err(e) => {
                    tracing::warn!(error = %e, "batch_is_exist failed");
                    return keys.into_iter().map(|h| (h, None)).collect();
                }
            };

            // Separate existing (need size query) from non-existing keys.
            // Non-existing keys must still appear in the result as (hash, None).
            // Process size queries sequentially to preserve 1:1 cardinality and
            // deterministic ordering (callers like SpmdParallelWorkers index by position).
            if exist_results.len() != keys.len() {
                tracing::warn!(
                    expected = keys.len(),
                    got = exist_results.len(),
                    "batch_is_exist size mismatch"
                );
                return keys.into_iter().map(|h| (h, None)).collect();
            }

            let mut results: Vec<(SequenceHash, Option<usize>)> = Vec::with_capacity(keys.len());
            for ((hash, key), exists) in keys.into_iter().zip(formatted).zip(exist_results) {
                if exists {
                    let size = match store.get_size(&key) {
                        Ok(size) => Some(size as usize),
                        Err(e) => {
                            tracing::warn!(key = %key, error = %e, "get_size failed");
                            None
                        }
                    };
                    results.push((hash, size));
                } else {
                    results.push((hash, None));
                }
            }
            results
        })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        tracing::error!(
            "MooncakeObjectBlockClient::put_blocks called with LogicalLayoutHandle - \
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
        tracing::error!(
            "MooncakeObjectBlockClient::get_blocks called with LogicalLayoutHandle - \
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
        MooncakeObjectBlockClient::put_blocks_with_layout(self, keys, layout, block_ids)
    }

    fn get_blocks_with_layout(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        MooncakeObjectBlockClient::get_blocks_with_layout(self, keys, layout, block_ids)
    }
}

#[cfg(all(test, feature = "mooncake", feature = "testing"))]
mod tests {
    use super::*;

    use kvbm_config::MooncakeObjectConfig;
    use kvbm_physical::testing::{create_fc_layout_system, create_lw_layout_system};
    use kvbm_physical::transfer::{FillPattern, compute_block_checksums, fill_blocks};

    #[allow(dead_code)]
    fn test_config() -> kvbm_physical::layout::LayoutConfig {
        kvbm_physical::layout::LayoutConfig::builder()
            .num_blocks(4)
            .num_layers(2)
            .outer_dim(2)
            .page_size(16)
            .inner_dim(128)
            .dtype_width_bytes(2)
            .build()
            .expect("standard config should build")
    }

    fn create_test_client() -> MooncakeObjectBlockClient {
        let config = MooncakeObjectConfig {
            metadata_server: "http://127.0.0.1:8080/metadata".to_string(),
            master_server_addr: "127.0.0.1:50051".to_string(),
            local_hostname: "localhost".to_string(),
            protocol: kvbm_config::MooncakeProtocol::Tcp,
            device_name: "".to_string(),
            global_segment_size: 512 * 1024 * 1024,
            local_buffer_size: 128 * 1024 * 1024,
            namespace: Some("test-blocks".to_string()),
            max_concurrent_requests: 4,
        };
        MooncakeObjectBlockClient::new(&config).expect("create client")
    }

    #[tokio::test]
    #[ignore = "Requires Mooncake server and NIXL environment"]
    async fn test_put_get_roundtrip_fully_contiguous() {
        let client = create_test_client();
        let layout = create_fc_layout_system(4);

        let block_ids = vec![0, 1, 2];
        fill_blocks(&layout, &block_ids, FillPattern::Sequential).expect("fill blocks");
        let src_checksums = compute_block_checksums(&layout, &block_ids).expect("checksum");

        let hashes: Vec<SequenceHash> = block_ids
            .iter()
            .map(|&id| SequenceHash::new(id as u64, None, id as u64))
            .collect();

        let put_results = client
            .put_blocks_with_layout(hashes.clone(), layout.clone(), block_ids.clone())
            .await;
        assert_eq!(put_results.len(), 3);
        for r in &put_results {
            assert!(r.is_ok(), "put failed: {:?}", r);
        }

        // Clear local memory
        fill_blocks(&layout, &block_ids, FillPattern::Constant(0)).expect("clear blocks");

        let get_results = client
            .get_blocks_with_layout(hashes.clone(), layout.clone(), block_ids.clone())
            .await;
        assert_eq!(get_results.len(), 3);
        for r in &get_results {
            assert!(r.is_ok(), "get failed: {:?}", r);
        }

        let dst_checksums =
            compute_block_checksums(&layout, &block_ids).expect("checksum after get");
        for id in &block_ids {
            assert_eq!(
                src_checksums[id], dst_checksums[id],
                "checksum mismatch for block {}",
                id
            );
        }
    }

    #[tokio::test]
    #[ignore = "Requires Mooncake server and NIXL environment"]
    async fn test_put_get_roundtrip_layer_separate() {
        let client = create_test_client();
        let layout = create_lw_layout_system(4);

        let block_ids = vec![0, 1];
        fill_blocks(&layout, &block_ids, FillPattern::Sequential).expect("fill blocks");
        let src_checksums = compute_block_checksums(&layout, &block_ids).expect("checksum");

        let hashes: Vec<SequenceHash> = block_ids
            .iter()
            .map(|&id| SequenceHash::new(id as u64 + 100, None, id as u64))
            .collect();

        let put_results = client
            .put_blocks_with_layout(hashes.clone(), layout.clone(), block_ids.clone())
            .await;
        assert_eq!(put_results.len(), 2);
        for r in &put_results {
            assert!(r.is_ok(), "put failed: {:?}", r);
        }

        fill_blocks(&layout, &block_ids, FillPattern::Constant(0)).expect("clear blocks");

        let get_results = client
            .get_blocks_with_layout(hashes.clone(), layout.clone(), block_ids.clone())
            .await;
        assert_eq!(get_results.len(), 2);
        for r in &get_results {
            assert!(r.is_ok(), "get failed: {:?}", r);
        }

        let dst_checksums =
            compute_block_checksums(&layout, &block_ids).expect("checksum after get");
        for id in &block_ids {
            assert_eq!(
                src_checksums[id], dst_checksums[id],
                "checksum mismatch for block {}",
                id
            );
        }
    }

    #[tokio::test]
    #[ignore = "Requires Mooncake server and NIXL environment"]
    async fn test_put_get_roundtrip_zero_copy_fully_contiguous() {
        let client = create_test_client().with_zero_copy(true);
        let layout = create_fc_layout_system(4);

        client.register_layout(&layout).expect("register layout");

        let block_ids = vec![0, 1, 2];
        fill_blocks(&layout, &block_ids, FillPattern::Sequential).expect("fill blocks");
        let src_checksums = compute_block_checksums(&layout, &block_ids).expect("checksum");

        let hashes: Vec<SequenceHash> = block_ids
            .iter()
            .map(|&id| SequenceHash::new(id as u64 + 200, None, id as u64))
            .collect();

        let put_results = client
            .put_blocks_with_layout(hashes.clone(), layout.clone(), block_ids.clone())
            .await;
        assert_eq!(put_results.len(), 3);
        for r in &put_results {
            assert!(r.is_ok(), "put failed: {:?}", r);
        }

        fill_blocks(&layout, &block_ids, FillPattern::Constant(0)).expect("clear blocks");

        let get_results = client
            .get_blocks_with_layout(hashes.clone(), layout.clone(), block_ids.clone())
            .await;
        assert_eq!(get_results.len(), 3);
        for r in &get_results {
            assert!(r.is_ok(), "get failed: {:?}", r);
        }

        let dst_checksums =
            compute_block_checksums(&layout, &block_ids).expect("checksum after get");
        for id in &block_ids {
            assert_eq!(
                src_checksums[id], dst_checksums[id],
                "checksum mismatch for block {}",
                id
            );
        }
    }
}
