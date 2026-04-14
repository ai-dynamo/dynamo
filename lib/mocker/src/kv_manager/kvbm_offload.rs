// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KVBM G1↔G2 KV cache offload integration for the mocker.
//!
//! [`MockOffloadEngine`] wires up a real `kvbm-engine` [`OffloadEngine`] +
//! [`InstanceLeader`] in-process (single node, no GPUs).  [`MockWorker`]
//! replaces `DirectWorker` so that transfers complete with simulated delays
//! based on configurable bandwidth parameters — no actual data is moved.

use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::future::BoxFuture;

use dynamo_tokens::PositionalLineageHash;
use kvbm_engine::{
    BlockId, G1, G2, InstanceId, LogicalLayoutHandle, SequenceHash,
    leader::InstanceLeader,
    object::ObjectBlockOps,
    offload::{ExternalBlock, OffloadEngine, PipelineBuilder, PresenceFilter, SourceBlocks},
    worker::{
        ConnectRemoteResponse, ImportMetadataResponse, LayoutHandle, RemoteDescriptor,
        SerializedLayout, SerializedLayoutResponse, Worker, WorkerTransfers,
    },
};
use kvbm_logical::blocks::BlockRegistry;
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};
use kvbm_physical::transfer::{TransferCompleteNotification, TransferOptions};

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the KVBM offload engine used by the mocker.
pub struct KvbmOffloadConfig {
    pub num_g2_blocks: usize,
    pub offload_batch_size: usize,
    /// KV cache bytes per block.  `None` → transfers complete instantly.
    pub block_size_bytes: Option<usize>,
    /// G1↔G2 bandwidth in GB/s (default: 14.0, PCIe Gen4 x16).
    pub bandwidth_g1_g2_gbps: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Create a local-only Messenger on a random TCP port.
/// Required by `InstanceLeader` even in single-node mode.
async fn create_local_messenger() -> Result<Arc<velo::Messenger>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let transport: Arc<dyn velo::backend::Transport> = Arc::new(
        velo::backend::tcp::TcpTransportBuilder::new()
            .from_listener(listener)?
            .build()?,
    );
    let messenger = velo::Messenger::builder()
        .add_transport(transport)
        .build()
        .await?;
    tokio::time::sleep(Duration::from_millis(100)).await;
    Ok(messenger)
}

fn build_registry() -> BlockRegistry {
    BlockRegistry::builder()
        .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker())
        .build()
}

fn build_block_manager<T: kvbm_logical::blocks::BlockMetadata>(
    block_count: usize,
    registry: &BlockRegistry,
) -> BlockManager<T> {
    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(1)
        .registry(registry.clone())
        .with_lru_backend()
        .build()
        .expect("BlockManager build should not fail with valid config")
}

// ─────────────────────────────────────────────────────────────────────────────
// MockWorker
// ─────────────────────────────────────────────────────────────────────────────

/// Worker that simulates transfer delays without moving real data.
/// Returns [`TransferCompleteNotification`] that resolves after a delay
/// computed from bandwidth parameters and block count.
struct MockWorker {
    block_size_bytes: Option<usize>,
    bandwidth_g1_g2_gbps: f64,
    event_manager: Arc<velo::EventManager>,
    runtime_handle: tokio::runtime::Handle,
}

impl MockWorker {
    fn new(config: &KvbmOffloadConfig) -> Self {
        tracing::info!(
            block_size_bytes = ?config.block_size_bytes,
            bw_g1_g2 = config.bandwidth_g1_g2_gbps,
            "MockWorker initialized"
        );
        Self {
            block_size_bytes: config.block_size_bytes,
            bandwidth_g1_g2_gbps: config.bandwidth_g1_g2_gbps,
            event_manager: Arc::new(velo::EventManager::local()),
            runtime_handle: tokio::runtime::Handle::current(),
        }
    }

    pub(crate) fn transfer_delay(
        &self,
        _src: LogicalLayoutHandle,
        _dst: LogicalLayoutHandle,
        num_blocks: usize,
    ) -> Option<Duration> {
        let block_bytes = self.block_size_bytes?;
        let total_bytes = num_blocks * block_bytes;
        Some(Duration::from_secs_f64(
            total_bytes as f64 / (self.bandwidth_g1_g2_gbps * 1e9),
        ))
    }

    fn delayed_notification(
        &self,
        delay: Option<Duration>,
    ) -> Result<TransferCompleteNotification> {
        let delay = match delay {
            Some(d) if !d.is_zero() => d,
            _ => return Ok(TransferCompleteNotification::completed()),
        };
        let event = self.event_manager.new_event()?;
        let handle = event.handle();
        let awaiter = self.event_manager.awaiter(handle)?;
        let em = self.event_manager.clone();
        self.runtime_handle.spawn(async move {
            tokio::time::sleep(delay).await;
            let _ = em.trigger(handle);
            drop(event);
        });
        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

impl WorkerTransfers for MockWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let delay = self.transfer_delay(src, dst, src_block_ids.len());
        tracing::info!(
            ?src,
            ?dst,
            num_blocks = src_block_ids.len(),
            delay_us = delay.map(|d| d.as_micros()).unwrap_or(0),
            "MockWorker: local transfer"
        );
        self.delayed_notification(delay)
    }

    fn execute_remote_onboard(
        &self,
        _src: RemoteDescriptor,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn execute_remote_offload(
        &self,
        _src: LogicalLayoutHandle,
        _src_block_ids: Arc<[BlockId]>,
        _dst: RemoteDescriptor,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }

    fn connect_remote(
        &self,
        _instance_id: InstanceId,
        _metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        Ok(ConnectRemoteResponse::ready())
    }

    fn has_remote_metadata(&self, _instance_id: InstanceId) -> bool {
        false
    }

    fn execute_remote_onboard_for_instance(
        &self,
        _instance_id: InstanceId,
        _remote_logical_type: LogicalLayoutHandle,
        _src_block_ids: Vec<BlockId>,
        _dst: LogicalLayoutHandle,
        _dst_block_ids: Arc<[BlockId]>,
        _options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        Ok(TransferCompleteNotification::completed())
    }
}

impl Worker for MockWorker {
    fn g1_handle(&self) -> Option<LayoutHandle> {
        None
    }
    fn g2_handle(&self) -> Option<LayoutHandle> {
        None
    }
    fn g3_handle(&self) -> Option<LayoutHandle> {
        None
    }
    fn export_metadata(&self) -> Result<SerializedLayoutResponse> {
        Ok(SerializedLayoutResponse::ready(
            SerializedLayout::from_bytes(vec![]),
        ))
    }
    fn import_metadata(&self, _metadata: SerializedLayout) -> Result<ImportMetadataResponse> {
        Ok(ImportMetadataResponse::ready(vec![]))
    }
}

impl ObjectBlockOps for MockWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        Box::pin(async move { keys.into_iter().map(|k| (k, None)).collect() })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _src_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        _dst_layout: LogicalLayoutHandle,
        _block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        Box::pin(async move { keys.into_iter().map(Err).collect() })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwapInHandle
// ─────────────────────────────────────────────────────────────────────────────

/// Returned by `enqueue_g1_eviction_virtual()`. The caller (offline replay)
/// must call `complete_offload()` when virtual time reaches `complete_at_ms`.
pub struct PendingOffload {
    pub complete_at_ms: f64,
    pub block_id: BlockId,
    pub seq_hash: dynamo_tokens::SequenceHash,
}

/// Non-blocking handle for a pending G2→G1 swap-in transfer.
/// Mirrors vLLM's async `transfer_async()` + `get_finished()` pattern.
pub struct SwapInHandle {
    inner: SwapInInner,
}

enum SwapInInner {
    Completed,
    /// Live/online: polls a oneshot channel fed by tokio::spawn(sleep)
    Async(tokio::sync::oneshot::Receiver<()>),
    /// Offline: completes when virtual time reaches deadline
    VirtualTime {
        complete_at_ms: f64,
    },
}

impl SwapInHandle {
    pub(crate) fn completed() -> Self {
        Self {
            inner: SwapInInner::Completed,
        }
    }

    /// Live/online mode: non-blocking poll.
    /// Mirrors vLLM's `OffloadingHandler.get_finished()`.
    pub fn is_complete(&mut self) -> bool {
        match &mut self.inner {
            SwapInInner::Completed => true,
            SwapInInner::Async(rx) => rx.try_recv().is_ok(),
            SwapInInner::VirtualTime { .. } => {
                panic!("VirtualTime SwapInHandle must use is_complete_at(now_ms)")
            }
        }
    }

    /// Offline mode: check against virtual time.
    pub fn is_complete_at(&self, now_ms: f64) -> bool {
        match &self.inner {
            SwapInInner::Completed => true,
            SwapInInner::VirtualTime { complete_at_ms } => now_ms >= *complete_at_ms,
            SwapInInner::Async(_) => {
                panic!("Async SwapInHandle must use is_complete()")
            }
        }
    }

    /// Return the virtual-time deadline, if this is a VirtualTime handle.
    pub fn complete_at_ms(&self) -> Option<f64> {
        match &self.inner {
            SwapInInner::VirtualTime { complete_at_ms } => Some(*complete_at_ms),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MockOffloadEngine
// ─────────────────────────────────────────────────────────────────────────────

/// In-process offload engine backed by [`MockWorker`].
///
/// Supports two modes:
/// - **Async** (live/online): full kvbm-engine pipeline with `OffloadEngine` +
///   `InstanceLeader` + `MockWorker`. Built via [`build()`].
/// - **Sync** (offline replay): no Messenger/Pipeline/Leader, only
///   `BlockManager<G2>` for capacity/LRU tracking. Built via [`build_sync()`].
pub struct MockOffloadEngine {
    /// Present in async mode only (live/online).
    offload_engine: Option<OffloadEngine>,
    leader: Option<Arc<InstanceLeader>>,
    worker: Option<Arc<MockWorker>>,
    /// Present in both modes — for capacity/LRU tracking.
    g2_manager: Arc<BlockManager<G2>>,
    /// For transfer_delay calculation in both modes.
    block_size_bytes: Option<usize>,
    bandwidth_g1_g2_gbps: f64,
}

impl MockOffloadEngine {
    /// Async build for live/online modes.
    pub async fn build(config: &KvbmOffloadConfig) -> Result<Self> {
        let messenger = create_local_messenger().await?;
        let registry = Arc::new(build_registry());
        let g2_manager = Arc::new(build_block_manager::<G2>(config.num_g2_blocks, &registry));

        // InstanceLeader with MockWorker
        let mock_worker = Arc::new(MockWorker::new(config));
        let worker_for_leader: Arc<dyn Worker> = mock_worker.clone();
        let leader = Arc::new(
            InstanceLeader::builder()
                .messenger(messenger)
                .registry((*registry).clone())
                .g2_manager(g2_manager.clone())
                .worker(worker_for_leader)
                .build()?,
        );

        // OffloadEngine: G1→G2 pipeline with PresenceFilter
        let g1_to_g2_pipeline = PipelineBuilder::<G1, G2>::new()
            .policy(Arc::new(PresenceFilter::<G1, G2>::new(registry.clone())))
            .batch_size(config.offload_batch_size)
            .build();

        let offload_engine = OffloadEngine::builder(leader.clone())
            .with_registry(registry)
            .with_g2_manager(g2_manager.clone())
            .with_g1_to_g2_pipeline(g1_to_g2_pipeline)
            .build()?;

        Ok(Self {
            offload_engine: Some(offload_engine),
            leader: Some(leader),
            worker: Some(mock_worker),
            g2_manager,
            block_size_bytes: config.block_size_bytes,
            bandwidth_g1_g2_gbps: config.bandwidth_g1_g2_gbps,
        })
    }

    /// Synchronous build for offline replay.
    /// No Messenger, no Pipeline, no InstanceLeader — only `BlockManager<G2>`
    /// for capacity tracking and delay config for transfer time calculation.
    pub fn build_sync(config: &KvbmOffloadConfig) -> Result<Self> {
        let registry = Arc::new(build_registry());
        let g2_manager = Arc::new(build_block_manager::<G2>(config.num_g2_blocks, &registry));
        Ok(Self {
            offload_engine: None,
            leader: None,
            worker: None,
            g2_manager,
            block_size_bytes: config.block_size_bytes,
            bandwidth_g1_g2_gbps: config.bandwidth_g1_g2_gbps,
        })
    }

    /// Enqueue a G1 block eviction for offload to G2 (async pipeline).
    /// Only valid in async mode.
    pub fn enqueue_g1_eviction(&self, block_id: BlockId, seq_hash: dynamo_tokens::SequenceHash) {
        let offload_engine = self
            .offload_engine
            .as_ref()
            .expect("enqueue_g1_eviction requires async mode");
        let kvbm_hash = PositionalLineageHash::new(seq_hash, None, 0);
        let block = ExternalBlock::<G1>::new(block_id, kvbm_hash);
        let blocks: SourceBlocks<G1> = vec![block].into();

        match offload_engine.enqueue_g1_to_g2(blocks) {
            Ok(_handle) => {
                tracing::debug!(block_id, seq_hash, "KVBM: enqueued G1→G2 offload");
            }
            Err(e) => {
                tracing::warn!("KVBM: G1→G2 offload failed for block {block_id}: {e}");
            }
        }
    }

    /// Offline: calculate offload completion time without running the async pipeline.
    /// Returns a [`PendingOffload`] that the caller processes when virtual time arrives.
    pub fn enqueue_g1_eviction_virtual(
        &self,
        block_id: BlockId,
        seq_hash: dynamo_tokens::SequenceHash,
        now_ms: f64,
    ) -> PendingOffload {
        let delay_ms = self.transfer_delay_ms(1);
        PendingOffload {
            complete_at_ms: now_ms + delay_ms,
            block_id,
            seq_hash,
        }
    }

    /// Offline: virtual time reached — register block in G2 `BlockManager` directly.
    /// Mirrors what the pipeline's `TransferExecutor` does after transfer completes:
    /// `allocate_blocks(1)` → `stage(seq_hash, block_size)` → `register_block()` → drop.
    /// `allocate_blocks()` auto-evicts LRU from InactivePool if G2 is full.
    pub fn complete_offload(&self, _block_id: BlockId, seq_hash: dynamo_tokens::SequenceHash) {
        let Some(mut blocks) = self.g2_manager.allocate_blocks(1) else {
            tracing::warn!(seq_hash, "KVBM offline: G2 full, cannot complete offload");
            return;
        };
        let mutable = blocks.pop().unwrap();
        let kvbm_hash = PositionalLineageHash::new(seq_hash, None, 0);
        match mutable.stage(kvbm_hash, self.g2_manager.block_size()) {
            Ok(complete) => {
                let imm = self.g2_manager.register_block(complete);
                drop(imm);
                tracing::debug!(seq_hash, "KVBM offline: G2 offload complete");
            }
            Err(_) => {
                tracing::warn!(seq_hash, "KVBM offline: G2 stage failed");
            }
        }
    }

    /// Start an async G2→G1 swap-in transfer. Returns a [`SwapInHandle`]
    /// that the scheduler polls on subsequent passes. Only valid in async mode.
    pub fn start_swap_in(&self, num_blocks: usize) -> SwapInHandle {
        let worker = self
            .worker
            .as_ref()
            .expect("start_swap_in requires async mode");
        let delay = worker
            .transfer_delay(LogicalLayoutHandle::G2, LogicalLayoutHandle::G1, num_blocks)
            .unwrap_or(Duration::ZERO);

        if delay.is_zero() {
            tracing::debug!(num_blocks, "KVBM: swap-in (instant)");
            return SwapInHandle::completed();
        }

        tracing::debug!(
            num_blocks,
            delay_us = delay.as_micros(),
            "KVBM: swap-in started"
        );
        let (tx, rx) = tokio::sync::oneshot::channel();
        worker.runtime_handle.spawn(async move {
            crate::common::utils::sleep_precise(delay).await;
            let _ = tx.send(());
        });
        SwapInHandle {
            inner: SwapInInner::Async(rx),
        }
    }

    /// Offline: return a `SwapInHandle` based on virtual time.
    pub fn start_swap_in_virtual(&self, num_blocks: usize, now_ms: f64) -> SwapInHandle {
        let delay_ms = self.transfer_delay_ms(num_blocks);
        if delay_ms == 0.0 {
            return SwapInHandle::completed();
        }
        SwapInHandle {
            inner: SwapInInner::VirtualTime {
                complete_at_ms: now_ms + delay_ms,
            },
        }
    }

    /// Check whether `seq_hash` exists in G2.
    /// Async mode queries through `InstanceLeader`, sync mode queries
    /// `BlockManager<G2>::match_blocks()` directly.
    pub fn find_in_tiers(&self, seq_hash: dynamo_tokens::SequenceHash) -> bool {
        if let Some(leader) = &self.leader {
            // Async mode: query through InstanceLeader
            use kvbm_engine::leader::{FindMatchesOptions, FindMatchesResult, Leader, StagingMode};
            let kvbm_hash = PositionalLineageHash::new(seq_hash, None, 0);
            let options = FindMatchesOptions {
                search_remote: false,
                staging_mode: StagingMode::Hold,
            };
            match leader.find_matches_with_options(&[kvbm_hash], options) {
                Ok(FindMatchesResult::Ready(result)) => {
                    let found = result.g2_count() > 0;
                    if found {
                        tracing::debug!(seq_hash, "KVBM: G2 hit for onboard");
                    }
                    found
                }
                Ok(FindMatchesResult::AsyncSession(_)) => false,
                Err(e) => {
                    tracing::warn!("KVBM: find_matches failed for {seq_hash}: {e}");
                    false
                }
            }
        } else {
            // Sync mode: query BlockManager<G2> directly.
            // Use scan_matches (not match_blocks) because match_blocks does
            // prefix-sequential matching that breaks on first miss, while
            // scan_matches checks each hash independently.
            let kvbm_hash = PositionalLineageHash::new(seq_hash, None, 0);
            let found = !self.g2_manager.scan_matches(&[kvbm_hash], false).is_empty();
            if found {
                tracing::debug!(seq_hash, "KVBM offline: G2 hit for onboard");
            }
            found
        }
    }

    /// Batch version: check which `seq_hashes` exist in G2.
    /// Returns a set of indices (into the input slice) that were found.
    pub fn find_in_tiers_batch(&self, seq_hashes: &[dynamo_tokens::SequenceHash]) -> Vec<bool> {
        if seq_hashes.is_empty() {
            return Vec::new();
        }
        if self.leader.is_some() {
            // Async mode: fall back to per-hash queries
            return seq_hashes.iter().map(|h| self.find_in_tiers(*h)).collect();
        }
        // Sync mode: non-destructive batch check via has_blocks (uses peek, not pop).
        // has_blocks checks inactive pool only; this is correct because
        // complete_offload() drops the ImmutableBlock immediately, so blocks
        // always reside in the inactive pool at query time.
        let kvbm_hashes: Vec<_> = seq_hashes
            .iter()
            .map(|h| PositionalLineageHash::new(*h, None, 0))
            .collect();
        self.g2_manager.has_blocks(&kvbm_hashes)
    }

    fn transfer_delay_ms(&self, num_blocks: usize) -> f64 {
        match self.block_size_bytes {
            Some(bsz) => {
                let total_bytes = num_blocks * bsz;
                (total_bytes as f64 / (self.bandwidth_g1_g2_gbps * 1e9)) * 1000.0
            }
            None => 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_g1_eviction_offloads_to_g2() -> Result<()> {
        let config = KvbmOffloadConfig {
            num_g2_blocks: 64,
            offload_batch_size: 8,
            block_size_bytes: None,
            bandwidth_g1_g2_gbps: 14.0,
        };
        let engine = MockOffloadEngine::build(&config).await?;

        engine.enqueue_g1_eviction(0, 0xdeadbeef_cafebabe);
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok(())
    }

    #[tokio::test]
    async fn test_find_in_tiers_after_offload() -> Result<()> {
        let config = KvbmOffloadConfig {
            num_g2_blocks: 64,
            offload_batch_size: 8,
            block_size_bytes: None,
            bandwidth_g1_g2_gbps: 14.0,
        };
        let engine = MockOffloadEngine::build(&config).await?;

        let seq_hash: dynamo_tokens::SequenceHash = 0xdeadbeef;
        engine.enqueue_g1_eviction(0, seq_hash);
        tokio::time::sleep(Duration::from_millis(200)).await;

        assert!(
            engine.find_in_tiers(seq_hash),
            "block should be found in G2"
        );
        assert!(
            !engine.find_in_tiers(0x12345678),
            "unknown block should not be found"
        );
        Ok(())
    }

    #[tokio::test]
    async fn test_sync_offload_via_pending() {
        // Simulate the full integration path:
        // enqueue_g1_eviction_virtual → PendingOffload → complete_offload → find_in_tiers
        let config = KvbmOffloadConfig {
            num_g2_blocks: 64,
            offload_batch_size: 8,
            block_size_bytes: Some(1024),
            bandwidth_g1_g2_gbps: 14.0,
        };
        let engine = MockOffloadEngine::build_sync(&config).unwrap();

        let seq_hash: dynamo_tokens::SequenceHash = 0xdeadbeef;
        let pending = engine.enqueue_g1_eviction_virtual(0, seq_hash, 100.0);
        assert!(pending.complete_at_ms > 100.0);

        // Before completion, should not be findable
        assert!(
            !engine.find_in_tiers(seq_hash),
            "should not find before complete"
        );

        // Complete the offload
        engine.complete_offload(pending.block_id, pending.seq_hash);

        // Now should be findable
        assert!(engine.find_in_tiers(seq_hash), "should find after complete");
    }

    #[tokio::test]
    async fn test_sync_offload_exceeding_capacity() {
        // Test with more blocks than G2 capacity to check LRU eviction behavior
        let config = KvbmOffloadConfig {
            num_g2_blocks: 100,
            offload_batch_size: 8,
            block_size_bytes: None,
            bandwidth_g1_g2_gbps: 14.0,
        };
        let engine = MockOffloadEngine::build_sync(&config).unwrap();

        // Offload 200 blocks (2x capacity)
        for i in 0..200u64 {
            engine.complete_offload(i as BlockId, i + 1000);
        }

        // Only the last ~100 should be findable (LRU evicts oldest)
        let mut found_recent = 0;
        for i in 100..200u64 {
            if engine.find_in_tiers(i + 1000) {
                found_recent += 1;
            }
        }

        let mut found_old = 0;
        for i in 0..100u64 {
            if engine.find_in_tiers(i + 1000) {
                found_old += 1;
            }
        }

        println!("found_recent={found_recent}, found_old={found_old}");
        assert!(found_recent > 0, "recent blocks should be findable in G2");
    }

    #[tokio::test]
    async fn test_sync_scan_matches_directly() {
        // Test that scan_matches finds blocks stored via complete_offload
        let config = KvbmOffloadConfig {
            num_g2_blocks: 64,
            offload_batch_size: 8,
            block_size_bytes: None,
            bandwidth_g1_g2_gbps: 14.0,
        };
        let engine = MockOffloadEngine::build_sync(&config).unwrap();

        let seq_hash: dynamo_tokens::SequenceHash = 0xdeadbeef;
        engine.complete_offload(0, seq_hash);

        let kvbm_hash = PositionalLineageHash::new(seq_hash, None, 0);
        let matches = engine.g2_manager.scan_matches(&[kvbm_hash], false);
        assert!(
            !matches.is_empty(),
            "scan_matches should find the offloaded block"
        );
    }

    #[tokio::test]
    async fn test_sync_offload_and_find_many() {
        // Test that blocks stored via complete_offload are findable,
        // even after many offloads (mimicking realistic usage).
        let config = KvbmOffloadConfig {
            num_g2_blocks: 1000,
            offload_batch_size: 8,
            block_size_bytes: None,
            bandwidth_g1_g2_gbps: 14.0,
        };
        let engine = MockOffloadEngine::build_sync(&config).unwrap();

        // Offload 100 blocks
        let hashes: Vec<dynamo_tokens::SequenceHash> = (1000..1100).collect();
        for (i, &h) in hashes.iter().enumerate() {
            engine.complete_offload(i as BlockId, h);
        }

        // All should be findable
        let mut found = 0;
        for &h in &hashes {
            if engine.find_in_tiers(h) {
                found += 1;
            }
        }
        assert_eq!(
            found,
            hashes.len(),
            "all offloaded blocks should be findable"
        );

        // Unknown hashes should not be found
        assert!(!engine.find_in_tiers(9999));
    }

    #[tokio::test]
    async fn test_sync_offload_and_find() {
        let config = KvbmOffloadConfig {
            num_g2_blocks: 64,
            offload_batch_size: 8,
            block_size_bytes: None,
            bandwidth_g1_g2_gbps: 14.0,
        };
        let engine = MockOffloadEngine::build_sync(&config).unwrap();

        let seq_hash: dynamo_tokens::SequenceHash = 0xdeadbeef;
        engine.complete_offload(0, seq_hash);

        assert!(
            engine.find_in_tiers(seq_hash),
            "block should be findable in G2 after complete_offload"
        );
        assert!(
            !engine.find_in_tiers(0x12345678),
            "unknown block should not be found"
        );
    }
}
