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

/// Non-blocking handle for a pending G2→G1 swap-in transfer.
/// Mirrors vLLM's async `transfer_async()` + `get_finished()` pattern.
pub struct SwapInHandle {
    rx: Option<tokio::sync::oneshot::Receiver<()>>,
}

impl SwapInHandle {
    fn completed() -> Self {
        Self { rx: None }
    }

    /// Non-blocking check if the transfer has completed.
    /// Mirrors vLLM's `OffloadingHandler.get_finished()`.
    pub fn is_complete(&mut self) -> bool {
        match &mut self.rx {
            None => true,
            Some(rx) => rx.try_recv().is_ok(),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MockOffloadEngine
// ─────────────────────────────────────────────────────────────────────────────

/// In-process offload engine backed by [`MockWorker`].
/// Wraps a real kvbm-engine [`OffloadEngine`] so that mocker's G1 evictions
/// feed into G2 [`BlockManager`] for logical metadata tracking.
pub struct MockOffloadEngine {
    offload_engine: OffloadEngine,
    leader: Arc<InstanceLeader>,
    worker: Arc<MockWorker>,
}

impl MockOffloadEngine {
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
            .with_g2_manager(g2_manager)
            .with_g1_to_g2_pipeline(g1_to_g2_pipeline)
            .build()?;

        Ok(Self {
            offload_engine,
            leader,
            worker: mock_worker,
        })
    }

    /// Enqueue a G1 block eviction for offload to G2.
    /// `seq_hash` is mocker's `SequenceHash` (u64), wrapped into
    /// `PositionalLineageHash` for kvbm-engine block identity tracking.
    pub fn enqueue_g1_eviction(&self, block_id: BlockId, seq_hash: dynamo_tokens::SequenceHash) {
        let kvbm_hash = PositionalLineageHash::new(seq_hash, None, 0);
        let block = ExternalBlock::<G1>::new(block_id, kvbm_hash);
        let blocks: SourceBlocks<G1> = vec![block].into();

        match self.offload_engine.enqueue_g1_to_g2(blocks) {
            Ok(_handle) => {
                tracing::debug!(block_id, seq_hash, "KVBM: enqueued G1→G2 offload");
            }
            Err(e) => {
                tracing::warn!("KVBM: G1→G2 offload failed for block {block_id}: {e}");
            }
        }
    }

    /// Start an async G2→G1 swap-in transfer. Returns a [`SwapInHandle`]
    /// that the scheduler polls on subsequent passes. Mirrors vLLM's
    /// `transfer_async()` which runs on a separate CUDA stream.
    pub fn start_swap_in(&self, num_blocks: usize) -> SwapInHandle {
        let delay = self
            .worker
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
        self.worker.runtime_handle.spawn(async move {
            crate::common::utils::sleep_precise(delay).await;
            let _ = tx.send(());
        });
        SwapInHandle { rx: Some(rx) }
    }

    /// Check whether `seq_hash` exists in G2.
    pub fn find_in_tiers(&self, seq_hash: dynamo_tokens::SequenceHash) -> bool {
        use kvbm_engine::leader::{FindMatchesOptions, FindMatchesResult, Leader, StagingMode};

        let kvbm_hash = PositionalLineageHash::new(seq_hash, None, 0);
        let options = FindMatchesOptions {
            search_remote: false,
            staging_mode: StagingMode::Hold,
        };
        match self.leader.find_matches_with_options(&[kvbm_hash], options) {
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
}
