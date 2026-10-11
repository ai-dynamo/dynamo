// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_llm::block_manager::{
    ManagedBlockPool,
    block::{
        GlobalRegistry,
        factory::{IntoBlocks, logical::LogicalBlockFactory},
    },
    kv_consolidator::StorageTier,
    layout::LayoutConfig,
    pool::AsyncBlockPoolController,
    storage::StorageType,
};

const BS: usize = 16;
type Pool<S> = ManagedBlockPool<S, VllmLocality, BasicMetadata>;

struct Fixture {
    slot: VllmConnectorSlot,
    host: Arc<Pool<PinnedStorage>>,
    disk: Arc<Pool<DiskStorage>>,
    rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
}

fn pool<S: Storage>(
    storage_type: StorageType,
    tier: StorageTier,
    block_set: usize,
) -> Arc<Pool<S>> {
    let config = LayoutConfig::builder()
        .num_blocks(32)
        .num_layers(1)
        .outer_dim(1)
        .page_size(BS)
        .inner_dim(1)
        .dtype_width_bytes(2)
        .build()
        .unwrap();
    let resources = DistributedLeaderWorkerResources::new(None, CancellationToken::new()).unwrap();
    let factory = LogicalBlockFactory::<S, _>::new(
        Arc::new(config),
        block_set,
        1,
        Arc::new(resources),
        storage_type,
        None,
    );
    Arc::new(
        ManagedBlockPool::builder()
            .blocks(
                factory
                    .into_blocks_with_metadata(BasicMetadata::default())
                    .unwrap(),
            )
            .global_registry(GlobalRegistry::default())
            .storage_tier(tier)
            .build()
            .unwrap(),
    )
}

impl Fixture {
    fn new(total_tokens: usize) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let mut slot = VllmConnectorSlot::new_for_test(
            "repoll".into(),
            Tokens::from((1..=total_tokens as u32).collect::<Vec<_>>()),
            123,
            BS,
            tx,
            Arc::new(CacheStatsTracker::new(None)),
            0,
        );
        let host = pool(StorageType::Pinned, StorageTier::HostPinned, 1);
        let disk = pool(StorageType::Disk(0), StorageTier::Disk, 2);
        slot.apc_test_host = Some(host.clone());
        slot.apc_test_disk = Some(disk.clone());
        Self {
            slot,
            host,
            disk,
            rx,
        }
    }

    fn seed<S: Storage>(&self, pool: &Pool<S>, start: usize, count: usize) {
        tokio::task::block_in_place(|| {
            let blocks = pool.allocate_blocks_blocking(count).unwrap();
            let prepared = blocks
                .into_iter()
                .enumerate()
                .map(|(i, mut block)| {
                    block
                        .apply_token_block(self.slot.sequence.blocks()[start + i].clone())
                        .unwrap();
                    block
                })
                .collect();
            drop(pool.register_blocks_blocking(prepared).unwrap());
        });
    }

    fn query(&mut self, local_tokens: usize) {
        tokio::task::block_in_place(|| self.slot.acquire_local_matches(local_tokens)).unwrap();
    }

    fn cache_six(&self) {
        self.seed(self.host.as_ref(), 0, 4);
        self.seed(self.disk.as_ref(), 4, 2);
    }

    fn staged_hashes(&self) -> Vec<SequenceHash> {
        self.slot
            .staging_from_host
            .iter()
            .flatten()
            .map(|b| b.sequence_hash())
            .chain(
                self.slot
                    .staging_from_disk
                    .iter()
                    .flatten()
                    .map(|b| b.sequence_hash()),
            )
            .collect()
    }

    fn sequence_hashes(&self, start: usize, end: usize) -> Vec<SequenceHash> {
        self.slot.sequence.blocks()[start..end]
            .iter()
            .map(|b| b.sequence_hash())
            .collect()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_growth_updates_sources_and_transfer_destinations() {
    let mut f = Fixture::new(7 * BS);
    f.cache_six();
    f.query(0);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(6 * BS));
    f.query(2 * BS);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(4 * BS));
    assert_eq!(f.staged_hashes(), f.sequence_hashes(2, 6));
    f.slot
        .append_mutable_device_blocks(&(100..106).collect::<Vec<_>>())
        .unwrap();
    f.slot.advance_computed_position(2 * BS).unwrap();
    f.slot.trigger_onboarding(4 * BS).unwrap();
    let LocalTransferRequest::Onboard(host) = f.rx.try_recv().unwrap() else {
        panic!("host transfer")
    };
    let LocalTransferRequest::Onboard(disk) = f.rx.try_recv().unwrap() else {
        panic!("disk transfer")
    };
    assert_eq!(host.src_blocks.block_ids(), vec![2, 3]);
    assert_eq!(host.dst_block_ids, vec![102, 103]);
    assert_eq!(disk.src_blocks.block_ids(), vec![0, 1]);
    assert_eq!(disk.dst_block_ids, vec![104, 105]);
    assert_eq!(f.slot.computed_tokens(), 6 * BS);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_growth_crosses_host_disk_boundary() {
    let mut f = Fixture::new(7 * BS);
    f.cache_six();
    f.query(0);
    f.query(5 * BS);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(BS));
    assert!(f.slot.staging_from_host.is_none());
    assert_eq!(f.staged_hashes(), f.sequence_hashes(5, 6));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_decrease_reacquires_the_missing_leading_blocks() {
    let mut f = Fixture::new(7 * BS);
    f.cache_six();
    f.query(2 * BS);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(4 * BS));
    f.query(0);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(6 * BS));
    assert_eq!(f.staged_hashes(), f.sequence_hashes(0, 6));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gpu_covers_the_staged_prefix_and_releases_its_references() {
    let mut f = Fixture::new(7 * BS);
    f.cache_six();
    f.query(0);
    f.query(6 * BS);
    assert_eq!(f.slot.state(), SlotState::Initialized);
    assert!(f.staged_hashes().is_empty());
    assert_eq!(f.slot.tokens_cached_from_host, 0);
    assert_eq!(f.slot.tokens_cached_from_disk, 0);
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if f.host.status().await.unwrap().active_blocks == 0
                && f.disk.status().await.unwrap().active_blocks == 0
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unchanged_gpu_prefix_preserves_matching_blocks() {
    let mut f = Fixture::new(7 * BS);
    f.cache_six();
    f.query(2 * BS);
    let before = f.staged_hashes();
    f.query(2 * BS);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(4 * BS));
    assert_eq!(f.staged_hashes(), before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_block_boundary_keeps_one_block_for_compute() {
    let mut f = Fixture::new(6 * BS);
    f.cache_six();
    f.query(0);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(5 * BS));
    f.query(4 * BS);
    assert_eq!(f.slot.state(), SlotState::OnboardStaged(BS));
    assert_eq!(f.staged_hashes(), f.sequence_hashes(4, 5));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn partial_tail_with_no_external_block_clears_staging() {
    let mut f = Fixture::new(6 * BS + 1);
    f.cache_six();
    f.query(0);
    f.query(6 * BS);
    assert_eq!(f.slot.state(), SlotState::Initialized);
    assert!(f.staged_hashes().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_drop_returns_staged_blocks_to_both_pools() {
    let mut f = Fixture::new(7 * BS);
    f.cache_six();
    f.query(0);
    drop(f.slot);
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if f.host.status().await.unwrap().active_blocks == 0
                && f.disk.status().await.unwrap().active_blocks == 0
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
