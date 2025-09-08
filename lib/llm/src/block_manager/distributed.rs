// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod transfer;
mod utils;
mod zmq;

mod leader;
mod worker;

pub use leader::{KvbmLeader, KvbmLeaderConfig, KvbmLeaderNumBlocksConfig};
pub use transfer::BlockTransferHandler;
pub use utils::{
    BlockTransferPool, BlockTransferRequest, ConnectorRequestLeader, ConnectorTransferType,
};
pub use worker::{KvbmWorker, KvbmWorkerConfig};
pub use zmq::Handler;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskReady {
    Continue,
    Cancel,
}
#[async_trait::async_trait]
pub trait ScheduledTaskHandle: Send + Sync {
    async fn ready(&self) -> TaskReady;
    fn mark_complete(&self);
}

pub struct SchedulerRequest<T> {
    pub handle_tx: tokio::sync::oneshot::Sender<Box<dyn ScheduledTaskHandle>>,
    pub task: T,
}

// impl<T> SchedulerRequest<T> {
//     pub fn new(task: T) -> (Self, tokio::sync::oneshot::Sender<Box<dyn ScheduledTaskHandle>>) {
//         let (handle_tx, handle_rx) = tokio::sync::oneshot::channel();
//         Self { handle_tx, task }
//     }
// }

#[cfg(all(test, feature = "testing-cuda", feature = "testing-etcd"))]
mod tests {
    use super::*;

    use crate::block_manager::KvBlockManager;
    use crate::block_manager::block::BasicMetadata;
    use crate::block_manager::block::data::logical::distributed_leader_worker::DistributedLeaderWorkerResources;
    use crate::block_manager::config::*;
    use crate::block_manager::locality::Logical;
    use crate::block_manager::storage::{
        DeviceAllocator, Storage, StorageAllocator,
        torch::{TorchDevice, TorchTensor},
    };

    use anyhow::Result;
    use rstest::*;

    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tokio_util::sync::CancellationToken;

    use dynamo_runtime::logging::init as init_logging;

    const NUM_BLOCKS: usize = 8;

    #[derive(Clone, Debug)]
    struct MockTensor {
        ptr: u64,
        size: usize,
        shape: Vec<usize>,
    }

    impl MockTensor {
        fn new(shape: Vec<usize>) -> Self {
            let allocator = DeviceAllocator::new(0).unwrap();

            // Multiply by 2 for fp16.
            let size = shape.iter().product::<usize>() * 2;

            let device_storage = std::mem::ManuallyDrop::new(allocator.allocate(size).unwrap());

            let ptr = device_storage.addr();
            Self { ptr, size, shape }
        }
    }

    impl TorchTensor for MockTensor {
        fn device(&self) -> TorchDevice {
            TorchDevice::Cuda(0)
        }

        fn data_ptr(&self) -> u64 {
            self.ptr
        }

        fn size_bytes(&self) -> usize {
            self.size
        }

        fn shape(&self) -> Vec<usize> {
            self.shape.clone()
        }

        fn stride(&self) -> Vec<usize> {
            // Generate the stride on the assumption that it is contiguous.
            let mut stride = vec![1];
            for i in (0..self.shape.len() - 1).rev() {
                stride.push(stride.last().unwrap() * self.shape[i]);
            }
            stride.reverse();
            stride
        }
    }

    fn get_unique_barrier_id() -> String {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);

        COUNTER.fetch_add(1, Ordering::Relaxed).to_string()
    }

    async fn build_leader_and_workers(num_workers: usize) -> Result<(KvbmLeader, Vec<KvbmWorker>)> {
        let mut workers = Vec::new();
        let barrier_id = get_unique_barrier_id();

        for i in 0..num_workers {
            let tensors: Vec<Arc<dyn TorchTensor>> =
                vec![Arc::new(MockTensor::new(vec![2, NUM_BLOCKS, 4096]))];

            let config = KvbmWorkerConfig::builder()
                .barrier_id_prefix(barrier_id.clone())
                .num_device_blocks(NUM_BLOCKS)
                .tensors(tensors)
                .device_id(i)
                .build()?;

            let worker = KvbmWorker::new(config, false).await?;
            workers.push(worker);
        }

        let host_blocks = KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 1.0,
            num_blocks_overriden: NUM_BLOCKS,
        };

        let disk_blocks = KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 1.0,
            num_blocks_overriden: NUM_BLOCKS,
        };

        let leader_config = KvbmLeaderConfig::builder()
            .barrier_id_prefix(barrier_id)
            .world_size(num_workers)
            .host_blocks_config(host_blocks)
            .disk_blocks_config(disk_blocks)
            .build()?;

        // When/if this returns, we know that all the workers were also successful.
        let leader = KvbmLeader::new(leader_config).await?;

        Ok((leader, workers))
    }

    #[tokio::test]
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    async fn test_leader_worker_sync_and_transfer(#[case] num_workers: usize) -> Result<()> {
        init_logging();

        let (leader, _workers) = build_leader_and_workers(num_workers).await?;

        // Do a whole bunch of distributed transfers.

        for block_idx in 0..NUM_BLOCKS {
            leader
                .transfer_blocks_request(utils::BlockTransferRequest::new(
                    utils::BlockTransferPool::Device,
                    utils::BlockTransferPool::Host,
                    vec![(block_idx, block_idx)],
                ))
                .await?
                .await?;
        }

        for block_idx in 0..NUM_BLOCKS {
            leader
                .transfer_blocks_request(utils::BlockTransferRequest::new(
                    utils::BlockTransferPool::Host,
                    utils::BlockTransferPool::Disk,
                    vec![(block_idx, block_idx)],
                ))
                .await?
                .await?;
        }

        for block_idx in 0..NUM_BLOCKS {
            leader
                .transfer_blocks_request(utils::BlockTransferRequest::new(
                    utils::BlockTransferPool::Disk,
                    utils::BlockTransferPool::Device,
                    vec![(block_idx, block_idx)],
                ))
                .await?
                .await?;
        }

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[case(1)]
    #[case(2)]
    #[case(4)]
    #[case(8)]
    async fn test_leader_worker_transfer_e2e(#[case] num_workers: usize) -> Result<()> {
        init_logging();

        const BLOCK_SIZE: usize = 4;

        let (leader, _workers) = build_leader_and_workers(num_workers).await?;

        let cancel_token = CancellationToken::new();

        let config = KvBlockManagerConfig::builder()
            .runtime(
                KvManagerRuntimeConfig::builder()
                    .worker_id(0)
                    .cancellation_token(cancel_token.clone())
                    .build()?,
            )
            .model(
                KvManagerModelConfig::builder()
                    .num_layers(1)
                    .outer_dim(1)
                    .page_size(BLOCK_SIZE)
                    .inner_dim(1)
                    .build()?,
            )
            .device_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(NUM_BLOCKS)
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            )
            .host_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(NUM_BLOCKS)
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            )
            .disk_layout(
                KvManagerLayoutConfig::builder()
                    .num_blocks(NUM_BLOCKS)
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            )
            .build()?;

        let resources = DistributedLeaderWorkerResources::new(
            Some(Arc::new(leader)),
            cancel_token.child_token(),
        )?;

        let block_manager = KvBlockManager::<
            Logical<DistributedLeaderWorkerResources>,
            BasicMetadata,
        >::new(config, resources)
        .await
        .unwrap();

        let device_pool = block_manager.device().unwrap();
        let host_pool = block_manager.host().unwrap();
        let disk_pool = block_manager.disk().unwrap();

        let mut device_blocks = device_pool.allocate_blocks(NUM_BLOCKS).await?;

        let mut sequence_hashes = Vec::new();
        for block in &mut device_blocks {
            block.init_sequence(42).unwrap();

            for _ in 0..BLOCK_SIZE {
                block.add_token(42).unwrap();
            }

            block.commit().unwrap();

            sequence_hashes.push(block.sequence_hash().unwrap());
        }

        // Register our blocks on the device.
        let immutable_device_blocks = device_pool.register_blocks(device_blocks).await?;

        // Wait for the blocks to be offloaded.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Now, all blocks should be on the host.
        let host_blocks = host_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;

        assert_eq!(host_blocks.len(), NUM_BLOCKS);

        let disk_blocks = disk_pool
            .match_sequence_hashes(sequence_hashes.as_slice())
            .await?;

        assert_eq!(disk_blocks.len(), NUM_BLOCKS);

        // Return the device blocks to the pool.
        drop(immutable_device_blocks);

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Clear out the device pool.
        let _ = device_pool.allocate_blocks(NUM_BLOCKS).await?;

        // Now, all the blocks should be gone.
        assert_eq!(
            device_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            0
        );

        // Wait for the device blocks to be returned to the pool.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Now, onboard them back to the device.
        let new_device_blocks = block_manager.onboard_blocks(host_blocks, None).await??;

        assert_eq!(new_device_blocks.len(), NUM_BLOCKS);

        Ok(())
    }
}
