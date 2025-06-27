// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod transfer;
mod utils;
mod zmq;

mod leader;
mod worker;

pub use leader::{KvbmLeader, KvbmLeaderConfig};
pub use utils::{BlockTransferPool, BlockTransferRequest};
pub use worker::{KvbmWorker, KvbmWorkerConfig};

#[cfg(all(test, feature = "testing-cuda", feature = "testing-etcd"))]
mod tests {
    use crate::block_manager::storage::{
        torch::{TorchDevice, TorchTensor},
        DeviceAllocator, Storage, StorageAllocator,
    };

    use anyhow::Result;
    use rstest::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use dynamo_runtime::logging::init as init_logging;

    use super::*;

    const NUM_DEVICE_BLOCKS: usize = 8;
    const NUM_HOST_BLOCKS: usize = 8;

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
            let tensors: Vec<Box<dyn TorchTensor>> =
                vec![Box::new(MockTensor::new(vec![2, NUM_DEVICE_BLOCKS, 4096]))];

            let config = KvbmWorkerConfig::builder()
                .barrier_id(barrier_id.clone())
                .num_device_blocks(NUM_DEVICE_BLOCKS)
                .tensors(tensors)
                .worker_id(i)
                .build()?;

            let worker = KvbmWorker::new(config).await?;
            workers.push(worker);
        }

        let leader_config = KvbmLeaderConfig::builder()
            .barrier_id(barrier_id)
            .world_size(num_workers)
            .num_host_blocks(NUM_HOST_BLOCKS)
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

        for block_idx in 0..std::cmp::min(NUM_DEVICE_BLOCKS, NUM_HOST_BLOCKS) {
            leader
                .transfer_blocks_request(utils::BlockTransferRequest::new(
                    utils::BlockTransferPool::Device,
                    utils::BlockTransferPool::Host,
                    vec![(block_idx, block_idx)],
                ))
                .await?
                .await?;
        }

        for block_idx in 0..std::cmp::min(NUM_DEVICE_BLOCKS, NUM_HOST_BLOCKS) {
            leader
                .transfer_blocks_request(utils::BlockTransferRequest::new(
                    utils::BlockTransferPool::Host,
                    utils::BlockTransferPool::Device,
                    vec![(block_idx, block_idx)],
                ))
                .await?
                .await?;
        }

        Ok(())
    }
}
