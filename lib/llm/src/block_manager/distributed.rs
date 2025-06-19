// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod transfer;
mod utils;
mod zmq;

mod leader;
mod worker;

pub use leader::{KvbmLeader, KvbmLeaderConfig};
pub use worker::{KvbmWorker, KvbmWorkerConfig};

#[cfg(all(test, feature = "testing-cuda", feature = "testing-etcd"))]
mod tests {
    use crate::block_manager::storage::{
        torch::{TorchDevice, TorchTensor},
        DeviceAllocator, Storage, StorageAllocator,
    };

    use anyhow::Result;

    use dynamo_runtime::logging::init as init_logging;

    use super::*;

    #[derive(Clone, Debug)]
    struct MockTensor {
        ptr: u64,
    }

    impl MockTensor {
        fn new() -> Self {
            let allocator = DeviceAllocator::new(0).unwrap();

            let device_storage = std::mem::ManuallyDrop::new(allocator.allocate(4096).unwrap());

            let ptr = device_storage.addr();
            Self { ptr }
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
            1024 * 1024 * 1024
        }

        fn shape(&self) -> Vec<usize> {
            vec![2, 8, 32]
        }

        fn stride(&self) -> Vec<usize> {
            vec![512, 64, 1]
        }
    }

    #[test]
    fn test_leader_worker_sync() -> Result<()> {
        init_logging();

        const NUM_WORKERS: usize = 4;

        let mut workers = Vec::new();

        // We're actually able to test this all in a single thread.
        // Worker startup is async. It returns immediately, and spins up a worker which waits on etcd + zmq init.
        // On the other hand, the leader startup is fully synchronous. It will only return once it's established a zmq connection with all workers.
        for i in 0..NUM_WORKERS {
            let tensors: Vec<Box<dyn TorchTensor>> = vec![Box::new(MockTensor::new())];

            let config = KvbmWorkerConfig::builder()
                .num_device_blocks(8)
                .tensors(tensors)
                .worker_id(i)
                .build()?;

            let worker = KvbmWorker::new(config)?;
            workers.push(worker);
        }

        let leader_config = KvbmLeaderConfig::builder()
            .world_size(NUM_WORKERS)
            .bytes_per_block(1)
            .build()?;

        // When/if this returns, we know that all the workers were also successful.
        let _ = KvbmLeader::new(leader_config)?;

        Ok(())
    }
}
