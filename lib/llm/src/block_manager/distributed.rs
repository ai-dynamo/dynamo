// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod transfer;
mod utils;
mod zmq;

mod leader;
mod worker;

pub use leader::KvbmLeader;
pub use worker::KvbmWorker;

#[cfg(all(test, feature = "testing-cuda", feature = "testing-etcd"))]
mod tests {
    use crate::block_manager::storage::{
        torch::{TorchDevice, TorchTensor},
        DeviceAllocator, Storage, StorageAllocator,
    };
    use crate::common::dtype::DType;

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
            vec![2, 8, 16]
        }

        fn stride(&self) -> Vec<usize> {
            vec![256, 32, 1]
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
            let worker = KvbmWorker::new(8, 4, tensors, 0, i, DType::FP16, "kvbm".to_string())?;
            workers.push(worker);
        }

        // When/if this returns, we know that all the workers were also successful.
        let _ = KvbmLeader::new("kvbm".to_string(), 1, NUM_WORKERS)?;

        Ok(())
    }
}
