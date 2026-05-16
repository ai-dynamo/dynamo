// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod layout;
pub mod manager;
pub mod transfer;

// The device abstraction (DeviceContext, DeviceStream, DeviceEvent,
// DeviceMemPool, the *Ops trait surface, and CUDA / SYCL impls) lives
// in the standalone `dynamo-device` crate. Re-export it as `device` so
// existing call sites that say `kvbm_physical::device::DeviceContext`
// continue to compile.
pub use dynamo_device as device;

pub use manager::TransferManager;
pub use transfer::{TransferConfig, TransferOptions};

pub use kvbm_common::BlockId;
pub type SequenceHash = kvbm_common::SequenceHash;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

#[cfg(test)]
#[cfg(not(feature = "testing-kvbm"))]
mod sentinel {
    #[test]
    #[allow(non_snake_case)]
    fn all_functional_tests_skipped___enable_testing_kvbm() {
        eprintln!(
            "kvbm-physical functional tests require feature `testing-kvbm`. \
             Run with: cargo test -p kvbm-physical --features testing-kvbm"
        );
    }
}
