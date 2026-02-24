// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod layout;
pub mod manager;
pub mod transfer;

pub use manager::TransferManager;
pub use transfer::{TransferConfig, TransferOptions};

pub use kvbm_common::BlockId;
pub type SequenceHash = kvbm_common::SequenceHash;

#[cfg(test)]
#[cfg(not(all(feature = "testing-nixl", feature = "testing-cuda")))]
mod sentinel {
    #[test]
    #[allow(non_snake_case)]
    fn all_functional_tests_skipped___enable_testing_nixl_and_testing_cuda() {
        eprintln!(
            "kvbm-physical functional tests require features `testing-nixl` and `testing-cuda`. \
             Run with: cargo test -p kvbm-physical --features testing-nixl,testing-cuda"
        );
    }
}
