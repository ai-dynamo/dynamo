// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend storage strategies for InactivePool.

use super::*;

mod fifo;
mod hashmap_backend;
mod lineage;
mod lru_backend;
mod multi_lru_backend;
mod reuse_policy;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use fifo::FifoReusePolicy;

pub(crate) use hashmap_backend::HashMapBackend;
pub(crate) use lineage::LineageBackend;
pub(crate) use lru_backend::LruBackend;
pub(crate) use multi_lru_backend::MultiLruBackend; // Not used widely yet

pub use reuse_policy::{ReusePolicy, ReusePolicyError};

use super::SequenceHash;
