// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend storage strategies for InactivePool.

use super::*;

mod fifo;
mod hashmap_backend;
mod lru_backend;
mod multi_lru_backend;
mod reuse_policy;

#[cfg(test)]
mod tests;

pub(crate) use fifo::FifoReusePolicy;
pub(crate) use hashmap_backend::HashMapBackend;
pub(crate) use lru_backend::LruBackend;
pub(crate) use multi_lru_backend::MultiLruBackend;

pub use reuse_policy::{ReusePolicy, ReusePolicyError};

use super::SequenceHash;
