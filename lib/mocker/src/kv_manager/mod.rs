// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable KV cache block managers.

#[cfg(feature = "kvbm-offload")]
use crate::kvbm_offload::OffloadId;

#[cfg(not(feature = "kvbm-offload"))]
type OffloadId = u64;

/// Result of an atomic G1 capacity acquisition, shared by both G1 backends.
#[cfg_attr(not(feature = "kvbm-offload"), allow(dead_code))]
pub(crate) enum G1Acquire<T> {
    Ready(T),
    CapacityExhausted,
    BlockedOnOffload {
        offload_id: OffloadId,
        deadline_ms: Option<f64>,
    },
    RetryNow {
        capacity_generation: u64,
        released_slots: usize,
    },
}

impl<T> G1Acquire<T> {
    pub(crate) fn map<U>(self, f: impl FnOnce(T) -> U) -> G1Acquire<U> {
        match self {
            Self::Ready(value) => G1Acquire::Ready(f(value)),
            Self::CapacityExhausted => G1Acquire::CapacityExhausted,
            Self::BlockedOnOffload {
                offload_id,
                deadline_ms,
            } => G1Acquire::BlockedOnOffload {
                offload_id,
                deadline_ms,
            },
            Self::RetryNow {
                capacity_generation,
                released_slots,
            } => G1Acquire::RetryNow {
                capacity_generation,
                released_slots,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct OffloadDependency {
    pub(crate) offload_id: OffloadId,
    pub(crate) deadline_ms: Option<f64>,
}

mod g1_manager;
pub mod kvbm_backend;
pub mod sglang_backend;
mod vllm_backend;
#[cfg(test)]
mod vllm_firewall_tests;

pub(crate) use g1_manager::DestinationReservation;
pub use g1_manager::G1Manager;
pub use kvbm_backend::KvManager;
pub use sglang_backend::SglangKvManager;
