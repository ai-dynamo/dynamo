// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable KV cache block managers.

/// Result of an atomic native-G1 capacity acquisition.
pub(crate) enum G1Acquire<T> {
    Ready(T),
    CapacityExhausted,
}

impl<T> G1Acquire<T> {
    pub(crate) fn map<U>(self, f: impl FnOnce(T) -> U) -> G1Acquire<U> {
        match self {
            Self::Ready(value) => G1Acquire::Ready(f(value)),
            Self::CapacityExhausted => G1Acquire::CapacityExhausted,
        }
    }
}

mod g1_manager;
pub(crate) mod sglang_backend;
mod vllm_backend;
#[cfg(test)]
mod vllm_firewall_tests;

pub(crate) use g1_manager::DestinationReservation;
pub(crate) use g1_manager::G1Manager;
pub(crate) use sglang_backend::SglangKvManager;
pub(crate) use vllm_backend::BlockRequestLease;
