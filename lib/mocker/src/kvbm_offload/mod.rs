// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G1↔G2 offload simulation for the vLLM mocker.
//!
//! Drives a real kvbm-engine `OffloadEngine` + `InstanceLeader` in process
//! without touching real GPU/CPU memory. Bandwidth is modelled as a
//! processor-sharing queue so concurrent transfers on the same link fair-share
//! throughput rather than all getting peak bandwidth.
//!
//! Gated behind `#[cfg(feature = "kvbm-offload")]`.

pub mod accountant;
pub mod config;
pub mod engine;
pub mod worker;

pub use accountant::{BandwidthAccountant, TransferId};
pub use config::KvbmOffloadConfig;
pub use engine::{MockOffloadEngine, SwapInHandle};
pub use worker::{MockWorker, TransferDirection};

/// Timing mode for `MockOffloadEngine` / `MockWorker`. Runtime logic is the
/// same for both variants; only the caller's source of `now_ms` differs
/// (wall clock vs virtual replay time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockSource {
    Real,
    Virtual,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_source_round_trip() {
        assert_eq!(ClockSource::Real, ClockSource::Real);
        assert_ne!(ClockSource::Real, ClockSource::Virtual);
    }
}
