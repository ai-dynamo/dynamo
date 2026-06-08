// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G1↔G2, G2↔G3, and G2↔G4 offload simulation for the vLLM mocker.
//!
//! Drives a real kvbm-engine `OffloadEngine` + `InstanceLeader` in process
//! without touching real GPU/CPU memory. Bandwidth is modelled as a
//! processor-sharing queue so concurrent transfers on the same link fair-share
//! throughput rather than all getting peak bandwidth.
//!
//! Gated behind `#[cfg(feature = "kvbm-offload")]`.

pub mod bandwidth_sharing_model;
pub(crate) mod capacity_reservation;
pub mod config;
pub mod engine;
pub(crate) mod shared_g3;
pub(crate) mod shared_g4;
pub mod worker;

pub use bandwidth_sharing_model::{BandwidthSharingModel, TransferId};
pub use config::KvbmOffloadConfig;
pub(crate) use engine::{G2BlockEventMetadata, G2OffloadBlock, G2RouterEvent};
pub use engine::{MockOffloadEngine, SwapInHandle};
pub use worker::{MockWorker, TransferDirection};
