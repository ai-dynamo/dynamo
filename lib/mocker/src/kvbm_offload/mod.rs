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

pub mod bandwidth_sharing_model;
pub mod config;
pub mod engine;
pub mod worker;

pub use bandwidth_sharing_model::{BandwidthSharingModel, TransferId};
pub use config::KvbmOffloadConfig;
pub use engine::{MockOffloadEngine, SwapInHandle};
pub use worker::{MockWorker, TransferDirection};
