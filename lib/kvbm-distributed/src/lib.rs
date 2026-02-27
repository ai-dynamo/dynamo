// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distributed coordination primitives for KVBM.

pub use kvbm_common::{BlockId, LogicalLayoutHandle, SequenceHash};
pub use velo::{InstanceId, PeerInfo, WorkerAddress};

/// G1 marker: GPU/device tier
#[derive(Clone, Copy, Debug)]
pub struct G1;
/// G2 marker: CPU/host tier
#[derive(Clone, Copy, Debug)]
pub struct G2;
/// G3 marker: Disk tier
#[derive(Clone, Copy, Debug)]
pub struct G3;
/// G4 marker: Object store tier
#[derive(Clone, Copy, Debug)]
pub struct G4;

#[cfg(feature = "collectives")]
pub mod collectives;
pub mod leader;
pub mod object;
pub mod offload;
pub mod pubsub;
pub mod runtime;
pub mod worker;
pub mod workers;

pub use runtime::{KvbmRuntime, KvbmRuntimeBuilder, RuntimeHandle};
