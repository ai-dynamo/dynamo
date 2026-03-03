// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../docs/architecture.md")]

pub use kvbm_common::{BlockId, LogicalLayoutHandle, SequenceHash};
pub use velo::{InstanceId, PeerInfo, WorkerAddress};

/// GPU/device tier -- HBM KV cache. Fastest access, smallest capacity.
/// Blocks here are actively used by attention kernels.
#[derive(Clone, Copy, Debug)]
pub struct G1;
/// CPU/host tier -- pinned DRAM cache. Microsecond-latency staging area
/// for RDMA transfers and G3/G4 promotion.
#[derive(Clone, Copy, Debug)]
pub struct G2;
/// Disk tier -- NVMe/SSD cache. Millisecond-latency persistent storage
/// for warm blocks.
#[derive(Clone, Copy, Debug)]
pub struct G3;
/// Object store tier -- S3/MinIO. Highest latency but unlimited capacity
/// for cold/archival blocks.
#[derive(Clone, Copy, Debug)]
pub struct G4;

#[cfg(feature = "collectives")]
pub mod collectives;
#[doc = include_str!("../docs/leader.md")]
pub mod leader;
#[doc = include_str!("../docs/object.md")]
pub mod object;
#[doc = include_str!("../docs/offload.md")]
pub mod offload;
pub mod pubsub;
#[doc = include_str!("../docs/runtime.md")]
pub mod runtime;
#[doc = include_str!("../docs/worker.md")]
pub mod worker;

#[cfg(any(test, feature = "testing"))]
pub mod testing;

pub use runtime::{KvbmRuntime, KvbmRuntimeBuilder, RuntimeHandle};
