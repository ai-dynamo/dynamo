// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_global_common::PoolKey;
pub use dynamo_global_common::{Capacity, PoolListing};

use super::watch::{Directory, WatchEvent};

/// The pools discovery knows. A pool's state (load, KV, readiness) is not part
/// of its listing. Router replicas derive the routing `PoolId` from each
/// listing's `key` with the shared `PoolIdDeriver`.
pub type PoolDirectory = dyn Directory<PoolListing, PoolKey>;
pub type PoolEvent = WatchEvent<PoolListing, PoolKey>;
