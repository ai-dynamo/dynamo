// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use url::Url;

use super::watch::{Directory, WatchEvent};
pub use dynamo_global_common::Capacity;
use dynamo_global_common::{PoolKey, PoolLocation};

/// Orders updates to one pool's record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Revision(pub u64);

/// One pool as discovery knows it. Pool state (load, KV, readiness) is not
/// part of the record. Router replicas derive the routing `PoolId` from `key`
/// with the shared `PoolIdDeriver`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolRecord {
    pub key: PoolKey,
    pub location: PoolLocation,
    // The fields below are empty until the pool registers them. The MCS store
    // knows only the pool's name and its PoolRelay.
    pub model: Option<String>,
    pub capabilities: Vec<String>,
    /// URLs of the pool's local frontends, which a Global Router sends
    /// requests to.
    pub frontends: Vec<Url>,
    pub capacity: Vec<Capacity>,
    /// The only PoolRelay allowed to stream this pool's state. On a trusted
    /// network it is the PoolRelay's address, as
    /// [`dynamo_global_common::address_identity`] writes it.
    pub relay_identity: Option<String>,
    /// Set by the store.
    pub revision: Revision,
}

pub type PoolDirectory = dyn Directory<PoolRecord, PoolKey>;
pub type PoolEvent = WatchEvent<PoolRecord, PoolKey>;
