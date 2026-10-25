// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use url::Url;

use super::identity::RelayIdentity;
use super::watch::{Directory, WatchEvent};
use dynamo_global_router_shared::PoolKey;
use dynamo_global_router_shared::PoolLocation;

/// Orders updates to one pool's record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Revision(pub u64);

/// One pool as discovery knows it. Pool state (load, KV, readiness) is not
/// part of the record. Router replicas derive the routing `PoolId` from `key`
/// with the shared `PoolIdDeriver`.
#[derive(Clone, Debug, PartialEq, Eq, Builder)]
#[builder(setter(into))]
pub struct PoolRecord {
    pub key: PoolKey,
    pub location: PoolLocation,
    // The fields below are empty until the pool registers them or the Relay's
    // catalog reports them. Read directly, Kubernetes (MCS) mode knows only the
    // pool's name and frontends.
    /// With `frontend_endpoint`, becomes `RelayPoolScope` in dynamo-llm.
    #[builder(default)]
    pub runtime_namespace: Option<String>,
    /// Dynamo endpoint path of the pool's frontend, for example
    /// `frontend.generate`.
    #[builder(default)]
    pub frontend_endpoint: Option<String>,
    #[builder(default)]
    pub model: Option<String>,
    #[builder(default)]
    pub capabilities: Vec<String>,
    /// URLs of the pool's local frontends, which a Global Router sends
    /// requests to.
    #[builder(default)]
    pub frontends: Vec<Url>,
    #[builder(default)]
    pub capacity: Vec<Capacity>,
    /// The only identity allowed to register this pool and stream its state.
    #[builder(default)]
    pub relay_identity: Option<RelayIdentity>,
    #[builder(default)]
    pub revision: Revision,
}

/// The workers of one hardware SKU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capacity {
    pub sku: String,
    pub count: u32,
}

pub type PoolDirectory = dyn Directory<PoolRecord, PoolKey>;
pub type PoolEvent = WatchEvent<PoolRecord, PoolKey>;
