// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;

use super::identity::RelayIdentity;
use super::watch::{Directory, WatchEvent};
use crate::global_view::PoolKey;
use crate::global_view::state::PoolLocation;

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
    // The fields below are `None` until the Relay's catalog reports them, in
    // modes where discovery only knows the pool's name (Kubernetes / MCS).
    /// With `frontend_endpoint`, becomes `RelayPoolScope` in dynamo-llm.
    #[builder(default)]
    pub runtime_namespace: Option<String>,
    #[builder(default)]
    pub frontend_endpoint: Option<String>,
    #[builder(default)]
    pub model: Option<String>,
    /// The only identity allowed to register this pool and stream its state.
    #[builder(default)]
    pub relay_identity: Option<RelayIdentity>,
    #[builder(default)]
    pub revision: Revision,
}

pub type PoolDirectory = dyn Directory<PoolRecord, PoolKey>;
pub type PoolEvent = WatchEvent<PoolRecord, PoolKey>;
