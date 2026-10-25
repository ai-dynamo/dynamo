// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON exchanged with the Global Discovery Service. The client ignores
//! response fields it does not know, so the service can add fields first.

use serde::{Deserialize, Serialize};
use url::Url;

use crate::replicas::ReplicaEndpoint;

#[derive(Debug, Deserialize)]
pub(crate) struct RoutersResponse {
    pub routers: Vec<ReplicaEndpoint>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PoolsResponse {
    pub pools: Vec<PoolListing>,
    /// True when the service lists every pool.
    pub complete: bool,
}

/// One pool in the `GET /pools` response.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
pub struct PoolListing {
    pub key: PoolKey,
    pub location: PoolLocation,
    /// `None` when the service knows only the pool's name, as with MCS.
    pub model: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
    /// Local frontends the Global Router dispatches requests to.
    #[serde(default)]
    pub frontends: Vec<Url>,
    #[serde(default)]
    pub capacity: Vec<Capacity>,
    /// The only Relay allowed to stream this pool's state, when the service
    /// knows it.
    #[serde(default)]
    pub relay_identity: Option<String>,
}

/// `POST /pools` request body. A PoolRelay sends it on every registration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolRegistration {
    pub key: PoolKey,
    pub location: PoolLocation,
    /// The model or checkpoint the pool serves.
    pub model: String,
    pub capabilities: Vec<String>,
    /// Local frontends the Global Router dispatches requests to.
    pub frontends: Vec<Url>,
    pub capacity: Vec<Capacity>,
}

/// `namespace` is the Kubernetes namespace containing the DGD, not the
/// Dynamo runtime namespace derived by the operator.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolKey {
    pub site_id: String,
    pub namespace: String,
    pub dgd_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolLocation {
    pub region: String,
    pub availability_zone: Option<String>,
    pub cluster: Option<String>,
    pub datacenter: Option<String>,
}

/// The workers of one hardware SKU.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capacity {
    pub sku: String,
    pub count: u32,
}
