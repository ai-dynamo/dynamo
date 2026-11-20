// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Router replicas. Every replica needs every pool's state, so a
//! PoolRelay opens one connection per replica.

use serde::{Deserialize, Serialize};
use url::Url;

use crate::watch::WatchEvent;

/// In JSON, a plain string.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouterReplicaId(pub String);

/// A Global Router replica and the address workload clusters reach it at.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RouterReplica {
    pub replica: RouterReplicaId,
    pub url: Url,
}

pub type RouterReplicaEvent = WatchEvent<RouterReplica, RouterReplicaId>;
