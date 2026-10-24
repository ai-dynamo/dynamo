// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Router replicas a PoolRelay connects to. Every replica needs every
//! pool's state, so a PoolRelay opens one connection per replica.

use serde::{Deserialize, Serialize};
use url::Url;

use crate::watch::WatchEvent;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReplicaId(pub String);

/// A replica's address as seen from workload clusters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplicaEndpoint {
    pub replica: ReplicaId,
    pub url: Url,
}

pub type ReplicaEvent = WatchEvent<ReplicaEndpoint, ReplicaId>;
