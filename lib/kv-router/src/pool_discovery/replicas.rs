// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router replicas a pool must connect to. Every replica needs every pool's
//! state, so a Relay opens one connection per replica.
//!
//! Replicas register themselves, so a `ReplicaDirectory` can be backed by
//! `dynamo_runtime::discovery::Discovery`. A replica must be listed before it
//! is ready, or pools can never reach it.

use url::Url;

use super::watch::{Directory, WatchEvent};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplicaId(pub String);

/// A replica's address as seen from workload clusters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplicaEndpoint {
    pub replica: ReplicaId,
    pub url: Url,
}

pub type ReplicaDirectory = dyn Directory<ReplicaEndpoint, ReplicaId>;
pub type ReplicaEvent = WatchEvent<ReplicaEndpoint, ReplicaId>;
