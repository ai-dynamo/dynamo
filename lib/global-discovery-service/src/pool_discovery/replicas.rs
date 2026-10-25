// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router replicas a pool must connect to. Every replica needs every pool's
//! state, so a Relay opens one connection per replica.
//!
//! The operator deploys the Global Router, so each replica registers itself
//! with the hub's Dynamo runtime discovery, like a worker. Relays get the
//! list from the Global Discovery Service, as `GET /routers`.
//!
//! Runtime discovery lists only Ready pods, and Relays dial only listed
//! replicas. So a replica's readiness must not wait for its pools, or no
//! pool can ever reach it. Pool completeness is a separate signal
//! ([`super::Completeness`]).
//!
//! The Relay reads a `ReplicaDirectory` in every mode, filled from
//! `GET /routers`. In Kubernetes mode the service's own directory reads the
//! router's EndpointSlices that MCS imports.

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
