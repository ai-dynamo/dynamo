// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Router replicas a pool must connect to. Every replica needs every
//! pool's state, so a PoolRelay opens one connection per replica, and gets
//! the list from the service as `GET /routers`.
//!
//! A replica's readiness must not wait for its pools: Relays dial only
//! listed (Ready) replicas, so otherwise no pool could ever reach it. Pool
//! completeness is a separate signal ([`super::Completeness`]).

pub use dynamo_global_common::{ReplicaEndpoint, ReplicaEvent, ReplicaId};

use super::watch::Directory;

pub type ReplicaDirectory = dyn Directory<ReplicaEndpoint, ReplicaId>;
