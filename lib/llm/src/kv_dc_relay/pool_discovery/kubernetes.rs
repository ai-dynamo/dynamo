// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes mode, using SIG Multicluster (DEP-1277). Sketch only:
//! signatures, no bodies.
//!
//! V1 assumes a trusted network where pod addresses are reachable between
//! clusters (for example Azure CNI with peered VNets, or the AWS VPC CNI with
//! peered VPCs), and an MCS implementation, Karmada first.
//!
//! The operator creates a `ServiceExport` for each pool's PoolRelay Service.
//! Every router replica watches the imported EndpointSlices: in the hub for
//! standard MCS, or on the Karmada control plane. The hub exports the
//! router's Service, so each Relay finds every router replica in its own
//! cluster. No registrar, no announcer, no leases.
//!
//! Router replica in the hub:
//!
//! ```ignore
//! let discovery = McsPoolDiscovery {
//!     hub, imports, mcs, pool_service_suffix, router,
//!     relay_auth: RelayAuth::TrustedNetwork, locations,
//! }
//! .build()
//! .await?;
//! let mut pools = discovery.pools.list_and_watch(Some(cancel.clone())).await?;
//! // WatchEvent::{Added, Removed, Resync} update the GlobalViewRuntime's pool set.
//! // RelayListener::accept uses discovery.authenticator and discovery.admission.
//! ```
//!
//! Relay in a workload cluster:
//!
//! ```ignore
//! let replicas = EndpointSliceReplicaDirectory::imported(local, &router, mcs);
//! let mut events = replicas.list_and_watch(Some(cancel.clone())).await?;
//! // Added(replica): dialer.dial(&replica, &pool_key, publication.clone(), cancel.clone())
//! // Removed(id): close that connection
//! ```

// Sketch: fields and parameters are unused until the bodies exist.
#![allow(dead_code, unused_variables)]

pub mod endpoints;
pub mod hub;
pub mod mcs;

pub use endpoints::{EndpointSliceReplicaDirectory, RouterService};
pub use hub::{
    DirectoryAdmission, McsPoolDirectory, McsPoolDiscovery, MeshAuthenticator, RelayAuth,
    TrustedNetworkAuthenticator, address_identity,
};
