// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes mode, using SIG Multicluster: pools ([`hub`]) and router
//! replicas ([`endpoints`]) from the EndpointSlices that MCS imports, named
//! as [`mcs`] describes. The Global Discovery Service
//! (`crate::service`) serves both as its MCS store.
//!
//! V1 assumes a trusted network where pod addresses are reachable between
//! clusters (for example Azure CNI with peered VNets, or the AWS VPC CNI with
//! peered VPCs), and an MCS implementation, Karmada first.
//!
//! The operator creates a `ServiceExport` for each pool's PoolRelay Service,
//! named `<DGD name>-pool-relay` by default. The hub exports the router's
//! Service, so its replicas appear in every importing cluster. Read
//! directly, a pool is live while its PoolRelay endpoint is Ready. The Global
//! Discovery Service also needs a live registration, which says what the
//! pool serves ([`crate::backends::registered`]), and reads the pool's
//! frontend addresses from its exported frontend Service, named
//! `<DGD name>-frontend` by default. Router replicas still register with the
//! hub's Dynamo runtime discovery, as in every mode.
//!
//! Create the PoolRelay Service and its `ServiceExport` in the workload
//! cluster, not through Karmada propagation. Kubernetes copies a Service's
//! labels onto its EndpointSlices, so a Karmada-propagated Service gives its
//! slices Karmada's labels. In a test with Karmada v1.19, deleting such a
//! `ServiceExport` made karmada-agent delete the Service's EndpointSlice
//! too. Exporting the Service again then found no slice until the Service
//! changed.
//!
//! Router replica in the hub, reading pools itself:
//!
//! ```ignore
//! let discovery = McsPoolDiscovery {
//!     replicas, imports, mcs, pool_service_suffix,
//!     relay_auth: RelayAuth::TrustedNetwork, locations,
//! }
//! .build()
//! .await?;
//! let mut pools = discovery.pools.list_and_watch(Some(cancel.clone())).await?;
//! // WatchEvent::{Added, Removed, Resync} update the GlobalViewRuntime's pool set.
//! // StateReceiver::accept checks each Relay with discovery.authenticator
//! // and admit(&*discovery.pools, ..).
//! ```
//!
//! Global Discovery Service with the MCS store:
//!
//! ```ignore
//! let registrations = Arc::new(MemoryPoolStore::new(lease)?);
//! let imported =
//!     McsPoolDirectory::with_frontends(client.clone(), mcs, suffix, true, locations, frontends)?;
//! let store = DiscoveryStore {
//!     pools: Arc::new(RegisteredPoolDirectory::new(Arc::new(imported), registrations.clone(), lease)),
//!     routers: Arc::new(EndpointSliceReplicaDirectory::imported(client, &router, mcs)),
//!     registrar: Some(registrations),
//!     authenticator: Arc::new(TrustedNetworkAuthenticator),
//!     // Or `Some`, with a `crate::peers::PeerSync` task.
//!     sharing: None,
//! };
//! service::serve(listener, store, shutdown).await?;
//! ```

mod cache;
pub mod endpoints;
pub mod hub;
pub mod mcs;

pub use endpoints::{EndpointSliceReplicaDirectory, RouterService};
pub use hub::{
    FrontendService, McsPoolDirectory, McsPoolDiscovery, MeshAuthenticator, RelayAuth,
    TrustedNetworkAuthenticator, address_identity,
};
