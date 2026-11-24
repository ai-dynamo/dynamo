// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The MCS backend, using SIG Multicluster: pools ([`McsPoolDirectory`]) and
//! Global Router replicas ([`McsReplicaDirectory`]) from the EndpointSlices
//! that MCS imports, named as [`McsImplementation`] describes.
//!
//! It assumes a trusted network where pod addresses are reachable between
//! clusters (for example Azure CNI with peered VNets, or the AWS VPC CNI with
//! peered VPCs), and an MCS implementation, Karmada first.
//!
//! Each pool's PoolRelay Service, named `<DGD name>-pool-relay` by default,
//! needs a `ServiceExport`, and so does the router's Service in the hub, so
//! its replicas appear in every importing cluster. A pool is live while a
//! PoolRelay endpoint is Ready, and its `relay_identities` are the addresses
//! of every Ready PoolRelay pod, for routers on a trusted network.
//!
//! A pool's frontends come from its exported `<DGD name>-frontend` Service,
//! and its region from [`ClusterRegions`]. Routers learn what it serves from
//! its PoolRelays' state streams, so this backend takes no registrations.
//!
//! With Karmada, a Service name may have at most 55 characters: DGD names up
//! to 44 with the default `-pool-relay` suffix. Karmada imports a Service as
//! `derived-<name>`, and that name may have at most 63 characters, so
//! Karmada imports nothing for a longer name, and the backend never sees that
//! Service's endpoints. [`McsReplicaDirectory::new`] logs an error for a
//! Service whose name is too long; [`McsImplementation::can_import`] lets a
//! caller refuse that name first.
//!
//! Create the PoolRelay Service and its `ServiceExport` in the workload
//! cluster, not through Karmada propagation. Kubernetes copies a Service's
//! labels onto its EndpointSlices, so a Karmada-propagated Service gives its
//! slices Karmada's labels. In a test with Karmada v1.19, deleting such a
//! `ServiceExport` made karmada-agent delete the Service's EndpointSlice
//! too. Exporting the Service again then found no slice until the Service
//! changed.
//!
//! Karmada watches a member's `ServiceExport`s only while it has propagated
//! at least one `ServiceExport` there, so propagate one to each member.
//!
//! ```ignore
//! let regions = ClusterRegions::new(configured, Some(client.clone()));
//! let pools = McsPoolDirectory::new(client.clone(), mcs, suffix, frontends, regions)?;
//! let routers = McsReplicaDirectory::new(client, &router, mcs);
//! ```

mod cache;
mod endpoints;
mod hub;
mod mcs;
mod regions;

pub use endpoints::{ExportedService, McsReplicaDirectory};
pub use hub::{FrontendService, McsPoolDirectory};
pub use mcs::{KARMADA_MAX_SERVICE_NAME, McsImplementation};
pub use regions::ClusterRegions;
