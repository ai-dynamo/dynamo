// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The MCS store, using SIG Multicluster: pools ([`McsPoolDirectory`]) and
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
//! PoolRelay endpoint is Ready. Its `relay_identities` are that PoolRelay
//! pod's addresses, for routers on a trusted network to admit it.
//!
//! With Karmada, a Service name may have at most 57 characters: DGD names up
//! to 46 with the default suffix. Kubernetes shortens a longer Service's
//! EndpointSlice names, and Karmada's names then no longer say which cluster
//! a slice came from. Such slices are skipped, with a warning.
//!
//! Create the PoolRelay Service and its `ServiceExport` in the workload
//! cluster, not through Karmada propagation. Kubernetes copies a Service's
//! labels onto its EndpointSlices, so a Karmada-propagated Service gives its
//! slices Karmada's labels. In a test with Karmada v1.19, deleting such a
//! `ServiceExport` made karmada-agent delete the Service's EndpointSlice
//! too. Exporting the Service again then found no slice until the Service
//! changed.

mod cache;
mod endpoints;
mod hub;
mod mcs;

pub use endpoints::{ExportedService, McsReplicaDirectory};
pub use hub::McsPoolDirectory;
pub use mcs::McsImplementation;
