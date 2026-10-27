// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The MCS store, using SIG Multicluster: pools ([`hub`]) and Global Router
//! replicas ([`endpoints`]) from the EndpointSlices that MCS imports, named as
//! [`mcs`] describes.
//!
//! It assumes a trusted network where pod addresses are reachable between
//! clusters (for example Azure CNI with peered VNets, or the AWS VPC CNI with
//! peered VPCs), and an MCS implementation, Karmada first.
//!
//! The operator creates a `ServiceExport` for each pool's PoolRelay Service,
//! named `<DGD name>-pool-relay` by default, and the hub exports the router's
//! Service, so its replicas appear in every importing cluster. A pool is live
//! while its PoolRelay endpoint is Ready, and its `relay_identity` is that
//! endpoint's address, which routers on a trusted network use to admit the
//! PoolRelay.
//!
//! Create the PoolRelay Service and its `ServiceExport` in the workload
//! cluster, not through Karmada propagation. Kubernetes copies a Service's
//! labels onto its EndpointSlices, so a Karmada-propagated Service gives its
//! slices Karmada's labels. In a test with Karmada v1.19, deleting such a
//! `ServiceExport` made karmada-agent delete the Service's EndpointSlice
//! too. Exporting the Service again then found no slice until the Service
//! changed.

mod cache;
pub mod endpoints;
pub mod hub;
pub mod mcs;

pub use endpoints::{EndpointSliceReplicaDirectory, ExportedService};
pub use hub::McsPoolDirectory;
pub use mcs::McsImplementation;
