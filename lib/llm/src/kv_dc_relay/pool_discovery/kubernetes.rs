// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Full Kubernetes mode (DEP-1277). Sketch only: signatures, no bodies.
//!
//! Each pool writes a `DynamoPoolExport` and a `Lease` into its own namespace
//! on the hub. Every router replica watches them. There is no registrar.
//!
//! Router replica in the hub:
//!
//! ```ignore
//! let discovery = KubernetesPoolDiscovery { hub, lease_duration, pool_selector, audience }
//!     .build()
//!     .await?;
//! let mut pools = discovery.pools.list_and_watch(Some(cancel.clone())).await?;
//! // WatchEvent::{Added, Removed, Resync} update the GlobalViewRuntime's pool set.
//! // RelayListener::accept uses discovery.authenticator and discovery.admission.
//! ```
//!
//! Relay in a workload cluster:
//!
//! ```ignore
//! let announcer = KubernetesPoolAnnouncer::new(hub, replicas, lease_duration);
//! let announcement = announcer.announce(record.clone()).await?;
//! let mut replicas = announcement.replicas();
//! // For each replica in *replicas.borrow():
//! //     dialer.dial(&replica, &record.key, publication.clone(), cancel.clone())
//! // On shutdown:
//! announcement.withdraw().await?;
//! ```

// Sketch: fields and parameters are unused until the bodies exist.
#![allow(dead_code, unused_variables)]

pub mod crd;
pub mod hub;
pub mod workload;

pub use crd::{DynamoPoolExport, DynamoPoolExportSpec};
pub use hub::{
    ExportAdmission, KubernetesPoolDirectory, KubernetesPoolDiscovery, RuntimeReplicaDirectory,
    TokenReviewAuthenticator,
};
pub use workload::{KubernetesAnnouncement, KubernetesPoolAnnouncer};
