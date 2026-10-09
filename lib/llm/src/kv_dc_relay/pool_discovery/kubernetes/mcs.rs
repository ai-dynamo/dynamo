// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The parts of the MCS API (`multicluster.x-k8s.io/v1alpha1`) this mode
//! reads. Declared here because no Rust crate ships them. The MCS
//! implementation installs the CRDs and writes these objects.

use kube::CustomResource;
use serde::{Deserialize, Serialize};

/// About API cluster ID of the cluster an imported EndpointSlice came from.
/// Used as `PoolKey::site_id`.
pub const SOURCE_CLUSTER_LABEL: &str = "multicluster.kubernetes.io/source-cluster";

/// Name of the `ServiceImport` an imported EndpointSlice belongs to.
pub const SERVICE_NAME_LABEL: &str = "multicluster.kubernetes.io/service-name";

#[derive(CustomResource, Clone, Debug, Deserialize, Serialize)]
#[kube(
    group = "multicluster.x-k8s.io",
    version = "v1alpha1",
    kind = "ServiceImport",
    namespaced,
    schema = "disabled"
)]
#[serde(rename_all = "camelCase")]
pub struct ServiceImportSpec {
    /// `ClusterSetIP` or `Headless`.
    #[serde(rename = "type")]
    pub import_type: String,
}
