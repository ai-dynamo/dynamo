// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The hub record for one pool. Its `Lease` has the same name and namespace.
//! A `ValidatingAdmissionPolicy` on the hub checks that the key fields and
//! `relay_identity` match the namespace.

use dynamo_kv_router::pool_discovery::{PoolDiscoveryError, PoolRecord};
use kube::CustomResource;
use serde::{Deserialize, Serialize};

#[derive(CustomResource, Clone, Debug, Deserialize, Serialize)]
#[kube(
    group = "nvidia.com",
    version = "v1alpha1",
    kind = "DynamoPoolExport",
    namespaced,
    schema = "disabled"
)]
#[serde(rename_all = "camelCase")]
pub struct DynamoPoolExportSpec {
    /// With `dgd_namespace` and `dgd_name`, forms the `PoolKey`.
    pub site_id: String,
    pub dgd_namespace: String,
    pub dgd_name: String,
    pub runtime_namespace: String,
    pub frontend_endpoint: String,
    pub relay_identity: String,
    pub location: ExportLocation,
    pub model: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportLocation {
    pub region: String,
    pub availability_zone: Option<String>,
    pub cluster: Option<String>,
    pub datacenter: Option<String>,
}

/// `metadata.generation` becomes the record's `Revision`.
pub fn to_record(export: &DynamoPoolExport) -> Result<PoolRecord, PoolDiscoveryError> {
    todo!()
}

pub fn to_spec(record: &PoolRecord) -> DynamoPoolExportSpec {
    todo!()
}
