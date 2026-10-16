// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How each MCS implementation names imported EndpointSlices. Only the
//! slices are read: they hold the endpoints, their readiness, and the source
//! cluster.

use dynamo_kv_router::pool_discovery::McsImplementation;
use k8s_openapi::api::discovery::v1::EndpointSlice;

/// Standard MCS: About API cluster ID of the cluster a slice came from.
pub const SOURCE_CLUSTER_LABEL: &str = "multicluster.kubernetes.io/source-cluster";

/// Standard MCS: name of the imported Service a slice belongs to.
pub const SERVICE_NAME_LABEL: &str = "multicluster.kubernetes.io/service-name";

/// Karmada: imported slices belong to a Service named `derived-<service>`,
/// through the usual `kubernetes.io/service-name` label.
pub const KARMADA_DERIVED_PREFIX: &str = "derived-";

/// Karmada: the source cluster's execution namespace,
/// `karmada-es-<cluster>`.
pub const KARMADA_WORK_NAMESPACE_ANNOTATION: &str = "work.karmada.io/namespace";

pub const KARMADA_EXECUTION_SPACE_PREFIX: &str = "karmada-es-";

/// Karmada's `endpointslice.kubernetes.io/managed-by` value on the slices it
/// collects.
pub const KARMADA_SLICE_MANAGER: &str = "endpointslice-controller.karmada.io";

/// Label selector for every imported slice. Narrowed further by name, since
/// not every implementation copies `exportedLabels`.
pub fn imported_selector(mcs: McsImplementation) -> String {
    todo!()
}

/// Label selector for one imported Service's slices.
pub fn service_selector(mcs: McsImplementation, service: &str) -> String {
    todo!()
}

/// The imported Service's name as exported, without Karmada's prefix.
pub fn service_name(mcs: McsImplementation, slice: &EndpointSlice) -> Option<&str> {
    todo!()
}

/// The cluster the slice came from. With Karmada this is the Karmada member
/// name, which must equal the About API cluster ID.
pub fn source_cluster(mcs: McsImplementation, slice: &EndpointSlice) -> Option<&str> {
    todo!()
}
