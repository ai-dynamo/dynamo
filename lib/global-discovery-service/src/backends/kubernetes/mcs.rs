// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How each MCS implementation names imported EndpointSlices. Only the
//! slices are read: they hold the endpoints, their readiness, and the source
//! cluster.

use k8s_openapi::api::discovery::v1::EndpointSlice;

/// How the MCS implementation names imported EndpointSlices. Not
/// `non_exhaustive`: the MCS store handles every variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum McsImplementation {
    /// The `multicluster.kubernetes.io` labels (Submariner, Cilium, GKE).
    Standard,
    /// Karmada's own names: a `derived-<service>` Service, and slices named
    /// `imported-<source cluster>-<source slice>`.
    Karmada,
}

/// Standard MCS: About API cluster ID of the cluster a slice came from.
pub const SOURCE_CLUSTER_LABEL: &str = "multicluster.kubernetes.io/source-cluster";

/// Standard MCS: name of the imported Service a slice belongs to.
pub const SERVICE_NAME_LABEL: &str = "multicluster.kubernetes.io/service-name";

/// Kubernetes: name of the Service a slice belongs to.
pub const KUBERNETES_SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";

/// Kubernetes: the controller that manages a slice.
pub const MANAGED_BY_LABEL: &str = "endpointslice.kubernetes.io/managed-by";

/// Karmada: imported slices belong to a Service named `derived-<service>`,
/// through the usual `kubernetes.io/service-name` label.
pub const KARMADA_DERIVED_PREFIX: &str = "derived-";

/// Karmada names each imported slice `imported-<cluster>-<source slice>`.
pub const KARMADA_IMPORTED_PREFIX: &str = "imported-";

/// Karmada's `endpointslice.kubernetes.io/managed-by` value on the slices it
/// collects.
pub const KARMADA_SLICE_MANAGER: &str = "endpointslice-controller.karmada.io";

/// Label selector for every imported slice. Narrowed further by name, since
/// not every implementation copies `exportedLabels`.
pub fn imported_selector(mcs: McsImplementation) -> String {
    match mcs {
        McsImplementation::Standard => SOURCE_CLUSTER_LABEL.to_owned(),
        McsImplementation::Karmada => format!("{MANAGED_BY_LABEL}={KARMADA_SLICE_MANAGER}"),
    }
}

/// Label selector for one imported Service's slices.
pub fn service_selector(mcs: McsImplementation, service: &str) -> String {
    match mcs {
        McsImplementation::Standard => format!("{SERVICE_NAME_LABEL}={service}"),
        McsImplementation::Karmada => {
            format!("{KUBERNETES_SERVICE_NAME_LABEL}={KARMADA_DERIVED_PREFIX}{service}")
        }
    }
}

/// The imported Service's name as exported, without Karmada's prefix.
pub fn service_name(mcs: McsImplementation, slice: &EndpointSlice) -> Option<&str> {
    let labels = slice.metadata.labels.as_ref()?;
    match mcs {
        McsImplementation::Standard => labels.get(SERVICE_NAME_LABEL).map(String::as_str),
        McsImplementation::Karmada => labels
            .get(KUBERNETES_SERVICE_NAME_LABEL)?
            .strip_prefix(KARMADA_DERIVED_PREFIX),
    }
}

/// The longest Service name whose slices Karmada can name a cluster for.
/// Kubernetes names a Service's slices `<service>-<5 random characters>`, but
/// shortens `<service>-` to 58 characters, and Karmada's names build on those.
pub const KARMADA_MAX_SERVICE_NAME: usize = 57;

/// Logs a slice of a Service the directory wants, but from an unknown
/// cluster. Such a slice is skipped, so its pool or router replica is not
/// listed.
pub fn warn_unknown_cluster(slice: &EndpointSlice, service: &str) {
    tracing::warn!(
        slice = slice.metadata.name.as_deref().unwrap_or_default(),
        service,
        karmada_max_service_name = KARMADA_MAX_SERVICE_NAME,
        "skipping an imported EndpointSlice that does not say which cluster it came from; with Karmada, check that the Service name is not too long"
    );
}

/// The cluster the slice came from. With Karmada this is the Karmada member
/// name, which must equal the About API cluster ID.
///
/// Karmada has no source-cluster label, so the cluster comes from the slice
/// name: `imported-<cluster>-<source slice>`, where the source slice name
/// starts with `<service>-`. The name is the same on the Karmada control
/// plane and in member clusters. The `work.karmada.io/namespace` annotation
/// is not used: it names the source cluster (`karmada-es-<cluster>`) only on
/// the control plane, and the destination cluster on copies that Karmada
/// propagates into members.
pub fn source_cluster(mcs: McsImplementation, slice: &EndpointSlice) -> Option<&str> {
    match mcs {
        McsImplementation::Standard => slice
            .metadata
            .labels
            .as_ref()?
            .get(SOURCE_CLUSTER_LABEL)
            .map(String::as_str),
        McsImplementation::Karmada => {
            let service = service_name(mcs, slice)?;
            let rest = slice
                .metadata
                .name
                .as_deref()?
                .strip_prefix(KARMADA_IMPORTED_PREFIX)?;
            // The last `-<service>-` ends the cluster name.
            rest.rmatch_indices(service)
                .find_map(|(start, _)| {
                    let cluster = rest[..start].strip_suffix('-')?;
                    rest[start + service.len()..]
                        .starts_with('-')
                        .then_some(cluster)
                })
                .filter(|cluster| !cluster.is_empty())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

    use super::*;

    fn slice(name: &str, labels: &[(&str, &str)]) -> EndpointSlice {
        EndpointSlice {
            address_type: "IPv4".to_owned(),
            endpoints: Vec::new(),
            metadata: ObjectMeta {
                name: Some(name.to_owned()),
                labels: Some(
                    labels
                        .iter()
                        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
                        .collect::<BTreeMap<_, _>>(),
                ),
                ..Default::default()
            },
            ports: None,
        }
    }

    fn karmada_slice(name: &str, service: &str) -> EndpointSlice {
        slice(
            name,
            &[
                (MANAGED_BY_LABEL, KARMADA_SLICE_MANAGER),
                (
                    KUBERNETES_SERVICE_NAME_LABEL,
                    &format!("{KARMADA_DERIVED_PREFIX}{service}"),
                ),
            ],
        )
    }

    #[test]
    fn selectors() {
        assert_eq!(
            imported_selector(McsImplementation::Standard),
            "multicluster.kubernetes.io/source-cluster"
        );
        assert_eq!(
            imported_selector(McsImplementation::Karmada),
            "endpointslice.kubernetes.io/managed-by=endpointslice-controller.karmada.io"
        );
        assert_eq!(
            service_selector(McsImplementation::Standard, "pool-relay"),
            "multicluster.kubernetes.io/service-name=pool-relay"
        );
        assert_eq!(
            service_selector(McsImplementation::Karmada, "pool-relay"),
            "kubernetes.io/service-name=derived-pool-relay"
        );
    }

    #[test]
    fn standard_reads_labels() {
        let slice = slice(
            "pool-relay-x7k2p",
            &[
                (SOURCE_CLUSTER_LABEL, "west"),
                (SERVICE_NAME_LABEL, "pool-relay"),
            ],
        );
        assert_eq!(
            service_name(McsImplementation::Standard, &slice),
            Some("pool-relay")
        );
        assert_eq!(
            source_cluster(McsImplementation::Standard, &slice),
            Some("west")
        );
    }

    // Names as Karmada v1.19 created them in a two-cluster test.
    #[test]
    fn karmada_reads_service_and_cluster() {
        let slice = karmada_slice(
            "imported-aws-west-relay-mocker-west-1-28g2h",
            "relay-mocker-west-1",
        );
        assert_eq!(
            service_name(McsImplementation::Karmada, &slice),
            Some("relay-mocker-west-1")
        );
        assert_eq!(
            source_cluster(McsImplementation::Karmada, &slice),
            Some("aws-west")
        );
    }

    #[test]
    fn karmada_cluster_name_may_contain_the_service_name() {
        let slice = karmada_slice("imported-a-svc-b-svc-x1y2z", "svc");
        assert_eq!(
            source_cluster(McsImplementation::Karmada, &slice),
            Some("a-svc-b")
        );
    }

    #[test]
    fn karmada_rejects_unexpected_names() {
        // Not imported by Karmada.
        let local = karmada_slice("relay-mocker-west-1-28g2h", "relay-mocker-west-1");
        assert_eq!(source_cluster(McsImplementation::Karmada, &local), None);
        // The source slice name does not start with the service name, as
        // when Kubernetes shortens a long Service name.
        let shortened = karmada_slice("imported-west-relay-mock-28g2h", "relay-mocker-west-1");
        assert_eq!(source_cluster(McsImplementation::Karmada, &shortened), None);
        // No cluster before the service name.
        let empty = karmada_slice("imported--svc-x1y2z", "svc");
        assert_eq!(source_cluster(McsImplementation::Karmada, &empty), None);
        // Not a derived Service.
        let plain = slice(
            "imported-west-svc-x1y2z",
            &[(KUBERNETES_SERVICE_NAME_LABEL, "svc")],
        );
        assert_eq!(service_name(McsImplementation::Karmada, &plain), None);
    }
}
