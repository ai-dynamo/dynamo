// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Kubernetes Lease-backed Mooncake master leader discovery.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use k8s_openapi::api::coordination::v1::Lease;
use kube::{Api, Client as KubeClient};

use super::{MooncakeLeaderResolver, MooncakeLeaderUnavailable};

struct KubernetesMooncakeLeaderResolver {
    leases: Api<Lease>,
    lease_name: String,
}

#[async_trait]
impl MooncakeLeaderResolver for KubernetesMooncakeLeaderResolver {
    async fn current_leader(&self) -> anyhow::Result<String> {
        let lease = self
            .leases
            .get(&self.lease_name)
            .await
            .map_err(|error| map_lease_get_error(&self.lease_name, error))?;
        mooncake_leader_from_lease(&lease).ok_or_else(|| {
            MooncakeLeaderUnavailable {
                message: format!(
                    "Mooncake master Lease {} has no active holderIdentity",
                    self.lease_name
                ),
            }
            .into()
        })
    }
}

fn map_lease_get_error(lease_name: &str, error: kube::Error) -> anyhow::Error {
    if matches!(&error, kube::Error::Api(response) if response.code == 404) {
        return MooncakeLeaderUnavailable {
            message: format!("Mooncake master Lease {lease_name} does not exist"),
        }
        .into();
    }

    anyhow::anyhow!("failed to read Mooncake master Lease {lease_name}: {error}")
}

pub(super) async fn build_leader_resolver(
    namespace: &str,
    lease_name: &str,
) -> anyhow::Result<Arc<dyn MooncakeLeaderResolver>> {
    let client = KubeClient::try_default().await?;
    Ok(Arc::new(KubernetesMooncakeLeaderResolver {
        leases: Api::namespaced(client, namespace),
        lease_name: lease_name.to_string(),
    }))
}

fn mooncake_leader_from_lease(lease: &Lease) -> Option<String> {
    mooncake_leader_from_lease_at(lease, Utc::now())
}

pub(super) fn mooncake_leader_from_lease_at(
    lease: &Lease,
    now: chrono::DateTime<Utc>,
) -> Option<String> {
    let spec = lease.spec.as_ref()?;
    let holder = spec
        .holder_identity
        .as_deref()
        .map(str::trim)
        .filter(|holder| !holder.is_empty())
        .map(str::to_string)?;

    // Match Mooncake's K8sLeaseGetHolder contract: holderIdentity is inactive after
    // renewTime + leaseDurationSeconds, even if a dead leader never released the Lease.
    if let (Some(renew_time), Some(lease_duration_seconds)) =
        (&spec.renew_time, spec.lease_duration_seconds)
    {
        let expires_at =
            renew_time.0 + chrono::Duration::seconds(i64::from(lease_duration_seconds));
        if now > expires_at {
            return None;
        }
    }

    Some(holder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lease_not_found_means_leader_unavailable() {
        let error = map_lease_get_error(
            "mooncake-master",
            kube::Error::Api(kube::core::ErrorResponse {
                status: "Failure".to_string(),
                message: "leases.coordination.k8s.io not found".to_string(),
                reason: "NotFound".to_string(),
                code: 404,
            }),
        );

        assert!(super::super::is_mooncake_leader_unavailable(&error));
    }

    #[test]
    fn lease_forbidden_remains_a_backend_error() {
        let error = map_lease_get_error(
            "mooncake-master",
            kube::Error::Api(kube::core::ErrorResponse {
                status: "Failure".to_string(),
                message: "leases.coordination.k8s.io is forbidden".to_string(),
                reason: "Forbidden".to_string(),
                code: 403,
            }),
        );

        assert!(!super::super::is_mooncake_leader_unavailable(&error));
        assert!(
            error
                .to_string()
                .contains("failed to read Mooncake master Lease")
        );
    }
}
