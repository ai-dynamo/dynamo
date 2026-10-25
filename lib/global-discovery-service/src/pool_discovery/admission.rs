// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::PoolDiscoveryError;
use super::identity::VerifiedRelayIdentity;
use super::record::{PoolDirectory, PoolRecord};
use dynamo_global_router_shared::PoolKey;

/// Decides whether an authenticated Relay may stream state for a pool. The
/// rule is the same in every mode; only the source of `relay_identity`
/// differs. A record without one admits any authenticated Relay.
///
/// Uses `PoolDirectory::get`, so it reads the watch cache and never lists.
/// The runtime keys the stream by its `PlaneLease`, not by the claimed key.
pub async fn admit(
    pools: &PoolDirectory,
    key: &PoolKey,
    caller: &VerifiedRelayIdentity,
) -> Result<PoolRecord, PoolDiscoveryError> {
    let record = pools
        .get(key)
        .await?
        .ok_or_else(|| PoolDiscoveryError::UnknownPool(key.clone()))?;
    match &record.relay_identity {
        Some(expected) if expected != caller.identity() => {
            Err(PoolDiscoveryError::PermissionDenied(key.clone()))
        }
        _ => Ok(record),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_trait::async_trait;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::pool_discovery::identity::RelayIdentity;
    use crate::pool_discovery::record::PoolRecordBuilder;
    use crate::pool_discovery::watch::{Directory, Snapshot, WatchStream};
    use dynamo_global_router_shared::PoolLocation;

    struct Pools(HashMap<PoolKey, PoolRecord>);

    #[async_trait]
    impl Directory<PoolRecord, PoolKey> for Pools {
        async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
            Ok(())
        }

        async fn list(&self) -> Result<Snapshot<PoolRecord>, PoolDiscoveryError> {
            unreachable!("admit must not list")
        }

        async fn get(&self, key: &PoolKey) -> Result<Option<PoolRecord>, PoolDiscoveryError> {
            Ok(self.0.get(key).cloned())
        }

        async fn list_and_watch(
            &self,
            _cancel_token: Option<CancellationToken>,
        ) -> Result<WatchStream<PoolRecord, PoolKey>, PoolDiscoveryError> {
            unreachable!("admit must not watch")
        }
    }

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("west", "dynamo", dgd).unwrap()
    }

    fn record(dgd: &str, relay: Option<&str>) -> PoolRecord {
        PoolRecordBuilder::default()
            .key(key(dgd))
            .location(PoolLocation {
                region: "us-west-2".to_owned(),
                availability_zone: None,
                cluster: Some("west".to_owned()),
                datacenter: None,
            })
            .relay_identity(relay.map(|id| RelayIdentity(id.to_owned())))
            .build()
            .unwrap()
    }

    fn caller(id: &str) -> VerifiedRelayIdentity {
        VerifiedRelayIdentity(RelayIdentity(id.to_owned()))
    }

    fn pools() -> Pools {
        Pools(HashMap::from([
            (key("bound"), record("bound", Some("ip:10.43.0.1"))),
            (key("open"), record("open", None)),
        ]))
    }

    #[tokio::test]
    async fn admits_the_bound_relay() {
        let record = admit(&pools(), &key("bound"), &caller("ip:10.43.0.1"))
            .await
            .unwrap();
        assert_eq!(record.key, key("bound"));
    }

    #[tokio::test]
    async fn refuses_another_relay() {
        assert!(matches!(
            admit(&pools(), &key("bound"), &caller("ip:10.43.0.2")).await,
            Err(PoolDiscoveryError::PermissionDenied(k)) if k == key("bound")
        ));
    }

    #[tokio::test]
    async fn record_without_identity_admits_any_relay() {
        assert!(
            admit(&pools(), &key("open"), &caller("ip:10.43.0.2"))
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn unknown_pool_is_refused() {
        assert!(matches!(
            admit(&pools(), &key("missing"), &caller("ip:10.43.0.1")).await,
            Err(PoolDiscoveryError::UnknownPool(_))
        ));
    }
}
