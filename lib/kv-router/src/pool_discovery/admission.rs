// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::PoolDiscoveryError;
use super::identity::VerifiedRelayIdentity;
use super::record::{PoolDirectory, PoolRecord};
use crate::global_view::PoolKey;

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
        Some(expected) if *expected != caller.0 => {
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
    use crate::global_view::state::PoolLocation;
    use crate::pool_discovery::identity::RelayIdentity;
    use crate::pool_discovery::record::PoolRecordBuilder;
    use crate::pool_discovery::watch::{Directory, Snapshot, WatchStream};

    struct Fixed(HashMap<PoolKey, PoolRecord>);

    #[async_trait]
    impl Directory<PoolRecord, PoolKey> for Fixed {
        async fn check_connection(&self) -> Result<(), PoolDiscoveryError> {
            unimplemented!()
        }

        async fn list(&self) -> Result<Snapshot<PoolRecord>, PoolDiscoveryError> {
            unimplemented!()
        }

        async fn get(&self, key: &PoolKey) -> Result<Option<PoolRecord>, PoolDiscoveryError> {
            Ok(self.0.get(key).cloned())
        }

        async fn list_and_watch(
            &self,
            _cancel_token: Option<CancellationToken>,
        ) -> Result<WatchStream<PoolRecord, PoolKey>, PoolDiscoveryError> {
            unimplemented!()
        }
    }

    fn key(dgd: &str) -> PoolKey {
        PoolKey::new("site-a", "ns", dgd).unwrap()
    }

    fn record(dgd: &str, relay: Option<&str>) -> PoolRecord {
        PoolRecordBuilder::default()
            .key(key(dgd))
            .location(PoolLocation {
                region: "r1".into(),
                availability_zone: None,
                cluster: None,
                datacenter: None,
            })
            .relay_identity(relay.map(|id| RelayIdentity(id.into())))
            .build()
            .unwrap()
    }

    fn caller(id: &str) -> VerifiedRelayIdentity {
        VerifiedRelayIdentity(RelayIdentity(id.into()))
    }

    fn pools() -> Fixed {
        Fixed(
            [record("bound", Some("relay-a")), record("open", None)]
                .into_iter()
                .map(|r| (r.key.clone(), r))
                .collect(),
        )
    }

    #[tokio::test]
    async fn admits_the_bound_relay() {
        let record = admit(&pools(), &key("bound"), &caller("relay-a"))
            .await
            .unwrap();
        assert_eq!(record.key, key("bound"));
    }

    #[tokio::test]
    async fn rejects_another_relay() {
        let err = admit(&pools(), &key("bound"), &caller("relay-b"))
            .await
            .unwrap_err();
        assert!(matches!(err, PoolDiscoveryError::PermissionDenied(k) if k == key("bound")));
    }

    #[tokio::test]
    async fn admits_any_relay_without_a_bound_identity() {
        assert!(
            admit(&pools(), &key("open"), &caller("relay-b"))
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn rejects_an_unknown_pool() {
        let err = admit(&pools(), &key("missing"), &caller("relay-a"))
            .await
            .unwrap_err();
        assert!(matches!(err, PoolDiscoveryError::UnknownPool(k) if k == key("missing")));
    }
}
