// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recently removed worker instances, shared by the response planes.

use std::{collections::HashMap, time::Duration};

use tokio::time::Instant;

use crate::discovery::EndpointInstanceId;

/// Tombstone lifetime. Bridges the `register()` → `associate_instance()`
/// window (sub-millisecond in practice); 5s bounds the set by recent worker
/// churn rather than process lifetime, since etcd lease IDs are unique per
/// restart and never get cleared by an `Added` event for the same identity.
pub(crate) const TOMBSTONE_TTL: Duration = Duration::from_secs(5);

/// Instances removed from discovery in the last [`TOMBSTONE_TTL`]. They close
/// the race between `cancel_instance_streams` and a later `associate_instance`
/// for the same instance. Expired entries are pruned on every insert and lookup.
#[derive(Default)]
pub(crate) struct InstanceTombstones(HashMap<EndpointInstanceId, Instant>);

impl InstanceTombstones {
    pub(crate) fn insert(&mut self, instance: &EndpointInstanceId) {
        let now = Instant::now();
        self.prune(now);
        self.0.insert(instance.clone(), now);
    }

    pub(crate) fn contains(&mut self, instance: &EndpointInstanceId) -> bool {
        self.prune(Instant::now());
        self.0.contains_key(instance)
    }

    pub(crate) fn remove(&mut self, instance: &EndpointInstanceId) {
        self.0.remove(instance);
    }

    fn prune(&mut self, now: Instant) {
        self.0
            .retain(|_, removed| now.saturating_duration_since(*removed) < TOMBSTONE_TTL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instance(instance_id: u64) -> EndpointInstanceId {
        EndpointInstanceId {
            namespace: "ns".into(),
            component: "worker".into(),
            endpoint: "generate".into(),
            instance_id,
        }
    }

    #[tokio::test(start_paused = true)]
    async fn tombstones_expire_and_prune() {
        let mut tombstones = InstanceTombstones::default();
        tombstones.insert(&instance(1));
        assert!(tombstones.contains(&instance(1)));
        assert!(!tombstones.contains(&instance(2)));

        tokio::time::advance(TOMBSTONE_TTL + Duration::from_secs(1)).await;
        tombstones.insert(&instance(2));
        // Inserting pruned the expired entry, so the map holds only the new one.
        assert_eq!(tombstones.0.len(), 1);
        assert!(!tombstones.contains(&instance(1)));

        tombstones.remove(&instance(2));
        assert!(!tombstones.contains(&instance(2)));
    }
}
