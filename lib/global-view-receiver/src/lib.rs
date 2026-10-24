// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconciles complete logical-pool snapshots and subsequent updates from Pool Relay replicas.
//!
//! The transport adapter supplies Relay identity, logical pool keys, and validated pool payloads.
//! A logical routing pool (for example, a DGD) may contain multiple KV catalog `PoolId`s.
//! The adapter reconciles the Relay's independent catalog, readiness, and load streams before
//! passing a per-pool view here; this crate defines no wire format or transport.
//! This crate keeps Relay replicas separate and does not merge or sum their payloads.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

/// Identifies one connection attempt to a Relay replica.
///
/// A replacement connection receives a new token. Late updates or disconnects from the old
/// connection cannot change the replacement's state.
#[derive(Debug, Clone)]
pub struct StreamToken<R> {
    relay: R,
    epoch: Arc<()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ApplyError {
    #[error("relay stream is no longer current")]
    StaleStream,
    #[error("a complete snapshot is required before updates")]
    SnapshotRequired,
    #[error("snapshot contains a duplicate pool")]
    DuplicatePool,
}

struct RelayStream<P> {
    epoch: Arc<()>,
    pools: Option<HashSet<P>>,
}

/// The current per-pool views from connected Relay replicas.
///
/// `R` identifies one Relay replica, `P` identifies a logical routing pool across clusters and
/// namespaces, and `V` is a validated per-pool view supplied by the transport adapter. A pool
/// is visible while at least one connected Relay advertises it. Discovery alone does not make
/// a withdrawn pool live.
pub struct GlobalViewReceiver<R, P, V> {
    relays: HashMap<R, RelayStream<P>>,
    pools: HashMap<P, HashMap<R, V>>,
}

impl<R, P, V> Default for GlobalViewReceiver<R, P, V> {
    fn default() -> Self {
        Self {
            relays: HashMap::new(),
            pools: HashMap::new(),
        }
    }
}

impl<R: Clone + Eq + Hash, P: Clone + Eq + Hash, V> GlobalViewReceiver<R, P, V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Begin a new stream, invalidating the previous stream and its pool views for this Relay.
    /// The replacement is not live until its first complete snapshot arrives.
    pub fn open(&mut self, relay: R) -> StreamToken<R> {
        if let Some(previous) = self.relays.remove(&relay) {
            self.remove_relay_views(&relay, previous.pools);
        }
        let epoch = Arc::new(());
        self.relays.insert(
            relay.clone(),
            RelayStream {
                epoch: epoch.clone(),
                pools: None,
            },
        );
        StreamToken { relay, epoch }
    }

    /// Replace one Relay's entire pool view. A pool omitted here is withdrawn from that Relay.
    /// Duplicate logical pool keys reject the snapshot without modifying the current view.
    pub fn replace_snapshot(
        &mut self,
        stream: &StreamToken<R>,
        pools: impl IntoIterator<Item = (P, V)>,
    ) -> Result<(), ApplyError> {
        self.current_mut(stream)?;
        let mut replacement = HashMap::new();
        for (pool_id, view) in pools {
            if replacement.insert(pool_id, view).is_some() {
                return Err(ApplyError::DuplicatePool);
            }
        }
        let old_pools = self
            .current_mut(stream)?
            .pools
            .replace(replacement.keys().cloned().collect());
        self.remove_relay_views(&stream.relay, old_pools);
        for (pool_id, view) in replacement {
            self.pools
                .entry(pool_id)
                .or_default()
                .insert(stream.relay.clone(), view);
        }
        Ok(())
    }

    /// Apply a per-pool update after this stream has delivered a complete snapshot.
    pub fn upsert(
        &mut self,
        stream: &StreamToken<R>,
        pool_id: P,
        view: V,
    ) -> Result<(), ApplyError> {
        self.current_mut(stream)?
            .pools
            .as_mut()
            .ok_or(ApplyError::SnapshotRequired)?
            .insert(pool_id.clone());
        self.pools
            .entry(pool_id)
            .or_default()
            .insert(stream.relay.clone(), view);
        Ok(())
    }

    /// Withdraw one pool from this Relay without affecting other Relay replicas.
    pub fn remove(&mut self, stream: &StreamToken<R>, pool_id: P) -> Result<(), ApplyError> {
        self.current_mut(stream)?
            .pools
            .as_mut()
            .ok_or(ApplyError::SnapshotRequired)?
            .remove(&pool_id);
        self.remove_relay_pool(&stream.relay, &pool_id);
        Ok(())
    }

    /// Drop all views from a disconnected Relay. An old stream cannot disconnect its replacement.
    /// The transport adapter must call this on EOF, error, or cancellation; dropping the token
    /// alone does not change the receiver's state.
    pub fn disconnect(&mut self, stream: &StreamToken<R>) -> Result<(), ApplyError> {
        self.current_mut(stream)?;
        if let Some(relay) = self.relays.remove(&stream.relay) {
            self.remove_relay_views(&stream.relay, relay.pools);
        }
        Ok(())
    }

    /// Return every live Relay view of one pool. Consumers can reconcile redundant observations
    /// without accidentally adding the same pool's load from multiple Relays.
    pub fn pool_views(&self, pool_id: &P) -> impl Iterator<Item = (&R, &V)> + use<'_, R, P, V> {
        self.pools
            .get(pool_id)
            .into_iter()
            .flat_map(|views| views.iter())
    }

    /// Logical pool keys advertised by at least one live Relay stream.
    pub fn pool_ids(&self) -> impl Iterator<Item = &P> {
        self.pools.keys()
    }

    fn current_mut(&mut self, stream: &StreamToken<R>) -> Result<&mut RelayStream<P>, ApplyError> {
        let relay = self
            .relays
            .get_mut(&stream.relay)
            .ok_or(ApplyError::StaleStream)?;
        if !Arc::ptr_eq(&relay.epoch, &stream.epoch) {
            return Err(ApplyError::StaleStream);
        }
        Ok(relay)
    }

    fn remove_relay_views(&mut self, relay: &R, pool_ids: Option<HashSet<P>>) {
        for pool_id in pool_ids.into_iter().flatten() {
            self.remove_relay_pool(relay, &pool_id);
        }
    }

    fn remove_relay_pool(&mut self, relay: &R, pool_id: &P) {
        if let Some(views) = self.pools.get_mut(pool_id) {
            views.remove(relay);
            if views.is_empty() {
                self.pools.remove(pool_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_relay_loss_does_not_brown_out_a_pool_seen_by_another() {
        let mut receiver = GlobalViewReceiver::<&str, String, u64>::new();
        let a = receiver.open("relay-a");
        let b = receiver.open("relay-b");
        let pool = "blue-dgd".to_owned();
        receiver.replace_snapshot(&a, [(pool.clone(), 10)]).unwrap();
        receiver.replace_snapshot(&b, [(pool.clone(), 11)]).unwrap();

        assert_eq!(receiver.pool_views(&pool).count(), 2);
        receiver.disconnect(&a).unwrap();
        assert_eq!(receiver.pool_views(&pool).count(), 1);
        receiver.disconnect(&b).unwrap();
        assert_eq!(receiver.pool_views(&pool).count(), 0);
    }

    #[test]
    fn reconnect_requires_fresh_snapshot_and_fences_the_old_stream() {
        let mut receiver = GlobalViewReceiver::<&str, u64, u64>::new();
        let old = receiver.open("relay-a");
        let pool = 1;
        receiver.replace_snapshot(&old, [(pool, 10)]).unwrap();

        let new = receiver.open("relay-a");
        assert_eq!(receiver.pool_views(&pool).count(), 0);
        assert_eq!(
            receiver.upsert(&old, pool, 20),
            Err(ApplyError::StaleStream)
        );
        assert_eq!(receiver.disconnect(&old), Err(ApplyError::StaleStream));
        assert_eq!(
            receiver.upsert(&new, pool, 21),
            Err(ApplyError::SnapshotRequired)
        );
        receiver.replace_snapshot(&new, [(pool, 22)]).unwrap();
        assert_eq!(
            receiver.pool_views(&pool).next().map(|(_, value)| *value),
            Some(22)
        );
    }

    #[test]
    fn invalid_snapshot_does_not_replace_the_current_view() {
        let mut receiver = GlobalViewReceiver::<&str, u64, u64>::new();
        let stream = receiver.open("relay-a");
        let pool = 1;
        receiver.replace_snapshot(&stream, [(pool, 10)]).unwrap();

        assert_eq!(
            receiver.replace_snapshot(&stream, [(pool, 20), (pool, 21)]),
            Err(ApplyError::DuplicatePool),
        );
        assert_eq!(
            receiver.pool_views(&pool).next().map(|(_, value)| *value),
            Some(10)
        );
    }
}
