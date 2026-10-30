// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use dynamo_global_common::PoolId;
use dynamo_global_router::{Pool, PoolSignals, RelaySource, ReportReceipt};
use dynamo_global_view_receiver::{ApplyError, PoolViews, StreamToken};
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::{PoolSnapshot, RelayPoolState};

type View = Option<Arc<RelayPoolState>>;
type Views = PoolViews<RelaySourceKey, PoolId, ReceivedView>;

#[derive(Clone)]
struct ReceivedView {
    state: View,
    serving: Option<(u64, ReportReceipt)>,
    kv: Option<(u64, ReportReceipt)>,
}

impl ReceivedView {
    fn new(
        state: View,
        previous: Option<&Self>,
        pool: &PoolId,
        signals: Option<&PoolSignals>,
    ) -> Self {
        let previous =
            previous.filter(|old| old.source() == state.as_ref().map(|state| state.source()));
        let (serving_sequence, kv_sequence) = state
            .as_ref()
            .map_or((None, None), |state| state.window_sequences());
        let receipt = signals.map(|signals| signals.receive_report(pool));
        let retain = |sequence, old: Option<(u64, ReportReceipt)>| match sequence {
            Some(sequence) if old.is_some_and(|(old, _)| old == sequence) => old,
            Some(sequence) => receipt.map(|receipt| (sequence, receipt)),
            None => old,
        };
        Self {
            state,
            serving: retain(serving_sequence, previous.and_then(|old| old.serving)),
            kv: retain(kv_sequence, previous.and_then(|old| old.kv)),
        }
    }

    fn source(&self) -> Option<RelaySource> {
        self.state.as_ref().map(|state| state.source())
    }
    fn ready_for(&self, model: &str) -> bool {
        self.state
            .as_ref()
            .is_none_or(|state| state.ready_for(model))
    }
    fn project(&self, pool: &Pool, signals: &PoolSignals) {
        if let Some(state) = &self.state {
            state.project(
                &pool.id,
                &pool.model,
                signals,
                self.serving.map(|(_, receipt)| receipt),
                self.kv.map(|(_, receipt)| receipt),
            );
        }
    }
}

/// One Relay Pod reporting one routing pool. Replica names need only be unique within a pool.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RelaySourceKey {
    pub pool: PoolId,
    pub service: String,
    pub replica: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ReceiveError {
    #[error(transparent)]
    Apply(#[from] ApplyError),
    #[error("relay view revision gap: expected {expected}, got {received}")]
    RevisionGap { expected: u64, received: u64 },
    #[error("relay view snapshot revision is not newer: current {current}, got {received}")]
    StaleSnapshot { current: u64, received: u64 },
}

struct Inner {
    discovery: watch::Receiver<Arc<PoolSnapshot>>,
    snapshot: watch::Sender<Arc<PoolSnapshot>>,
    views: Mutex<Views>,
    signals: Option<Arc<PoolSignals>>,
    requires_view: fn(&Pool) -> bool,
}

impl Inner {
    fn project_pool(&self, pool: &PoolId, views: &Views) {
        if let Some(signals) = &self.signals
            && let Some(candidate) = self
                .discovery
                .borrow()
                .pools()
                .iter()
                .find(|candidate| &candidate.id == pool)
        {
            for (_, view) in views.pool_views(pool) {
                view.project(candidate, signals);
            }
        }
    }

    fn forget_source(&self, pool: &PoolId, source: Option<RelaySource>) {
        if let (Some(signals), Some(source)) = (&self.signals, source) {
            signals.forget_source(pool, source);
        }
    }

    fn refresh(&self) {
        let views = self.views.lock();
        let discovery = self.discovery.borrow();
        if let Some(signals) = &self.signals {
            let candidate_ids = discovery
                .pools()
                .iter()
                .map(|pool| &pool.id)
                .collect::<HashSet<_>>();
            for pool in self.snapshot.borrow().pools() {
                if !candidate_ids.contains(&pool.id) {
                    signals.forget(&pool.id);
                }
            }
            for pool in discovery.pools() {
                for (_, view) in views.pool_views(&pool.id) {
                    view.project(pool, signals);
                }
            }
        }
        let pools = discovery
            .pools()
            .iter()
            .filter(|pool| {
                !(self.requires_view)(pool)
                    || views
                        .pool_views(&pool.id)
                        .any(|(_, view)| view.ready_for(&pool.model))
            })
            .cloned()
            .collect();
        let next = PoolSnapshot::new(pools);
        self.snapshot.send_if_modified(|current| {
            if **current == next {
                return false;
            }
            *current = Arc::new(next);
            true
        });
    }
}

/// Discovery pools with a current Relay view, gated by model readiness when state is supplied.
pub struct RelayGatedPoolCatalog {
    inner: Arc<Inner>,
}

impl RelayGatedPoolCatalog {
    pub fn new(discovery: watch::Receiver<Arc<PoolSnapshot>>, cancel: CancellationToken) -> Self {
        Self::with_signals(discovery, cancel, None)
    }
    pub fn with_signals(
        discovery: watch::Receiver<Arc<PoolSnapshot>>,
        cancel: CancellationToken,
        signals: Option<Arc<PoolSignals>>,
    ) -> Self {
        Self::build(discovery, cancel, signals, |_| true)
    }
    /// Static pools without a configured Relay retain their existing eligibility.
    pub fn for_subscriptions(
        discovery: watch::Receiver<Arc<PoolSnapshot>>,
        cancel: CancellationToken,
        signals: Arc<PoolSignals>,
    ) -> Self {
        Self::build(discovery, cancel, Some(signals), |pool| {
            pool.relay.is_some()
        })
    }
    fn build(
        discovery: watch::Receiver<Arc<PoolSnapshot>>,
        cancel: CancellationToken,
        signals: Option<Arc<PoolSignals>>,
        requires_view: fn(&Pool) -> bool,
    ) -> Self {
        let (snapshot, _) = watch::channel(Arc::new(PoolSnapshot::default()));
        let inner = Arc::new(Inner {
            discovery: discovery.clone(),
            snapshot,
            views: Mutex::new(Views::new()),
            signals,
            requires_view,
        });
        inner.refresh();
        let watcher = Arc::clone(&inner);
        tokio::spawn(async move {
            let mut discovery = discovery;
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    changed = discovery.changed() => { if changed.is_err() { break; } watcher.refresh(); }
                }
            }
        });
        Self { inner }
    }
    /// A replacement stream withdraws its predecessor until it sends a snapshot.
    pub fn open(&self, relay: RelaySourceKey) -> RelayStreamGuard {
        let token = {
            let mut views = self.inner.views.lock();
            let previous = views
                .pool_views(&relay.pool)
                .find(|(source, _)| *source == &relay)
                .and_then(|(_, view)| view.source());
            let token = views.open(relay.clone());
            self.inner.forget_source(&relay.pool, previous);
            token
        };
        self.inner.refresh();
        RelayStreamGuard {
            inner: Arc::clone(&self.inner),
            token,
            revision: None,
            source: relay,
        }
    }
    pub fn subscribe(&self) -> watch::Receiver<Arc<PoolSnapshot>> {
        self.inner.snapshot.subscribe()
    }
}

/// A receiving task owns this guard. Dropping the task withdraws its Relay view.
pub struct RelayStreamGuard {
    inner: Arc<Inner>,
    token: StreamToken<RelaySourceKey>,
    revision: Option<u64>,
    source: RelaySourceKey,
}

impl RelayStreamGuard {
    /// Presence-only helper for transport-independent view consumers.
    pub fn replace_snapshot(
        &mut self,
        revision: u64,
        pools: impl IntoIterator<Item = PoolId>,
    ) -> Result<(), ReceiveError> {
        self.replace_state_snapshot(revision, pools.into_iter().map(|pool| (pool, None)))
    }
    pub(crate) fn replace_state_snapshot(
        &mut self,
        revision: u64,
        pools: impl IntoIterator<Item = (PoolId, View)>,
    ) -> Result<(), ReceiveError> {
        if let Some(current) = self.revision
            && revision <= current
        {
            return Err(ReceiveError::StaleSnapshot {
                current,
                received: revision,
            });
        }
        {
            let inner = Arc::clone(&self.inner);
            let mut views = inner.views.lock();
            let previous = views
                .pool_views(&self.source.pool)
                .find(|(source, _)| *source == &self.source)
                .and_then(|(_, view)| view.source());
            let pools = pools
                .into_iter()
                .map(|(pool, state)| {
                    let old = views
                        .pool_views(&pool)
                        .find(|(source, _)| *source == &self.source)
                        .map(|(_, view)| view);
                    let view = ReceivedView::new(state, old, &pool, inner.signals.as_deref());
                    (pool, view)
                })
                .collect::<Vec<_>>();
            views.replace_snapshot(&self.token, pools)?;
            let next = self.current_source(&views);
            if previous != next {
                inner.forget_source(&self.source.pool, previous);
            }
            inner.project_pool(&self.source.pool, &views);
        }
        self.revision = Some(revision);
        self.inner.refresh();
        Ok(())
    }
    pub fn upsert(&mut self, revision: u64, pool: PoolId) -> Result<(), ReceiveError> {
        self.upsert_state(revision, pool, None)
    }
    pub(crate) fn upsert_state(
        &mut self,
        revision: u64,
        pool: PoolId,
        state: View,
    ) -> Result<(), ReceiveError> {
        self.check_revision(revision)?;
        let inner = Arc::clone(&self.inner);
        let mut views = inner.views.lock();
        let previous = views
            .pool_views(&pool)
            .find(|(source, _)| *source == &self.source)
            .map(|(_, view)| view);
        let readiness_changed = previous.is_none_or(|old| {
            old.state.as_ref().map(|state| state.ready_models())
                != state.as_ref().map(|state| state.ready_models())
        });
        let old_source = previous.and_then(|view| view.source());
        let received = ReceivedView::new(state, previous, &pool, inner.signals.as_deref());
        let new_source = received.source();
        views.upsert(&self.token, pool.clone(), received)?;
        if old_source != new_source {
            inner.forget_source(&pool, old_source);
        }
        inner.project_pool(&pool, &views);
        drop(views);
        self.revision = Some(revision);
        if readiness_changed {
            self.inner.refresh();
        }
        Ok(())
    }
    pub fn remove(&mut self, revision: u64, pool: &PoolId) -> Result<(), ReceiveError> {
        self.check_revision(revision)?;
        let inner = Arc::clone(&self.inner);
        let mut views = inner.views.lock();
        let previous = self.current_source(&views);
        views.remove(&self.token, pool)?;
        inner.forget_source(pool, previous);
        drop(views);
        self.revision = Some(revision);
        self.inner.refresh();
        Ok(())
    }
    fn check_revision(&mut self, received: u64) -> Result<(), ReceiveError> {
        let Some(expected) = self.revision.and_then(|value| value.checked_add(1)) else {
            return Err(ApplyError::SnapshotRequired.into());
        };
        if received != expected {
            {
                let inner = Arc::clone(&self.inner);
                let mut views = inner.views.lock();
                let previous = self.current_source(&views);
                views.replace_snapshot(&self.token, [])?;
                inner.forget_source(&self.source.pool, previous);
            }
            self.revision = None;
            self.inner.refresh();
            return Err(ReceiveError::RevisionGap { expected, received });
        }
        Ok(())
    }
    fn current_source(&self, views: &Views) -> Option<RelaySource> {
        views
            .pool_views(&self.source.pool)
            .find(|(source, _)| *source == &self.source)
            .and_then(|(_, view)| view.source())
    }
}

impl Drop for RelayStreamGuard {
    fn drop(&mut self) {
        let disconnected = {
            let inner = Arc::clone(&self.inner);
            let mut views = inner.views.lock();
            let previous = self.current_source(&views);
            let disconnected = views.disconnect(&self.token).is_ok();
            if disconnected {
                inner.forget_source(&self.source.pool, previous);
            }
            disconnected
        };
        if disconnected {
            self.inner.refresh();
        }
    }
}

#[cfg(test)]
#[path = "../tests/common/mod.rs"]
mod fixtures;

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_global_common::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
    use dynamo_global_router::{Pool, RelaySource};
    use dynamo_llm::pool_relay::wan::grpc::protocol as wire;

    fn state(replica: u64, ready: bool) -> Arc<RelayPoolState> {
        let identity = wire::RelayIdentity {
            drt_instance_id: replica,
            relay_incarnation: 1,
        };
        let catalog = wire::KvPoolCatalogUpdate {
            protocol_version: wire::RELAY_PROTOCOL_VERSION,
            contract_marker: wire::RELAY_CONTRACT_MARKER,
            relay: Some(identity),
            snapshot: Some(wire::KvPoolCatalogSnapshot::default()),
            ..Default::default()
        };
        let readiness = wire::ServingReadinessUpdate {
            protocol_version: wire::RELAY_PROTOCOL_VERSION,
            contract_marker: wire::RELAY_CONTRACT_MARKER,
            relay: Some(identity),
            entries: vec![wire::TopologyEntry {
                canonical_model_id: "model".into(),
                state: if ready {
                    wire::ServingReadinessState::Ready
                } else {
                    wire::ServingReadinessState::Unavailable
                } as i32,
                ..Default::default()
            }],
            ..Default::default()
        };
        Arc::new(
            RelayPoolState::from_updates(
                Arc::new(catalog),
                &readiness,
                None,
                None,
                RelaySource {
                    runtime_id: replica,
                    incarnation: 1,
                },
            )
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn removing_only_ready_view_withdraws_pool_even_when_an_unready_view_remains() {
        let key = PoolKey::new("east", "dynamo", "blue").unwrap();
        let pool = V1PoolIdDeriver.derive(&key);
        let (_sender, candidates) = watch::channel(Arc::new(PoolSnapshot::new(vec![Pool::new(
            &key,
            "model",
            "east",
            vec!["http://pool:8000".parse().unwrap()],
        )])));
        let cancel = CancellationToken::new();
        let catalog = RelayGatedPoolCatalog::new(candidates, cancel.clone());
        let current = catalog.subscribe();
        let source = |replica: &str| RelaySourceKey {
            pool: pool.clone(),
            service: pool.to_string(),
            replica: replica.into(),
        };
        let mut ready = catalog.open(source("1"));
        ready
            .replace_state_snapshot(0, [(pool.clone(), Some(state(1, true)))])
            .unwrap();
        let mut unready = catalog.open(source("2"));
        unready
            .replace_state_snapshot(0, [(pool.clone(), Some(state(2, false)))])
            .unwrap();
        assert_eq!(current.borrow().pools().len(), 1);
        ready.remove(1, &pool).unwrap();
        assert!(current.borrow().pools().is_empty());
        cancel.cancel();
    }
    #[tokio::test]
    async fn subscription_mode_preserves_static_pool_while_relay_pool_waits_for_readiness() {
        let key = PoolKey::new("east", "dynamo", "blue").unwrap();
        let mut subscribed = Pool::new(
            &key,
            "model",
            "east",
            vec!["http://blue:8000".parse().unwrap()],
        );
        subscribed.relay = Some("http://relay:50051".parse().unwrap());
        let static_key = PoolKey::new("east", "dynamo", "static").unwrap();
        let static_pool = Pool::new(
            &static_key,
            "model",
            "east",
            vec!["http://static:8000".parse().unwrap()],
        );
        let static_id = static_pool.id.clone();
        let pool = subscribed.id.clone();
        let (_sender, candidates) =
            watch::channel(Arc::new(PoolSnapshot::new(vec![subscribed, static_pool])));
        let cancel = CancellationToken::new();
        let signals = Arc::new(PoolSignals::new(Arc::new(
            dynamo_global_router::InFlightTracker::new(),
        )));
        let catalog = RelayGatedPoolCatalog::for_subscriptions(candidates, cancel.clone(), signals);
        let current = catalog.subscribe();
        assert_eq!(
            current
                .borrow()
                .pools()
                .iter()
                .map(|p| &p.id)
                .collect::<Vec<_>>(),
            vec![&static_id]
        );
        let mut view = catalog.open(RelaySourceKey {
            pool: pool.clone(),
            service: pool.to_string(),
            replica: "1".into(),
        });
        view.replace_state_snapshot(0, [(pool.clone(), Some(state(1, false)))])
            .unwrap();
        assert_eq!(current.borrow().pools().len(), 1);
        view.upsert_state(1, pool.clone(), Some(state(1, true)))
            .unwrap();
        assert_eq!(current.borrow().pools().len(), 2);
        drop(view);
        assert_eq!(
            current
                .borrow()
                .pools()
                .iter()
                .map(|p| &p.id)
                .collect::<Vec<_>>(),
            vec![&static_id]
        );
        cancel.cancel();
    }
    fn loaded_state(state: dynamo_global_common::grpc::PoolState) -> Arc<RelayPoolState> {
        use prost::Message;
        Arc::new(
            RelayPoolState::from_updates(
                Arc::new(wire::KvPoolCatalogUpdate::decode(state.catalog).unwrap()),
                &wire::ServingReadinessUpdate::decode(state.readiness).unwrap(),
                Some(Arc::new(
                    wire::ServingLoadUpdate::decode(state.serving_load).unwrap(),
                )),
                Some(Arc::new(
                    wire::KvPoolLoadUpdate::decode(state.kv_load).unwrap(),
                )),
                RelaySource {
                    runtime_id: 1,
                    incarnation: 1,
                },
            )
            .unwrap(),
        )
    }

    #[tokio::test]
    async fn metadata_and_readiness_changes_do_not_renew_expired_load_windows() {
        use prost::Message;
        use std::time::Duration;
        let key = PoolKey::new("east", "dynamo", "blue").unwrap();
        let pool = V1PoolIdDeriver.derive(&key);
        let (_sender, candidates) = watch::channel(Arc::new(PoolSnapshot::new(vec![Pool::new(
            &key,
            "qwen",
            "east",
            vec!["http://pool:8000".parse().unwrap()],
        )])));
        let cancel = CancellationToken::new();
        let signals = Arc::new(PoolSignals::new(Arc::new(
            dynamo_global_router::InFlightTracker::new(),
        )));
        let catalog =
            RelayGatedPoolCatalog::with_signals(candidates, cancel.clone(), Some(signals.clone()));
        let mut view = catalog.open(RelaySourceKey {
            pool: pool.clone(),
            service: pool.to_string(),
            replica: "1".into(),
        });
        let mut raw = super::fixtures::state(true, 1, 1, 10);
        let mut serving = wire::ServingLoadUpdate::decode(raw.serving_load).unwrap();
        serving.window_ms = 20;
        raw.serving_load = serving.encode_to_vec().into();
        let mut kv = wire::KvPoolLoadUpdate::decode(raw.kv_load).unwrap();
        kv.window_ms = 20;
        raw.kv_load = kv.encode_to_vec().into();
        view.replace_state_snapshot(0, [(pool.clone(), Some(loaded_state(raw.clone())))])
            .unwrap();
        assert!(signals.occupancy(&pool).is_some());
        tokio::time::timeout(Duration::from_secs(2), async {
            while signals.occupancy(&pool).is_some() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .unwrap();
        let mut metadata = wire::KvPoolCatalogUpdate::decode(raw.catalog).unwrap();
        metadata.snapshot.as_mut().unwrap().pools[0].registrations[0]
            .aliases
            .push("another-alias".into());
        raw.catalog = metadata.encode_to_vec().into();
        view.upsert_state(1, pool.clone(), Some(loaded_state(raw.clone())))
            .unwrap();
        assert!(
            signals.occupancy(&pool).is_none(),
            "metadata must not revive old reports"
        );
        let mut ready = wire::ServingReadinessUpdate::decode(raw.readiness).unwrap();
        ready.entries[0].state = wire::ServingReadinessState::Unavailable as i32;
        raw.readiness = ready.encode_to_vec().into();
        view.upsert_state(2, pool.clone(), Some(loaded_state(raw.clone())))
            .unwrap();
        ready.entries[0].state = wire::ServingReadinessState::Ready as i32;
        raw.readiness = ready.encode_to_vec().into();
        view.upsert_state(3, pool.clone(), Some(loaded_state(raw)))
            .unwrap();
        assert!(
            signals.occupancy(&pool).is_none(),
            "readiness recovery must preserve report age"
        );
        view.upsert_state(
            4,
            pool.clone(),
            Some(loaded_state(super::fixtures::state(true, 1, 1, 11))),
        )
        .unwrap();
        assert!(
            signals.occupancy(&pool).is_some(),
            "a new load sequence restores freshness"
        );
        cancel.cancel();
    }

    #[tokio::test]
    async fn discovery_reprojects_retained_reports_for_late_candidates_and_model_changes() {
        use prost::Message;
        use std::time::Duration;
        let key = PoolKey::new("east", "dynamo", "blue").unwrap();
        let pool = V1PoolIdDeriver.derive(&key);
        let (sender, candidates) = watch::channel(Arc::new(PoolSnapshot::default()));
        let cancel = CancellationToken::new();
        let in_flight = Arc::new(dynamo_global_router::InFlightTracker::new());
        let signals = Arc::new(PoolSignals::new(in_flight.clone()));
        let catalog =
            RelayGatedPoolCatalog::with_signals(candidates, cancel.clone(), Some(signals.clone()));
        let mut current = catalog.subscribe();
        let mut view = catalog.open(RelaySourceKey {
            pool: pool.clone(),
            service: pool.to_string(),
            replica: "1".into(),
        });
        let mut raw = super::fixtures::state(true, 1, 1, 10);
        let mut readiness = wire::ServingReadinessUpdate::decode(raw.readiness).unwrap();
        let mut other = readiness.entries[0].clone();
        other.canonical_model_id = "other".into();
        readiness.entries.push(other);
        raw.readiness = readiness.encode_to_vec().into();
        let mut serving = wire::ServingLoadUpdate::decode(raw.serving_load).unwrap();
        let mut other = serving.models[0].clone();
        other.canonical_model_id = "other".into();
        other
            .load
            .as_mut()
            .unwrap()
            .requests
            .as_mut()
            .unwrap()
            .requests_generating = Some(32);
        serving.models.push(other);
        raw.serving_load = serving.encode_to_vec().into();
        view.replace_state_snapshot(0, [(pool.clone(), Some(loaded_state(raw)))])
            .unwrap();
        assert!(signals.occupancy(&pool).is_none());
        let candidate = |model| {
            Pool::new(
                &key,
                model,
                "east",
                vec!["http://pool:8000".parse().unwrap()],
            )
        };
        // Requests admitted after receipt must remain in the load correction after reprojection.
        let request = in_flight.start(&pool);
        sender.send_replace(Arc::new(PoolSnapshot::new(vec![candidate("qwen")])));
        tokio::time::timeout(Duration::from_secs(2), async {
            while current.borrow().pools().is_empty() {
                current.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(signals.occupancy(&pool).unwrap().slots, 9.0 / 64.0);
        sender.send_replace(Arc::new(PoolSnapshot::new(vec![candidate("other")])));
        tokio::time::timeout(Duration::from_secs(2), async {
            while current.borrow().pools()[0].model != "other" {
                current.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert_eq!(signals.occupancy(&pool).unwrap().slots, 33.0 / 64.0);
        drop(request);
        sender.send_replace(Arc::new(PoolSnapshot::default()));
        tokio::time::timeout(Duration::from_secs(2), async {
            while !current.borrow().pools().is_empty() {
                current.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert!(signals.occupancy(&pool).is_none());
        cancel.cancel();
    }
}
