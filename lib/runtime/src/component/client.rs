// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::atomic::{AtomicBool, Ordering};
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock, Mutex as StdMutex},
    time::Duration,
};

use anyhow::Result;
use arc_swap::ArcSwap;
use futures::StreamExt;

use crate::component::{Endpoint, Instance};
use crate::config::environment_names::runtime as env_runtime;
use crate::discovery::{DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId};
use crate::routing_policy::{RoutingOccupancyState, get_or_create_routing_occupancy_state};
use crate::traits::DistributedRuntimeProvider;

/// Default interval for periodic reconciliation of instance_avail with instance_source
const DEFAULT_INHIBITED_DURATION_SECS: u64 = 5;

/// Process-wide inhibited duration, resolved from the environment on first client construction.
static INHIBITED_DURATION: LazyLock<Duration> =
    LazyLock::new(|| inhibited_duration_from_env(|name| std::env::var(name).ok()));

fn inhibited_duration_from_env(mut lookup: impl FnMut(&str) -> Option<String>) -> Duration {
    let seconds = match lookup(env_runtime::DYN_RUNTIME_INHIBITED_DURATION_SECS) {
        None => DEFAULT_INHIBITED_DURATION_SECS,
        Some(raw) => match raw.parse::<u64>() {
            Ok(seconds) => seconds,
            Err(err) => {
                tracing::warn!(
                    value = raw,
                    %err,
                    "invalid {}; using the default of {} seconds",
                    env_runtime::DYN_RUNTIME_INHIBITED_DURATION_SECS,
                    DEFAULT_INHIBITED_DURATION_SECS,
                );
                DEFAULT_INHIBITED_DURATION_SECS
            }
        },
    };
    Duration::from_secs(seconds)
}

/// Shared endpoint discovery state for a single endpoint query.
///
/// This wraps both the coalesced instance snapshot used for routing decisions
/// and a raw, lossless per-subscriber event feed used by the response-stream
/// cancellation watcher. Both outputs are driven by a single underlying
/// discovery `list_and_watch` task so clients do not multiply control-plane
/// watches.
#[derive(Debug)]
pub(crate) struct EndpointDiscoverySource {
    instance_source: tokio::sync::watch::Receiver<Vec<Instance>>,
    event_subscribers: StdMutex<Vec<tokio::sync::mpsc::UnboundedSender<DiscoveryEvent>>>,
}

pub(crate) struct DiscoveryEventReceiver {
    receiver: tokio::sync::mpsc::UnboundedReceiver<DiscoveryEvent>,
    _source: Arc<EndpointDiscoverySource>,
}

impl std::ops::Deref for DiscoveryEventReceiver {
    type Target = tokio::sync::mpsc::UnboundedReceiver<DiscoveryEvent>;

    fn deref(&self) -> &Self::Target {
        &self.receiver
    }
}

impl std::ops::DerefMut for DiscoveryEventReceiver {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.receiver
    }
}

impl EndpointDiscoverySource {
    fn new(instance_source: tokio::sync::watch::Receiver<Vec<Instance>>) -> Self {
        Self {
            instance_source,
            event_subscribers: StdMutex::new(Vec::new()),
        }
    }

    fn instance_receiver(&self) -> tokio::sync::watch::Receiver<Vec<Instance>> {
        self.instance_source.clone()
    }

    fn subscribe_events(&self) -> tokio::sync::mpsc::UnboundedReceiver<DiscoveryEvent> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        self.event_subscribers.lock().unwrap().push(tx);
        rx
    }

    fn broadcast_event(&self, event: &DiscoveryEvent) {
        let subscribers = &mut *self.event_subscribers.lock().unwrap();
        subscribers.retain(|tx| tx.send(event.clone()).is_ok());
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RoutingInstanceCounts {
    pub discovered: usize,
    pub routable: usize,
    pub overloaded: usize,
    /// IDs not currently reported overloaded, derived from `discovered - overloaded`.
    pub free: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct RoutingInstances {
    discovered_ids: Vec<u64>,
    routable_ids: Vec<u64>,
    overloaded_ids: HashSet<u64>,
    free_ids: Vec<u64>,
    routable_id_set: Arc<HashSet<u64>>,
    /// True after this client has observed at least one discovered instance.
    /// Once set, a later empty snapshot is authoritative rather than startup
    /// absence of information.
    availability_initialized: bool,
}

impl RoutingInstances {
    fn new(discovered_ids: Vec<u64>) -> Self {
        let availability_initialized = !discovered_ids.is_empty();
        Self::from_parts(
            discovered_ids.clone(),
            discovered_ids,
            HashSet::new(),
            availability_initialized,
        )
    }

    fn from_parts(
        mut discovered_ids: Vec<u64>,
        mut routable_ids: Vec<u64>,
        overloaded_ids: HashSet<u64>,
        availability_initialized: bool,
    ) -> Self {
        discovered_ids.sort_unstable();
        discovered_ids.dedup();
        routable_ids.sort_unstable();
        routable_ids.dedup();
        let free_ids = Self::derive_free_ids(&routable_ids, &overloaded_ids);
        let routable_id_set = Arc::new(routable_ids.iter().copied().collect());
        Self {
            discovered_ids,
            routable_ids,
            overloaded_ids,
            free_ids,
            routable_id_set,
            availability_initialized,
        }
    }

    pub(crate) fn discovered_ids(&self) -> &[u64] {
        &self.discovered_ids
    }

    pub(crate) fn routable_ids(&self) -> &[u64] {
        &self.routable_ids
    }

    fn available_ids(&self) -> Option<Arc<HashSet<u64>>> {
        self.availability_initialized
            .then(|| Arc::clone(&self.routable_id_set))
    }

    pub(crate) fn free_ids(&self) -> &[u64] {
        &self.free_ids
    }

    pub(crate) fn counts(&self) -> RoutingInstanceCounts {
        RoutingInstanceCounts {
            discovered: self.discovered_ids.len(),
            routable: self.routable_ids.len(),
            overloaded: self.overloaded_ids.len(),
            free: self.free_ids.len(),
        }
    }

    pub(crate) fn is_overloaded(&self, instance_id: u64) -> bool {
        self.overloaded_ids.contains(&instance_id)
    }

    fn overloaded_ids(&self) -> Option<HashSet<u64>> {
        if self.overloaded_ids.is_empty() {
            return None;
        }

        Some(self.overloaded_ids.clone())
    }

    fn reconcile_discovered(&self, discovered_ids: Vec<u64>) -> Self {
        let old_discovered_ids = self.discovered_ids.iter().copied().collect::<HashSet<_>>();
        let new_discovered_ids = discovered_ids.iter().copied().collect::<HashSet<_>>();
        let mut overloaded_ids = self.overloaded_ids.clone();
        overloaded_ids
            .retain(|id| !old_discovered_ids.contains(id) || new_discovered_ids.contains(id));

        let availability_initialized = self.availability_initialized || !discovered_ids.is_empty();
        Self::from_parts(
            discovered_ids.clone(),
            discovered_ids,
            overloaded_ids,
            availability_initialized,
        )
    }

    fn report_instance_down(&self, instance_id: u64) -> Self {
        let routable_ids: Vec<u64> = self
            .routable_ids
            .iter()
            .copied()
            .filter(|id| *id != instance_id)
            .collect();

        Self::from_parts(
            self.discovered_ids.clone(),
            routable_ids,
            self.overloaded_ids.clone(),
            self.availability_initialized,
        )
    }

    #[cfg(any(test, feature = "testing"))]
    fn override_routable_ids(&self, routable_ids: Vec<u64>) -> Self {
        // Route through from_parts so `free_ids` is recomputed from the new
        // routable set instead of carrying the stale value forward.
        Self::from_parts(
            self.discovered_ids.clone(),
            routable_ids,
            self.overloaded_ids.clone(),
            self.availability_initialized,
        )
    }

    fn set_overloaded(&self, overloaded_ids: HashSet<u64>) -> Self {
        Self::from_parts(
            self.discovered_ids.clone(),
            self.routable_ids.clone(),
            overloaded_ids,
            self.availability_initialized,
        )
    }

    /// Add a single instance to the overloaded set (immediate
    /// backpressure mark). Short-lived: the next metric-driven
    /// `set_overloaded` recompute overwrites the whole set.
    fn mark_overloaded(&self, instance_id: u64) -> Self {
        let mut overloaded_ids = self.overloaded_ids.clone();
        overloaded_ids.insert(instance_id);
        Self::from_parts(
            self.discovered_ids.clone(),
            self.routable_ids.clone(),
            overloaded_ids,
            self.availability_initialized,
        )
    }

    fn clear_overloaded_for_removed(&self, removed_ids: &HashSet<u64>) -> Self {
        let mut overloaded_ids = self.overloaded_ids.clone();
        overloaded_ids.retain(|id| !removed_ids.contains(id));
        Self::from_parts(
            self.discovered_ids.clone(),
            self.routable_ids.clone(),
            overloaded_ids,
            self.availability_initialized,
        )
    }

    fn derive_free_ids(routable_ids: &[u64], overloaded_ids: &HashSet<u64>) -> Vec<u64> {
        if overloaded_ids.is_empty() {
            return routable_ids.to_vec();
        }

        routable_ids
            .iter()
            .copied()
            .filter(|id| !overloaded_ids.contains(id))
            .collect()
    }
}

#[derive(Debug)]
struct RoutingInstancesState {
    snapshot: ArcSwap<RoutingInstances>,
    update_lock: StdMutex<()>,
    overload_reconciliation_needed: AtomicBool,
    instance_avail_tx: tokio::sync::watch::Sender<Vec<u64>>,
}

impl RoutingInstancesState {
    fn new(discovered_ids: Vec<u64>) -> (Self, tokio::sync::watch::Receiver<Vec<u64>>) {
        let snapshot = RoutingInstances::new(discovered_ids);
        let (instance_avail_tx, instance_avail_rx) =
            tokio::sync::watch::channel(snapshot.routable_ids().to_vec());
        (
            Self {
                snapshot: ArcSwap::from_pointee(snapshot),
                update_lock: StdMutex::new(()),
                overload_reconciliation_needed: AtomicBool::new(false),
                instance_avail_tx,
            },
            instance_avail_rx,
        )
    }

    fn snapshot(&self) -> arc_swap::Guard<Arc<RoutingInstances>> {
        self.snapshot.load()
    }

    fn update(
        &self,
        update: impl FnOnce(&RoutingInstances) -> RoutingInstances,
        publish_routable_ids: bool,
    ) -> Arc<RoutingInstances> {
        let _guard = self.update_lock.lock().unwrap();
        let current = self.snapshot.load();
        let next = Arc::new(update(&current));
        self.snapshot.store(next.clone());
        if publish_routable_ids {
            self.publish_routable_ids(&next);
        }
        next
    }

    fn publish_routable_ids(&self, routing_instances: &RoutingInstances) {
        let _ = self
            .instance_avail_tx
            .send(routing_instances.routable_ids().to_vec());
    }

    fn routable_ids(&self) -> Vec<u64> {
        self.snapshot().routable_ids().to_vec()
    }

    fn available_ids(&self) -> Option<Arc<HashSet<u64>>> {
        self.snapshot().available_ids()
    }

    fn free_ids(&self) -> Vec<u64> {
        self.snapshot().free_ids.clone()
    }

    fn counts(&self) -> RoutingInstanceCounts {
        self.snapshot().counts()
    }

    fn overloaded_ids(&self) -> Option<HashSet<u64>> {
        self.snapshot().overloaded_ids()
    }

    fn report_instance_down(&self, instance_id: u64) {
        self.update(|current| current.report_instance_down(instance_id), true);
    }

    fn set_overloaded_instances(&self, overloaded_instance_ids: &[u64]) -> bool {
        let overloaded_ids = overloaded_instance_ids
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let _guard = self.update_lock.lock().unwrap();
        self.overload_reconciliation_needed
            .store(false, Ordering::Release);
        let current = self.snapshot.load();
        if current.overloaded_ids == overloaded_ids {
            return false;
        }

        let next = Arc::new(current.set_overloaded(overloaded_ids));
        self.snapshot.store(next);
        true
    }

    fn mark_overloaded_immediate(&self, instance_id: u64) {
        let _guard = self.update_lock.lock().unwrap();
        let current = self.snapshot.load();
        let next = Arc::new(current.mark_overloaded(instance_id));
        self.snapshot.store(next);
        self.overload_reconciliation_needed
            .store(true, Ordering::Release);
    }

    fn overload_reconciliation_needed(&self) -> bool {
        self.overload_reconciliation_needed.load(Ordering::Acquire)
    }

    fn clear_overloaded_for_removed(&self, removed_instance_ids: &[u64]) {
        if removed_instance_ids.is_empty() {
            return;
        }

        let removed_ids = removed_instance_ids.iter().copied().collect::<HashSet<_>>();
        self.update(
            move |current| current.clear_overloaded_for_removed(&removed_ids),
            false,
        );
    }

    fn reconcile_discovered(&self, discovered_ids: Vec<u64>) -> Arc<RoutingInstances> {
        self.update(
            move |current| current.reconcile_discovered(discovered_ids),
            true,
        )
    }

    #[cfg(any(test, feature = "testing"))]
    fn override_routable_ids(&self, ids: Vec<u64>) {
        self.update(move |current| current.override_routable_ids(ids), true);
    }
}

#[derive(Clone, Debug)]
pub struct Client {
    // This is me
    pub endpoint: Endpoint,
    // Shared endpoint discovery source backing both snapshots and raw events.
    endpoint_discovery_source: Arc<EndpointDiscoverySource>,
    // These are the remotes I know about from watching key-value store
    pub instance_source: Arc<tokio::sync::watch::Receiver<Vec<Instance>>>,
    // Immutable routing snapshot. Free IDs are derived from discovered IDs and overloaded IDs.
    routing_instances: Arc<RoutingInstancesState>,
    // Client clones and standalone watchers jointly own the reconciliation task.
    instance_avail_owner: Arc<tokio::sync::watch::Receiver<Vec<u64>>>,
    /// Interval for periodic reconciliation of instance_avail with instance_source.
    /// This ensures instances removed via `report_instance_down` are eventually restored.
    /// A zero value disables local worker inhibition.
    reconcile_interval: Duration,
}

impl Client {
    // Client with auto-discover instances using key-value store
    pub(crate) async fn new(endpoint: Endpoint) -> Result<Self> {
        Self::with_reconcile_interval(endpoint, *INHIBITED_DURATION).await
    }

    /// Like [`Self::new`], but the `monitor_instance_source` background task
    /// is bound to `cancel_token` instead of the process-wide primary token.
    /// See [`Self::with_reconcile_interval_and_cancellation`] for why a
    /// caller whose own scope is narrower than the process needs this.
    pub(crate) async fn with_cancellation(
        endpoint: Endpoint,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> Result<Self> {
        Self::with_reconcile_interval_and_cancellation(endpoint, *INHIBITED_DURATION, cancel_token)
            .await
    }

    /// Create a client with a custom reconcile interval.
    /// The reconcile interval controls how often `instance_avail` is reset to match
    /// `instance_source`, restoring any instances removed via `report_instance_down`.
    pub(crate) async fn with_reconcile_interval(
        endpoint: Endpoint,
        reconcile_interval: Duration,
    ) -> Result<Self> {
        let cancel_token = endpoint.drt().primary_token();
        Self::with_reconcile_interval_and_cancellation(endpoint, reconcile_interval, cancel_token)
            .await
    }

    /// Like [`Self::with_reconcile_interval`], but the `monitor_instance_source`
    /// background task is bound to `cancel_token` rather than the process-wide
    /// primary token.
    ///
    /// A caller that builds a `Client` scoped to something narrower than the
    /// process — a monitor bound to one `WorkerSet`'s lifecycle, say — must use
    /// this constructor. `Client` is `Clone`, and `monitor_instance_source`
    /// captures its own clone before returning, so dropping every `Client`
    /// handle the caller holds does not stop that task; only cancelling its
    /// token does. Built through [`Self::new`] or [`Self::with_reconcile_interval`]
    /// instead, that task runs until process shutdown regardless of how long
    /// the caller actually keeps the `Client` around.
    pub(crate) async fn with_reconcile_interval_and_cancellation(
        endpoint: Endpoint,
        reconcile_interval: Duration,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> Result<Self> {
        tracing::trace!(
            "Client::new_dynamic: Creating dynamic client for endpoint: {}",
            endpoint.id()
        );
        let endpoint_discovery_source =
            Self::get_or_create_dynamic_discovery_source(&endpoint).await?;
        let instance_source = Arc::new(endpoint_discovery_source.instance_receiver());

        // Seed instance_avail from the current instance_source snapshot so that
        // callers who proceed immediately after wait_for_instances (which reads
        // instance_source directly) will also find instances in instance_avail
        // (which is read by the routing methods like random/round_robin).
        let initial_ids: Vec<u64> = instance_source
            .borrow()
            .iter()
            .map(|instance| instance.id())
            .collect();
        let (routing_instances, instance_avail_owner) = RoutingInstancesState::new(initial_ids);
        let client = Client {
            endpoint: endpoint.clone(),
            endpoint_discovery_source,
            instance_source: instance_source.clone(),
            routing_instances: Arc::new(routing_instances),
            instance_avail_owner: Arc::new(instance_avail_owner),
            reconcile_interval,
        };
        client.monitor_instance_source_with_cancellation(cancel_token, true);
        Ok(client)
    }

    /// Instances available from watching key-value store
    pub fn instances(&self) -> Vec<Instance> {
        self.instance_source.borrow().clone()
    }

    pub fn instance_ids(&self) -> Vec<u64> {
        self.instances().into_iter().map(|ep| ep.id()).collect()
    }

    pub fn instance_ids_avail(&self) -> Vec<u64> {
        self.routing_instances.routable_ids()
    }

    /// Routable instance ids excluding those currently flagged overloaded — the set used
    /// for load-aware (random / round-robin) worker selection.
    pub fn instance_ids_free(&self) -> Vec<u64> {
        self.routing_instances.free_ids()
    }

    pub(crate) fn routing_instances(&self) -> arc_swap::Guard<Arc<RoutingInstances>> {
        self.routing_instances.snapshot()
    }

    pub fn routing_instance_counts(&self) -> RoutingInstanceCounts {
        self.routing_instances.counts()
    }

    /// Get a watcher for available instance IDs
    pub fn instance_avail_watcher(&self) -> tokio::sync::watch::Receiver<Vec<u64>> {
        self.instance_avail_owner.as_ref().clone()
    }

    /// Create a client view whose routable instances are restricted by a caller-owned
    /// admission set.
    ///
    /// Endpoint discovery remains the source of connection metadata and hard availability. The
    /// returned client publishes only the intersection of that endpoint membership and
    /// `admitted_ids`, allowing a higher-level controller to keep discovered-but-unvalidated
    /// instances out of a routing group. The view has independent overload and fault-inhibition
    /// state, just like a freshly constructed client.
    pub fn with_admitted_instances(
        &self,
        admitted_ids: tokio::sync::watch::Receiver<Vec<u64>>,
    ) -> Self {
        self.with_admitted_instances_and_cancellation(
            admitted_ids,
            self.endpoint.drt().primary_token(),
        )
    }

    /// Like [`Self::with_admitted_instances`], with a lifecycle token for construction-time
    /// cancellation by an owning controller.
    pub fn with_admitted_instances_and_cancellation(
        &self,
        mut admitted_ids: tokio::sync::watch::Receiver<Vec<u64>>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        let mut endpoint_instances = self.instance_source.as_ref().clone();
        let initial = Self::filter_admitted_instances(
            endpoint_instances.borrow().as_slice(),
            admitted_ids.borrow().as_slice(),
        );
        let initial_ids = initial.iter().map(Instance::id).collect::<Vec<_>>();
        let (instance_tx, instance_rx) = tokio::sync::watch::channel(initial);
        let updater_cancel = cancel_token.clone();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = updater_cancel.cancelled() => break,
                    _ = instance_tx.closed() => break,
                    result = endpoint_instances.changed() => {
                        if result.is_err() {
                            break;
                        }
                    }
                    result = admitted_ids.changed() => {
                        if result.is_err() {
                            break;
                        }
                    }
                }

                let next = Self::filter_admitted_instances(
                    endpoint_instances.borrow_and_update().as_slice(),
                    admitted_ids.borrow_and_update().as_slice(),
                );
                let changed = *instance_tx.borrow() != next;
                if changed && instance_tx.send(next).is_err() {
                    break;
                }
            }
        });

        let (routing_instances, instance_avail_owner) = RoutingInstancesState::new(initial_ids);
        let client = Self {
            endpoint: self.endpoint.clone(),
            endpoint_discovery_source: self.endpoint_discovery_source.clone(),
            instance_source: Arc::new(instance_rx),
            routing_instances: Arc::new(routing_instances),
            instance_avail_owner: Arc::new(instance_avail_owner),
            reconcile_interval: self.reconcile_interval,
        };
        client.monitor_instance_source_with_cancellation(cancel_token, false);
        client
    }

    fn filter_admitted_instances(instances: &[Instance], admitted_ids: &[u64]) -> Vec<Instance> {
        if admitted_ids.is_empty() {
            return Vec::new();
        }

        let admitted = admitted_ids.iter().copied().collect::<HashSet<_>>();
        instances
            .iter()
            .filter(|instance| admitted.contains(&instance.id()))
            .cloned()
            .collect()
    }

    /// Subscribe to raw discovery events for this endpoint.
    ///
    /// Unlike `instance_source`, this feed does not coalesce remove→add pairs,
    /// so consumers can react to every removal event exactly once.
    pub(crate) fn subscribe_discovery_events(&self) -> DiscoveryEventReceiver {
        DiscoveryEventReceiver {
            receiver: self.endpoint_discovery_source.subscribe_events(),
            _source: self.endpoint_discovery_source.clone(),
        }
    }

    /// Wait for at least one Instance to be available for this Endpoint
    pub async fn wait_for_instances(&self) -> Result<Vec<Instance>> {
        tracing::trace!(
            "wait_for_instances: Starting wait for endpoint: {}",
            self.endpoint.id()
        );
        let mut rx = self.instance_source.as_ref().clone();
        // wait for there to be 1 or more endpoints
        let mut instances: Vec<Instance>;
        loop {
            instances = rx.borrow_and_update().to_vec();
            if instances.is_empty() {
                rx.changed().await?;
            } else {
                tracing::info!(
                    "wait_for_instances: Found {} instance(s) for endpoint: {}",
                    instances.len(),
                    self.endpoint.id()
                );
                break;
            }
        }
        Ok(instances)
    }

    /// Mark an instance as down/unavailable
    pub fn report_instance_down(&self, instance_id: u64) {
        if self.reconcile_interval.is_zero() {
            tracing::debug!(
                instance_id,
                "local worker inhibition is disabled; leaving instance routable"
            );
            return;
        }

        self.routing_instances.report_instance_down(instance_id);
        tracing::debug!("inhibiting instance {instance_id}");
    }

    /// Replace the set of overloaded instances reported by the worker monitor.
    /// Returns true when this changes the routing snapshot.
    pub fn set_overloaded_instances(&self, overloaded_instance_ids: &[u64]) -> bool {
        self.routing_instances
            .set_overloaded_instances(overloaded_instance_ids)
    }

    /// Whether request-path backpressure changed overload state after the monitor's
    /// most recent metric publication.
    pub fn overload_reconciliation_needed(&self) -> bool {
        self.routing_instances.overload_reconciliation_needed()
    }

    /// Mark an instance overloaded immediately after a worker-scoped
    /// `WorkerOverloaded` response. This is backpressure, not a fault, so it
    /// does not call `report_instance_down`. The next worker-monitor
    /// reconciliation replaces this short-lived global routing hint.
    pub fn mark_overloaded_immediate(&self, instance_id: u64) {
        self.routing_instances
            .mark_overloaded_immediate(instance_id);
        tracing::debug!(
            instance_id,
            "marking instance overloaded (backpressure); next metric event will re-evaluate"
        );
    }

    pub fn clear_overloaded_instances_for_removed(&self, removed_instance_ids: &[u64]) {
        self.routing_instances
            .clear_overloaded_for_removed(removed_instance_ids);
    }

    pub fn overloaded_instance_ids(&self) -> Option<HashSet<u64>> {
        self.routing_instances.overloaded_ids()
    }

    /// Workers currently eligible for selection: discovered and not locally
    /// inhibited by [`Self::report_instance_down`].
    ///
    /// This hard-availability snapshot is separate from transient overload.
    /// `None` means this client has not discovered an instance yet. After the
    /// first discovery, `Some` is authoritative, including `Some(empty)` when
    /// the last previously discovered worker is removed.
    pub fn available_instance_ids(&self) -> Option<Arc<HashSet<u64>>> {
        self.routing_instances.available_ids()
    }

    /// Monitor the key-value instance source and update instance_avail.
    ///
    /// This function also performs periodic reconciliation: if `instance_source` hasn't
    /// changed for `reconcile_interval`, we reset `instance_avail` to match
    /// `instance_source`. This ensures instances removed via `report_instance_down`
    /// are eventually restored even if the discovery source doesn't emit updates.
    ///
    /// The spawned task runs until `cancel_token` cancels. A caller that wants
    /// this task to outlive nothing shorter than the process should pass
    /// `self.endpoint.drt().primary_token()`, as [`Self::new`] does.
    fn monitor_instance_source_with_cancellation(
        &self,
        cancel_token: tokio_util::sync::CancellationToken,
        prune_shared_occupancy: bool,
    ) {
        let reconcile_interval = self.reconcile_interval;
        let endpoint = self.endpoint.clone();
        let endpoint_discovery_source = self.endpoint_discovery_source.clone();
        let routing_instances = self.routing_instances.clone();
        let instance_source = self.instance_source.clone();
        let endpoint_id = self.endpoint.id();
        tokio::task::spawn(async move {
            let mut rx = instance_source.as_ref().clone();
            while !cancel_token.is_cancelled() {
                let instance_ids: Vec<u64> = rx
                    .borrow_and_update()
                    .iter()
                    .map(|instance| instance.id())
                    .collect();

                let snapshot = routing_instances.reconcile_discovered(instance_ids);

                // Clean up stale occupancy counters for instances that no longer exist.
                if prune_shared_occupancy {
                    let registry = endpoint.drt().routing_occupancy_states();
                    if let Ok(registry) = registry.try_lock()
                        && let Some(weak) = registry.get(&endpoint)
                        && let Some(state) = weak.upgrade()
                    {
                        state.retain(snapshot.discovered_ids());
                    }
                }

                tokio::select! {
                    _ = cancel_token.cancelled() => break,
                    _ = routing_instances.instance_avail_tx.closed() => break,
                    result = rx.changed() => {
                        if let Err(err) = result {
                            tracing::error!(
                                "monitor_instance_source: The Sender is dropped: {err}, endpoint={endpoint_id}",
                            );
                            cancel_token.cancel();
                        }
                    }
                    _ = tokio::time::sleep(reconcile_interval), if !reconcile_interval.is_zero() => {
                        tracing::trace!(
                            "monitor_instance_source: periodic reconciliation for endpoint={endpoint_id}",
                        );
                    }
                }
            }
            drop(endpoint_discovery_source);
        });
    }

    /// Simulate a complete discovery snapshot for testing.
    #[cfg(any(test, feature = "testing"))]
    pub fn override_discovered_instances(&self, ids: Vec<u64>) {
        self.reconcile_discovered_instances(ids);
    }

    /// Override routable IDs for testing while preserving discovery membership.
    #[cfg(any(test, feature = "testing"))]
    pub fn override_instance_avail(&self, ids: Vec<u64>) {
        self.routing_instances.override_routable_ids(ids);
    }

    fn reconcile_discovered_instances(&self, discovered_ids: Vec<u64>) -> Arc<RoutingInstances> {
        self.routing_instances.reconcile_discovered(discovered_ids)
    }

    async fn get_or_create_dynamic_discovery_source(
        endpoint: &Endpoint,
    ) -> Result<Arc<EndpointDiscoverySource>> {
        let drt = endpoint.drt();
        let sources = drt.endpoint_discovery_sources();
        let mut sources = sources.lock().await;

        if let Some(source) = sources.get(endpoint) {
            if let Some(source) = source.upgrade() {
                return Ok(source);
            } else {
                sources.remove(endpoint);
            }
        }

        let discovery = drt.discovery();
        let discovery_query = crate::discovery::DiscoveryQuery::Endpoint {
            namespace: endpoint.component.namespace.name.clone(),
            component: endpoint.component.name.clone(),
            endpoint: endpoint.name.clone(),
        };

        let mut discovery_stream = discovery
            .list_and_watch(discovery_query.clone(), None)
            .await?;
        let (watch_tx, watch_rx) = tokio::sync::watch::channel(vec![]);
        let discovery_source = Arc::new(EndpointDiscoverySource::new(watch_rx));

        let secondary = endpoint.component.drt.runtime().secondary().clone();
        let discovery_source_task = Arc::downgrade(&discovery_source);

        secondary.spawn(async move {
            tracing::trace!("endpoint_watcher: Starting for discovery query: {:?}", discovery_query);
            let mut map: HashMap<u64, Instance> = HashMap::new();

            loop {
                let discovery_event = tokio::select! {
                    _ = watch_tx.closed() => {
                        break;
                    }
                    discovery_event = discovery_stream.next() => {
                        match discovery_event {
                            Some(Ok(event)) => {
                                event
                            },
                            Some(Err(e)) => {
                                tracing::error!("endpoint_watcher: discovery stream error: {}; shutting down for discovery query: {:?}", e, discovery_query);
                                break;
                            }
                            None => {
                                break;
                            }
                        }
                    }
                };

                if let Some(discovery_source) = discovery_source_task.upgrade() {
                    discovery_source.broadcast_event(&discovery_event);
                }

                match discovery_event {
                    DiscoveryEvent::Added(DiscoveryInstance::Endpoint(instance)) => {
                        map.insert(instance.instance_id, instance);
                    }
                    DiscoveryEvent::Added(_) => {}
                    DiscoveryEvent::ModelTaintsUpdated(_) => {}
                    DiscoveryEvent::Removed(id) => {
                        if let DiscoveryInstanceId::Endpoint(endpoint_id) = id {
                            map.remove(&endpoint_id.instance_id);
                        }
                    }
                }

                let instances: Vec<Instance> = map.values().cloned().collect();
                if watch_tx.send(instances).is_err() {
                    break;
                }
            }
            let _ = watch_tx.send(vec![]);
        });

        sources.insert(endpoint.clone(), Arc::downgrade(&discovery_source));
        Ok(discovery_source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DistributedRuntime, Runtime, distributed::DistributedConfig};

    async fn wait_for_discovery_event(
        receiver: &mut DiscoveryEventReceiver,
        predicate: impl Fn(&DiscoveryEvent) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let event = receiver.recv().await.expect("discovery event feed closed");
                if predicate(&event) {
                    return;
                }
            }
        })
        .await
        .expect("expected discovery event was not received");
    }

    async fn wait_for_watch_state<T>(
        receiver: &mut tokio::sync::watch::Receiver<Vec<T>>,
        predicate: impl Fn(&[T]) -> bool,
    ) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if predicate(receiver.borrow_and_update().as_slice()) {
                    return;
                }
                receiver
                    .changed()
                    .await
                    .expect("instance availability feed closed");
            }
        })
        .await
        .expect("expected instance availability state was not observed");
    }

    #[test]
    fn test_inhibited_duration_from_env() {
        assert_eq!(
            inhibited_duration_from_env(|_| None),
            Duration::from_secs(DEFAULT_INHIBITED_DURATION_SECS)
        );
        assert_eq!(
            inhibited_duration_from_env(|_| Some("17".to_string())),
            Duration::from_secs(17)
        );
        assert_eq!(
            inhibited_duration_from_env(|_| Some("0".to_string())),
            Duration::ZERO
        );
        assert_eq!(
            inhibited_duration_from_env(|_| Some("invalid".to_string())),
            Duration::from_secs(DEFAULT_INHIBITED_DURATION_SECS)
        );
    }

    #[tokio::test]
    async fn dropping_last_client_releases_routing_state() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let endpoint = drt
            .namespace("test_client_lifecycle".to_string())
            .unwrap()
            .component("test_component".to_string())
            .unwrap()
            .endpoint("decode".to_string());
        let client = endpoint.client().await.unwrap();
        let routing_instances = Arc::downgrade(&client.routing_instances);
        let discovery_source = Arc::downgrade(&client.endpoint_discovery_source);
        let mut raw_watcher = client.instance_source.as_ref().clone();
        let mut watcher = client.instance_avail_watcher();
        let mut event_watcher = client.subscribe_discovery_events();
        raw_watcher.borrow_and_update();
        watcher.borrow_and_update();

        drop(client);
        assert!(routing_instances.upgrade().is_some());
        assert!(discovery_source.upgrade().is_some());

        endpoint.register_endpoint_instance().await.unwrap();
        wait_for_watch_state(&mut watcher, |instances| instances.len() == 1).await;
        wait_for_discovery_event(&mut event_watcher, |event| {
            matches!(event, DiscoveryEvent::Added(DiscoveryInstance::Endpoint(_)))
        })
        .await;

        endpoint.unregister_endpoint_instance().await.unwrap();
        wait_for_watch_state(&mut watcher, |instances| instances.is_empty()).await;
        assert!(raw_watcher.borrow_and_update().is_empty());
        wait_for_discovery_event(&mut event_watcher, |event| {
            matches!(event, DiscoveryEvent::Removed(_))
        })
        .await;

        drop(watcher);

        tokio::time::timeout(Duration::from_secs(1), async {
            while routing_instances.strong_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("client monitor retained state after its last observer was dropped");
        assert!(discovery_source.upgrade().is_some());

        endpoint.register_endpoint_instance().await.unwrap();
        wait_for_watch_state(&mut raw_watcher, |instances| instances.len() == 1).await;
        wait_for_discovery_event(&mut event_watcher, |event| {
            matches!(event, DiscoveryEvent::Added(DiscoveryInstance::Endpoint(_)))
        })
        .await;
        endpoint.unregister_endpoint_instance().await.unwrap();
        wait_for_watch_state(&mut raw_watcher, |instances| instances.is_empty()).await;
        wait_for_discovery_event(&mut event_watcher, |event| {
            matches!(event, DiscoveryEvent::Removed(_))
        })
        .await;

        drop(event_watcher);
        tokio::time::timeout(Duration::from_secs(1), async {
            while discovery_source.strong_count() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("discovery event receiver retained its source after being dropped");

        rt.shutdown();
    }

    /// Test that instances removed via report_instance_down are restored after
    /// the reconciliation interval elapses.
    #[tokio::test]
    async fn test_instance_reconciliation() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(100);

        let rt = Runtime::from_current().unwrap();
        // Use process_local config to avoid needing etcd/nats
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_reconciliation".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        // Use a short reconcile interval for faster tests
        let client = Client::with_reconcile_interval(endpoint, TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();

        // Initially, instance_avail should be empty (no registered instances)
        assert!(client.instance_ids_avail().is_empty());

        // For this test, we'll directly manipulate instance_avail and verify reconciliation
        // Store some test IDs
        client.override_instance_avail(vec![1, 2, 3]);

        assert_eq!(client.instance_ids_avail(), vec![1u64, 2, 3]);

        // Simulate report_instance_down removing instance 2
        client.report_instance_down(2);
        assert_eq!(client.instance_ids_avail(), vec![1u64, 3]);

        // Wait for reconciliation interval + buffer
        // The monitor_instance_source will reset instance_avail to match instance_source
        // Since instance_source is empty, after reconciliation instance_avail should be empty
        tokio::time::sleep(TEST_RECONCILE_INTERVAL + Duration::from_millis(50)).await;

        // After reconciliation, instance_avail should match instance_source (which is empty)
        assert!(
            client.instance_ids_avail().is_empty(),
            "After reconciliation, instance_avail should match instance_source"
        );

        rt.shutdown();
    }

    /// A zero inhibited duration disables local worker inhibition.
    #[tokio::test]
    async fn test_zero_inhibited_duration_leaves_instance_routable() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_disabled_inhibition".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = Client::with_reconcile_interval(endpoint, Duration::ZERO)
            .await
            .unwrap();

        client.override_instance_avail(vec![1, 2, 3]);
        client.report_instance_down(2);

        assert_eq!(
            client.instance_ids_avail(),
            vec![1, 2, 3],
            "a zero inhibited duration should leave the reported instance routable"
        );

        rt.shutdown();
    }

    /// Test that report_instance_down correctly removes an instance from instance_avail.
    #[tokio::test]
    async fn test_report_instance_down() {
        let rt = Runtime::from_current().unwrap();
        // Use process_local config to avoid needing etcd/nats
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_report_down".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = endpoint.client().await.unwrap();

        // Manually set up instance_avail with test instances
        client.override_instance_avail(vec![1, 2, 3]);
        assert_eq!(client.instance_ids_avail(), vec![1u64, 2, 3]);

        // Report instance 2 as down
        client.report_instance_down(2);

        // Verify instance 2 is removed
        let avail = client.instance_ids_avail();
        assert!(avail.contains(&1), "Instance 1 should still be available");
        assert!(
            !avail.contains(&2),
            "Instance 2 should be removed after report_instance_down"
        );
        assert!(avail.contains(&3), "Instance 3 should still be available");

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_overloaded_instance_ids_returns_none_when_empty() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_overloaded_ids".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();

        assert_eq!(client.overloaded_instance_ids(), None);

        assert!(client.set_overloaded_instances(&[7]));
        assert_eq!(client.overloaded_instance_ids(), Some(HashSet::from([7])));
        assert!(!client.set_overloaded_instances(&[7]));

        assert!(client.set_overloaded_instances(&[]));
        assert_eq!(client.overloaded_instance_ids(), None);
        assert!(!client.set_overloaded_instances(&[]));

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_instance_reconciliation_preserves_overloaded_existing_instances() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(50);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_overloaded_reconciliation".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let worker_id = instances[0].id();

        for _ in 0..10 {
            if client.instance_ids_free().contains(&worker_id) {
                break;
            }
            tokio::time::sleep(TEST_RECONCILE_INTERVAL).await;
        }
        assert!(
            client.instance_ids_free().contains(&worker_id),
            "worker should be free after initial discovery reconciliation"
        );

        client.set_overloaded_instances(&[worker_id]);
        assert!(
            client.instance_ids_free().is_empty(),
            "worker should be overloaded before periodic reconciliation"
        );

        tokio::time::sleep(TEST_RECONCILE_INTERVAL + Duration::from_millis(50)).await;

        assert!(
            client.instance_ids_free().is_empty(),
            "periodic reconciliation should not mark an existing overloaded worker free"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_report_instance_down_preserves_overloaded_state() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(50);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_report_down_preserves_overloaded".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let worker_id = instances[0].id();

        for _ in 0..10 {
            if client.instance_ids_avail().contains(&worker_id) {
                break;
            }
            tokio::time::sleep(TEST_RECONCILE_INTERVAL).await;
        }

        client.set_overloaded_instances(&[worker_id]);
        client.report_instance_down(worker_id);

        assert!(
            !client.instance_ids_avail().contains(&worker_id),
            "reported-down worker should leave routable availability"
        );
        assert_eq!(
            client.routing_instance_counts().overloaded,
            1,
            "reported-down worker should remain overloaded while still discovered"
        );
        assert!(
            client.instance_ids_free().is_empty(),
            "reported-down overloaded worker should not become free"
        );

        endpoint.unregister_endpoint_instance().await.unwrap();
        for _ in 0..10 {
            if client.routing_instance_counts().overloaded == 0 {
                break;
            }
            tokio::time::sleep(TEST_RECONCILE_INTERVAL).await;
        }

        assert_eq!(
            client.routing_instance_counts().overloaded,
            0,
            "stable discovery removal should clear overloaded state"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_instance_reconciliation_prunes_removed_overloaded_instances() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(50);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_removed_overloaded_cleanup".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let worker_id = instances[0].id();

        client.set_overloaded_instances(&[worker_id]);
        assert_eq!(client.routing_instance_counts().overloaded, 1);
        assert!(client.instance_ids_free().is_empty());

        endpoint.unregister_endpoint_instance().await.unwrap();
        for _ in 0..10 {
            if client.routing_instance_counts().overloaded == 0 {
                break;
            }
            tokio::time::sleep(TEST_RECONCILE_INTERVAL).await;
        }

        assert_eq!(
            client.routing_instance_counts().overloaded,
            0,
            "removed discovered workers should not remain in overloaded state"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_instance_ids_free_excludes_overloaded_new_instances() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(50);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let worker_id = drt.connection_id();
        let ns = drt
            .namespace("test_new_overloaded_reconciliation".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();
        client.set_overloaded_instances(&[worker_id]);

        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        assert_eq!(instances[0].id(), worker_id);
        assert!(
            client.instance_ids_free().is_empty(),
            "newly discovered overloaded worker should not be free"
        );

        tokio::time::sleep(TEST_RECONCILE_INTERVAL + Duration::from_millis(50)).await;

        assert!(
            client.instance_ids_free().is_empty(),
            "discovery reconciliation should not affect recomputed free workers"
        );

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_discovery_add_updates_free_without_overloaded_publish() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(50);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_free_updates_on_discovery_add".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let instances = client.wait_for_instances().await.unwrap();
        let worker_id = instances[0].id();

        for _ in 0..10 {
            if client.instance_ids_free().contains(&worker_id) {
                break;
            }
            tokio::time::sleep(TEST_RECONCILE_INTERVAL).await;
        }

        assert_eq!(
            client.instance_ids_free(),
            vec![worker_id],
            "newly discovered non-overloaded workers should appear free without an overload update"
        );

        rt.shutdown();
    }

    /// Test that instance_avail_watcher receives updates when instances change.
    #[tokio::test]
    async fn test_instance_avail_watcher() {
        let rt = Runtime::from_current().unwrap();
        // Use process_local config to avoid needing etcd/nats
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_watcher".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = endpoint.client().await.unwrap();
        let watcher = client.instance_avail_watcher();

        // Set initial instances
        client.override_instance_avail(vec![1, 2, 3]);

        // Report instance down - this should notify the watcher
        client.report_instance_down(2);

        // The watcher should receive the update
        // Note: We need to check if changed() was signaled
        let current = watcher.borrow().clone();
        assert_eq!(current, vec![1, 3]);

        rt.shutdown();
    }

    /// Regression test: `monitor_instance_source_with_cancellation`'s task must
    /// exit on its own `cancel_token`, not only at process shutdown.
    ///
    /// `Client::new` bound this task to the process-wide primary token
    /// unconditionally. A caller building a `Client` scoped to something
    /// narrower — a monitor bound to one `WorkerSet`'s lifecycle, say — had no
    /// way to stop the task before then: dropping every `Client` handle does
    /// not stop it, since it holds its own clone. Every WorkerSet rebuild
    /// leaked one.
    ///
    /// The observable is the strong count of `routing_instances`: the spawned
    /// task captures a clone of it, so the count returning to 1 proves the
    /// task actually exited and dropped that capture.
    #[tokio::test]
    async fn monitor_instance_source_exits_on_its_own_cancellation_token() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt
            .namespace("test_monitor_instance_source_cancellation".to_string())
            .unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let cancel_token = tokio_util::sync::CancellationToken::new();
        let client = Client::with_cancellation(endpoint.clone(), cancel_token.clone())
            .await
            .unwrap();

        // Negative control, first: the task must still be alive, and still
        // holding its capture, before cancellation.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(
            Arc::strong_count(&client.routing_instances) > 1,
            "monitor task must be running (and holding its capture) before cancellation"
        );

        cancel_token.cancel();
        tokio::time::timeout(Duration::from_secs(5), async {
            while Arc::strong_count(&client.routing_instances) > 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("monitor_instance_source task must exit when its cancel_token cancels");

        rt.shutdown();
    }

    #[tokio::test]
    async fn admitted_client_never_routes_unadmitted_endpoint_instances() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_admitted_client".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());
        let endpoint_client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = endpoint_client.wait_for_instances().await.unwrap()[0].id();

        let (admission_tx, admission_rx) = tokio::sync::watch::channel(Vec::new());
        let admitted_client = endpoint_client.with_admitted_instances(admission_rx);
        let mut admitted = admitted_client.instance_avail_watcher();
        assert!(admitted.borrow().is_empty());

        admission_tx.send_replace(vec![worker_id]);
        tokio::time::timeout(Duration::from_secs(1), admitted.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(admitted.borrow_and_update().clone(), vec![worker_id]);

        admission_tx.send_replace(Vec::new());
        tokio::time::timeout(Duration::from_secs(1), admitted.changed())
            .await
            .unwrap()
            .unwrap();
        assert!(admitted.borrow_and_update().is_empty());

        rt.shutdown();
    }

    /// Test that concurrent select_and_increment distributes load correctly.
    #[tokio::test]
    async fn test_concurrent_select_and_increment() {
        let state = Arc::new(RoutingOccupancyState::default());
        let instance_ids: Vec<u64> = vec![100, 200, 300];
        let num_requests = 90;

        let mut handles = Vec::new();
        for _ in 0..num_requests {
            let state = state.clone();
            let ids = instance_ids.clone();
            handles.push(tokio::spawn(async move {
                state.select_exact_min_and_increment(&ids).await
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        assert_eq!(state.load(100), 30);
        assert_eq!(state.load(200), 30);
        assert_eq!(state.load(300), 30);
    }

    #[tokio::test]
    async fn test_select_exact_min_and_increment_randomizes_ties() {
        let mut selected = [false; 3];

        for _ in 0..120 {
            let state = RoutingOccupancyState::default();
            let picked = state
                .select_exact_min_and_increment(&[10, 20, 30])
                .await
                .unwrap();
            match picked {
                10 => selected[0] = true,
                20 => selected[1] = true,
                30 => selected[2] = true,
                _ => panic!("unexpected worker id: {picked}"),
            }
        }

        let selected_count = selected.into_iter().filter(|seen| *seen).count();
        assert!(
            selected_count > 1,
            "tie-breaking should not always select the first minimum-load worker"
        );
    }

    #[tokio::test]
    async fn test_connection_counts() {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_ll_counts".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let state1 = get_or_create_routing_occupancy_state(&endpoint).await;
        let state2 = get_or_create_routing_occupancy_state(&endpoint).await;

        let picked1 = state1
            .select_exact_min_and_increment(&[10, 20, 30])
            .await
            .unwrap();
        assert_eq!(state1.load(picked1), 1);

        let picked2 = state1
            .select_exact_min_and_increment(&[10, 20, 30])
            .await
            .unwrap();
        assert_ne!(picked1, picked2);

        // state2 should see the same counts (same underlying Arc)
        assert_eq!(state2.load(10), state1.load(10));
        assert_eq!(state2.load(20), state1.load(20));
        assert_eq!(state2.load(30), state1.load(30));

        state2.decrement(picked1);
        assert_eq!(state1.load(picked1), if picked1 == picked2 { 1 } else { 0 });

        rt.shutdown();
    }

    #[tokio::test]
    async fn test_least_loaded_state_retain_preserves_live_counts() {
        let state = RoutingOccupancyState::default();

        // Add some connections
        state.select_exact_min_and_increment(&[1, 2, 3]).await;
        state.select_exact_min_and_increment(&[1, 2, 3]).await;
        state.select_exact_min_and_increment(&[1, 2, 3]).await;
        // Each instance should have 1 connection
        assert_eq!(state.load(1), 1);
        assert_eq!(state.load(2), 1);
        assert_eq!(state.load(3), 1);

        // Discovery removal must not delete guard-owned accounting.
        state.retain(&[1, 3]);

        assert_eq!(state.load(1), 1);
        assert_eq!(state.load(2), 1);
        assert_eq!(state.load(3), 1);
    }

    #[tokio::test]
    async fn test_monitor_instance_source_defers_removed_worker_cleanup() {
        const TEST_RECONCILE_INTERVAL: Duration = Duration::from_millis(50);

        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
            .await
            .unwrap();
        let ns = drt.namespace("test_occupancy_cleanup".to_string()).unwrap();
        let component = ns.component("test_component".to_string()).unwrap();
        let endpoint = component.endpoint("test_endpoint".to_string());

        let client = Client::with_reconcile_interval(endpoint.clone(), TEST_RECONCILE_INTERVAL)
            .await
            .unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        client.wait_for_instances().await.unwrap();

        let worker_id = client.instance_ids_avail()[0];
        let state = get_or_create_routing_occupancy_state(&endpoint).await;
        state.increment(worker_id);
        assert_eq!(state.load(worker_id), 1);

        endpoint.unregister_endpoint_instance().await.unwrap();

        for _ in 0..10 {
            if !client.instance_ids().contains(&worker_id) {
                break;
            }
            tokio::time::sleep(TEST_RECONCILE_INTERVAL).await;
        }

        assert_eq!(
            state.load(worker_id),
            1,
            "discovery absence must retain live accounting"
        );
        state.decrement(worker_id);
        assert_eq!(state.load(worker_id), 0);

        rt.shutdown();
    }
}
