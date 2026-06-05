// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registry state machine.
//!
//! Empty → SingleKey { key, clients, used_slots } → Empty
//!                                                ↘ Draining { ... } (shutdown)
//!
//! Tenancy is enforced by the opaque [`RegistrationKey`] computed from the
//! active arm of [`RegistrationInstance`] — equal instances produce equal
//! keys, different arms hash into disjoint key spaces. Slot accounting is
//! per-instance (`RegistrationInstance::slot_count`).
//!
//! ## Lifecycle and metrics
//!
//! Registration is two-phase to keep counters honest when an external
//! component (e.g. [`crate::container::ServiceContainer`]) rejects a
//! registration **after** the registry has already accepted it:
//!
//! 1. [`Registry::try_register`] reserves a slot and updates the live
//!    gauges (`used_slots`, `registered_clients`). It does **not** touch
//!    `register_total` yet.
//! 2. After the container hook returns, the caller invokes
//!    [`Registry::commit_register`] (success) or
//!    [`Registry::rollback_register`] (failure). Commit increments
//!    `register_total`; rollback increments `register_rejected_total` and
//!    decrements the gauges.
//! 3. [`Registry::unregister`] handles clean disconnects of committed
//!    registrations (increments `unregister_total` and, if it was the last
//!    client, `reset_total`).

mod client;
mod key;
pub mod lifecycle;

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;
use serde::Serialize;
use tokio::sync::Notify;
use tracing::warn;

pub use self::client::{ClientEntry, RegistrationId};
pub use self::key::RegistrationKey;
pub use self::lifecycle::{NoopLifecycle, StreamLifecycle};

use crate::error::{ServiceError, ServiceResult};
use crate::instance::RegistrationInstance;
use crate::metrics::ServiceMetrics;

/// Primary type. Hands out `RegistrationId`s, enforces the state machine,
/// and updates metrics. Cheap to clone — internal state is an `RwLock`
/// inside an `Arc`.
pub struct Registry {
    inner: Arc<RegistryInner>,
}

struct RegistryInner {
    state: RwLock<RegistryState>,
    capacity_slots: u32,
    metrics: ServiceMetrics,
    /// Notified whenever the client count reaches zero (in `SingleKey` or
    /// `Draining`). Subscribers double-check the count under the lock.
    empty_notify: Notify,
}

#[derive(Default)]
enum RegistryState {
    #[default]
    Empty,
    SingleKey {
        key: RegistrationKey,
        instance: RegistrationInstance,
        clients: HashMap<RegistrationId, ClientSlot>,
        used_slots: u32,
    },
    Draining {
        // Last-known key/instance, kept for observability while clients drain.
        key: Option<RegistrationKey>,
        instance: Option<RegistrationInstance>,
        clients: HashMap<RegistrationId, ClientSlot>,
        used_slots: u32,
    },
}

/// Internal pairing of the registration record and the per-stream lifecycle
/// object the shutdown coordinator broadcasts to.
///
/// `lifecycle` is `None` between [`Registry::try_register`] (slot reserved)
/// and [`Registry::commit_register`] (lifecycle attached). The shutdown
/// coordinator never broadcasts to a slot in the pre-commit window — this
/// is what guarantees `ServerShutdownInitiated` cannot land on a stream
/// before its first `Accepted` event.
///
/// The active [`RegistrationInstance`] is held once at the `SingleKey` /
/// `Draining` state level (all attached clients share an equal instance by
/// key), so we don't carry a per-slot copy.
struct ClientSlot {
    entry: ClientEntry,
    lifecycle: Option<Arc<dyn StreamLifecycle>>,
    /// `true` once [`Registry::commit_register`] has been called. A slot
    /// that disappears while still uncommitted counts as a rejected
    /// registration in the metrics, not an unregister.
    committed: bool,
}

impl Registry {
    pub fn new(capacity_slots: u32, metrics: ServiceMetrics) -> Self {
        metrics.capacity_slots.set(capacity_slots as i64);
        metrics.used_slots.set(0);
        metrics.registered_clients.set(0);
        Self {
            inner: Arc::new(RegistryInner {
                state: RwLock::new(RegistryState::Empty),
                capacity_slots,
                metrics,
                empty_notify: Notify::new(),
            }),
        }
    }

    pub fn capacity_slots(&self) -> u32 {
        self.inner.capacity_slots
    }

    pub fn metrics(&self) -> &ServiceMetrics {
        &self.inner.metrics
    }

    /// Whether the registry is in the `Draining` phase. The HTTP `/ready`
    /// endpoint can use this to flip to `503` during shutdown.
    pub fn is_draining(&self) -> bool {
        matches!(*self.inner.state.read(), RegistryState::Draining { .. })
    }

    /// Reserve a slot for `instance`. Caller is responsible for
    /// `instance.validate(capacity_slots)` ahead of this. Updates the
    /// `used_slots` and `registered_clients` gauges; does **not** touch
    /// `register_total` (that happens in [`Self::commit_register`]).
    ///
    /// The slot starts with **no attached lifecycle**. The shutdown
    /// coordinator skips lifecycle-less slots when broadcasting, which
    /// means `ServerShutdownInitiated` cannot land on a stream before the
    /// handler has had a chance to enqueue `Accepted`. Attach the
    /// lifecycle (and bump `register_total`) only after the first
    /// `Accepted` is in the channel — see [`Self::commit_register`].
    ///
    /// The returned [`ClientEntry`] is the registration id the caller
    /// passes to [`Self::commit_register`] / [`Self::rollback_register`].
    pub fn try_register(
        &self,
        instance: RegistrationInstance,
        client_id: String,
    ) -> ServiceResult<ClientEntry> {
        let metrics = &self.inner.metrics;
        let slots = instance.slot_count();
        let new_key = instance.key();
        let mut state = self.inner.state.write();
        match &mut *state {
            RegistryState::Draining { .. } => {
                metrics.register_rejected_total.inc();
                Err(ServiceError::ShuttingDown(
                    "service is draining; new registrations are not accepted".into(),
                ))
            }
            RegistryState::Empty => {
                let entry = ClientEntry::new(client_id, slots);
                let slot = ClientSlot {
                    entry: entry.clone(),
                    lifecycle: None,
                    committed: false,
                };
                let mut clients = HashMap::new();
                clients.insert(entry.id, slot);
                *state = RegistryState::SingleKey {
                    key: new_key,
                    instance,
                    clients,
                    used_slots: slots,
                };
                metrics.used_slots.set(slots as i64);
                metrics.registered_clients.set(1);
                Ok(entry)
            }
            RegistryState::SingleKey {
                key: active_key,
                instance: active_instance,
                clients,
                used_slots,
            } => {
                if active_key != &new_key {
                    metrics.register_rejected_total.inc();
                    return Err(ServiceError::KeyConflict(format!(
                        "active tenant kind={} key={}; requested kind={} key={}",
                        active_instance.kind_str(),
                        active_key.short_hex(),
                        instance.kind_str(),
                        new_key.short_hex(),
                    )));
                }
                let new_used = used_slots.saturating_add(slots);
                if new_used > self.inner.capacity_slots {
                    metrics.register_rejected_total.inc();
                    return Err(ServiceError::NoCapacity(format!(
                        "used {} + requested {} > capacity {}",
                        used_slots, slots, self.inner.capacity_slots,
                    )));
                }
                let entry = ClientEntry::new(client_id, slots);
                clients.insert(
                    entry.id,
                    ClientSlot {
                        entry: entry.clone(),
                        lifecycle: None,
                        committed: false,
                    },
                );
                *used_slots = new_used;
                metrics.used_slots.set(new_used as i64);
                metrics.registered_clients.set(clients.len() as i64);
                Ok(entry)
            }
        }
    }

    /// Attach `lifecycle` to a previously-reserved slot and increment
    /// `register_total`. The lifecycle becomes reachable to the shutdown
    /// coordinator via [`Self::begin_drain`] only **after** this call —
    /// the caller is expected to have already enqueued the `Accepted`
    /// event on the response stream, so any subsequent
    /// `ServerShutdownInitiated` is guaranteed to land after `Accepted`.
    ///
    /// Errors:
    /// - [`ServiceError::ShuttingDown`] if [`Self::begin_drain`] ran
    ///   during the caller's between-`try_register`-and-`commit_register`
    ///   window. The slot is rolled back inline (gauges decremented,
    ///   `register_rejected_total` incremented).
    /// - [`ServiceError::Internal`] if the id is not present. This should
    ///   never happen in practice — the caller obtained the id from a
    ///   successful `try_register` and has not called `rollback_register`.
    pub fn commit_register(
        &self,
        id: RegistrationId,
        lifecycle: Arc<dyn StreamLifecycle>,
    ) -> ServiceResult<()> {
        let metrics = &self.inner.metrics;
        let mut state = self.inner.state.write();
        match &mut *state {
            RegistryState::Empty => {
                drop(lifecycle);
                Err(ServiceError::internal(
                    "commit_register: registry is Empty (slot vanished)",
                ))
            }
            RegistryState::SingleKey { clients, .. } => {
                let Some(slot) = clients.get_mut(&id) else {
                    drop(lifecycle);
                    return Err(ServiceError::internal(format!(
                        "commit_register: registration id {id} not found"
                    )));
                };
                slot.lifecycle = Some(lifecycle);
                slot.committed = true;
                metrics.register_total.inc();
                Ok(())
            }
            RegistryState::Draining {
                clients,
                used_slots,
                ..
            } => {
                // Concurrent shutdown beat the caller to the punch. Roll
                // back the slot inline so the in-flight handler can return
                // `unavailable` to its client cleanly.
                let Some(slot) = clients.remove(&id) else {
                    drop(lifecycle);
                    return Err(ServiceError::ShuttingDown(
                        "service began draining; slot was already removed".into(),
                    ));
                };
                *used_slots = used_slots.saturating_sub(slot.entry.reserved_slots);
                metrics.used_slots.set(*used_slots as i64);
                metrics.registered_clients.set(clients.len() as i64);
                metrics.register_rejected_total.inc();
                if clients.is_empty() {
                    self.inner.empty_notify.notify_waiters();
                }
                drop(lifecycle);
                Err(ServiceError::ShuttingDown(
                    "service began draining during registration; rolling back".into(),
                ))
            }
        }
    }

    /// Release a slot that never committed (e.g. the container hook
    /// rejected the registration). Increments `register_rejected_total`
    /// instead of `unregister_total`; never fires `reset_total` because
    /// the slot was never a real client. Returns `true` if the id was found.
    pub fn rollback_register(&self, id: RegistrationId) -> bool {
        self.remove_slot(id, RemovalReason::Rollback)
    }

    /// Remove a committed registration. Increments `unregister_total`
    /// (and `reset_total` if it was the last client in `SingleKey`).
    /// Returns `true` if the id was found.
    pub fn unregister(&self, id: RegistrationId) -> bool {
        self.remove_slot(id, RemovalReason::Unregister)
    }

    /// Shared removal path for both clean unregister and pre-commit
    /// rollback. Branches on `reason` for metric accounting and on
    /// committed-state for safety.
    fn remove_slot(&self, id: RegistrationId, reason: RemovalReason) -> bool {
        let metrics = &self.inner.metrics;
        let mut state = self.inner.state.write();

        let (removed_slot, became_empty, was_draining) = match &mut *state {
            RegistryState::Empty => (None, false, false),
            RegistryState::SingleKey {
                clients,
                used_slots,
                ..
            } => match clients.remove(&id) {
                None => (None, false, false),
                Some(slot) => {
                    *used_slots = used_slots.saturating_sub(slot.entry.reserved_slots);
                    let empty = clients.is_empty();
                    if !empty {
                        metrics.used_slots.set(*used_slots as i64);
                        metrics.registered_clients.set(clients.len() as i64);
                    }
                    (Some(slot), empty, false)
                }
            },
            RegistryState::Draining {
                clients,
                used_slots,
                ..
            } => match clients.remove(&id) {
                None => (None, false, true),
                Some(slot) => {
                    *used_slots = used_slots.saturating_sub(slot.entry.reserved_slots);
                    metrics.used_slots.set(*used_slots as i64);
                    metrics.registered_clients.set(clients.len() as i64);
                    (Some(slot), clients.is_empty(), true)
                }
            },
        };

        let Some(slot) = removed_slot else {
            return false;
        };

        // Metric accounting.
        match (reason, slot.committed) {
            (RemovalReason::Rollback, _) | (RemovalReason::Unregister, false) => {
                if reason == RemovalReason::Unregister && !slot.committed {
                    warn!(
                        registration_id = %slot.entry.id,
                        "unregister called on never-committed slot; counting as rejection"
                    );
                }
                metrics.register_rejected_total.inc();
            }
            (RemovalReason::Unregister, true) => {
                metrics.unregister_total.inc();
            }
        }

        // State machine: SingleKey → Empty on last-out. `reset_total` only
        // fires for clean unregister-driven resets, never for rollback (the
        // slot was never a real client).
        if became_empty && !was_draining {
            *state = RegistryState::Empty;
            metrics.used_slots.set(0);
            metrics.registered_clients.set(0);
            if matches!(reason, RemovalReason::Unregister) && slot.committed {
                metrics.reset_total.inc();
            }
        }

        if became_empty {
            self.inner.empty_notify.notify_waiters();
        }

        true
    }

    /// Flip into `Draining` state, return every **committed** lifecycle
    /// for broadcast, and refuse all subsequent `try_register` calls.
    /// Pre-commit slots (where the handler hasn't yet sent `Accepted` on
    /// the stream) are intentionally skipped — broadcasting to them would
    /// inject `ServerShutdownInitiated` before `Accepted`, violating the
    /// stream protocol. Those handlers will instead see
    /// [`ServiceError::ShuttingDown`] from their pending
    /// [`Self::commit_register`] call and return `unavailable` to their
    /// clients cleanly.
    ///
    /// Calling this multiple times is safe: the first call transitions
    /// the state, later calls just snapshot the currently-attached
    /// lifecycles.
    pub fn begin_drain(&self) -> Vec<Arc<dyn StreamLifecycle>> {
        let mut state = self.inner.state.write();
        match &mut *state {
            RegistryState::Empty => {
                *state = RegistryState::Draining {
                    key: None,
                    instance: None,
                    clients: HashMap::new(),
                    used_slots: 0,
                };
                Vec::new()
            }
            RegistryState::SingleKey { .. } => {
                let prior = std::mem::replace(&mut *state, RegistryState::Empty);
                let RegistryState::SingleKey {
                    key,
                    instance,
                    clients,
                    used_slots,
                } = prior
                else {
                    unreachable!("matched SingleKey above")
                };
                let lifecycles = collect_lifecycles(&clients);
                *state = RegistryState::Draining {
                    key: Some(key),
                    instance: Some(instance),
                    clients,
                    used_slots,
                };
                lifecycles
            }
            RegistryState::Draining { clients, .. } => collect_lifecycles(clients),
        }
    }

    /// Wait until the client count reaches zero. Returns immediately when
    /// the registry is already empty (in `Empty` or in `Draining` with no
    /// clients).
    pub async fn wait_until_empty(&self) {
        loop {
            // Subscribe BEFORE checking the count to avoid the wake-lost
            // race when the last unregister fires between our check and
            // the `.notified()` await.
            let notified = self.inner.empty_notify.notified();
            if self.client_count() == 0 {
                return;
            }
            notified.await;
        }
    }

    fn client_count(&self) -> usize {
        match &*self.inner.state.read() {
            RegistryState::Empty => 0,
            RegistryState::SingleKey { clients, .. } | RegistryState::Draining { clients, .. } => {
                clients.len()
            }
        }
    }

    /// Read-only snapshot suitable for serialization in the HTTP sidecar.
    pub fn snapshot(&self) -> RegistrySnapshot {
        let state = self.inner.state.read();
        match &*state {
            RegistryState::Empty => RegistrySnapshot {
                state: "Empty",
                capacity_slots: self.inner.capacity_slots,
                used_slots: 0,
                key: None,
                instance: None,
                clients: Vec::new(),
            },
            RegistryState::SingleKey {
                key,
                instance,
                clients,
                used_slots,
            } => RegistrySnapshot {
                state: "SingleKey",
                capacity_slots: self.inner.capacity_slots,
                used_slots: *used_slots,
                key: Some(key.to_string()),
                instance: Some(InstanceView::from(instance)),
                clients: client_views(clients),
            },
            RegistryState::Draining {
                key,
                instance,
                clients,
                used_slots,
            } => RegistrySnapshot {
                state: "Draining",
                capacity_slots: self.inner.capacity_slots,
                used_slots: *used_slots,
                key: key.as_ref().map(|k| k.to_string()),
                instance: instance.as_ref().map(InstanceView::from),
                clients: client_views(clients),
            },
        }
    }
}

impl Clone for Registry {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RemovalReason {
    Unregister,
    Rollback,
}

/// Snapshot the lifecycles of committed slots. Slots that haven't yet had
/// [`Registry::commit_register`] called on them are skipped — the shutdown
/// coordinator can't broadcast to them without violating the
/// `Accepted`-then-everything-else stream-protocol invariant.
fn collect_lifecycles(
    clients: &HashMap<RegistrationId, ClientSlot>,
) -> Vec<Arc<dyn StreamLifecycle>> {
    clients
        .values()
        .filter_map(|slot| slot.lifecycle.clone())
        .collect()
}

fn client_views(clients: &HashMap<RegistrationId, ClientSlot>) -> Vec<ClientView> {
    let mut entries: Vec<ClientView> = clients
        .values()
        .map(|slot| ClientView {
            id: slot.entry.id.to_string(),
            client_id: slot.entry.client_id.clone(),
            reserved_slots: slot.entry.reserved_slots,
            committed: slot.committed,
            registered_at_unix_ms: slot
                .entry
                .registered_at
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or_default(),
        })
        .collect();
    entries.sort_by(|a, b| a.registered_at_unix_ms.cmp(&b.registered_at_unix_ms));
    entries
}

/// Serializable read-only view of the registry. Returned by the
/// `/v1/registrations` HTTP endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct RegistrySnapshot {
    pub state: &'static str,
    pub capacity_slots: u32,
    pub used_slots: u32,
    /// Hex-encoded current [`RegistrationKey`] if any.
    pub key: Option<String>,
    /// Active [`RegistrationInstance`] view, if any.
    pub instance: Option<InstanceView>,
    pub clients: Vec<ClientView>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum InstanceView {
    Kvbm {
        model_name: String,
        layout: &'static str,
        tp_size: u32,
        block_size: u32,
        mode: crate::mode::ServiceMode,
    },
}

impl From<&RegistrationInstance> for InstanceView {
    fn from(inst: &RegistrationInstance) -> Self {
        match inst {
            RegistrationInstance::Kvbm(k) => Self::Kvbm {
                model_name: k.model_name.clone(),
                layout: k.layout.kind_str(),
                tp_size: k.tp_size,
                block_size: k.block_size,
                mode: k.mode,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ClientView {
    pub id: String,
    pub client_id: String,
    pub reserved_slots: u32,
    pub committed: bool,
    pub registered_at_unix_ms: u64,
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::instance::{KvbmInstance, LayoutShape};
    use crate::mode::ServiceMode;

    fn kvbm(tp: u32, model: &str) -> RegistrationInstance {
        RegistrationInstance::Kvbm(KvbmInstance {
            model_name: model.into(),
            layout: LayoutShape::UniversalTp1Canonical { bytes: vec![0, 1] },
            tp_size: tp,
            block_size: 64,
            mode: ServiceMode::Kvbm,
        })
    }

    fn noop() -> Arc<dyn StreamLifecycle> {
        Arc::new(NoopLifecycle)
    }

    fn fresh(cap: u32) -> Registry {
        Registry::new(cap, ServiceMetrics::new())
    }

    fn metric_value(g: &prometheus::IntCounter) -> u64 {
        g.get()
    }

    /// Shortcut: reserve + commit with a noop lifecycle, mirroring what the
    /// gRPC handler does once `Accepted` is queued.
    fn register_and_commit(
        r: &Registry,
        instance: RegistrationInstance,
        client_id: &str,
    ) -> ClientEntry {
        let entry = r
            .try_register(instance, client_id.into())
            .expect("try_register");
        r.commit_register(entry.id, noop())
            .expect("commit_register");
        entry
    }

    #[test]
    fn empty_to_single_key_then_empty_again() {
        let r = fresh(8);
        let a = register_and_commit(&r, kvbm(4, "llm"), "c-a");
        let b = register_and_commit(&r, kvbm(4, "llm"), "c-b");
        assert_eq!(r.snapshot().used_slots, 8);
        assert!(r.unregister(a.id));
        assert!(r.unregister(b.id));
        let s = r.snapshot();
        assert_eq!(s.state, "Empty");
        assert_eq!(s.used_slots, 0);
        assert_eq!(s.clients.len(), 0);
    }

    #[test]
    fn rejects_different_key() {
        let r = fresh(8);
        r.try_register(kvbm(4, "llm-a"), "c1".into()).unwrap();
        let err = r.try_register(kvbm(4, "llm-b"), "c2".into()).unwrap_err();
        assert!(matches!(err, ServiceError::KeyConflict(_)));
    }

    #[test]
    fn rejects_when_capacity_full() {
        let r = fresh(8);
        r.try_register(kvbm(4, "llm"), "c1".into()).unwrap();
        r.try_register(kvbm(4, "llm"), "c2".into()).unwrap();
        let err = r.try_register(kvbm(4, "llm"), "c3".into()).unwrap_err();
        assert!(matches!(err, ServiceError::NoCapacity(_)));
    }

    #[test]
    fn reset_allows_new_key_after_last_detach() {
        let r = fresh(8);
        let entry = register_and_commit(&r, kvbm(2, "llm-a"), "c1");
        assert!(r.unregister(entry.id));
        r.try_register(kvbm(2, "llm-b"), "c2".into()).unwrap();
        let snap = r.snapshot();
        assert!(matches!(
            snap.instance,
            Some(InstanceView::Kvbm { ref model_name, .. }) if model_name == "llm-b"
        ));
    }

    #[test]
    fn unregister_unknown_id_is_a_noop() {
        let r = fresh(8);
        assert!(!r.unregister(RegistrationId::new()));
        r.try_register(kvbm(2, "llm"), "c1".into()).unwrap();
        assert!(!r.unregister(RegistrationId::new()));
    }

    #[test]
    fn used_slots_tracks_correctly() {
        let r = fresh(8);
        let a = register_and_commit(&r, kvbm(2, "llm"), "c1");
        let b = register_and_commit(&r, kvbm(2, "llm"), "c2");
        let _c = register_and_commit(&r, kvbm(2, "llm"), "c3");
        assert_eq!(r.snapshot().used_slots, 6);
        assert!(r.unregister(a.id));
        assert_eq!(r.snapshot().used_slots, 4);
        assert!(r.unregister(b.id));
        assert_eq!(r.snapshot().used_slots, 2);
    }

    #[test]
    fn rollback_decrements_gauges_and_increments_rejected() {
        let r = fresh(8);
        let entry = r.try_register(kvbm(4, "llm"), "c1".into()).unwrap();
        assert_eq!(r.snapshot().used_slots, 4);
        let rejected_before = metric_value(&r.metrics().register_rejected_total);
        let unregister_before = metric_value(&r.metrics().unregister_total);
        assert!(r.rollback_register(entry.id));
        assert_eq!(r.snapshot().used_slots, 0);
        assert_eq!(r.snapshot().state, "Empty");
        assert_eq!(
            metric_value(&r.metrics().register_rejected_total),
            rejected_before + 1
        );
        // Rollback must NOT increment unregister_total.
        assert_eq!(
            metric_value(&r.metrics().unregister_total),
            unregister_before
        );
        // And it must NOT increment register_total (we never committed).
        assert_eq!(metric_value(&r.metrics().register_total), 0);
    }

    #[test]
    fn rollback_does_not_increment_reset_total() {
        let r = fresh(8);
        let entry = r.try_register(kvbm(2, "llm"), "c1".into()).unwrap();
        let reset_before = metric_value(&r.metrics().reset_total);
        assert!(r.rollback_register(entry.id));
        assert_eq!(metric_value(&r.metrics().reset_total), reset_before);
    }

    #[test]
    fn commit_then_unregister_increments_both_counters() {
        let r = fresh(8);
        let entry = register_and_commit(&r, kvbm(2, "llm"), "c1");
        assert_eq!(metric_value(&r.metrics().register_total), 1);
        assert!(r.unregister(entry.id));
        assert_eq!(metric_value(&r.metrics().unregister_total), 1);
        assert_eq!(metric_value(&r.metrics().reset_total), 1);
    }

    #[test]
    fn begin_drain_from_empty_returns_empty_vec_and_flips_state() {
        let r = fresh(8);
        let lifecycles = r.begin_drain();
        assert!(lifecycles.is_empty());
        assert!(r.is_draining());
        assert_eq!(r.snapshot().state, "Draining");
    }

    #[test]
    fn begin_drain_returns_committed_lifecycles_and_flips_state() {
        let r = fresh(8);
        register_and_commit(&r, kvbm(2, "llm"), "c1");
        register_and_commit(&r, kvbm(2, "llm"), "c2");
        let lifecycles = r.begin_drain();
        assert_eq!(lifecycles.len(), 2);
        assert!(r.is_draining());
        let snap = r.snapshot();
        assert_eq!(snap.state, "Draining");
        assert_eq!(snap.used_slots, 4);
        assert!(snap.instance.is_some());
    }

    /// Pre-commit slots (after try_register, before commit_register) MUST
    /// be invisible to the shutdown broadcaster — otherwise
    /// `ServerShutdownInitiated` could land on the stream before
    /// `Accepted`.
    #[test]
    fn begin_drain_skips_uncommitted_slots() {
        let r = fresh(8);
        let _entry = r.try_register(kvbm(2, "llm"), "c1".into()).unwrap();
        // No commit_register yet.
        let lifecycles = r.begin_drain();
        assert!(
            lifecycles.is_empty(),
            "uncommitted slot must not surface a lifecycle"
        );
        assert!(r.is_draining());
        // Used slots still reflects the reservation.
        assert_eq!(r.snapshot().used_slots, 2);
    }

    #[test]
    fn commit_register_during_drain_rolls_back_and_errors() {
        let r = fresh(8);
        let entry = r.try_register(kvbm(2, "llm"), "c1".into()).unwrap();
        r.begin_drain();
        let err = r.commit_register(entry.id, noop()).unwrap_err();
        assert!(matches!(err, ServiceError::ShuttingDown(_)));
        assert_eq!(metric_value(&r.metrics().register_rejected_total), 1);
        assert_eq!(metric_value(&r.metrics().register_total), 0);
        // Slot is gone.
        let snap = r.snapshot();
        assert_eq!(snap.used_slots, 0);
        assert_eq!(snap.clients.len(), 0);
    }

    #[test]
    fn register_during_drain_is_rejected_with_shutting_down() {
        let r = fresh(8);
        r.begin_drain();
        let err = r.try_register(kvbm(2, "llm"), "c1".into()).unwrap_err();
        assert!(matches!(err, ServiceError::ShuttingDown(_)));
    }

    #[test]
    fn unregister_during_drain_does_not_reset_state() {
        let r = fresh(8);
        let entry = register_and_commit(&r, kvbm(2, "llm"), "c1");
        r.begin_drain();
        assert!(r.unregister(entry.id));
        assert_eq!(r.snapshot().state, "Draining");
        assert_eq!(r.snapshot().clients.len(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn wait_until_empty_resolves_when_last_unregisters() {
        let r = fresh(8);
        let entry = register_and_commit(&r, kvbm(2, "llm"), "c1");
        let r2 = r.clone();
        let waiter = tokio::spawn(async move {
            tokio::time::timeout(Duration::from_secs(5), r2.wait_until_empty())
                .await
                .expect("wait_until_empty must resolve");
        });
        tokio::task::yield_now().await;
        assert!(r.unregister(entry.id));
        waiter.await.unwrap();
    }

    #[tokio::test]
    async fn wait_until_empty_immediate_when_already_empty() {
        let r = fresh(8);
        tokio::time::timeout(Duration::from_millis(100), r.wait_until_empty())
            .await
            .expect("must resolve immediately when empty");
    }
}
