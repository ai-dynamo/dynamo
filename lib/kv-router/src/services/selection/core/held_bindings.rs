// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer session bindings that arrived before this replica discovered their
//! partition, affinity table or worker. They wait here, outside the affinity
//! table so the worker never looks bound or eligible, and are applied through
//! the normal version rule once a worker upsert makes them placeable.

use tokio::time::Instant;

use super::*;

/// Total held bindings across partitions, the same bound as one binding channel.
const MAX_HELD_BINDINGS: usize = AFFINITY_EVENT_CHANNEL_CAPACITY;

struct HeldBinding {
    event: AffinityBindingEvent,
    received: Instant,
}

/// The newest unplaced binding per `(partition, session_id)`.
#[derive(Default)]
pub(super) struct HeldBindings {
    partitions: HashMap<RoutingPartitionId, HashMap<String, HeldBinding>>,
    len: usize,
}

impl HeldBindings {
    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.len
    }

    /// Keep `event` unless a newer version is already held for its session.
    /// An equal version refreshes the arrival time, as the owner's refresh does
    /// for a placed binding. When full, the oldest held binding is dropped.
    pub(super) fn hold(&mut self, event: AffinityBindingEvent, now: Instant) {
        if let Some(held) = self
            .partitions
            .get_mut(&event.partition)
            .and_then(|sessions| sessions.get_mut(&event.session_id))
        {
            if event.version() >= held.event.version() {
                *held = HeldBinding {
                    event,
                    received: now,
                };
            }
            return;
        }
        if self.len >= MAX_HELD_BINDINGS {
            self.evict_oldest();
        }
        self.partitions
            .entry(event.partition.clone())
            .or_default()
            .insert(
                event.session_id.clone(),
                HeldBinding {
                    event,
                    received: now,
                },
            );
        self.len += 1;
    }

    /// Forget a held binding that `placed` supersedes, so it cannot be applied
    /// later into a slot the newer binding has since left.
    pub(super) fn discard_superseded(&mut self, placed: &AffinityBindingEvent) {
        let Some(sessions) = self.partitions.get_mut(&placed.partition) else {
            return;
        };
        if sessions
            .get(&placed.session_id)
            .is_some_and(|held| held.event.version() <= placed.version())
        {
            sessions.remove(&placed.session_id);
            self.len -= 1;
            if sessions.is_empty() {
                self.partitions.remove(&placed.partition);
            }
        }
    }

    /// Remove and return the partition's bindings whose worker is now placeable,
    /// with their arrival time. Bindings held for `ttl` or longer are dropped:
    /// the owner's copy would already have expired.
    pub(super) fn take_placeable(
        &mut self,
        partition: &RoutingPartitionId,
        ttl: Duration,
        now: Instant,
        placeable: impl Fn(WorkerId) -> bool,
    ) -> Vec<(AffinityBindingEvent, Instant)> {
        let Some(sessions) = self.partitions.get_mut(partition) else {
            return Vec::new();
        };
        let before = sessions.len();
        let mut ready = Vec::new();
        for (_, held) in sessions.extract_if(|_, held| {
            now.saturating_duration_since(held.received) >= ttl || placeable(held.event.worker_id)
        }) {
            if now.saturating_duration_since(held.received) < ttl {
                ready.push((held.event, held.received));
            }
        }
        self.len -= before - sessions.len();
        if sessions.is_empty() {
            self.partitions.remove(partition);
        }
        ready
    }

    fn evict_oldest(&mut self) {
        let Some((partition, session_id)) = self
            .partitions
            .iter()
            .flat_map(|(partition, sessions)| {
                sessions
                    .iter()
                    .map(move |(session_id, held)| (held.received, partition, session_id))
            })
            .min_by_key(|(received, ..)| *received)
            .map(|(_, partition, session_id)| (partition.clone(), session_id.clone()))
        else {
            return;
        };
        tracing::debug!(
            key = %partition,
            "Dropping oldest held session affinity replica update: hold limit reached"
        );
        if let Some(sessions) = self.partitions.get_mut(&partition) {
            sessions.remove(&session_id);
            self.len -= 1;
            if sessions.is_empty() {
                self.partitions.remove(&partition);
            }
        }
    }
}

impl SelectionCore {
    pub(super) fn worker_in_partition(
        &self,
        worker_id: WorkerId,
        partition: &RoutingPartitionId,
    ) -> bool {
        self.catalog
            .get(worker_id)
            .is_some_and(|record| record.key() == *partition)
    }

    /// Apply the partition's held bindings whose worker the catalog now has.
    /// Callers commit catalog changes before calling this; `dispatch_affinity_event`
    /// checks placement under the same lock, so no binding falls between them.
    pub(super) fn apply_held_bindings(&self, partition: &RoutingPartitionId) {
        let mut held = self.held_bindings.lock();
        if held.is_empty() {
            return;
        }
        let Some(entry) = self.entry(partition) else {
            return;
        };
        let Some(table) = entry.affinity.get() else {
            return;
        };
        let now = Instant::now();
        let ready = held.take_placeable(partition, table.ttl(), now, |worker_id| {
            self.worker_in_partition(worker_id, partition)
        });
        for (event, received) in ready {
            let (target, version, worker_id) = (event.target(), event.version(), event.worker_id);
            let outcome = table.apply_held_replica_update(
                event.session_id.clone(),
                target,
                version,
                now.saturating_duration_since(received),
            );
            // A local request is binding the session; keep the peer binding in
            // case it is cancelled, and retry on the next drain.
            if outcome == ReplicaApplyOutcome::IgnoredInitializing {
                held.hold(event, received);
            }
            tracing::trace!(
                key = %partition,
                worker_id,
                ?outcome,
                "Applied held session affinity replica update"
            );
        }
    }
}
