// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryInstanceId, DiscoveryMetadata,
    reconcile_discovery_snapshot,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct CachedCrMetadata {
    pub(super) metadata: Arc<DiscoveryMetadata>,
    pub(super) uid: Option<String>,
    pub(super) owner_pod_uid: Option<String>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(super) struct ReadyEntry {
    pub(super) instance_id: u64,
    pub(super) pod_uid: String,
}

impl ReadyEntry {
    pub(super) fn new(instance_id: u64, pod_uid: String) -> Self {
        Self {
            instance_id,
            pod_uid,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReadinessContribution {
    cr_key: String,
    ready: ReadyEntry,
}

/// Tracks the readiness entries contributed by each watched Kubernetes object.
///
/// EndpointSlices can temporarily duplicate a Pod while they are resharded. The
/// reverse index keeps the Pod ready until every identical contribution is gone.
/// Conflicting Pod incarnations for the same CR key fail closed.
#[derive(Default)]
pub(super) struct ReadinessIndex {
    by_object: HashMap<String, HashSet<ReadinessContribution>>,
    by_cr_key: HashMap<String, HashMap<ReadyEntry, usize>>,
}

impl ReadinessIndex {
    pub(super) fn replace_object(
        &mut self,
        object_key: String,
        entries: Vec<(String, ReadyEntry)>,
    ) -> HashSet<String> {
        let mut affected = self.remove_object(&object_key);
        let contributions: HashSet<_> = entries
            .into_iter()
            .map(|(cr_key, ready)| ReadinessContribution { cr_key, ready })
            .collect();

        for contribution in &contributions {
            affected.insert(contribution.cr_key.clone());
            *self
                .by_cr_key
                .entry(contribution.cr_key.clone())
                .or_default()
                .entry(contribution.ready.clone())
                .or_default() += 1;
        }

        if !contributions.is_empty() {
            self.by_object.insert(object_key, contributions);
        }
        affected
    }

    pub(super) fn remove_object(&mut self, object_key: &str) -> HashSet<String> {
        let Some(contributions) = self.by_object.remove(object_key) else {
            return HashSet::new();
        };

        let mut affected = HashSet::with_capacity(contributions.len());
        for contribution in contributions {
            affected.insert(contribution.cr_key.clone());
            let remove_cr_key = if let Some(entries) = self.by_cr_key.get_mut(&contribution.cr_key)
            {
                if let Some(count) = entries.get_mut(&contribution.ready) {
                    *count -= 1;
                    if *count == 0 {
                        entries.remove(&contribution.ready);
                    }
                }
                entries.is_empty()
            } else {
                false
            };
            if remove_cr_key {
                self.by_cr_key.remove(&contribution.cr_key);
            }
        }
        affected
    }

    pub(super) fn resolved(&self, cr_key: &str) -> Option<ReadyEntry> {
        let entries = self.by_cr_key.get(cr_key)?;
        if entries.len() != 1 {
            tracing::warn!(
                cr_key,
                incarnations = entries.len(),
                "Conflicting readiness contributions; excluding worker"
            );
            return None;
        }
        entries.keys().next().cloned()
    }

    pub(super) fn resolved_entries(&self) -> HashMap<String, ReadyEntry> {
        self.by_cr_key
            .keys()
            .filter_map(|cr_key| self.resolved(cr_key).map(|ready| (cr_key.clone(), ready)))
            .collect()
    }
}

pub(super) struct JoinTable {
    left: HashMap<String, ReadyEntry>,
    right: HashMap<String, CachedCrMetadata>,
    pub(super) known: HashMap<u64, Arc<DiscoveryMetadata>>,
}

impl JoinTable {
    pub(super) fn new() -> Self {
        Self {
            left: HashMap::new(),
            right: HashMap::new(),
            known: HashMap::new(),
        }
    }

    pub(super) fn set_readiness(
        &mut self,
        cr_key: String,
        ready: Option<ReadyEntry>,
        changes: &mut BatchChanges,
    ) {
        if self.left.get(&cr_key) == ready.as_ref() {
            return;
        }

        if let Some(old) = self.left.get(&cr_key) {
            changes.capture(old.instance_id, &self.known);
        }
        if let Some(new) = ready.as_ref() {
            changes.capture(new.instance_id, &self.known);
        }

        let old = match ready {
            Some(ready) => self.left.insert(cr_key.clone(), ready),
            None => self.left.remove(&cr_key),
        };
        if let Some(old) = old
            && self
                .left
                .get(&cr_key)
                .is_none_or(|current| current.instance_id != old.instance_id)
        {
            self.known.remove(&old.instance_id);
        }

        self.reconcile_key(&cr_key);
    }

    pub(super) fn set_cr(
        &mut self,
        cr_key: String,
        cached: Option<CachedCrMetadata>,
        changes: &mut BatchChanges,
    ) {
        if let Some(ready) = self.left.get(&cr_key) {
            changes.capture(ready.instance_id, &self.known);
        }

        match cached {
            Some(cached) => {
                self.right.insert(cr_key.clone(), cached);
            }
            None => {
                self.right.remove(&cr_key);
            }
        }
        self.reconcile_key(&cr_key);
    }

    pub(super) fn replace_readiness(
        &mut self,
        mut next: HashMap<String, ReadyEntry>,
        changes: &mut BatchChanges,
    ) {
        let keys: HashSet<String> = self.left.keys().chain(next.keys()).cloned().collect();
        for cr_key in keys {
            self.set_readiness(cr_key.clone(), next.remove(&cr_key), changes);
        }
    }

    pub(super) fn replace_crs(
        &mut self,
        mut next: HashMap<String, CachedCrMetadata>,
        changes: &mut BatchChanges,
    ) {
        let keys: HashSet<String> = self.right.keys().chain(next.keys()).cloned().collect();
        for cr_key in keys {
            self.set_cr(cr_key.clone(), next.remove(&cr_key), changes);
        }
    }

    #[cfg(test)]
    pub(super) fn apply_readiness_scan(
        &mut self,
        next: HashMap<String, (u64, String)>,
    ) -> Vec<DiscoveryEvent> {
        let next = next
            .into_iter()
            .map(|(cr_key, (instance_id, pod_uid))| (cr_key, ReadyEntry::new(instance_id, pod_uid)))
            .collect();
        let mut changes = BatchChanges::default();
        self.replace_readiness(next, &mut changes);
        changes.finish(self).events
    }

    #[cfg(test)]
    pub(super) fn apply_cr_scan(
        &mut self,
        next: HashMap<String, CachedCrMetadata>,
    ) -> Vec<DiscoveryEvent> {
        let mut changes = BatchChanges::default();
        self.replace_crs(next, &mut changes);
        changes.finish(self).events
    }

    #[cfg(test)]
    pub(super) fn cr_uid(&self, cr_key: &str) -> Option<&str> {
        self.right.get(cr_key)?.uid.as_deref()
    }

    fn reconcile_key(&mut self, cr_key: &str) {
        let Some(ready) = self.left.get(cr_key) else {
            return;
        };

        let Some(cached) = self
            .right
            .get(cr_key)
            .filter(|cached| cached.owner_pod_uid.as_deref() == Some(ready.pod_uid.as_str()))
        else {
            self.known.remove(&ready.instance_id);
            return;
        };

        self.known
            .insert(ready.instance_id, cached.metadata.clone());
    }
}

#[derive(Default)]
pub(super) struct BatchChanges {
    before: HashMap<u64, Option<Arc<DiscoveryMetadata>>>,
}

impl BatchChanges {
    fn capture(&mut self, instance_id: u64, known: &HashMap<u64, Arc<DiscoveryMetadata>>) {
        self.before
            .entry(instance_id)
            .or_insert_with(|| known.get(&instance_id).cloned());
    }

    pub(super) fn finish(self, join_table: &JoinTable) -> Publication {
        let mut events = Vec::new();
        let mut state_changes = Vec::with_capacity(self.before.len());

        for (instance_id, before) in self.before {
            let after = join_table.known.get(&instance_id).cloned();
            if matches!((&before, &after), (Some(old), Some(new)) if Arc::ptr_eq(old, new)) {
                continue;
            }

            let old_instances = flatten_metadata(before.as_deref());
            let new_instances = flatten_metadata(after.as_deref());
            // Check raw instance equality before moving new_instances.
            // reconcile_discovery_snapshot intentionally suppresses same-ID EventChannel
            // diffs, so using diff.is_empty() as the Upsert gate would leave list_state
            // stale when an EventChannel transport changes.
            let instances_changed = old_instances != new_instances;
            let (diff, _) = reconcile_discovery_snapshot(&old_instances, new_instances);
            events.extend(diff);

            match after {
                Some(metadata) if instances_changed => {
                    state_changes.push(StateChange::Upsert(instance_id, metadata));
                }
                None if before.is_some() => {
                    state_changes.push(StateChange::Remove(instance_id));
                }
                _ => {}
            }
        }

        Publication {
            events,
            state_changes,
        }
    }
}

fn flatten_metadata(
    metadata: Option<&DiscoveryMetadata>,
) -> HashMap<DiscoveryInstanceId, DiscoveryInstance> {
    metadata
        .into_iter()
        .flat_map(DiscoveryMetadata::get_all)
        .map(|instance| (instance.id(), instance))
        .collect()
}

pub(super) enum StateChange {
    Upsert(u64, Arc<DiscoveryMetadata>),
    Remove(u64),
}

pub(super) struct Publication {
    pub(super) events: Vec<DiscoveryEvent>,
    pub(super) state_changes: Vec<StateChange>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::{Instance, TransportType};

    const TEST_POD_UID: &str = "pod-uid-test";

    fn ready(instance_id: u64, pod_uid: &str) -> ReadyEntry {
        ReadyEntry::new(instance_id, pod_uid.to_string())
    }

    fn cached(cr_uid: &str, owner_pod_uid: &str) -> CachedCrMetadata {
        let mut metadata = DiscoveryMetadata::new();
        metadata
            .register_endpoint(DiscoveryInstance::Endpoint(Instance {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep".to_string(),
                instance_id: 99,
                transport: TransportType::Tcp("127.0.0.1:1234".to_string()),
                device_type: None,
                request_plane_codec: None,
            }))
            .unwrap();
        CachedCrMetadata {
            metadata: Arc::new(metadata),
            uid: Some(cr_uid.to_string()),
            owner_pod_uid: Some(owner_pod_uid.to_string()),
        }
    }

    fn apply_readiness(table: &mut JoinTable, next: HashMap<String, ReadyEntry>) -> Publication {
        let mut changes = BatchChanges::default();
        table.replace_readiness(next, &mut changes);
        changes.finish(table)
    }

    fn apply_crs(table: &mut JoinTable, next: HashMap<String, CachedCrMetadata>) -> Publication {
        let mut changes = BatchChanges::default();
        table.replace_crs(next, &mut changes);
        changes.finish(table)
    }

    #[test]
    fn joins_only_matching_pod_incarnations() {
        let mut table = JoinTable::new();
        apply_readiness(
            &mut table,
            HashMap::from([("worker".to_string(), ready(1, "U2"))]),
        );

        let publication = apply_crs(
            &mut table,
            HashMap::from([("worker".to_string(), cached("C1", "U1"))]),
        );
        assert!(publication.events.is_empty());
        assert!(!table.known.contains_key(&1));

        let publication = apply_crs(
            &mut table,
            HashMap::from([("worker".to_string(), cached("C2", "U2"))]),
        );
        assert!(
            publication
                .events
                .iter()
                .any(|event| matches!(event, DiscoveryEvent::Added(_)))
        );
        assert!(table.known.contains_key(&1));
    }

    #[test]
    fn readiness_index_keeps_duplicate_slice_contribution() {
        let mut index = ReadinessIndex::default();
        let contribution = ("worker".to_string(), ready(1, TEST_POD_UID));
        index.replace_object("slice-a".to_string(), vec![contribution.clone()]);
        index.replace_object("slice-b".to_string(), vec![contribution]);

        index.remove_object("slice-a");
        assert_eq!(index.resolved("worker"), Some(ready(1, TEST_POD_UID)));
    }

    #[test]
    fn readiness_index_fails_closed_on_conflicting_incarnations() {
        let mut index = ReadinessIndex::default();
        index.replace_object(
            "slice-a".to_string(),
            vec![("worker".to_string(), ready(1, "U1"))],
        );
        index.replace_object(
            "slice-b".to_string(),
            vec![("worker".to_string(), ready(1, "U2"))],
        );

        assert_eq!(index.resolved("worker"), None);
        index.remove_object("slice-a");
        assert_eq!(index.resolved("worker"), Some(ready(1, "U2")));
    }

    #[test]
    fn batch_coalesces_add_then_remove() {
        let mut table = JoinTable::new();
        let mut changes = BatchChanges::default();
        table.set_readiness(
            "worker".to_string(),
            Some(ready(1, TEST_POD_UID)),
            &mut changes,
        );
        table.set_cr(
            "worker".to_string(),
            Some(cached("C1", TEST_POD_UID)),
            &mut changes,
        );
        table.set_cr("worker".to_string(), None, &mut changes);

        let publication = changes.finish(&table);
        assert!(publication.events.is_empty());
        assert!(publication.state_changes.is_empty());
    }

    #[test]
    fn readiness_rebuild_removes_missing_worker() {
        let mut table = JoinTable::new();
        apply_readiness(
            &mut table,
            HashMap::from([("worker".to_string(), ready(1, TEST_POD_UID))]),
        );
        apply_crs(
            &mut table,
            HashMap::from([("worker".to_string(), cached("C1", TEST_POD_UID))]),
        );

        let publication = apply_readiness(&mut table, HashMap::new());
        assert!(
            publication
                .events
                .iter()
                .any(|event| matches!(event, DiscoveryEvent::Removed(_)))
        );
        assert!(
            publication
                .state_changes
                .iter()
                .any(|c| matches!(c, StateChange::Remove(1)))
        );
        assert!(!table.known.contains_key(&1));
    }

    #[test]
    fn cr_recreation_replaces_cached_cr_without_events() {
        let mut table = JoinTable::new();
        apply_readiness(
            &mut table,
            HashMap::from([("worker".to_string(), ready(1, TEST_POD_UID))]),
        );
        apply_crs(
            &mut table,
            HashMap::from([("worker".to_string(), cached("C1", TEST_POD_UID))]),
        );

        let publication = apply_crs(
            &mut table,
            HashMap::from([("worker".to_string(), cached("C2", TEST_POD_UID))]),
        );
        assert!(publication.events.is_empty());
        assert!(publication.state_changes.is_empty());
        assert_eq!(table.cr_uid("worker"), Some("C2"));
    }

    #[test]
    fn cr_delta_touches_only_changed_worker() {
        let mut table = JoinTable::new();
        let readiness = (0..1_000)
            .map(|id| (format!("worker-{id}"), ready(id, TEST_POD_UID)))
            .collect();
        let crs = (0..1_000)
            .map(|id| {
                (
                    format!("worker-{id}"),
                    cached(&format!("C{id}"), TEST_POD_UID),
                )
            })
            .collect();
        apply_readiness(&mut table, readiness);
        apply_crs(&mut table, crs);

        let mut updated_metadata = DiscoveryMetadata::new();
        updated_metadata
            .register_endpoint(DiscoveryInstance::Endpoint(Instance {
                namespace: "ns".to_string(),
                component: "comp".to_string(),
                endpoint: "ep-updated".to_string(),
                instance_id: 99,
                transport: TransportType::Tcp("127.0.0.1:1234".to_string()),
                device_type: None,
                request_plane_codec: None,
            }))
            .unwrap();
        let updated_cr = CachedCrMetadata {
            metadata: Arc::new(updated_metadata),
            uid: Some("C-new".to_string()),
            owner_pod_uid: Some(TEST_POD_UID.to_string()),
        };

        let mut changes = BatchChanges::default();
        table.set_cr("worker-500".to_string(), Some(updated_cr), &mut changes);
        let publication = changes.finish(&table);

        assert_eq!(publication.state_changes.len(), 1);
        assert!(matches!(
            publication.state_changes.as_slice(),
            [StateChange::Upsert(500, _)]
        ));
    }

    #[test]
    fn event_channel_transport_change_updates_list_state() {
        // reconcile_discovery_snapshot intentionally emits no DiscoveryEvent for
        // same-ID EventChannel transport changes, so the Upsert gate must not rely
        // on the event diff being non-empty.
        use crate::discovery::{EventScope, EventTransport};

        let mut table = JoinTable::new();
        apply_readiness(
            &mut table,
            HashMap::from([("worker".to_string(), ready(1, TEST_POD_UID))]),
        );

        let mut meta = DiscoveryMetadata::new();
        meta.register_event_channel(DiscoveryInstance::EventChannel {
            scope: EventScope::Namespace {
                name: "ns".to_string(),
            },
            topic: "kv-events".to_string(),
            instance_id: 1,
            transport: EventTransport::zmq("tcp://127.0.0.1:5555"),
        })
        .unwrap();
        apply_crs(
            &mut table,
            HashMap::from([(
                "worker".to_string(),
                CachedCrMetadata {
                    metadata: Arc::new(meta),
                    uid: Some("C1".to_string()),
                    owner_pod_uid: Some(TEST_POD_UID.to_string()),
                },
            )]),
        );

        let mut updated_meta = DiscoveryMetadata::new();
        updated_meta
            .register_event_channel(DiscoveryInstance::EventChannel {
                scope: EventScope::Namespace {
                    name: "ns".to_string(),
                },
                topic: "kv-events".to_string(),
                instance_id: 1,
                transport: EventTransport::zmq("tcp://127.0.0.1:6666"),
            })
            .unwrap();
        let publication = apply_crs(
            &mut table,
            HashMap::from([(
                "worker".to_string(),
                CachedCrMetadata {
                    metadata: Arc::new(updated_meta),
                    uid: Some("C2".to_string()),
                    owner_pod_uid: Some(TEST_POD_UID.to_string()),
                },
            )]),
        );

        // reconcile emits no events (same-ID EventChannel transport change is
        // intentionally suppressed), but list_state must still be updated.
        assert!(publication.events.is_empty());
        assert!(matches!(
            publication.state_changes.as_slice(),
            [StateChange::Upsert(1, _)]
        ));
    }
}
