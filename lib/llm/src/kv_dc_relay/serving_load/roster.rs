// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishers registered for a serving-load stream, from discovery watches.
//!
//! A registered publisher is expected to report from when the Relay first sees
//! its record until one freshness window later; after that only its reports
//! keep it expected. A record left behind by a hard-killed process therefore
//! stops holding an aggregate incomplete once that window passes.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::time::{Duration, Instant};

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::discovery::{
    DiscoveryEvent, DiscoveryInstance, DiscoveryQuery, EventChannelQuery,
};
use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use tokio_stream::StreamMap;
use tokio_util::sync::CancellationToken;

use crate::utils::retry::{Backoff, FailureStreak};

const REWATCH_INITIAL: Duration = Duration::from_millis(100);
const REWATCH_MAX: Duration = Duration::from_secs(5);

/// Change to one roster, from its discovery watch.
pub(super) enum RosterEvent {
    Added(u64),
    Removed(u64),
    /// The complete registered set.
    Resync(HashSet<u64>),
    /// The watch failed; it is being re-established.
    Lost,
}

/// One roster per key, each fed by its own discovery watch.
pub(super) struct Rosters<K> {
    watches: StreamMap<K, BoxStream<'static, RosterEvent>>,
    /// First time the Relay saw each registered publisher. A key is absent
    /// until its watch delivers a complete set, and again after it is lost.
    registered: HashMap<K, HashMap<u64, Instant>>,
}

impl<K> Default for Rosters<K> {
    fn default() -> Self {
        Self {
            watches: StreamMap::new(),
            registered: HashMap::new(),
        }
    }
}

impl<K: Clone + Eq + Hash + Unpin> Rosters<K> {
    /// Watches exactly `keys`, keeping existing watches; `query` builds the
    /// discovery query of a new key.
    pub(super) fn sync(
        &mut self,
        keys: HashSet<K>,
        query: impl Fn(&K) -> EventChannelQuery,
        drt: &DistributedRuntime,
    ) {
        let stale = self
            .watches
            .keys()
            .filter(|key| !keys.contains(key))
            .cloned()
            .collect::<Vec<_>>();
        for key in stale {
            self.watches.remove(&key);
            self.registered.remove(&key);
        }
        for key in keys {
            if !self.watches.contains_key(&key) {
                let watch = watch_publishers(drt.clone(), query(&key)).boxed();
                self.watches.insert(key, watch);
            }
        }
    }

    pub(super) async fn next(&mut self) -> Option<(K, RosterEvent)> {
        self.watches.next().await
    }

    pub(super) fn apply(&mut self, key: K, event: RosterEvent, now: Instant) {
        match event {
            // Before the first `Resync` the roster is unknown, and that `Resync`
            // repeats every initial `Added`.
            RosterEvent::Added(id) => {
                if let Some(registered) = self.registered.get_mut(&key) {
                    registered.entry(id).or_insert(now);
                }
            }
            RosterEvent::Removed(id) => {
                if let Some(registered) = self.registered.get_mut(&key) {
                    registered.remove(&id);
                }
            }
            RosterEvent::Resync(ids) => {
                let previous = self.registered.remove(&key).unwrap_or_default();
                let registered = ids
                    .into_iter()
                    .map(|id| (id, previous.get(&id).copied().unwrap_or(now)))
                    .collect();
                self.registered.insert(key, registered);
            }
            RosterEvent::Lost => {
                self.registered.remove(&key);
            }
        }
    }

    /// Publishers first seen within `freshness`, per key with a known roster.
    /// Older ones are expected only while they report.
    pub(super) fn recent(&self, freshness: Duration, now: Instant) -> HashMap<K, HashSet<u64>> {
        self.registered
            .iter()
            .map(|(key, registered)| {
                let recent = registered
                    .iter()
                    .filter(|&(_, &since)| now.saturating_duration_since(since) <= freshness)
                    .map(|(&id, _)| id)
                    .collect();
                (key.clone(), recent)
            })
            .collect()
    }

    /// Whether `publisher` is registered under `key`, however long ago.
    pub(super) fn contains(&self, key: &K, publisher: u64) -> bool {
        self.registered
            .get(key)
            .is_some_and(|registered| registered.contains_key(&publisher))
    }

    /// Whether any key has a registered publisher, however old.
    pub(super) fn any_registered(&self) -> bool {
        self.registered
            .values()
            .any(|registered| !registered.is_empty())
    }
}

/// Roster changes for `query`. Never ends: a failed watch yields
/// [`RosterEvent::Lost`] and is re-established with backoff. Dropping the
/// stream cancels the watch.
fn watch_publishers(
    drt: DistributedRuntime,
    query: EventChannelQuery,
) -> impl Stream<Item = RosterEvent> + Send + 'static {
    async_stream::stream! {
        let cancel = CancellationToken::new();
        let _cancel_on_drop = cancel.clone().drop_guard();
        let mut retry = Backoff::new(REWATCH_INITIAL, REWATCH_MAX);
        let mut failures = FailureStreak::default();
        loop {
            let watch = drt
                .discovery()
                .list_and_watch(DiscoveryQuery::EventChannels(query.clone()), Some(cancel.child_token()))
                .await;
            let error = match watch {
                Ok(mut watch) => loop {
                    let event = match watch.next().await {
                        Some(Ok(event)) => event,
                        Some(Err(error)) => break error,
                        None => break anyhow::anyhow!("watch ended"),
                    };
                    match event {
                        DiscoveryEvent::Added(instance) => {
                            if let Some(id) = event_channel_id(&instance) {
                                yield RosterEvent::Added(id);
                            }
                        }
                        DiscoveryEvent::Removed(id) => yield RosterEvent::Removed(id.instance_id()),
                        DiscoveryEvent::Resync(instances) => {
                            retry.reset();
                            if let Some(failures) = failures.recover() {
                                tracing::info!(?query, failures, "KV DC Relay serving-load discovery recovered");
                            }
                            yield RosterEvent::Resync(instances.iter().filter_map(event_channel_id).collect());
                        }
                        DiscoveryEvent::ModelTaintsUpdated(_) => {}
                    }
                },
                Err(error) => error,
            };
            yield RosterEvent::Lost;
            let delay = retry.next_delay();
            if failures.fail() {
                tracing::warn!(?query, error = %format!("{error:#}"), retry_ms = delay.as_millis(), "KV DC Relay serving-load discovery failed; affected coverage cannot be proven complete");
            } else {
                tracing::debug!(?query, error = %format!("{error:#}"), failures = failures.failures(), retry_ms = delay.as_millis(), "KV DC Relay serving-load discovery still failing");
            }
            tokio::time::sleep(delay).await;
        }
    }
}

fn event_channel_id(instance: &DiscoveryInstance) -> Option<u64> {
    match instance {
        DiscoveryInstance::EventChannel { instance_id, .. } => Some(*instance_id),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRESHNESS: Duration = Duration::from_secs(3);

    fn recent(rosters: &Rosters<&'static str>, now: Instant) -> Option<HashSet<u64>> {
        rosters.recent(FRESHNESS, now).remove("ns")
    }

    #[test]
    fn registered_publishers_age_out_after_one_freshness_window() {
        let start = Instant::now();
        let mut rosters = Rosters::default();
        assert_eq!(recent(&rosters, start), None, "unknown before a resync");

        rosters.apply("ns", RosterEvent::Resync(HashSet::from([1])), start);
        rosters.apply("ns", RosterEvent::Added(2), start + Duration::from_secs(2));
        assert_eq!(
            recent(&rosters, start + Duration::from_secs(3)),
            Some(HashSet::from([1, 2]))
        );
        // A resync keeps first-seen times, so it does not revive 1.
        rosters.apply(
            "ns",
            RosterEvent::Resync(HashSet::from([1, 2])),
            start + Duration::from_secs(4),
        );
        assert_eq!(
            recent(&rosters, start + Duration::from_secs(4)),
            Some(HashSet::from([2]))
        );
        assert_eq!(
            recent(&rosters, start + Duration::from_secs(6)),
            Some(HashSet::new())
        );
        assert!(rosters.any_registered());

        rosters.apply("ns", RosterEvent::Lost, start + Duration::from_secs(6));
        assert_eq!(recent(&rosters, start + Duration::from_secs(6)), None);
        assert!(!rosters.any_registered());
    }
}
