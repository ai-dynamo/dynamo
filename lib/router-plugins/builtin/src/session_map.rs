// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded least-recently-used map from session ID to the worker that served its last request.
//!
//! Offline replay never sets a host affinity target, so session-aware policies keep this state
//! themselves. It is policy-instance local: each routing partition and worker role has its own.

use std::collections::{BTreeMap, HashMap};

use dynamo_kv_router::protocols::WorkerWithDpRank;

pub(crate) const DEFAULT_MAX_SESSIONS: usize = 65_536;

pub(crate) struct SessionMap {
    capacity: usize,
    clock: u64,
    entries: HashMap<String, (WorkerWithDpRank, u64)>,
    /// Last-use stamp → session, oldest first.
    recency: BTreeMap<u64, String>,
}

impl SessionMap {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0);
        Self {
            capacity,
            clock: 0,
            entries: HashMap::new(),
            recency: BTreeMap::new(),
        }
    }

    /// The worker that served `session`'s most recent request, if it is still remembered.
    pub(crate) fn get(&self, session: &str) -> Option<WorkerWithDpRank> {
        self.entries.get(session).map(|(worker, _)| *worker)
    }

    /// Record that `worker` served `session`, marking it most recently used and evicting the least
    /// recently used session beyond capacity.
    pub(crate) fn bind(&mut self, session: &str, worker: WorkerWithDpRank) {
        self.clock += 1;
        let stamp = self.clock;
        if let Some((bound, last_used)) = self.entries.get_mut(session) {
            *bound = worker;
            let previous = std::mem::replace(last_used, stamp);
            if let Some(key) = self.recency.remove(&previous) {
                self.recency.insert(stamp, key);
            }
            return;
        }
        self.entries.insert(session.to_owned(), (worker, stamp));
        self.recency.insert(stamp, session.to_owned());
        while self.entries.len() > self.capacity {
            let Some((_, oldest)) = self.recency.pop_first() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(id: u64) -> WorkerWithDpRank {
        WorkerWithDpRank::from_worker_id(id)
    }

    #[test]
    fn evicts_the_least_recently_used_session() {
        let mut sessions = SessionMap::new(2);
        sessions.bind("a", worker(1));
        sessions.bind("b", worker(2));
        // Touching `a` makes `b` the oldest, so `c` evicts `b`.
        sessions.bind("a", worker(3));
        sessions.bind("c", worker(4));
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions.get("a"), Some(worker(3)));
        assert_eq!(sessions.get("b"), None);
        assert_eq!(sessions.get("c"), Some(worker(4)));
    }
}
