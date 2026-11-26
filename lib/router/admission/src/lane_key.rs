// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{BuildHasher, RandomState};
use std::sync::OnceLock;

/// Key of one keyed queue lane, derived from a string the request classifier chose: a session
/// id, a program, a tenant.
///
/// A class queue has one shared heap plus one lane per key that has queued requests. Lanes
/// share one order: the best head across every lane of a class dispatches first. A lane whose
/// head cannot dispatch, because the dispatch gate holds it or its eligible workers are full,
/// holds only that lane; the shared heap and the other lanes keep draining. A request waits in
/// a keyed lane only when its classifier set a lane key; without one it waits in its pinned or
/// exclusively targeted worker's lane, or in the shared heap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LaneKey(u64);

impl LaneKey {
    /// Derive the lane key for a string, with a hash keyed once per process: the string comes
    /// from a client (a session id, say), so one client must not be able to pick a value that
    /// lands in another's lane. Two strings that still collide share a lane, its order and its
    /// holds.
    pub fn new(key: &str) -> Self {
        static KEYS: OnceLock<RandomState> = OnceLock::new();
        Self(KEYS.get_or_init(RandomState::new).hash_one(key))
    }
}
