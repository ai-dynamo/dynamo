// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::hash::{BuildHasher, RandomState};
use std::sync::OnceLock;

/// Key of one per-session queue lane, derived from the session identifier.
///
/// A class queue has one shared lane plus one lane per session that has queued requests. Lanes
/// share one order: the best head across every lane of a class dispatches first. A lane whose
/// head cannot dispatch, because the dispatch gate holds it or its eligible workers are full,
/// holds only that lane; the shared lane and the other sessions keep draining. The actor
/// assigns a session lane to every request that carries a session context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SessionLane(u64);

impl SessionLane {
    /// Derive the lane key for a session identifier, with a hash keyed once per process:
    /// session identifiers come from clients, so one client must not be able to pick an
    /// identifier that lands in another session's lane. Two identifiers that still collide
    /// share a lane, its order and its holds.
    pub fn new(session_id: &str) -> Self {
        static KEYS: OnceLock<RandomState> = OnceLock::new();
        Self(KEYS.get_or_init(RandomState::new).hash_one(session_id))
    }
}
