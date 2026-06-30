// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BinaryHeap, HashMap, VecDeque, hash_map::Entry};

use super::PolicyQueueEntry;

pub(super) struct SessionQueue<T> {
    sessions: HashMap<Option<String>, BinaryHeap<PolicyQueueEntry<T>>>,
    order: VecDeque<Option<String>>,
}

impl<T> SessionQueue<T> {
    pub(super) fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub(super) fn push(&mut self, session_id: Option<String>, entry: PolicyQueueEntry<T>) {
        match self.sessions.entry(session_id.clone()) {
            Entry::Occupied(mut session) => session.get_mut().push(entry),
            Entry::Vacant(session) => {
                self.order.push_back(session_id);
                let mut pending = BinaryHeap::new();
                pending.push(entry);
                session.insert(pending);
            }
        }
    }

    pub(super) fn peek(&self) -> Option<&PolicyQueueEntry<T>> {
        let session_id = self.order.get(self.head_index()?)?;
        self.sessions.get(session_id)?.peek()
    }

    pub(super) fn pop(&mut self) -> Option<PolicyQueueEntry<T>> {
        let session_id = self.order.remove(self.head_index()?)?;
        let (entry, empty) = {
            let pending = self.sessions.get_mut(&session_id)?;
            let entry = pending.pop()?;
            (entry, pending.is_empty())
        };
        if empty {
            self.sessions.remove(&session_id);
        } else {
            self.order.push_back(session_id);
        }
        Some(entry)
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&PolicyQueueEntry<T>) -> bool) {
        for pending in self.sessions.values_mut() {
            pending.retain(&mut keep);
        }
        self.sessions.retain(|_, pending| !pending.is_empty());
        self.order
            .retain(|session_id| self.sessions.contains_key(session_id));
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &PolicyQueueEntry<T>> {
        self.sessions.values().flat_map(|queue| queue.iter())
    }

    pub(super) fn into_entries(self) -> impl Iterator<Item = PolicyQueueEntry<T>> {
        self.sessions.into_values().flat_map(BinaryHeap::into_iter)
    }

    fn head_index(&self) -> Option<usize> {
        let mut best = None;
        for (index, session_id) in self.order.iter().enumerate() {
            let strict_priority = self
                .sessions
                .get(session_id)
                .and_then(|pending| pending.peek())
                .expect("active session queue must not be empty")
                .priority
                .strict_priority;
            if best.is_none_or(|(_, best_priority)| strict_priority > best_priority) {
                best = Some((index, strict_priority));
            }
        }
        best.map(|(index, _)| index)
    }
}
