// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, VecDeque, hash_map::Entry};

use super::PolicyQueueEntry;

pub(super) struct SessionQueue<T> {
    pending: HashMap<String, PolicyQueueEntry<T>>,
    order: VecDeque<String>,
}

impl<T> SessionQueue<T> {
    pub(super) fn new() -> Self {
        Self {
            pending: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub(super) fn push(
        &mut self,
        session_id: String,
        entry: PolicyQueueEntry<T>,
    ) -> Result<(), (String, PolicyQueueEntry<T>)> {
        match self.pending.entry(session_id.clone()) {
            Entry::Vacant(pending) => {
                pending.insert(entry);
                self.order.push_back(session_id);
                Ok(())
            }
            Entry::Occupied(_) => Err((session_id, entry)),
        }
    }

    pub(super) fn peek(&self) -> Option<&PolicyQueueEntry<T>> {
        self.pending.get(self.order.get(self.head_index()?)?)
    }

    pub(super) fn pop(&mut self) -> Option<PolicyQueueEntry<T>> {
        let session_id = self.order.remove(self.head_index()?)?;
        self.pending.remove(&session_id)
    }

    pub(super) fn retain(&mut self, mut keep: impl FnMut(&PolicyQueueEntry<T>) -> bool) {
        self.pending.retain(|_, entry| keep(entry));
        self.order
            .retain(|session_id| self.pending.contains_key(session_id));
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = &PolicyQueueEntry<T>> {
        self.pending.values()
    }

    pub(super) fn into_entries(self) -> impl Iterator<Item = PolicyQueueEntry<T>> {
        self.pending.into_values()
    }

    fn head_index(&self) -> Option<usize> {
        let mut best = None;
        for (index, session_id) in self.order.iter().enumerate() {
            let strict_priority = self
                .pending
                .get(session_id)
                .expect("active session must have a pending request")
                .priority
                .strict_priority;
            if best.is_none_or(|(_, best_priority)| strict_priority > best_priority) {
                best = Some((index, strict_priority));
            }
        }
        best.map(|(index, _)| index)
    }
}
