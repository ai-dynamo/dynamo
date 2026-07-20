// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G2 eviction ordering independent of residency and capacity accounting.

use std::collections::BTreeSet;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap;

use super::{G2EvictionPolicy, HostBlockKey};

/// Ordering callbacks consumed by the G2 residency manager.
///
/// The manager remains authoritative for whether a block exists, is resident,
/// is pinned, or is protected by the cohort being admitted. A strategy only
/// tracks eviction order. Victim selection is a non-mutating preview so the
/// manager can reject an atomic admission without rolling strategy state back.
pub(super) trait G2EvictionStrategy: Send {
    /// A pending store became resident and eligible for future eviction.
    fn on_resident(&mut self, key: HostBlockKey);

    /// A resident, unpinned block was accessed.
    fn on_access(&mut self, key: HostBlockKey);

    /// A resident block transitioned from unpinned to pinned.
    fn on_pin(&mut self, key: HostBlockKey);

    /// A resident block transitioned from pinned to unpinned.
    fn on_unpin(&mut self, key: HostBlockKey);

    /// The manager committed removal of a resident block.
    fn on_remove(&mut self, key: HostBlockKey);

    /// Preview up to `count` victims in policy order.
    ///
    /// `is_eligible` is supplied by the manager and captures its canonical pin
    /// state plus admission-specific protection. This method must not mutate
    /// strategy state.
    fn select_victims(
        &self,
        count: usize,
        is_eligible: &mut dyn FnMut(HostBlockKey) -> bool,
    ) -> Vec<HostBlockKey>;
}

pub(super) fn build_g2_eviction_strategy(
    policy: G2EvictionPolicy,
) -> Result<Box<dyn G2EvictionStrategy>> {
    match policy {
        G2EvictionPolicy::Lru => Ok(Box::new(LruEviction::default())),
        unsupported => bail!("unsupported G2 eviction policy: {unsupported:?}"),
    }
}

#[derive(Default)]
struct LruEviction {
    recency_clock: u64,
    last_touch: FxHashMap<HostBlockKey, u64>,
    oldest_first: BTreeSet<(u64, HostBlockKey)>,
}

impl LruEviction {
    fn mark_mru(&mut self, key: HostBlockKey) {
        if let Some(previous) = self.last_touch.remove(&key) {
            let removed = self.oldest_first.remove(&(previous, key));
            debug_assert!(removed);
        }
        self.recency_clock = self
            .recency_clock
            .checked_add(1)
            .expect("G2 recency counter exhausted");
        let previous = self.last_touch.insert(key, self.recency_clock);
        debug_assert!(previous.is_none());
        let inserted = self.oldest_first.insert((self.recency_clock, key));
        debug_assert!(inserted);
    }

    fn remove_from_order(&mut self, key: HostBlockKey) {
        if let Some(last_touch) = self.last_touch.remove(&key) {
            let removed = self.oldest_first.remove(&(last_touch, key));
            debug_assert!(removed);
        }
    }
}

impl G2EvictionStrategy for LruEviction {
    fn on_resident(&mut self, key: HostBlockKey) {
        debug_assert!(!self.last_touch.contains_key(&key));
        self.mark_mru(key);
    }

    fn on_access(&mut self, key: HostBlockKey) {
        debug_assert!(self.last_touch.contains_key(&key));
        self.mark_mru(key);
    }

    fn on_pin(&mut self, key: HostBlockKey) {
        debug_assert!(self.last_touch.contains_key(&key));
        self.remove_from_order(key);
    }

    fn on_unpin(&mut self, key: HostBlockKey) {
        debug_assert!(!self.last_touch.contains_key(&key));
        self.mark_mru(key);
    }

    fn on_remove(&mut self, key: HostBlockKey) {
        self.remove_from_order(key);
    }

    fn select_victims(
        &self,
        count: usize,
        is_eligible: &mut dyn FnMut(HostBlockKey) -> bool,
    ) -> Vec<HostBlockKey> {
        self.oldest_first
            .iter()
            .map(|(_, key)| *key)
            .filter(|key| is_eligible(*key))
            .take(count)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(value: u8) -> HostBlockKey {
        HostBlockKey::new(0, None, [value; 32])
    }

    #[test]
    fn lru_callbacks_control_order_without_owning_eligibility() {
        let mut lru = LruEviction::default();
        lru.on_resident(key(1));
        lru.on_resident(key(2));
        lru.on_resident(key(3));
        lru.on_access(key(1));

        lru.on_pin(key(2));
        let mut exclude_three = |candidate| candidate != key(3);
        assert_eq!(lru.select_victims(2, &mut exclude_three), vec![key(1)]);

        lru.on_unpin(key(2));
        assert_eq!(
            lru.select_victims(2, &mut exclude_three),
            vec![key(1), key(2)]
        );

        lru.on_remove(key(1));
        let mut all = |_| true;
        assert_eq!(lru.select_victims(3, &mut all), vec![key(3), key(2)]);
    }

    #[test]
    fn victim_preview_does_not_mutate_lru_order() {
        let mut lru = LruEviction::default();
        lru.on_resident(key(1));
        lru.on_resident(key(2));
        let mut all = |_| true;

        assert_eq!(lru.select_victims(2, &mut all), vec![key(1), key(2)]);
        assert_eq!(lru.select_victims(2, &mut all), vec![key(1), key(2)]);
    }

    #[test]
    fn unsupported_policy_is_rejected_by_the_strategy_factory() {
        assert!(build_g2_eviction_strategy(G2EvictionPolicy::Arc).is_err());
        assert!(build_g2_eviction_strategy(G2EvictionPolicy::RetentionPriorityLru).is_err());
    }
}
