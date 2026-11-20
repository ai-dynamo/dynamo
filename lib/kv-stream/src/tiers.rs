// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tier slots: where a block is resident, as small integers the host assigns.

/// How many tier slots one rank can use.
pub const MAX_TIERS: usize = 8;

/// One place a block can be resident on a rank. The slot number is the host's: SGLang maps its
/// media (GPU, pinned host) to fixed slots; vLLM interns `(medium, group, locality, ownership)`
/// scopes through a [`TierMap`]. The group belongs in the key: vLLM publishes a block of every
/// KV cache group under one hash, so without it a remove from one group drops the others. A
/// model with several groups spends that many of the [`MAX_TIERS`] slots per medium.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tier(u8);

impl Tier {
    /// `None` when `slot` is not below [`MAX_TIERS`].
    pub const fn new(slot: u8) -> Option<Self> {
        if (slot as usize) < MAX_TIERS {
            Some(Self(slot))
        } else {
            None
        }
    }

    pub const fn slot(self) -> usize {
        self.0 as usize
    }
}

/// A set of tier slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TierSet(pub(crate) u8);

impl TierSet {
    pub const EMPTY: Self = Self(0);
    /// Every slot, used or not: a clear of `ALL` leaves the rank holding nothing.
    pub const ALL: Self = Self(u8::MAX);

    pub const fn of(tier: Tier) -> Self {
        Self(1 << tier.0)
    }

    pub const fn with(self, tier: Tier) -> Self {
        Self(self.0 | (1 << tier.0))
    }

    pub const fn contains(self, tier: Tier) -> bool {
        self.0 & (1 << tier.0) != 0
    }

    pub const fn is_all(self) -> bool {
        self.0 == u8::MAX
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn iter(self) -> impl Iterator<Item = Tier> {
        (0..MAX_TIERS as u8)
            .map(Tier)
            .filter(move |tier| self.contains(*tier))
    }
}

/// How many times a block is held on each tier: 0 or 1 on a set-like tier, the number of
/// outstanding stores on a counted one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct TierCounts(pub(crate) [u16; MAX_TIERS]);

impl TierCounts {
    pub const fn get(self, tier: Tier) -> u16 {
        self.0[tier.slot()]
    }

    /// Builds counts for loading a snapshot.
    pub fn with(mut self, tier: Tier, count: u16) -> Self {
        self.0[tier.slot()] = count;
        self
    }

    pub fn is_empty(self) -> bool {
        self.0 == [0; MAX_TIERS]
    }

    /// The tiers that hold the block at least once.
    pub fn tiers(self) -> TierSet {
        let mut set = TierSet::EMPTY;
        for (slot, count) in self.0.iter().enumerate() {
            if *count != 0 {
                set = set.with(Tier(slot as u8));
            }
        }
        set
    }
}

/// Assigns tier slots to an engine's own scope keys in the order they first appear.
///
/// Slots are per stream and live as long as the map; they are not stable across restarts, so a
/// snapshot wire format carries the key, not the slot. A host that names counted tiers by slot
/// in [`StreamConfig`](crate::StreamConfig) interns those scopes before the first batch.
#[derive(Debug, Clone)]
pub struct TierMap<K> {
    keys: Vec<K>,
}

impl<K> Default for TierMap<K> {
    fn default() -> Self {
        Self { keys: Vec::new() }
    }
}

impl<K> TierMap<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// The slot for `key`, assigning the next free one on first sight. `None` once all
    /// [`MAX_TIERS`] slots are taken: the host then cannot fold that scope and must say so
    /// with [`BatchWriter::mark_incomplete`](crate::BatchWriter::mark_incomplete).
    pub fn intern(&mut self, key: &K) -> Option<Tier>
    where
        K: PartialEq + Clone,
    {
        self.intern_with(|known| known == key, || key.clone())
    }

    /// [`intern`](Self::intern) for a host that holds the scope in borrowed pieces (the
    /// strings of a decoded event): `is` compares a known key against them, and `make` builds
    /// an owned key only on first sight, so a known scope costs no allocation.
    pub fn intern_with(
        &mut self,
        is: impl Fn(&K) -> bool,
        make: impl FnOnce() -> K,
    ) -> Option<Tier> {
        if let Some(slot) = self.keys.iter().position(is) {
            return Some(Tier(slot as u8));
        }
        if self.keys.len() == MAX_TIERS {
            return None;
        }
        self.keys.push(make());
        Some(Tier(self.keys.len() as u8 - 1))
    }

    pub fn key(&self, tier: Tier) -> Option<&K> {
        self.keys.get(tier.slot())
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}
