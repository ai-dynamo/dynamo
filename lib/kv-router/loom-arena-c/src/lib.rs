// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! arena-c's child-table code (`src/indexer/arena_c/table.rs`), compiled unchanged
//! against loom's atomics, behind a small public facade for the models in `tests/`.

mod sync {
    pub(crate) use loom::sync::atomic::{AtomicU64, Ordering};
}

#[allow(dead_code)]
#[path = "../../src/indexer/arena_c/table.rs"]
mod table;

pub use loom::sync::atomic::AtomicU64;

/// Outcome of a lock-free claim, mirroring `table::Claim`.
#[derive(Debug, PartialEq, Eq)]
pub enum Claim {
    Claimed,
    Exists(u64),
    Busy,
    Full,
}

/// Outcome of a locked insert, mirroring `table::Locked`.
#[derive(Debug, PartialEq, Eq)]
pub enum Locked {
    Inserted,
    Exists(u64),
    Full,
}

/// An empty table of `slots` slots.
pub fn new_table(slots: u32) -> Vec<AtomicU64> {
    let words: Vec<AtomicU64> = (0..table::words_for(slots))
        .map(|_| AtomicU64::new(0))
        .collect();
    table::Table::init(&words, slots);
    words
}

pub fn child_key(offset: u32, head: u64) -> u64 {
    table::child_key(offset, head)
}

pub fn pack(run: u32, generation: u32) -> u64 {
    table::pack(run, generation)
}

pub fn find(words: &[AtomicU64], key: u64) -> Option<u64> {
    table::Table::new(words).find(key)
}

pub fn claim(words: &[AtomicU64], key: u64, value: u64) -> Claim {
    match table::Table::new(words).claim(key, value) {
        table::Claim::Claimed => Claim::Claimed,
        table::Claim::Exists(value) => Claim::Exists(value),
        table::Claim::Busy => Claim::Busy,
        table::Claim::Full => Claim::Full,
    }
}

pub fn insert_locked(words: &[AtomicU64], key: u64, value: u64) -> Locked {
    match table::Table::new(words).insert_locked(key, value) {
        table::Locked::Inserted => Locked::Inserted,
        table::Locked::Exists(value) => Locked::Exists(value),
        table::Locked::Full => Locked::Full,
    }
}

pub fn unlink(words: &[AtomicU64], key: u64, value: u64) -> bool {
    table::Table::new(words).unlink(key, value)
}

pub fn entries(words: &[AtomicU64]) -> Vec<(u64, u64)> {
    table::Table::new(words).entries().collect()
}

pub fn live(words: &[AtomicU64]) -> u32 {
    table::Table::new(words).live()
}

pub fn slots(words: &[AtomicU64]) -> u32 {
    table::Table::new(words).slots()
}
