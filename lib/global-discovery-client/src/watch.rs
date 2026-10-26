// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! List-and-watch events. A `Resync` replaces all earlier state.

#[derive(Clone, Debug)]
pub enum WatchEvent<T, K> {
    /// A new or changed item. It replaces any earlier item with the same key.
    Added(T),
    Removed(K),
    Resync(Snapshot<T>),
}

#[derive(Clone, Debug)]
pub struct Snapshot<T> {
    pub items: Vec<T>,
    pub completeness: Completeness,
}

/// Whether a snapshot lists every item, so a new router replica can tell a
/// complete list from a partial one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completeness {
    /// Read from a shared store, so it lists every existing item.
    Exact,
    /// Built from the updates this replica has received so far.
    Estimated,
}
