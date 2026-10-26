// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! List-and-watch, as `dynamo_runtime::discovery::Discovery` does it: a watch
//! starts with one `Added` per current item, then one `Resync` with the same
//! set, then changes. A `Resync` replaces all earlier state.

#[derive(Clone, Debug)]
pub enum WatchEvent<T, K> {
    /// A new or changed item. It replaces any earlier item with the same key.
    Added(T),
    /// An item went away. It can name a key the watcher never got, for an
    /// item that went away just before the watch started; ignore those.
    Removed(K),
    Resync(Snapshot<T>),
}

/// A list, and whether it is complete.
#[derive(Clone, Debug)]
pub struct Snapshot<T> {
    pub items: Vec<T>,
    pub completeness: Completeness,
}

/// Whether a snapshot lists every item, so a new router replica can tell a
/// complete list from a partial one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Completeness {
    /// Read from a shared store: every existing item is listed.
    Exact,
    /// Built from what this process has heard so far.
    Estimated,
}
