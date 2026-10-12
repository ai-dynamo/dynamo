// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::PoisonError;

use crossbeam_utils::sync::{ShardedLock, ShardedLockReadGuard, ShardedLockWriteGuard};

/// Reader-sharded `RwLock` for tables read by every lifecycle operation.
///
/// Each request operation read-locks the worker table and the derived load
/// table, while writes happen only on topology changes. With a single reader
/// count, every read acquisition writes the same cache line from every core.
/// `ShardedLock` spreads readers across eight per-thread shards, chosen by
/// thread index rather than CPU, which reduces reader contention; threads can
/// still share a shard. The rare writer locks every shard and still waits for
/// all in-flight readers, so the exclusion guarantees match `RwLock`.
/// Poisoning is ignored to match `parking_lot::RwLock`.
#[derive(Default)]
pub(super) struct ShardedRwLock<T>(ShardedLock<T>);

impl<T> ShardedRwLock<T> {
    pub(super) fn new(value: T) -> Self {
        Self(ShardedLock::new(value))
    }

    pub(super) fn read(&self) -> ShardedLockReadGuard<'_, T> {
        self.0.read().unwrap_or_else(PoisonError::into_inner)
    }

    pub(super) fn write(&self) -> ShardedLockWriteGuard<'_, T> {
        self.0.write().unwrap_or_else(PoisonError::into_inner)
    }
}
