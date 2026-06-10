// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::sync::Arc;

use rustc_hash::FxHashMap;

/// Update an `Arc` lookup map for keys that should point at `node`.
///
/// Duplicate store/repair paths often revisit keys that already resolve to the
/// same node. Skipping those replacements avoids unnecessary hash table writes
/// and `Arc` refcount churn while still repairing stale entries.
pub(crate) fn update_arc_lookup_for_keys<K, T>(
    lookup: &mut FxHashMap<K, Arc<T>>,
    keys: impl IntoIterator<Item = K>,
    node: &Arc<T>,
) -> bool
where
    K: Eq + Hash,
{
    let mut changed = false;

    for key in keys {
        match lookup.entry(key) {
            Entry::Occupied(mut entry) if !Arc::ptr_eq(entry.get(), node) => {
                entry.insert(Arc::clone(node));
                changed = true;
            }
            Entry::Occupied(_) => {}
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(node));
                changed = true;
            }
        }
    }

    changed
}
