// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::sync::Arc;

use rustc_hash::FxHashMap;

/// Update an `Arc` lookup map for keys that should point at `node`.
///
/// Duplicate store paths often revisit keys that already resolve to the same
/// node. Skipping those replacements avoids unnecessary hash table writes and
/// `Arc` refcount churn. Returns the number of entries inserted or changed.
pub(crate) fn update_arc_lookup_for_keys<K, T>(
    lookup: &mut FxHashMap<K, Arc<T>>,
    keys: impl IntoIterator<Item = K>,
    node: &Arc<T>,
) -> usize
where
    K: Eq + Hash,
{
    let mut changed = 0;

    for key in keys {
        match lookup.entry(key) {
            Entry::Occupied(mut entry) if !Arc::ptr_eq(entry.get(), node) => {
                entry.insert(Arc::clone(node));
                changed += 1;
            }
            Entry::Occupied(_) => {}
            Entry::Vacant(entry) => {
                entry.insert(Arc::clone(node));
                changed += 1;
            }
        }
    }

    changed
}

/// Update existing `Arc` lookup entries for keys that should point at `node`.
///
/// Unlike [`update_arc_lookup_for_keys`], this does not insert missing keys.
/// Lookup repair uses absence as meaningful state: a remove can scrub an entry
/// before another worker on the same event thread repairs a stale lookup.
/// Returns the number of existing entries that changed.
pub(crate) fn update_existing_arc_lookup_for_keys<K, T>(
    lookup: &mut FxHashMap<K, Arc<T>>,
    keys: impl IntoIterator<Item = K>,
    node: &Arc<T>,
) -> usize
where
    K: Eq + Hash,
{
    let mut changed = 0;

    for key in keys {
        if let Some(entry) = lookup.get_mut(&key)
            && !Arc::ptr_eq(entry, node)
        {
            *entry = Arc::clone(node);
            changed += 1;
        }
    }

    changed
}
