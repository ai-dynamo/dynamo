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

/// Point existing `Arc` lookup entries for `keys` that name `from` at `to`.
///
/// Unlike [`update_arc_lookup_for_keys`], this never inserts and leaves entries naming
/// any other node alone. A missing entry is meaningful: a remove can scrub it before
/// another worker on the same event thread repairs a stale lookup. An entry elsewhere
/// is left to lazy repair, so stale coverage on `to` cannot pull it over. Returns the
/// number of entries that changed.
pub(crate) fn redirect_arc_lookup_for_keys<K, T>(
    lookup: &mut FxHashMap<K, Arc<T>>,
    keys: impl IntoIterator<Item = K>,
    from: &Arc<T>,
    to: &Arc<T>,
) -> usize
where
    K: Eq + Hash,
{
    let mut changed = 0;

    for key in keys {
        if let Some(entry) = lookup.get_mut(&key)
            && Arc::ptr_eq(entry, from)
            && !Arc::ptr_eq(entry, to)
        {
            *entry = Arc::clone(to);
            changed += 1;
        }
    }

    changed
}
