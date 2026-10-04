// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crossbeam_epoch::{self as epoch, Atomic, Guard, Owned, Shared};
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use dashmap::mapref::entry::Entry;
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::node::Node;
use super::types::SharedNode;
use crate::protocols::LocalBlockHash;

const SMALL_CHILD_LIMIT: usize = 4;

type ShardedChildren = DashMap<LocalBlockHash, SharedNode, FxBuildHasher>;

#[derive(Clone)]
struct ChildEntry {
    hash: LocalBlockHash,
    node: SharedNode,
}

enum ChildrenState {
    Empty,
    Singleton(ChildEntry),
    Small(Box<[ChildEntry]>),
    Sharded(ShardedChildren),
}

impl ChildrenState {
    /// Entries of a compact snapshot. A sharded map is mutated in place and never
    /// retired, so it reports none.
    fn compact_entries(&self) -> &[ChildEntry] {
        match self {
            ChildrenState::Empty | ChildrenState::Sharded(_) => &[],
            ChildrenState::Singleton(entry) => std::slice::from_ref(entry),
            ChildrenState::Small(entries) => entries,
        }
    }

    fn into_child_arcs(self) -> Vec<SharedNode> {
        match self {
            ChildrenState::Empty => Vec::new(),
            ChildrenState::Singleton(entry) => vec![entry.node],
            ChildrenState::Small(entries) => entries
                .into_vec()
                .into_iter()
                .map(|entry| entry.node)
                .collect(),
            ChildrenState::Sharded(children) => {
                children.into_iter().map(|(_, child)| child).collect()
            }
        }
    }
}

thread_local! {
    /// Retired child snapshots and unlinked child `Arc`s whose epoch grace period ended
    /// during this thread's collections.
    ///
    /// Deferred destructors run on whichever thread collects, often a lookup, so they only
    /// bury their garbage here, if the thread drains ([`DrainScope`]). The thread frees it
    /// in small budgets after each lookup ([`LookupPin`]) or event, so no single call frees
    /// a whole detached subtree and no queue is shared on the hot path.
    static LOCAL_GRAVEYARD: LocalGraveyard = const {
        LocalGraveyard {
            graves: RefCell::new(Vec::new()),
            draining: Cell::new(false),
        }
    };
}

struct LocalGraveyard {
    graves: RefCell<Vec<Grave>>,
    /// Whether this thread frees its graveyard; see [`DrainScope`].
    draining: Cell<bool>,
}

impl Drop for LocalGraveyard {
    /// An exiting thread hands its garbage to the lanes, which free it iteratively and
    /// release the retired-snapshot counts it holds.
    fn drop(&mut self) {
        for grave in self.graves.get_mut().drain(..) {
            SHARED_GRAVEYARD.push(grave);
        }
    }
}

/// Garbage no draining thread holds: overflow past [`LOCAL_GRAVEYARD_CAP`], garbage
/// collected outside a [`DrainScope`], and garbage of exiting threads. Event lanes drain
/// it.
static SHARED_GRAVEYARD: SegQueue<Grave> = SegQueue::new();

const LOCAL_GRAVEYARD_CAP: usize = 4096;

/// Graveyard work a lookup does after unpinning: graves opened plus nodes freed.
const LOOKUP_GRAVEYARD_BUDGET: usize = 8;

fn bury(grave: Grave) {
    let mut grave = Some(grave);
    let _ = LOCAL_GRAVEYARD.try_with(|local| {
        if local.draining.get()
            && let Ok(mut graves) = local.graves.try_borrow_mut()
            && graves.len() < LOCAL_GRAVEYARD_CAP
        {
            graves.extend(grave.take());
        }
    });
    if let Some(grave) = grave {
        SHARED_GRAVEYARD.push(grave);
    }
}

fn unbury(include_shared: bool) -> Option<Grave> {
    LOCAL_GRAVEYARD
        .try_with(|local| {
            local
                .graves
                .try_borrow_mut()
                .ok()
                .and_then(|mut graves| graves.pop())
        })
        .ok()
        .flatten()
        .or_else(|| include_shared.then(|| SHARED_GRAVEYARD.pop()).flatten())
}

/// Marks this thread as one that frees its own graveyard while the scope lives: an event
/// lane for its whole run, or any thread for the length of a lookup. Garbage collected on
/// a thread outside every scope goes to the shared graveyard instead. Such a thread pins
/// the global collector for unrelated work, as rayon and moka do, and would never free it.
pub(super) struct DrainScope {
    was_draining: bool,
}

impl DrainScope {
    pub(super) fn enter() -> Self {
        let was_draining = LOCAL_GRAVEYARD
            .try_with(|local| local.draining.replace(true))
            .unwrap_or(false);
        Self { was_draining }
    }
}

impl Drop for DrainScope {
    fn drop(&mut self) {
        let _ = LOCAL_GRAVEYARD.try_with(|local| local.draining.set(self.was_draining));
    }
}

/// A lookup's epoch pin. Dropping it unpins, then pays down a little of this thread's
/// graveyard, so collections that lookups trigger are freed a bounded amount at a time.
pub(super) struct LookupPin {
    guard: Option<Guard>,
    _scope: DrainScope,
}

impl LookupPin {
    pub(super) fn new() -> Self {
        // Enter first, so a collection inside the pin buries its garbage here.
        let scope = DrainScope::enter();
        Self {
            guard: Some(epoch::pin()),
            _scope: scope,
        }
    }
}

impl std::ops::Deref for LookupPin {
    type Target = Guard;

    fn deref(&self) -> &Guard {
        self.guard.as_ref().expect("pinned until drop")
    }
}

impl Drop for LookupPin {
    fn drop(&mut self) {
        drop(self.guard.take());
        NodeChildren::drain_graveyard_with(LOOKUP_GRAVEYARD_BUDGET, false);
    }
}

enum Grave {
    /// A retired compact snapshot; its children still count it in
    /// `retired_snapshot_refs` until it is freed.
    Snapshot(Box<ChildrenState>),
    /// Children of a retired snapshot that a drain opened but ran out of budget before
    /// dropping; each still counts in `retired_snapshot_refs`.
    Retired(Vec<SharedNode>),
    Nodes(Vec<SharedNode>),
}

pub(super) enum ChildInsertResult {
    Existing(SharedNode),
    Inserted(SharedNode),
}

/// Child map published as immutable compact snapshots, promoted to a `DashMap` past
/// [`SMALL_CHILD_LIMIT`] children.
///
/// Snapshots are reclaimed through epochs: readers pin, load the current snapshot, and
/// read it in place; writers swap in a successor and retire the predecessor once every
/// pinned reader has moved on. Unlike a debt-based `ArcSwap`, a store does not walk
/// every thread's reader slots, so its cost does not grow with the thread count.
///
/// Reclamation invariant: every child `Arc` owned by a map is released through the
/// epoch while the map is reachable. A compact snapshot is retired whole, a sharded map
/// defers each `Arc` it unlinks in place, and a split moves the published state to the
/// suffix instead of dropping it. A child found in a map loaded under a guard therefore
/// stays allocated until that guard unpins, which is what lets [`Self::get_ref`] lend it
/// without touching its reference count. Other owners, such as writer lookups, only add
/// references. Dropping a `NodeChildren` frees its state inline; see `Drop`.
pub(super) struct NodeChildren {
    /// Never null. Unlinked snapshots are only retired through [`Self::retire`].
    state: Atomic<ChildrenState>,
}

impl fmt::Debug for NodeChildren {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeChildren")
            .field("len", &self.len())
            .finish()
    }
}

impl Drop for NodeChildren {
    fn drop(&mut self) {
        // SAFETY: `&mut self` rules out concurrent loads through this node, and a node is
        // dropped only after its last `Arc` is released. A reader borrows a child only
        // through a parent it holds for the borrow's lifetime (`get_ref` ties the borrow to
        // the parent), and every parent releases a child `Arc` through the epoch, so no
        // pinned reader can still reach this node or the snapshot it publishes, including
        // one `transfer_for_split` moved here from the node it split off.
        //
        // Freeing inline also keeps deferred destructors from pinning: a pin during
        // thread exit registers a fresh participant whose first pin collects, so pinning
        // here would recurse once per expired bag.
        unsafe {
            let state = self.state.load(Ordering::Relaxed, epoch::unprotected());
            drop(state.into_owned());
        }
    }
}

impl NodeChildren {
    pub(super) fn from_map(children: FxHashMap<LocalBlockHash, SharedNode>) -> Self {
        let state = if children.len() > SMALL_CHILD_LIMIT {
            let sharded = ShardedChildren::with_hasher(FxBuildHasher);
            for (key, child) in children {
                sharded.insert(key, child);
            }
            ChildrenState::Sharded(sharded)
        } else {
            let mut entries: Vec<_> = children
                .into_iter()
                .map(|(hash, node)| ChildEntry { hash, node })
                .collect();
            entries.sort_unstable_by_key(|entry| entry.hash);
            Self::compact_state(entries)
        };
        Self {
            state: Atomic::new(state),
        }
    }

    fn load<'g>(&self, guard: &'g Guard) -> &'g ChildrenState {
        let state = self.state.load(Ordering::Acquire, guard);
        // SAFETY: the pointer is never null, and a snapshot unlinked by a writer is
        // destroyed only after every guard pinned before the unlink is dropped.
        unsafe { state.deref() }
    }

    fn with_state<R>(&self, f: impl FnOnce(&ChildrenState) -> R) -> R {
        let guard = epoch::pin();
        f(self.load(&guard))
    }

    /// Publishes `next` unconditionally and retires the previous snapshot.
    fn replace(&self, next: ChildrenState, guard: &Guard) {
        let previous = self.state.swap(Owned::new(next), Ordering::AcqRel, guard);
        // SAFETY: the swap unlinked `previous`.
        unsafe { Self::retire(previous, guard) };
    }

    /// Destroys an unlinked snapshot once every thread pinned before the unlink has
    /// unpinned.
    ///
    /// Until it is freed its child `Arc`s are not live references. Each child counts them,
    /// so the stale-leaf sweep can still require an exact strong count: the count rises
    /// only after the snapshot is unlinked and falls only as the graveyard drops that `Arc`.
    ///
    /// # Safety
    ///
    /// `snapshot` must be unlinked and retired at most once.
    unsafe fn retire(snapshot: Shared<'_, ChildrenState>, guard: &Guard) {
        // SAFETY: the caller guarantees `snapshot` is unlinked but not yet destroyed.
        let entries = unsafe { snapshot.deref() }.compact_entries();
        for entry in entries {
            entry.node.note_retired_snapshot_ref();
        }
        // SAFETY: pinned readers keep the snapshot alive until they unpin, and no one can
        // reach it after that.
        unsafe {
            guard.defer_unchecked(move || {
                bury(Grave::Snapshot(snapshot.into_owned().into_box()));
            });
        }
    }

    /// Drops child `Arc`s unlinked in place from a sharded map once every thread pinned
    /// before the unlink has unpinned, so a reader that borrowed one through
    /// [`Self::get_ref`] keeps a live node.
    ///
    /// The deferred `Arc`s are not counted in `retired_snapshot_refs`: they belong to
    /// detached children, which the stale-leaf sweep skips because they are no longer in
    /// their parent's map, and a node is never linked again once unlinked.
    fn defer_drop(unlinked: Vec<SharedNode>, guard: &Guard) {
        if unlinked.is_empty() {
            return;
        }
        guard.defer(move || bury(Grave::Nodes(unlinked)));
    }

    /// Frees this thread's graveyard and the shared one, at most `budget` units of work
    /// (graves opened plus nodes freed). Event lanes call this. Returns whether both are
    /// empty.
    pub(super) fn drain_graveyard(budget: usize) -> bool {
        Self::drain_graveyard_with(budget, true)
    }

    /// Frees graveyard garbage iteratively, so deep detached subtrees cannot overflow the
    /// stack. Leftovers are buried again.
    fn drain_graveyard_with(budget: usize, include_shared: bool) -> bool {
        // Children of an opened snapshot. Each count is released only as its `Arc` is
        // dropped, so leftovers stay counted and the stale-leaf check stays exact.
        let mut retired: Vec<SharedNode> = Vec::new();
        let mut stack: Vec<SharedNode> = Vec::new();
        let mut work = 0;
        while work < budget {
            let node = if let Some(node) = retired.pop() {
                node.release_retired_snapshot_ref();
                node
            } else if let Some(node) = stack.pop() {
                node
            } else {
                let Some(grave) = unbury(include_shared) else {
                    return true;
                };
                work += 1;
                match grave {
                    // `retire` counted only compact entries; a sharded map is never retired.
                    Grave::Snapshot(snapshot) if snapshot.compact_entries().is_empty() => {
                        stack = snapshot.into_child_arcs();
                    }
                    Grave::Snapshot(snapshot) => retired = snapshot.into_child_arcs(),
                    Grave::Retired(nodes) => retired = nodes,
                    Grave::Nodes(nodes) => stack = nodes,
                }
                continue;
            };
            if let Some(node) = Arc::into_inner(node) {
                stack.extend(node.into_child_arcs());
                work += 1;
            }
        }
        if !retired.is_empty() {
            bury(Grave::Retired(retired));
        }
        if !stack.is_empty() {
            bury(Grave::Nodes(stack));
        }
        false
    }

    /// Takes every child `Arc` and frees the published state now rather than through the
    /// epoch.
    ///
    /// # Safety
    ///
    /// No thread may load this map concurrently or still hold a borrow from it.
    pub(super) unsafe fn take_exclusive(&self) -> Vec<SharedNode> {
        // SAFETY: the caller rules out every reader of the replaced state.
        let state = unsafe {
            self.state
                .swap(
                    Owned::new(ChildrenState::Empty),
                    Ordering::Relaxed,
                    epoch::unprotected(),
                )
                .into_owned()
        };
        state.into_box().into_child_arcs()
    }

    /// Takes every child `Arc` out of a map no one else can reach.
    pub(super) fn into_child_arcs(self) -> Vec<SharedNode> {
        // SAFETY: `self` is owned, so no thread can load this state.
        unsafe { self.take_exclusive() }
    }

    /// Publishes `next` only if `current` is still the published snapshot.
    fn compare_and_swap<'g>(
        &self,
        current: Shared<'g, ChildrenState>,
        next: ChildrenState,
        guard: &'g Guard,
    ) -> bool {
        match self.state.compare_exchange(
            current,
            Owned::new(next),
            Ordering::AcqRel,
            Ordering::Acquire,
            guard,
        ) {
            Ok(_) => {
                // SAFETY: the swap unlinked `current`.
                unsafe { Self::retire(current, guard) };
                true
            }
            Err(_) => false,
        }
    }

    /// Hands this thread's retired snapshots and deferred child drops to the collector and
    /// collects expired ones. Writers call this when idle so retired child `Arc`s do not
    /// linger in a quiet thread's local bag.
    pub(super) fn flush_retired() {
        epoch::pin().flush();
    }

    pub(super) fn is_empty(&self) -> bool {
        self.with_state(|state| match state {
            ChildrenState::Empty => true,
            ChildrenState::Singleton(_) => false,
            ChildrenState::Small(children) => children.is_empty(),
            ChildrenState::Sharded(children) => children.is_empty(),
        })
    }

    fn len(&self) -> usize {
        self.with_state(|state| match state {
            ChildrenState::Empty => 0,
            ChildrenState::Singleton(_) => 1,
            ChildrenState::Small(children) => children.len(),
            ChildrenState::Sharded(children) => children.len(),
        })
    }

    pub(super) fn get(&self, key: &LocalBlockHash) -> Option<SharedNode> {
        self.with_state(|state| match state {
            ChildrenState::Empty => None,
            ChildrenState::Singleton(entry) => (entry.hash == *key).then(|| entry.node.clone()),
            ChildrenState::Small(children) => children
                .binary_search_by_key(key, |entry| entry.hash)
                .ok()
                .map(|index| children[index].node.clone()),
            ChildrenState::Sharded(children) => {
                children.get(key).map(|entry| entry.value().clone())
            }
        })
    }

    /// Borrows the child for `key` while both this map and `guard` live, without cloning
    /// its `Arc`. Sound by the reclamation invariant on [`NodeChildren`]: the map releases
    /// its `Arc` only through the epoch, and the borrow cannot outlive the map's owner,
    /// whose drop frees inline. `guard` must be a real pin, never `epoch::unprotected()`.
    pub(super) fn get_ref<'g>(
        &'g self,
        key: &LocalBlockHash,
        guard: &'g Guard,
    ) -> Option<&'g Node> {
        let child = match self.load(guard) {
            ChildrenState::Empty => return None,
            ChildrenState::Singleton(entry) => (entry.hash == *key).then_some(&entry.node)?,
            ChildrenState::Small(children) => {
                let index = children
                    .binary_search_by_key(key, |entry| entry.hash)
                    .ok()?;
                &children[index].node
            }
            ChildrenState::Sharded(children) => {
                // Only the node's address leaves the shard lock: the `Arc` slot itself
                // can move when the shard rehashes.
                let child = children.get(key)?;
                // SAFETY: the map owns this `Arc` while the shard lock is held and
                // releases it only through the epoch (`defer_drop`, or `retire` of the
                // whole map), which waits for `guard`.
                return Some(unsafe { &*Arc::as_ptr(child.value()) });
            }
        };
        // SAFETY: `child` lives in a compact snapshot loaded under `guard`, which is
        // destroyed only after `guard` unpins; the node outlives that `Arc`.
        Some(unsafe { &*Arc::as_ptr(child) })
    }

    pub(super) fn values_snapshot(&self) -> Vec<SharedNode> {
        self.with_state(|state| match state {
            ChildrenState::Empty => Vec::new(),
            ChildrenState::Singleton(entry) => vec![entry.node.clone()],
            ChildrenState::Small(children) => {
                children.iter().map(|entry| entry.node.clone()).collect()
            }
            ChildrenState::Sharded(children) => {
                children.iter().map(|entry| entry.value().clone()).collect()
            }
        })
    }

    pub(super) fn entries_snapshot(&self) -> Vec<(LocalBlockHash, SharedNode)> {
        self.with_state(|state| match state {
            ChildrenState::Empty => Vec::new(),
            ChildrenState::Singleton(entry) => vec![(entry.hash, entry.node.clone())],
            ChildrenState::Small(children) => children
                .iter()
                .map(|entry| (entry.hash, entry.node.clone()))
                .collect(),
            ChildrenState::Sharded(children) => children
                .iter()
                .map(|entry| (*entry.key(), entry.value().clone()))
                .collect(),
        })
    }

    pub(super) fn extend_values(&self, queue: &mut VecDeque<SharedNode>) {
        self.with_state(|state| match state {
            ChildrenState::Empty => {}
            ChildrenState::Singleton(entry) => queue.push_back(entry.node.clone()),
            ChildrenState::Small(children) => {
                queue.extend(children.iter().map(|entry| entry.node.clone()));
            }
            ChildrenState::Sharded(children) => {
                queue.extend(children.iter().map(|entry| entry.value().clone()));
            }
        })
    }

    pub(super) fn insert(&self, key: LocalBlockHash, child: SharedNode) {
        let guard = epoch::pin();
        let current = self.load(&guard);
        if let ChildrenState::Sharded(children) = current {
            if let Some(previous) = children.insert(key, child) {
                Self::defer_drop(vec![previous], &guard);
            }
            return;
        }

        let mut entries = Self::clone_compact_entries(current, 1);
        match entries.binary_search_by_key(&key, |entry| entry.hash) {
            Ok(index) => entries[index].node = child,
            Err(index) => entries.insert(
                index,
                ChildEntry {
                    hash: key,
                    node: child,
                },
            ),
        }
        self.replace(Self::state_from_entries(entries), &guard);
    }

    pub(super) fn insert_if_absent(
        &self,
        key: LocalBlockHash,
        child: SharedNode,
    ) -> ChildInsertResult {
        let guard = epoch::pin();
        loop {
            let current = self.state.load(Ordering::Acquire, &guard);
            // SAFETY: see `load`.
            let current_ref = unsafe { current.deref() };
            let insert_index = match current_ref {
                ChildrenState::Empty => 0,
                ChildrenState::Singleton(entry) if entry.hash == key => {
                    return ChildInsertResult::Existing(entry.node.clone());
                }
                ChildrenState::Singleton(entry) => usize::from(entry.hash < key),
                ChildrenState::Small(entries) => {
                    match entries.binary_search_by_key(&key, |entry| entry.hash) {
                        Ok(index) => {
                            return ChildInsertResult::Existing(entries[index].node.clone());
                        }
                        Err(index) => index,
                    }
                }
                ChildrenState::Sharded(children) => {
                    return match children.entry(key) {
                        Entry::Occupied(entry) => ChildInsertResult::Existing(entry.get().clone()),
                        Entry::Vacant(entry) => {
                            entry.insert(child.clone());
                            ChildInsertResult::Inserted(child)
                        }
                    };
                }
            };

            let mut entries = Self::clone_compact_entries(current_ref, 1);
            entries.insert(
                insert_index,
                ChildEntry {
                    hash: key,
                    node: child.clone(),
                },
            );
            if self.compare_and_swap(current, Self::state_from_entries(entries), &guard) {
                return ChildInsertResult::Inserted(child);
            }
        }
    }

    pub(super) fn remove(&self, key: &LocalBlockHash) -> bool {
        let guard = epoch::pin();
        let current = self.load(&guard);
        let remove_index = match current {
            ChildrenState::Empty => return false,
            ChildrenState::Singleton(entry) => {
                if entry.hash != *key {
                    return false;
                }
                0
            }
            ChildrenState::Small(entries) => {
                let Ok(index) = entries.binary_search_by_key(key, |entry| entry.hash) else {
                    return false;
                };
                index
            }
            ChildrenState::Sharded(children) => {
                let Some((_, removed)) = children.remove(key) else {
                    return false;
                };
                Self::defer_drop(vec![removed], &guard);
                return true;
            }
        };

        let mut entries = Self::clone_compact_entries(current, 0);
        entries.remove(remove_index);
        self.replace(Self::compact_state(entries), &guard);
        true
    }

    pub(super) fn clear(&self) -> bool {
        let guard = epoch::pin();
        match self.load(&guard) {
            ChildrenState::Empty => false,
            ChildrenState::Sharded(children) => {
                let mut removed = Vec::with_capacity(children.len());
                children.retain(|_, child| {
                    removed.push(child.clone());
                    false
                });
                if removed.is_empty() {
                    return false;
                }
                Self::defer_drop(removed, &guard);
                true
            }
            ChildrenState::Singleton(_) | ChildrenState::Small(_) => {
                self.replace(ChildrenState::Empty, &guard);
                true
            }
        }
    }

    /// Transfers the current state while the owning node's exclusive shape gate is held.
    /// The suffix keeps its representation; the prefix restarts with compact children.
    pub(super) fn transfer_for_split(&self) -> Self {
        let guard = epoch::pin();
        let current = self
            .state
            .swap(Owned::new(ChildrenState::Empty), Ordering::AcqRel, &guard);
        // The snapshot moves to the suffix unchanged. Readers that loaded it from this
        // node keep reading a live allocation; it is retired when the suffix replaces it,
        // and the suffix itself is dropped only after this node's compact map, which the
        // caller publishes it into, retires its reference (see `Drop`).
        Self {
            state: Atomic::from(current),
        }
    }

    /// Copies compact entries with room for `spare` more, so inserting a new key neither
    /// grows the copy nor shrinks it again when it becomes a boxed slice.
    fn clone_compact_entries(state: &ChildrenState, spare: usize) -> Vec<ChildEntry> {
        let current: &[ChildEntry] = match state {
            ChildrenState::Empty => &[],
            ChildrenState::Singleton(entry) => std::slice::from_ref(entry),
            ChildrenState::Small(children) => children,
            ChildrenState::Sharded(_) => unreachable!("sharded children are mutated in place"),
        };
        let mut entries = Vec::with_capacity(current.len() + spare);
        entries.extend_from_slice(current);
        entries
    }

    fn state_from_entries(entries: Vec<ChildEntry>) -> ChildrenState {
        if entries.len() <= SMALL_CHILD_LIMIT {
            return Self::compact_state(entries);
        }

        let sharded = ShardedChildren::with_hasher(FxBuildHasher);
        for entry in entries {
            sharded.insert(entry.hash, entry.node);
        }
        ChildrenState::Sharded(sharded)
    }

    fn compact_state(mut entries: Vec<ChildEntry>) -> ChildrenState {
        debug_assert!(entries.windows(2).all(|pair| pair[0].hash < pair[1].hash));
        debug_assert!(entries.len() <= SMALL_CHILD_LIMIT);
        match entries.len() {
            0 => ChildrenState::Empty,
            1 => ChildrenState::Singleton(entries.remove(0)),
            _ => ChildrenState::Small(entries.into_boxed_slice()),
        }
    }

    #[cfg(test)]
    fn kind(&self) -> ChildrenKind {
        self.with_state(|state| match state {
            ChildrenState::Empty => ChildrenKind::Empty,
            ChildrenState::Singleton(_) => ChildrenKind::Singleton,
            ChildrenState::Small(_) => ChildrenKind::Small,
            ChildrenState::Sharded(_) => ChildrenKind::Sharded,
        })
    }
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
enum ChildrenKind {
    Empty,
    Singleton,
    Small,
    Sharded,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::{Barrier, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use super::*;

    fn child() -> SharedNode {
        Arc::new(Node::new())
    }

    #[test]
    fn compact_removals_shrink_to_singleton_then_empty() {
        let children = NodeChildren::from_map(FxHashMap::default());
        let nodes: Vec<_> = (0..SMALL_CHILD_LIMIT).map(|_| child()).collect();

        for (key, node) in nodes.iter().enumerate() {
            let result = children.insert_if_absent(LocalBlockHash(key as u64), node.clone());
            assert!(matches!(result, ChildInsertResult::Inserted(_)));
        }
        assert_eq!(children.kind(), ChildrenKind::Small);

        for key in (1..SMALL_CHILD_LIMIT).rev() {
            assert!(children.remove(&LocalBlockHash(key as u64)));
            assert!(children.get(&LocalBlockHash(key as u64)).is_none());
        }
        assert_eq!(children.kind(), ChildrenKind::Singleton);
        assert!(Arc::ptr_eq(
            &children.get(&LocalBlockHash(0)).unwrap(),
            &nodes[0]
        ));

        assert!(children.remove(&LocalBlockHash(0)));
        assert!(!children.remove(&LocalBlockHash(0)));
        assert_eq!(children.kind(), ChildrenKind::Empty);
        assert!(children.is_empty());
    }

    #[test]
    fn fifth_distinct_child_promotes_and_ordinary_mutations_do_not_demote() {
        let children = NodeChildren::from_map(FxHashMap::default());

        for key in 0..=SMALL_CHILD_LIMIT {
            let result = children.insert_if_absent(LocalBlockHash(key as u64), child());
            assert!(matches!(result, ChildInsertResult::Inserted(_)));
        }
        assert_eq!(children.kind(), ChildrenKind::Sharded);
        assert_eq!(children.len(), SMALL_CHILD_LIMIT + 1);

        for key in 0..=SMALL_CHILD_LIMIT {
            assert!(children.remove(&LocalBlockHash(key as u64)));
        }
        assert!(children.is_empty());
        assert_eq!(children.kind(), ChildrenKind::Sharded);

        children.insert(LocalBlockHash(99), child());
        children.clear();
        assert!(children.is_empty());
        assert_eq!(children.kind(), ChildrenKind::Sharded);
    }

    #[test]
    fn split_transfer_compacts_prefix_and_preserves_suffix_children() {
        let compact = NodeChildren::from_map(FxHashMap::default());
        compact.insert(LocalBlockHash(1), child());
        compact.insert(LocalBlockHash(2), child());
        let compact_suffix = compact.transfer_for_split();
        assert_eq!(compact.kind(), ChildrenKind::Empty);
        assert_eq!(compact_suffix.kind(), ChildrenKind::Small);
        assert_eq!(compact_suffix.len(), 2);
        compact.insert(LocalBlockHash(3), child());
        assert_eq!(compact.kind(), ChildrenKind::Singleton);

        let sharded = NodeChildren::from_map(FxHashMap::default());
        for key in 0..=SMALL_CHILD_LIMIT {
            sharded.insert(LocalBlockHash(key as u64), child());
        }
        assert_eq!(sharded.kind(), ChildrenKind::Sharded);
        let sharded_suffix = sharded.transfer_for_split();
        assert_eq!(sharded.kind(), ChildrenKind::Empty);
        assert_eq!(sharded_suffix.kind(), ChildrenKind::Sharded);
        assert_eq!(sharded_suffix.len(), SMALL_CHILD_LIMIT + 1);
        sharded.insert(LocalBlockHash(99), child());
        assert_eq!(sharded.kind(), ChildrenKind::Singleton);
    }

    #[test]
    fn stale_compact_remove_replace_cannot_overwrite_promotion() {
        let children = Arc::new(NodeChildren::from_map(FxHashMap::default()));
        let originals: Vec<_> = (0..SMALL_CHILD_LIMIT).map(|_| child()).collect();
        for (key, node) in originals.iter().enumerate() {
            children.insert(LocalBlockHash(key as u64), node.clone());
        }

        let guard = epoch::pin();
        let stale = children.state.load(Ordering::Acquire, &guard);
        // SAFETY: `guard` keeps the snapshot alive for the rest of the test.
        let mut stale_entries = NodeChildren::clone_compact_entries(unsafe { stale.deref() }, 0);
        stale_entries.remove(0);
        let replacement = child();
        let replacement_index = stale_entries
            .binary_search_by_key(&LocalBlockHash(1), |entry| entry.hash)
            .unwrap();
        stale_entries[replacement_index].node = replacement.clone();

        let start = Arc::new(Barrier::new(2));
        let promoted = Arc::new(Barrier::new(2));
        let promote_children = children.clone();
        let promote_start = start.clone();
        let promote_done = promoted.clone();
        let fifth = child();
        let fifth_for_thread = fifth.clone();
        let handle = thread::spawn(move || {
            promote_start.wait();
            let result = promote_children
                .insert_if_absent(LocalBlockHash(SMALL_CHILD_LIMIT as u64), fifth_for_thread);
            promote_done.wait();
            result
        });

        start.wait();
        promoted.wait();
        assert_eq!(children.kind(), ChildrenKind::Sharded);
        assert!(!children.compare_and_swap(
            stale,
            NodeChildren::compact_state(stale_entries),
            &guard
        ));
        assert!(matches!(
            handle.join().unwrap(),
            ChildInsertResult::Inserted(_)
        ));

        assert!(Arc::ptr_eq(
            &children.get(&LocalBlockHash(0)).unwrap(),
            &originals[0]
        ));
        assert!(Arc::ptr_eq(
            &children.get(&LocalBlockHash(1)).unwrap(),
            &originals[1]
        ));
        assert!(!Arc::ptr_eq(
            &children.get(&LocalBlockHash(1)).unwrap(),
            &replacement
        ));
        assert!(Arc::ptr_eq(
            &children
                .get(&LocalBlockHash(SMALL_CHILD_LIMIT as u64))
                .unwrap(),
            &fifth
        ));
    }

    #[test]
    fn concurrent_duplicate_insert_has_one_winner() {
        const THREADS: usize = 16;
        let children = Arc::new(NodeChildren::from_map(FxHashMap::default()));
        let barrier = Arc::new(Barrier::new(THREADS + 1));
        let inserted = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let children = children.clone();
            let barrier = barrier.clone();
            let inserted = inserted.clone();
            handles.push(thread::spawn(move || {
                let candidate = child();
                barrier.wait();
                let result = children.insert_if_absent(LocalBlockHash(7), candidate);
                match result {
                    ChildInsertResult::Inserted(node) => {
                        inserted.fetch_add(1, Ordering::Relaxed);
                        node
                    }
                    ChildInsertResult::Existing(node) => node,
                }
            }));
        }

        barrier.wait();
        let returned: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        let stored = children.get(&LocalBlockHash(7)).unwrap();
        assert_eq!(inserted.load(Ordering::Relaxed), 1);
        assert_eq!(children.len(), 1);
        assert!(returned.iter().all(|node| Arc::ptr_eq(node, &stored)));
    }

    #[test]
    fn concurrent_distinct_inserts_survive_promotion_races() {
        const THREADS: usize = 32;
        let children = Arc::new(NodeChildren::from_map(FxHashMap::default()));
        let barrier = Arc::new(Barrier::new(THREADS + 1));
        let mut handles = Vec::with_capacity(THREADS);

        for key in 0..THREADS {
            let children = children.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                children.insert_if_absent(LocalBlockHash(key as u64), child())
            }));
        }

        barrier.wait();
        for handle in handles {
            assert!(matches!(
                handle.join().unwrap(),
                ChildInsertResult::Inserted(_)
            ));
        }
        assert_eq!(children.kind(), ChildrenKind::Sharded);
        assert_eq!(children.len(), THREADS);
        for key in 0..THREADS {
            assert!(children.get(&LocalBlockHash(key as u64)).is_some());
        }
    }
    /// Flushes and collects until `node` has `expected` strong references. Other tests'
    /// pins on the global collector can delay reclamation, so this polls.
    fn wait_for_strong_count(node: &SharedNode, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while Arc::strong_count(node) != expected {
            assert!(Instant::now() < deadline, "deferred child drops never ran");
            NodeChildren::flush_retired();
            NodeChildren::drain_graveyard(usize::MAX);
            thread::yield_now();
        }
    }

    #[test]
    fn sharded_child_unlink_defers_arc_drop() {
        type Unlink = fn(&NodeChildren, LocalBlockHash);
        let unlinks: [(&str, Unlink); 3] = [
            ("insert_replace", |children, key| {
                children.insert(key, child())
            }),
            ("remove", |children, key| assert!(children.remove(&key))),
            ("clear", |children, _| assert!(children.clear())),
        ];

        for (name, unlink) in unlinks {
            let children = NodeChildren::from_map(FxHashMap::default());
            let nodes: Vec<_> = (0..=SMALL_CHILD_LIMIT).map(|_| child()).collect();
            for (key, node) in nodes.iter().enumerate() {
                children.insert(LocalBlockHash(key as u64), node.clone());
            }
            assert_eq!(children.kind(), ChildrenKind::Sharded, "{name}");
            let key = LocalBlockHash(0);
            let target = &nodes[0];
            // Drain the compact snapshots retired while filling the map.
            wait_for_strong_count(target, 2);

            let guard = epoch::pin();
            let borrowed = children.get_ref(&key, &guard).unwrap();
            assert!(std::ptr::eq(borrowed, Arc::as_ptr(target)), "{name}");

            unlink(&children, key);
            for _ in 0..8 {
                NodeChildren::flush_retired();
            }
            // The map's reference waits for `guard`, so the borrow stays valid.
            assert_eq!(Arc::strong_count(target), 2, "{name}");
            assert_eq!(borrowed.edge_len_for_test(), 0, "{name}");
            assert!(
                children
                    .get(&key)
                    .is_none_or(|current| !Arc::ptr_eq(&current, target)),
                "{name}"
            );

            drop(guard);
            wait_for_strong_count(target, 1);
        }
    }

    #[test]
    fn budgeted_drain_keeps_undropped_snapshot_children_counted() {
        let _draining = DrainScope::enter();
        let leaf = child();
        // What `retire` leaves for a retired snapshot that holds `leaf`.
        leaf.note_retired_snapshot_ref();
        bury(Grave::Snapshot(Box::new(ChildrenState::Singleton(
            ChildEntry {
                hash: LocalBlockHash(1),
                node: leaf.clone(),
            },
        ))));

        // Opening the grave spends the whole budget before the entry is dropped.
        assert!(!NodeChildren::drain_graveyard_with(1, false));
        assert_eq!(Arc::strong_count(&leaf), 2);
        assert_eq!(leaf.retired_snapshot_refs_for_test(), 1);

        assert!(NodeChildren::drain_graveyard_with(usize::MAX, false));
        assert_eq!(Arc::strong_count(&leaf), 1);
        assert_eq!(leaf.retired_snapshot_refs_for_test(), 0);
    }

    #[test]
    fn garbage_collected_outside_a_drain_scope_reaches_the_lanes() {
        let node = child();
        let (buried_tx, buried_rx) = mpsc::channel();
        let (exit_tx, exit_rx) = mpsc::channel::<()>();
        let foreign = {
            let node = node.clone();
            thread::spawn(move || {
                // A deferred destructor run by a collection on a thread that pins the
                // global collector for unrelated work, such as a rayon steal.
                bury(Grave::Nodes(vec![node]));
                buried_tx.send(()).unwrap();
                exit_rx.recv().unwrap();
            })
        };
        buried_rx.recv().unwrap();

        // The foreign thread never drains, so a lane must free the node while it lives.
        wait_for_strong_count(&node, 1);
        exit_tx.send(()).unwrap();
        foreign.join().unwrap();
    }
}
