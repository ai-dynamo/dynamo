// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loom models of arena-c's lock-free and lock-ordering protocols, over the backend's own
//! child-table code. The gates are loom `RwLock`s standing in for the runs' `parking_lot`
//! gates; everything the backend does to a child table goes through `table.rs`.
//!
//! 1. A claim under the shared gate versus a table rebuild under the exclusive gate: no
//!    claim is lost, and a lock-free reader always finds an existing child.
//! 2. An append's end-child re-probe versus a racing claim: never both an appended block
//!    and an end child with the same head (one path stored twice, S2).
//! 3. The eager unlink (parent shared gate, child `try_write`) versus a store descending
//!    into the child: the store never lands in a dead child.
//!
//! Each has a negative control that removes the rule and must fail.

use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use loom::sync::{Arc, RwLock};
use loom::thread;
use loom_arena_c::{
    Claim, Locked, child_key, claim, entries, find, insert_locked, new_table, pack, unlink,
};

fn model(f: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(f);
}

// ----------------------------------------------------------------------------
// 1. Claim versus rebuild
// ----------------------------------------------------------------------------

struct Growable {
    gate: RwLock<()>,
    tables: [Vec<loom_arena_c::AtomicU64>; 2],
    current: AtomicUsize,
}

fn claim_versus_rebuild(gated_rebuild: bool) {
    model(move || {
        let a = child_key(0, 1);
        let b = child_key(0, 2);
        let run = Arc::new(Growable {
            gate: RwLock::new(()),
            tables: [new_table(4), new_table(8)],
            current: AtomicUsize::new(0),
        });
        assert_eq!(
            insert_locked(&run.tables[0], a, pack(2, 1)),
            Locked::Inserted
        );

        let claimer = {
            let run = run.clone();
            thread::spawn(move || {
                let _gate = run.gate.read().unwrap();
                let current = run.current.load(Ordering::Acquire);
                assert_eq!(claim(&run.tables[current], b, pack(3, 1)), Claim::Claimed);
            })
        };
        let rebuilder = {
            let run = run.clone();
            thread::spawn(move || {
                let _gate = gated_rebuild.then(|| run.gate.write().unwrap());
                for (key, value) in entries(&run.tables[0]) {
                    insert_locked(&run.tables[1], key, value);
                }
                run.current.store(1, Ordering::Release);
            })
        };
        // A lock-free reader (ROOT's hop) finds the existing child in whichever table.
        let current = run.current.load(Ordering::Acquire);
        assert_eq!(find(&run.tables[current], a), Some(pack(2, 1)));

        claimer.join().unwrap();
        rebuilder.join().unwrap();
        let current = &run.tables[run.current.load(Ordering::Acquire)];
        assert_eq!(find(current, a), Some(pack(2, 1)));
        assert_eq!(find(current, b), Some(pack(3, 1)), "a claim was lost");
    });
}

#[test]
fn claim_under_shared_gate_versus_rebuild() {
    claim_versus_rebuild(true);
}

#[test]
#[should_panic(expected = "a claim was lost")]
fn negative_control_rebuild_without_exclusive_gate_loses_claims() {
    claim_versus_rebuild(false);
}

// ----------------------------------------------------------------------------
// 2. Append re-probe versus a racing claim
// ----------------------------------------------------------------------------

struct Appendable {
    gate: RwLock<()>,
    version: AtomicU64,
    table: Vec<loom_arena_c::AtomicU64>,
    appended: AtomicBool,
    claimed: AtomicBool,
}

fn append_versus_claim(reprobe: bool) {
    model(move || {
        // The run has three blocks and a child table; the end child would be (3, 9).
        let end = child_key(3, 9);
        let run = Arc::new(Appendable {
            gate: RwLock::new(()),
            version: AtomicU64::new(0),
            table: new_table(4),
            appended: AtomicBool::new(false),
            claimed: AtomicBool::new(false),
        });
        assert_eq!(
            insert_locked(&run.table, child_key(1, 5), pack(4, 1)),
            Locked::Inserted
        );

        let appender = {
            let run = run.clone();
            thread::spawn(move || {
                let (version, planned) = {
                    let _gate = run.gate.read().unwrap();
                    (run.version.load(Ordering::Acquire), find(&run.table, end))
                };
                if planned.is_some() {
                    return;
                }
                let _gate = run.gate.write().unwrap();
                if run.version.load(Ordering::Relaxed) != version {
                    return;
                }
                // Claims do not bump the version: only this re-probe sees them.
                if reprobe && find(&run.table, end).is_some() {
                    return;
                }
                run.appended.store(true, Ordering::Relaxed);
                run.version.fetch_add(1, Ordering::Release);
            })
        };
        let claimer = {
            let run = run.clone();
            thread::spawn(move || {
                // A store plans and claims under one hold of the shared gate. If the
                // append went first, planning matches the run's own new position 3
                // instead of looking for a child there.
                let _gate = run.gate.read().unwrap();
                if run.appended.load(Ordering::Relaxed) {
                    return;
                }
                if claim(&run.table, end, pack(6, 1)) == Claim::Claimed {
                    run.claimed.store(true, Ordering::Relaxed);
                }
            })
        };
        appender.join().unwrap();
        claimer.join().unwrap();
        assert!(
            !(run.appended.load(Ordering::Relaxed) && run.claimed.load(Ordering::Relaxed)),
            "one path stored twice"
        );
    });
}

#[test]
fn append_reprobes_the_end_child_against_claims() {
    append_versus_claim(true);
}

#[test]
#[should_panic(expected = "one path stored twice")]
fn negative_control_append_without_reprobe_duplicates_a_path() {
    append_versus_claim(false);
}

// ----------------------------------------------------------------------------
// 3. Eager unlink versus a store descending into the child
// ----------------------------------------------------------------------------

struct Child {
    gate: RwLock<()>,
    dead: AtomicBool,
    holders: AtomicU64,
}

struct Parent {
    gate: RwLock<()>,
    table: Vec<loom_arena_c::AtomicU64>,
}

const OLD: u32 = 2;
const NEW: u32 = 3;

fn unlink_versus_store(check_dead: bool) {
    model(move || {
        let key = child_key(4, 7);
        let old = pack(OLD, 1);
        let parent = Arc::new(Parent {
            gate: RwLock::new(()),
            table: new_table(4),
        });
        assert_eq!(insert_locked(&parent.table, key, old), Locked::Inserted);
        // The child is holder-less, so the unlinker may take it.
        let child = Arc::new(Child {
            gate: RwLock::new(()),
            dead: AtomicBool::new(false),
            holders: AtomicU64::new(0),
        });

        let unlinker = {
            let parent = parent.clone();
            let child = child.clone();
            thread::spawn(move || {
                let _parent_gate = parent.gate.read().unwrap();
                if find(&parent.table, key) != Some(old) {
                    return;
                }
                let Ok(_child_gate) = child.gate.try_write() else {
                    return; // pending_unlinks
                };
                if child.holders.load(Ordering::Relaxed) == 0 && !child.dead.load(Ordering::Relaxed)
                {
                    child.dead.store(true, Ordering::Release);
                    assert!(unlink(&parent.table, key, old));
                }
            })
        };
        let store = {
            let parent = parent.clone();
            let child = child.clone();
            thread::spawn(move || -> u32 {
                for _ in 0..4 {
                    let found = {
                        let _gate = parent.gate.read().unwrap();
                        find(&parent.table, key)
                    };
                    match found {
                        Some(value) if value == old => {
                            let _gate = child.gate.read().unwrap();
                            if check_dead && child.dead.load(Ordering::Acquire) {
                                continue;
                            }
                            child.holders.fetch_add(1, Ordering::Relaxed);
                            return OLD;
                        }
                        Some(_) => return NEW,
                        None => {
                            let _gate = parent.gate.read().unwrap();
                            match claim(&parent.table, key, pack(NEW, 1)) {
                                Claim::Claimed => return NEW,
                                Claim::Exists(_) | Claim::Busy => continue,
                                Claim::Full => panic!("the table has room"),
                            }
                        }
                    }
                }
                panic!("the store kept re-planning");
            })
        };
        unlinker.join().unwrap();
        let placed = store.join().unwrap();
        match placed {
            OLD => {
                assert!(
                    !child.dead.load(Ordering::Acquire),
                    "the store landed in a dead child"
                );
                assert_eq!(find(&parent.table, key), Some(old));
            }
            _ => assert_eq!(find(&parent.table, key), Some(pack(NEW, 1))),
        }
    });
}

#[test]
fn eager_unlink_versus_a_descending_store() {
    unlink_versus_store(true);
}

#[test]
#[should_panic(expected = "the store landed in a dead child")]
fn negative_control_store_ignoring_dead_lands_in_a_dead_child() {
    unlink_versus_store(false);
}

// ----------------------------------------------------------------------------
// 4. Claims of one key racing an unlink of that key
// ----------------------------------------------------------------------------

/// Two stores claim the same key while an unlink tombstones the old child with that key.
/// Claims reuse a same-key tombstone by compare-and-swap, so at most one claim wins and
/// the table never holds two live children for one key.
#[test]
fn same_key_claims_and_tombstone_reuse_publish_one_child() {
    model(|| {
        let key = child_key(2, 11);
        let old = pack(2, 1);
        let table = Arc::new(new_table(4));
        assert_eq!(insert_locked(&table, key, old), Locked::Inserted);
        let claimers: Vec<_> = [pack(3, 1), pack(4, 1)]
            .into_iter()
            .map(|value| {
                let table = table.clone();
                thread::spawn(move || (value, claim(&table, key, value)))
            })
            .collect();
        let unlinker = {
            let table = table.clone();
            thread::spawn(move || unlink(&table, key, old))
        };
        let outcomes: Vec<(u64, Claim)> = claimers
            .into_iter()
            .map(|claimer| claimer.join().unwrap())
            .collect();
        let unlinked = unlinker.join().unwrap();

        let winners: Vec<u64> = outcomes
            .iter()
            .filter(|(_, outcome)| *outcome == Claim::Claimed)
            .map(|&(value, _)| value)
            .collect();
        assert!(winners.len() <= 1, "two claims of one key won");
        let live: Vec<(u64, u64)> = entries(&table)
            .into_iter()
            .filter(|&(k, _)| k == key)
            .collect();
        assert!(live.len() <= 1, "two live children under one key");
        match winners.first() {
            Some(&value) => {
                assert!(unlinked);
                assert_eq!(find(&table, key), Some(value));
            }
            None if unlinked => assert_eq!(find(&table, key), None),
            None => assert_eq!(find(&table, key), Some(old)),
        }
    });
}
