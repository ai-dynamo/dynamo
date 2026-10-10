// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};

use loom::sync::Arc;
use loom::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use loom::thread;

use crate::protocol::{
    EMPTY, Mailbox, Step, TOMB, claim_enter_with, claim_exit, load_entry_with, load_len_with,
    promote, promote_with, publish_len_with, read_begin, read_validate, reuse_fence_with,
};
use crate::sync::Mutex;

fn model(f: impl Fn() + Sync + Send + 'static) {
    let mut builder = loom::model::Builder::new();
    builder.preemption_bound = Some(3);
    builder.check(f);
}

/// Runs `f`, which must find a bug: the negative control of a fix.
fn must_fail(name: &str, f: impl FnOnce()) {
    let failed = catch_unwind(AssertUnwindSafe(f)).is_err();
    assert!(
        failed,
        "negative control {name}: the model found no bug with the fix removed"
    );
}

// ----------------------------------------------------------------------------
// W1 (fix 1): the release fence after the odd step
// ----------------------------------------------------------------------------

fn torn_read<const FENCE: bool>() {
    model(|| {
        let version = Arc::new(AtomicU64::new(0));
        let flags = Arc::new(AtomicU32::new(0));
        let a = Arc::new(AtomicU64::new(0));
        let b = Arc::new(AtomicU64::new(0));
        let writer = {
            let (version, flags, a, b) = (version.clone(), flags.clone(), a.clone(), b.clone());
            thread::spawn(move || {
                let step = Step::<true>::open_with::<FENCE>(&version, &flags);
                a.store(1, Ordering::Relaxed);
                b.store(1, Ordering::Relaxed);
                drop(step);
            })
        };
        if let Some(v1) = read_begin(&version) {
            let rb = b.load(Ordering::Relaxed);
            let ra = a.load(Ordering::Relaxed);
            if read_validate(&version, v1) {
                assert_eq!(ra, rb, "a validated read mixed two states");
            }
        }
        writer.join().unwrap();
    });
}

#[test]
fn step_fence_rejects_torn_reads() {
    torn_read::<true>();
}

#[test]
fn step_without_fence_validates_a_torn_read() {
    must_fail("no step fence", torn_read::<false>);
}

// ----------------------------------------------------------------------------
// Fix 1: split promotions inside the step
// ----------------------------------------------------------------------------

/// A run of 4 positions; slot 0 holds the first 3 as a partial entry. A split at 2 makes
/// the run 2 long and promotes slot 0 to a whole holder. A validated read must never
/// credit slot 0 with more than 3 positions.
fn split_promotion<const INSIDE: bool>() {
    model(|| {
        let version = Arc::new(AtomicU64::new(0));
        let flags = Arc::new(AtomicU32::new(0));
        let len = Arc::new(AtomicU32::new(4));
        let whole = Arc::new(AtomicU64::new(0));
        let entry = Arc::new(AtomicU64::new(3 << 32));
        let splitter = {
            let (version, flags, len, whole, entry) = (
                version.clone(),
                flags.clone(),
                len.clone(),
                whole.clone(),
                entry.clone(),
            );
            thread::spawn(move || {
                if !INSIDE {
                    promote(&whole, 1, &entry);
                }
                let step = Step::<true>::open_with::<true>(&version, &flags);
                len.store(2, Ordering::Relaxed);
                if INSIDE {
                    promote(&whole, 1, &entry);
                }
                drop(step);
            })
        };
        if let Some(v1) = read_begin(&version) {
            let l = load_len_with::<true>(&len);
            let w1 = whole.load(Ordering::Relaxed);
            let e = load_entry_with::<true>(&entry);
            let w2 = whole.load(Ordering::Relaxed);
            if read_validate(&version, v1) {
                let credit = if (w1 | w2) & 1 != 0 {
                    l
                } else if e != EMPTY && e != TOMB {
                    (e >> 32) as u32
                } else {
                    0
                };
                assert!(credit <= 3, "credited {credit} positions of 3 held");
            }
        }
        splitter.join().unwrap();
    });
}

#[test]
fn split_promotions_inside_the_step_never_overcount() {
    split_promotion::<true>();
}

#[test]
fn split_promotions_before_the_step_overcount() {
    must_fail("promotion outside the step", split_promotion::<false>);
}

// ----------------------------------------------------------------------------
// A1 (fix 2): the release fence after a free-list pop
// ----------------------------------------------------------------------------

fn free_list_reuse<const FENCE: bool>() {
    model(|| {
        let version = Arc::new(AtomicU64::new(0));
        let flags = Arc::new(AtomicU32::new(0));
        // The run names address 1, whose word holds its data.
        let pointer = Arc::new(AtomicU32::new(1));
        let data = Arc::new(AtomicU64::new(10));
        let free: Arc<Mutex<Vec<u32>>> = Arc::new(Mutex::new(Vec::new()));
        let killer = {
            let (version, flags, pointer, free) = (
                version.clone(),
                flags.clone(),
                pointer.clone(),
                free.clone(),
            );
            thread::spawn(move || {
                let step = Step::<true>::open(&version, &flags);
                pointer.store(0, Ordering::Relaxed);
                drop(step);
                free.lock().push(1);
            })
        };
        let reuser = {
            let (data, free) = (data.clone(), free.clone());
            thread::spawn(move || {
                let popped = free.lock().pop();
                if popped.is_some() {
                    reuse_fence_with::<FENCE>();
                    data.store(20, Ordering::Relaxed);
                }
            })
        };
        if let Some(v1) = read_begin(&version) {
            let p = pointer.load(Ordering::Relaxed);
            if p == 1 {
                let d = data.load(Ordering::Relaxed);
                if read_validate(&version, v1) {
                    assert_eq!(d, 10, "a stale reader validated the new owner's data");
                }
            }
        }
        killer.join().unwrap();
        reuser.join().unwrap();
    });
}

#[test]
fn reuse_fence_stops_stale_readers() {
    free_list_reuse::<true>();
}

#[test]
fn reuse_without_fence_lets_a_stale_reader_validate() {
    must_fail("no reuse fence", free_list_reuse::<false>);
}

// ----------------------------------------------------------------------------
// W5: the stepless append and the reader's acquire load of `len`
// ----------------------------------------------------------------------------

fn sole_holder_append<const ORDERED: bool>() {
    model(|| {
        let version = Arc::new(AtomicU64::new(0));
        let len = Arc::new(AtomicU32::new(1));
        let data = Arc::new([AtomicU64::new(5), AtomicU64::new(0)]);
        let appender = {
            let (len, data) = (len.clone(), data.clone());
            thread::spawn(move || {
                data[1].store(7, Ordering::Relaxed);
                publish_len_with::<ORDERED>(&len, 2);
            })
        };
        if let Some(v1) = read_begin(&version) {
            let l = load_len_with::<ORDERED>(&len);
            if l == 2 {
                let d = data[1].load(Ordering::Relaxed);
                if read_validate(&version, v1) {
                    assert_eq!(d, 7, "an appended position read before its write");
                }
            }
        }
        appender.join().unwrap();
    });
}

#[test]
fn appended_positions_are_visible_with_their_length() {
    sole_holder_append::<true>();
}

#[test]
fn relaxed_length_exposes_unwritten_positions() {
    must_fail("relaxed len", sole_holder_append::<false>);
}

// ----------------------------------------------------------------------------
// 5.3: a child claim against a step that hands the table off
// ----------------------------------------------------------------------------

fn claim_handoff<const FENCE: bool>() {
    model(|| {
        let version = Arc::new(AtomicU64::new(0));
        let flags = Arc::new(AtomicU32::new(0));
        let inflight = Arc::new(AtomicU32::new(0));
        let key = Arc::new(AtomicU64::new(EMPTY));
        let value = Arc::new(AtomicU64::new(EMPTY));
        let copied = Arc::new(AtomicU64::new(EMPTY));
        let claimed = Arc::new(AtomicBool::new(false));
        let rebuilder = {
            let (version, flags, inflight, key, copied) = (
                version.clone(),
                flags.clone(),
                inflight.clone(),
                key.clone(),
                copied.clone(),
            );
            thread::spawn(move || {
                let step =
                    Step::<true>::open_excluding_claims_with::<FENCE>(&version, &flags, &inflight);
                copied.store(key.load(Ordering::Acquire), Ordering::Relaxed);
                drop(step);
            })
        };
        if claim_enter_with::<FENCE>(&inflight, &version, 0) {
            if key
                .compare_exchange(EMPTY, 42, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                value.store(1, Ordering::Release);
                claimed.store(true, Ordering::Relaxed);
            }
            claim_exit(&inflight);
        }
        rebuilder.join().unwrap();
        if claimed.load(Ordering::Relaxed) {
            assert_eq!(
                copied.load(Ordering::Relaxed),
                42,
                "the hand-off lost a claim"
            );
        }
    });
}

#[test]
fn claims_are_never_lost_in_a_table_handoff() {
    claim_handoff::<true>();
}

#[test]
fn claims_without_seqcst_fences_are_lost() {
    must_fail("no SeqCst fences in the handshake", claim_handoff::<false>);
}

// ----------------------------------------------------------------------------
// W5: promotion (bit, then tombstone) against the reader's double coverage read
// ----------------------------------------------------------------------------

fn promotion_double_read<const ORDERED: bool>() {
    model(|| {
        let whole = Arc::new(AtomicU64::new(0));
        let entry = Arc::new(AtomicU64::new(1 << 32));
        let promoter = {
            let (whole, entry) = (whole.clone(), entry.clone());
            thread::spawn(move || promote_with::<ORDERED>(&whole, 1, &entry))
        };
        let w1 = whole.load(Ordering::Relaxed);
        let e = load_entry_with::<ORDERED>(&entry);
        let w2 = whole.load(Ordering::Relaxed);
        let credited = (w1 | w2) & 1 != 0 || (e != EMPTY && e != TOMB);
        assert!(credited, "the slot was neither whole nor partial");
        promoter.join().unwrap();
    });
}

#[test]
fn promotion_never_drops_coverage_for_a_reader() {
    promotion_double_read::<true>();
}

#[test]
fn unordered_promotion_drops_coverage() {
    must_fail("relaxed tombstone", promotion_double_read::<false>);
}

// ----------------------------------------------------------------------------
// 8.1: the rank mailbox state machine, two lanes and two ranks
// ----------------------------------------------------------------------------

fn mailbox<const RECHECK: bool>() {
    model(|| {
        let cells: Arc<[Mailbox<(usize, u8)>; 2]> =
            Arc::new([Mailbox::default(), Mailbox::default()]);
        let lists: Arc<[Mutex<VecDeque<usize>>; 2]> =
            Arc::new([Mutex::new(VecDeque::new()), Mutex::new(VecDeque::new())]);
        let applied: Arc<Mutex<Vec<(usize, usize, u8)>>> = Arc::new(Mutex::new(Vec::new()));
        let running = Arc::new([AtomicBool::new(false), AtomicBool::new(false)]);
        let serve = {
            let (cells, lists, applied, running) = (
                cells.clone(),
                lists.clone(),
                applied.clone(),
                running.clone(),
            );
            move |me: usize, cell: usize| {
                assert!(cells[cell].take(), "fix 8: a listed cell was not READY");
                assert!(
                    !running[cell].swap(true, Ordering::AcqRel),
                    "two lanes ran one rank"
                );
                while let Some((producer, seq)) = cells[cell].pop() {
                    applied.lock().push((cell, producer, seq));
                }
                running[cell].store(false, Ordering::Release);
                if cells[cell].release_with::<RECHECK>() {
                    lists[me].lock().push_back(cell);
                }
            }
        };
        let lane = |me: usize, tasks: Vec<(usize, u8)>| {
            let (cells, lists, serve) = (cells.clone(), lists.clone(), serve.clone());
            thread::spawn(move || {
                for (cell, seq) in tasks {
                    if cells[cell].push((me, seq)) {
                        lists[me].lock().push_back(cell);
                    }
                }
                for _ in 0..2 {
                    let own = lists[me].lock().pop_front();
                    match own.or_else(|| lists[1 - me].lock().pop_front()) {
                        Some(cell) => serve(me, cell),
                        None => thread::yield_now(),
                    }
                }
            })
        };
        // Lane 0 feeds both ranks; lane 1 is rank 0's second producer.
        let a = lane(0, vec![(0, 1), (0, 2), (1, 1)]);
        let b = lane(1, vec![(0, 1)]);
        a.join().unwrap();
        b.join().unwrap();
        // Whatever is still READY sits on a list; nothing may be stranded.
        for me in 0..2 {
            loop {
                let next = lists[me].lock().pop_front();
                let Some(cell) = next else { break };
                serve(me, cell);
            }
        }
        let applied = applied.lock();
        assert_eq!(
            applied.len(),
            4,
            "a queued task was stranded: {:?}",
            *applied
        );
        let from_lane0: Vec<u8> = applied
            .iter()
            .filter(|&&(cell, producer, _)| cell == 0 && producer == 0)
            .map(|&(_, _, seq)| seq)
            .collect();
        assert_eq!(
            from_lane0,
            vec![1, 2],
            "a producer's tasks ran out of order"
        );
    });
}

#[test]
fn mailboxes_run_every_task_once_in_order() {
    mailbox::<true>();
}

#[test]
fn mailboxes_without_the_idle_recheck_strand_tasks() {
    must_fail("no IDLE re-check", mailbox::<false>);
}

// ----------------------------------------------------------------------------
// W7 (fix 4): a step that unwinds stays odd and the run is poisoned
// ----------------------------------------------------------------------------

fn poisoned_step<const POISON: bool>() {
    model(|| {
        let version = Arc::new(AtomicU64::new(0));
        let flags = Arc::new(AtomicU32::new(0));
        let a = Arc::new(AtomicU64::new(0));
        let b = Arc::new(AtomicU64::new(0));
        let lock = Arc::new(Mutex::new(()));
        let writer = {
            let (version, flags, a, lock) =
                (version.clone(), flags.clone(), a.clone(), lock.clone());
            thread::spawn(move || {
                let _guard = lock.lock();
                let step = Step::<POISON>::open_with::<true>(&version, &flags);
                a.store(1, Ordering::Relaxed);
                // The step panics before it writes `b`.
                step.unwind();
            })
        };
        // R7: bounded optimistic attempts, then the lock; odd under the lock is poisoned.
        let mut seen = None;
        for _ in 0..2 {
            if let Some(v1) = read_begin(&version) {
                let ra = a.load(Ordering::Relaxed);
                let rb = b.load(Ordering::Relaxed);
                if read_validate(&version, v1) {
                    seen = Some((ra, rb));
                    break;
                }
            }
            thread::yield_now();
        }
        if seen.is_none() {
            let _guard = lock.lock();
            let v = version.load(Ordering::Acquire);
            if v & 1 == 0 {
                seen = Some((a.load(Ordering::Relaxed), b.load(Ordering::Relaxed)));
            }
        }
        if let Some((ra, rb)) = seen {
            assert_eq!(ra, rb, "a reader validated a step that never finished");
        }
        writer.join().unwrap();
    });
}

#[test]
fn poisoned_steps_stop_readers() {
    poisoned_step::<true>();
}

#[test]
fn steps_closed_on_unwind_expose_torn_state() {
    must_fail("step closed on unwind", poisoned_step::<false>);
}
