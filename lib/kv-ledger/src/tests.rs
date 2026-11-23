// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{Applied, Chain, Effect, Ledger, Replay};

/// Stores `hashes` as a chain under `parent`, with `hash + 1000` as each block's content.
fn store(ledger: &mut Ledger, mut parent: Option<u64>, hashes: &[u64]) {
    for &hash in hashes {
        ledger.store(hash, hash + 1000, parent);
        parent = Some(hash);
    }
}

fn chain(parent: Option<u64>, hashes: &[u64]) -> Chain {
    Chain {
        parent,
        blocks: hashes.iter().map(|&hash| (hash, hash + 1000)).collect(),
    }
}

#[test]
fn chains_come_parents_first_and_split_where_a_sequence_branches() {
    let mut ledger = Ledger::default();
    store(&mut ledger, None, &[1, 2, 3]);
    store(&mut ledger, Some(2), &[4, 5]);
    store(&mut ledger, None, &[9]);

    assert_eq!(
        ledger.chains(),
        vec![
            chain(None, &[1, 2]),
            chain(None, &[9]),
            chain(Some(2), &[3]),
            chain(Some(2), &[4, 5]),
        ]
    );
}

#[test]
fn chains_leave_out_what_cannot_be_reached() {
    let mut ledger = Ledger::default();
    store(&mut ledger, None, &[1, 2, 3, 4]);

    // Block 2 goes: 3 and 4 are still held, out of reach. One remove drops a block.
    assert!(ledger.remove(2));
    assert!(!ledger.remove(2));
    assert_eq!(ledger.chains(), vec![chain(None, &[1])]);
    assert_eq!(ledger.iter().count(), 3);

    // The engine stores 2 again, and the rest of the chain is reachable again. A second
    // store of a held block changes nothing.
    store(&mut ledger, Some(1), &[2]);
    ledger.store(3, 0, None);
    assert_eq!(ledger.chains(), vec![chain(None, &[1, 2, 3, 4])]);

    // A block that names itself as its parent and a parent cycle are held, never reached.
    store(&mut ledger, Some(7), &[7]);
    store(&mut ledger, Some(9), &[8]);
    store(&mut ledger, Some(8), &[9]);
    assert_eq!(ledger.chains(), vec![chain(None, &[1, 2, 3, 4])]);
    assert_eq!(ledger.iter().count(), 7);
}

fn push(replay: &mut Replay<u64, Vec<u64>>, seq: u64) -> Applied {
    replay.apply(seq, seq, |batch, state| {
        state.push(*batch);
        Effect::Delta
    })
}

fn tail(replay: &Replay<u64, Vec<u64>>) -> Option<Vec<u64>> {
    replay.tail().map(|tail| tail.copied().collect())
}

#[test]
fn replay_serves_the_window_and_restarts_it_at_a_gap() {
    let mut replay = Replay::new(Vec::new(), 3);
    assert_eq!(replay.last_seq(), None);
    assert!(replay.since(0).is_none());

    // The first batch can carry any sequence.
    for seq in 5..10 {
        assert_eq!(push(&mut replay, seq), Applied::InOrder);
    }
    assert_eq!((replay.oldest_seq(), replay.last_seq()), (Some(7), Some(9)));
    assert_eq!(
        replay.since(8).unwrap().copied().collect::<Vec<_>>(),
        vec![8, 9]
    );
    assert!(replay.since(6).is_none());

    // A gap is applied all the same, and nothing before it can be replayed any more.
    assert_eq!(push(&mut replay, 12), Applied::OutOfOrder { expected: 10 });
    assert_eq!(replay.state().len(), 6);
    assert_eq!(
        (replay.oldest_seq(), replay.last_seq()),
        (Some(12), Some(12))
    );
    assert!(replay.since(9).is_none());
}

#[test]
fn a_cut_stays_open_while_its_tail_is_short() {
    // A window of 4 lets a cut carry a tail of 2.
    let mut replay = Replay::new(Vec::new(), 4);
    assert_eq!(tail(&replay), None);
    push(&mut replay, 1);
    assert_eq!(replay.cut(), &[1]);
    assert_eq!(tail(&replay), Some(vec![]));

    push(&mut replay, 2);
    push(&mut replay, 3);
    assert_eq!(tail(&replay), Some(vec![2, 3]));

    // The third batch on top closes it there and then, not at the next request.
    push(&mut replay, 4);
    assert_eq!(tail(&replay), None);

    // So does a batch that resets state, and a gap.
    replay.cut();
    replay.apply(5, 5, |_, state| {
        state.clear();
        Effect::Reset
    });
    assert_eq!(tail(&replay), None);
    replay.cut();
    push(&mut replay, 7);
    assert_eq!(tail(&replay), None);
}

#[derive(Clone)]
enum Event {
    Store(Option<u64>, Vec<u64>),
    Remove(Vec<u64>),
    Clear,
}

fn fold(event: &Event, ledger: &mut Ledger) -> Effect {
    match event {
        Event::Store(parent, hashes) => store(ledger, *parent, hashes),
        Event::Remove(hashes) => {
            for &hash in hashes {
                ledger.remove(hash);
            }
        }
        Event::Clear => {
            *ledger = Ledger::default();
            return Effect::Reset;
        }
    }
    Effect::Delta
}

#[test]
fn a_copy_cut_then_its_tail_equals_following_from_the_start() {
    let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
    let mut below = |bound: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % bound
    };

    let mut source = Replay::new(Ledger::default(), 8);
    // What a host keeps: the copy it took at the open cut.
    let mut copy = Ledger::default();
    let mut joined: Option<Ledger> = None;
    for seq in 0..400 {
        let event = match below(10) {
            0 => Event::Clear,
            1..=3 => Event::Remove(vec![below(40), below(40)]),
            _ => {
                let first = below(40);
                let parent = (below(3) != 0).then(|| below(40));
                Event::Store(parent, vec![first, first + 40])
            }
        };
        if let Some(joined) = &mut joined {
            fold(&event, joined);
        }
        source.apply(seq, event, fold);

        if below(7) == 0 {
            // A consumer joins: it loads the copy, applies the tail, then follows live.
            if source.tail().is_none() {
                copy = source.cut().clone();
            }
            let mut state = copy.clone();
            for event in source.tail().unwrap() {
                fold(event, &mut state);
            }
            joined = Some(state);
        }
        if let Some(joined) = &joined {
            assert_eq!(joined, source.state(), "after batch {seq}");
        }
    }
}
