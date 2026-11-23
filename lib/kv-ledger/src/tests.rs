// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use crate::{Applied, Chain, Effect, Ledger, Replay, Tier};

const GPU: Tier = Tier::new(0).unwrap();
const HOST: Tier = Tier::new(1).unwrap();

/// Stores `hashes` as a chain under `parent`, with `hash + 1000` as each block's content.
fn store(ledger: &mut Ledger, tier: Tier, mut parent: Option<u64>, hashes: &[u64]) {
    for &hash in hashes {
        ledger.store(tier, hash, hash + 1000, parent);
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
fn a_block_is_held_until_its_last_tier_goes() {
    let mut ledger = Ledger::new();
    store(&mut ledger, GPU, None, &[1]);
    store(&mut ledger, HOST, None, &[1]);
    // A second store on the same tier changes nothing: one remove takes it off.
    store(&mut ledger, GPU, None, &[1]);

    assert!(ledger.remove(GPU, 1));
    assert!(!ledger.remove(GPU, 1));
    assert_eq!(ledger.len(), 1);
    assert!(ledger.remove(HOST, 1));
    assert!(ledger.is_empty());
    assert!(!ledger.remove(HOST, 7));
}

#[test]
fn chains_come_parents_first_and_split_where_a_sequence_branches() {
    let mut ledger = Ledger::new();
    store(&mut ledger, GPU, None, &[1, 2, 3]);
    store(&mut ledger, GPU, Some(2), &[4, 5]);
    store(&mut ledger, GPU, None, &[9]);

    assert_eq!(
        ledger.chains(GPU),
        vec![
            chain(None, &[1, 2]),
            chain(None, &[9]),
            chain(Some(2), &[3]),
            chain(Some(2), &[4, 5]),
        ]
    );
}

#[test]
fn chains_leave_out_what_cannot_be_reached_on_the_tier() {
    let mut ledger = Ledger::new();
    store(&mut ledger, GPU, None, &[1, 2, 3, 4]);
    store(&mut ledger, HOST, None, &[1, 2]);

    // Block 2 leaves the GPU and stays on the host: 3 and 4 are still held, out of reach.
    assert!(ledger.remove(GPU, 2));
    assert_eq!(ledger.chains(GPU), vec![chain(None, &[1])]);
    assert_eq!(ledger.chains(HOST), vec![chain(None, &[1, 2])]);
    assert_eq!(ledger.len(), 4);

    // The engine stores 2 on the GPU again, and the rest of the chain is reachable again.
    store(&mut ledger, GPU, Some(1), &[2]);
    assert_eq!(ledger.chains(GPU), vec![chain(None, &[1, 2, 3, 4])]);

    // A block that names itself as its parent and a parent cycle are held, never reached.
    store(&mut ledger, GPU, Some(7), &[7]);
    store(&mut ledger, GPU, Some(9), &[8]);
    store(&mut ledger, GPU, Some(8), &[9]);
    assert_eq!(ledger.chains(GPU), vec![chain(None, &[1, 2, 3, 4])]);
    assert_eq!(ledger.iter().count(), 7);
}

fn push(replay: &mut Replay<u64, Vec<u64>>, seq: u64) -> Applied {
    replay.apply(seq, seq, |batch, state| {
        state.push(*batch);
        Effect::Delta
    })
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
fn a_snapshot_is_shared_while_its_tail_is_short() {
    // A window of 4 lets a snapshot carry a tail of 2.
    let mut replay = Replay::new(Vec::new(), 4);
    push(&mut replay, 1);
    let (first, tail) = replay.snapshot();
    assert_eq!(tail.count(), 0);

    push(&mut replay, 2);
    push(&mut replay, 3);
    let (second, tail) = replay.snapshot();
    assert_eq!(tail.copied().collect::<Vec<_>>(), vec![2, 3]);
    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(second.last_seq, Some(1));

    // The third batch on top retires it there and then, not at the next request.
    push(&mut replay, 4);
    assert!(!replay.has_shared_snapshot());
    assert_eq!(replay.snapshot().0.last_seq, Some(4));

    // So does a batch that resets state, and a gap.
    replay.apply(5, 5, |_, state| {
        state.clear();
        Effect::Reset
    });
    assert!(!replay.has_shared_snapshot());
    let _ = replay.snapshot();
    push(&mut replay, 7);
    assert!(!replay.has_shared_snapshot());
    assert_eq!(replay.snapshots_built(), 3);
}

#[derive(Clone)]
enum Event {
    Store(Tier, Option<u64>, Vec<u64>),
    Remove(Tier, Vec<u64>),
    Clear,
}

fn fold(event: &Event, ledger: &mut Ledger) -> Effect {
    match event {
        Event::Store(tier, parent, hashes) => store(ledger, *tier, *parent, hashes),
        Event::Remove(tier, hashes) => {
            for &hash in hashes {
                ledger.remove(*tier, hash);
            }
        }
        Event::Clear => {
            *ledger = Ledger::new();
            return Effect::Reset;
        }
    }
    Effect::Delta
}

#[test]
fn a_snapshot_then_its_tail_equals_following_from_the_start() {
    let mut rng = 0x9E37_79B9_7F4A_7C15_u64;
    let mut below = |bound: u64| {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng % bound
    };

    let mut source = Replay::new(Ledger::new(), 8);
    let mut joined: Option<Ledger> = None;
    for seq in 0..400 {
        let tier = if below(2) == 0 { GPU } else { HOST };
        let event = match below(10) {
            0 => Event::Clear,
            1..=3 => Event::Remove(tier, vec![below(40), below(40)]),
            _ => {
                let first = below(40);
                let parent = (below(3) != 0).then(|| below(40));
                Event::Store(tier, parent, vec![first, first + 40])
            }
        };
        if let Some(joined) = &mut joined {
            fold(&event, joined);
        }
        source.apply(seq, event, fold);

        if below(7) == 0 {
            // A consumer joins: it loads the snapshot, applies the tail, then follows live.
            let (snapshot, tail) = source.snapshot();
            let mut state = snapshot.state.clone();
            for event in tail {
                fold(event, &mut state);
            }
            joined = Some(state);
        }
        if let Some(joined) = &joined {
            assert_eq!(joined, source.state(), "after batch {seq}");
        }
    }
}
