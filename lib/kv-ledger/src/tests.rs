// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, VecDeque};

use crate::{Applied, Block, Chain, Effect, Ledger, Replay};

/// Stores `hashes` as a chain under `parent`, with `hash + 1000` as each block's content.
fn store(ledger: &mut Ledger<u64>, mut parent: Option<u64>, hashes: &[u64]) {
    for &hash in hashes {
        ledger.store(hash, hash + 1000, parent);
        parent = Some(hash);
    }
}

fn chain(parent: Option<u64>, hashes: &[u64]) -> Chain<u64> {
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

#[test]
fn chains_do_not_depend_on_where_blocks_sit() {
    // A value that is not a `u64` and has no `Default`.
    let content = |hash: u64| [hash as u32; 40];
    let chain = |parent: Option<u64>, hashes: &[u64]| Chain {
        parent,
        blocks: hashes.iter().map(|&hash| (hash, content(hash))).collect(),
    };

    // A parent stored after its child sits in the slot after it.
    let mut ledger = Ledger::default();
    ledger.store(2, content(2), Some(1));
    ledger.store(1, content(1), None);
    assert_eq!(ledger.chains(), vec![chain(None, &[1, 2])]);

    // Roots and branches come by hash, though 9 was stored before 1 and 5 before 4.
    let mut ledger = Ledger::default();
    ledger.store(9, content(9), None);
    ledger.store(1, content(1), None);
    ledger.store(5, content(5), Some(1));
    ledger.store(4, content(4), Some(1));
    assert_eq!(
        ledger.chains(),
        vec![
            chain(None, &[1]),
            chain(None, &[9]),
            chain(Some(1), &[4]),
            chain(Some(1), &[5]),
        ]
    );

    // 2 sits next to its parent, and 4 does not: both are still children of 1.
    let mut ledger = Ledger::default();
    ledger.store(1, content(1), None);
    ledger.store(2, content(2), Some(1));
    ledger.store(3, content(3), None);
    ledger.store(4, content(4), Some(1));
    assert_eq!(
        ledger.chains(),
        vec![
            chain(None, &[1]),
            chain(None, &[3]),
            chain(Some(1), &[2]),
            chain(Some(1), &[4]),
        ]
    );

    // Removing 2 moves 4 into its slot. 4 is still found there, and can be removed.
    assert!(ledger.remove(2));
    assert_eq!(
        ledger.chains(),
        vec![chain(None, &[1, 4]), chain(None, &[3])]
    );
    assert!(ledger.remove(4));
    assert_eq!(ledger.chains(), vec![chain(None, &[1]), chain(None, &[3])]);
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

fn fold(event: &Event, ledger: &mut Ledger<u64>) -> Effect {
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

/// The blocks a ledger holds, as a plain map.
fn held(ledger: &Ledger<u64>) -> BTreeMap<u64, Block<u64>> {
    ledger.iter().collect()
}

/// What `fold` does, on a plain map.
fn fold_plain(event: &Event, held: &mut BTreeMap<u64, Block<u64>>) {
    match event {
        Event::Store(parent, hashes) => {
            let mut parent = *parent;
            for &hash in hashes {
                let content = hash + 1000;
                held.entry(hash).or_insert(Block { parent, content });
                parent = Some(hash);
            }
        }
        Event::Remove(hashes) => {
            for hash in hashes {
                held.remove(hash);
            }
        }
        Event::Clear => held.clear(),
    }
}

/// What `Ledger::chains` means, on a plain map: a block joins its parent's chain when it is
/// the parent's only child, and starts a chain of its own otherwise.
fn plain_chains(held: &BTreeMap<u64, Block<u64>>) -> Vec<Chain<u64>> {
    let mut children: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    let mut queue = VecDeque::new();
    for (&hash, block) in held {
        match block.parent {
            None => queue.push_back(hash),
            Some(parent) => children.entry(parent).or_default().push(hash),
        }
    }
    let mut chains = Vec::new();
    while let Some(start) = queue.pop_front() {
        let mut blocks = vec![(start, held[&start].content)];
        let branches = loop {
            let tail = blocks[blocks.len() - 1].0;
            let kids = children.remove(&tail).unwrap_or_default();
            if let [only] = kids[..] {
                blocks.push((only, held[&only].content));
                continue;
            }
            break kids;
        };
        chains.push(Chain {
            parent: held[&start].parent,
            blocks,
        });
        queue.extend(branches);
    }
    chains
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
    // The same events on a plain map: what the ledger must hold and chain, wherever its
    // blocks have come to sit.
    let mut plain = BTreeMap::new();
    // What a host keeps: the copy it took at the open cut.
    let mut copy = Ledger::default();
    let mut joined: Option<Ledger<u64>> = None;
    for seq in 0..2000 {
        let event = match below(40) {
            0 => Event::Clear,
            1..=14 => Event::Remove(vec![below(80), below(80)]),
            _ => {
                // Two blocks in either hash order, under no parent or any hash, its own
                // included.
                let first = below(40);
                let parent = (below(3) != 0).then(|| below(80));
                let mut hashes = vec![first, first + 40];
                if below(2) == 0 {
                    hashes.reverse();
                }
                Event::Store(parent, hashes)
            }
        };
        if let Some(joined) = &mut joined {
            fold(&event, joined);
        }
        fold_plain(&event, &mut plain);
        source.apply(seq, event, fold);
        let ledger = source.state();
        assert_eq!(held(ledger), plain, "blocks held after batch {seq}");
        assert_eq!(ledger.chains(), plain_chains(&plain), "chains after batch {seq}");
        assert_eq!(ledger.is_empty(), plain.is_empty(), "after batch {seq}");

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
            assert_eq!(held(joined), plain, "a joined consumer after batch {seq}");
        }
    }
}
