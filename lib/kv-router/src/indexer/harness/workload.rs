// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared prefix pools: system prompt, then document, then user turn, with ids drawn
//! from small ranges so ranks share prefixes and diverge at every depth.

use crate::protocols::LocalBlockHash;

pub(super) fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub(super) fn hash_parts(parts: &[u64]) -> u64 {
    parts
        .iter()
        .fold(0xD1B5_4A32_D192_ED03, |acc, &p| mix(acc ^ mix(p)))
}

/// Pool shape for the differential: few ids per level, so every level is shared.
pub(super) struct Pool {
    pub(super) salt: u64,
    pub(super) doc_len: u64,
}

impl Pool {
    pub(super) const SYSTEMS: u64 = 3;
    pub(super) const DOCS: u64 = 4;
    pub(super) const TURNS: u64 = 3;

    /// Local hashes of one pool sequence. Document 3 of each system prompt is four times
    /// longer than the others, so edges also cross the 16-block edge-index threshold.
    pub(super) fn sequence(&self, system: u64, doc: u64, turn: u64) -> Vec<LocalBlockHash> {
        let salt = self.salt;
        let system_len = 1 + hash_parts(&[1, salt, system]) % 3;
        let doc_cap = if doc == 3 {
            self.doc_len * 4
        } else {
            self.doc_len
        };
        let doc_len = hash_parts(&[2, salt, system, doc]) % (doc_cap + 1);
        let turn_len = 1 + hash_parts(&[3, salt, system, doc, turn]) % 3;
        let mut seq = Vec::with_capacity((system_len + doc_len + turn_len) as usize);
        seq.extend((0..system_len).map(|i| hash_parts(&[10, salt, system, i])));
        seq.extend((0..doc_len).map(|i| hash_parts(&[11, salt, system, doc, i])));
        seq.extend((0..turn_len).map(|i| hash_parts(&[12, salt, system, doc, turn, i])));
        seq.into_iter().map(LocalBlockHash).collect()
    }

    pub(super) fn random(&self, rng: &mut fastrand::Rng) -> Vec<LocalBlockHash> {
        self.sequence(
            rng.u64(..Self::SYSTEMS),
            rng.u64(..Self::DOCS),
            rng.u64(..Self::TURNS),
        )
    }

    /// `count` decode blocks after `tail`. Variants 1 and 2 are shared by every rank, so
    /// ranks race to extend and split the same leaf; any other variant is private.
    pub(super) fn extension(&self, tail: u64, variant: u64, count: u64) -> Vec<LocalBlockHash> {
        (0..count)
            .map(|i| LocalBlockHash(hash_parts(&[20, self.salt, tail, variant, i])))
            .collect()
    }
}
