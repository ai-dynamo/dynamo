// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rendezvous (Highest Random Weight) hashing primitives.
//!
//! New routing uses [`score`] with explicit domain separation and length framing. The legacy
//! worker scorer is retained so moving LoRA's implementation into this lower-level crate does
//! not change existing adapter placement.

use crate::protocols::WorkerWithDpRank;

const FRAMING_VERSION: &[u8] = b"dynamo-rendezvous-v1";

fn first_u64(hash: blake3::Hash) -> u64 {
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&hash.as_bytes()[..8]);
    u64::from_le_bytes(bytes)
}

fn update_field(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

/// Score one logical target for `key` within `domain`.
///
/// `target_id` must be stable for as long as routing should survive process restarts. For a
/// Dynamo worker that normally means `stable_routing_id`, with a separately tagged ephemeral
/// worker ID only as a compatibility fallback. `local_dp_rank` is the rank ordinal within that
/// stable worker, not an ephemeral process-wide identifier.
pub fn score(domain: &[u8], key: &[u8], target_id: &[u8], local_dp_rank: u32) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(FRAMING_VERSION);
    update_field(&mut hasher, domain);
    update_field(&mut hasher, key);
    update_field(&mut hasher, target_id);
    hasher.update(&local_dp_rank.to_le_bytes());
    first_u64(hasher.finalize())
}

/// Compatibility scorer for the existing LoRA `(name, worker_id, dp_rank)` encoding.
///
/// Do not use this unframed encoding for new routing protocols.
pub fn legacy_worker_score(key: &str, worker: WorkerWithDpRank) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(key.as_bytes());
    hasher.update(&worker.worker_id.to_le_bytes());
    hasher.update(&worker.dp_rank.to_le_bytes());
    first_u64(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_legacy_score(key: &str, worker: WorkerWithDpRank) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(key.as_bytes());
        hasher.update(&worker.worker_id.to_le_bytes());
        hasher.update(&worker.dp_rank.to_le_bytes());
        first_u64(hasher.finalize())
    }

    #[test]
    fn legacy_worker_score_preserves_lora_encoding() {
        assert_eq!(
            legacy_worker_score("adapter-a", WorkerWithDpRank::new(7, 3)),
            0x5ead_a6df_812d_f3dc
        );
        for worker in [
            WorkerWithDpRank::new(0, 0),
            WorkerWithDpRank::new(7, 3),
            WorkerWithDpRank::new(u64::MAX, u32::MAX),
        ] {
            assert_eq!(
                legacy_worker_score("adapter-a", worker),
                reference_legacy_score("adapter-a", worker)
            );
        }
    }

    #[test]
    fn new_score_separates_domains_targets_and_ranks() {
        let base = score(b"session/aggregated", b"session-a", b"worker-0", 0);
        assert_eq!(base, 0x6082_f19d_5652_af58);
        assert_ne!(
            base,
            score(b"session/prefill", b"session-a", b"worker-0", 0)
        );
        assert_ne!(
            base,
            score(b"session/aggregated", b"session-b", b"worker-0", 0)
        );
        assert_ne!(
            base,
            score(b"session/aggregated", b"session-a", b"worker-1", 0)
        );
        assert_ne!(
            base,
            score(b"session/aggregated", b"session-a", b"worker-0", 1)
        );
    }

    #[test]
    fn length_framing_prevents_field_boundary_collisions() {
        assert_ne!(
            score(b"ab", b"c", b"target", 0),
            score(b"a", b"bc", b"target", 0)
        );
    }
}
