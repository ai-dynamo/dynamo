// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_kv_router::protocols::{BlockMmObjectInfo, compute_seq_hash_for_block};

fn direct_blocks(
    tokens: &[u32],
    block_size: u32,
    mm_infos: Option<&[Option<BlockExtraInfo>]>,
    lora_name: Option<&str>,
    cache_namespace: Option<&str>,
    is_eagle: bool,
) -> Vec<ApproximateLruBlock> {
    let local_hashes = compute_block_hash_for_seq(
        tokens,
        block_size,
        BlockHashOptions {
            block_mm_infos: mm_infos,
            lora_name,
            cache_namespace,
            is_eagle: Some(is_eagle),
        },
    );
    let sequence_hashes = compute_seq_hash_for_block(&local_hashes);
    local_hashes
        .into_iter()
        .zip(sequence_hashes)
        .map(|(local_hash, sequence_hash)| ApproximateLruBlock {
            local_hash,
            sequence_hash,
        })
        .collect()
}

#[test]
fn streamed_chunks_complete_prompt_tail_and_extend_canonical_chain() {
    let prompt = vec![1, 2, 3];
    let mut tracker = CanonicalOutputTracker::from_parts(&prompt, None, 4, false, None, None);
    tracker.set_prompt_parent(None);

    let first = tracker.observe(0, &[4, 5]).unwrap();
    assert_eq!(first.start_position, 0);
    assert_eq!(first.private_blocks, 0);
    let second = tracker.observe(0, &[6, 7, 8, 9]).unwrap();
    assert_eq!(second.start_position, 1);
    assert_eq!(second.private_blocks, 0);

    let expected = direct_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None, None, None, false);
    assert_eq!(
        first
            .blocks
            .into_iter()
            .chain(second.blocks)
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn incomplete_output_tail_is_not_materialized() {
    let mut tracker = CanonicalOutputTracker::from_parts(&[1], None, 4, false, None, None);
    assert!(tracker.observe(0, &[2, 3]).is_none());
    assert_eq!(tracker.initial_private_blocks(), 1);
}

#[test]
fn aligned_prompt_reports_partial_output_as_private_occupancy() {
    let prompt = [1, 2, 3, 4];
    let prompt_block = direct_blocks(&prompt, 4, None, None, None, false);
    let mut tracker = CanonicalOutputTracker::from_parts(&prompt, None, 4, false, None, None);
    tracker.set_prompt_parent(Some(prompt_block[0].sequence_hash));

    assert!(tracker.observe(0, &[5]).is_none());
    let partial = tracker.observe(0, &[6]).unwrap();
    assert!(partial.blocks.is_empty());
    assert_eq!(partial.private_blocks, 1);

    let completed = tracker.observe(0, &[7, 8, 9]).unwrap();
    assert_eq!(completed.blocks.len(), 1);
    assert_eq!(completed.private_blocks, 0);
}

#[test]
fn multiple_choice_streams_keep_independent_hash_tails() {
    let prompt = [1, 2, 3, 4];
    let prompt_block = direct_blocks(&prompt, 4, None, None, None, false);
    let mut tracker = CanonicalOutputTracker::from_parts(&prompt, None, 4, false, None, None);
    tracker.set_prompt_parent(Some(prompt_block[0].sequence_hash));

    let choice_zero = tracker.observe(0, &[5, 6, 7, 8, 13]).unwrap();
    let choice_one = tracker.observe(1, &[9, 10, 11, 12, 14]).unwrap();
    assert_eq!(choice_zero.start_position, 1);
    assert_eq!(choice_one.start_position, 1);
    assert_eq!(
        choice_zero.blocks[0],
        direct_blocks(&[1, 2, 3, 4, 5, 6, 7, 8], 4, None, None, None, false)[1]
    );
    assert_eq!(
        choice_one.blocks[0],
        direct_blocks(&[1, 2, 3, 4, 9, 10, 11, 12], 4, None, None, None, false)[1]
    );
}

#[test]
fn eagle_lora_namespace_and_multimodal_hashing_matches_canonical_path() {
    let prompt = vec![10, 11, 12];
    let mm_infos = vec![Some(BlockExtraInfo {
        mm_objects: vec![BlockMmObjectInfo {
            mm_hash: 42,
            offsets: vec![(0, 2)],
        }],
    })];
    let mut tracker = CanonicalOutputTracker::from_parts(
        &prompt,
        Some(&mm_infos),
        4,
        true,
        Some("adapter-a".to_string()),
        Some("tenant-a".to_string()),
    );

    let first = tracker.observe(0, &[13, 14]).unwrap();
    let second = tracker.observe(0, &[15, 16, 17, 18]).unwrap();
    let expected = direct_blocks(
        &[10, 11, 12, 13, 14, 15, 16, 17, 18],
        4,
        Some(&[mm_infos[0].clone(), None]),
        Some("adapter-a"),
        Some("tenant-a"),
        true,
    );
    assert_eq!(
        first
            .blocks
            .into_iter()
            .chain(second.blocks)
            .collect::<Vec<_>>(),
        expected
    );
}
