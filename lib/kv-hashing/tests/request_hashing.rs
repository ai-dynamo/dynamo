// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end request → block-hash tests.

use dynamo_kv_hashing::{
    KvHashingError, Request, RequestMmObjectInfo, SaltHash, Token, TokenBlockMmInfo,
    compute_block_hash,
};
use dynamo_tokens::{TokenBlockSequence, Tokens};

fn req(
    tokens: Vec<Token>,
    lora: Option<&str>,
    salt: Option<&str>,
    mm: Vec<RequestMmObjectInfo>,
) -> Request {
    Request::builder()
        .tokens(tokens)
        .lora_name(lora.map(|s| s.to_string()))
        .salt(salt.map(|s| s.to_string()))
        .mm_info(mm)
        .build()
        .expect("test fixture mm_info should validate")
}

const BS: u32 = 4;

// -----------------------------------------------------------------------------
// #1 determinism
// -----------------------------------------------------------------------------
#[test]
fn determinism_same_request_same_hashes() {
    let r1 = req((1..=12).collect(), Some("lora-a"), Some("model-x"), vec![]);
    let r2 = req((1..=12).collect(), Some("lora-a"), Some("model-x"), vec![]);
    let plh1 = r1.positional_lineage_hashes(BS).unwrap();
    let plh2 = r2.positional_lineage_hashes(BS).unwrap();
    assert_eq!(plh1, plh2);
    assert_eq!(plh1.len(), 3);
}

// -----------------------------------------------------------------------------
// #2 salt isolation
// -----------------------------------------------------------------------------
#[test]
fn salt_isolation_lora_change_diverges_from_block_zero() {
    let base = req((1..=12).collect(), None, None, vec![]);
    let lora_a = req((1..=12).collect(), Some("a"), None, vec![]);
    let lora_b = req((1..=12).collect(), Some("b"), None, vec![]);
    let salty = req((1..=12).collect(), None, Some("salt"), vec![]);

    let h_base = base.positional_lineage_hashes(BS).unwrap();
    let h_a = lora_a.positional_lineage_hashes(BS).unwrap();
    let h_b = lora_b.positional_lineage_hashes(BS).unwrap();
    let h_salt = salty.positional_lineage_hashes(BS).unwrap();

    // lora and salt change ⇒ all blocks differ from the no-salt baseline.
    for i in 0..h_base.len() {
        assert_ne!(h_base[i], h_a[i]);
        assert_ne!(h_base[i], h_b[i]);
        assert_ne!(h_base[i], h_salt[i]);
        assert_ne!(h_a[i], h_b[i]);
    }

    // identical (salt, lora) ⇒ identical hashes.
    let lora_a_again = req((1..=12).collect(), Some("a"), None, vec![]);
    let h_a2 = lora_a_again.positional_lineage_hashes(BS).unwrap();
    assert_eq!(h_a, h_a2);
}

// -----------------------------------------------------------------------------
// #3 mm_per_block: an MM run inside one block diverges only at that block (and downstream).
// -----------------------------------------------------------------------------
#[test]
fn mm_per_block_diverges_only_at_affected_block() {
    // 4 blocks of size 4 = 16 tokens. MM run inside block 3 (positions [12..14)).
    let tokens: Vec<Token> = (0..16).collect();
    let mm_a = vec![RequestMmObjectInfo {
        mm_hash: 0xAA,
        offset: 12,
        length: 2,
    }];
    let mm_b = vec![RequestMmObjectInfo {
        mm_hash: 0xBB,
        offset: 12,
        length: 2,
    }];
    let r_a = req(tokens.clone(), None, None, mm_a);
    let r_b = req(tokens.clone(), None, None, mm_b);

    let h_a = r_a.positional_lineage_hashes(BS).unwrap();
    let h_b = r_b.positional_lineage_hashes(BS).unwrap();
    assert_eq!(h_a.len(), 4);

    // Blocks 0..3 (covering positions [0..12)) are identical (prefix sharing).
    for i in 0..3 {
        assert_eq!(h_a[i], h_b[i], "block {i} should match — pre-MM prefix");
    }
    // Block 3 diverges (different mm_hash) — and lineage propagates downstream
    // (no downstream blocks here since we only have 4).
    assert_ne!(h_a[3], h_b[3]);

    // Cross-request sharing: same image at the same global offset → same block 3 hash.
    let r_a2 = req(
        tokens,
        None,
        None,
        vec![RequestMmObjectInfo {
            mm_hash: 0xAA,
            offset: 12,
            length: 2,
        }],
    );
    let h_a2 = r_a2.positional_lineage_hashes(BS).unwrap();
    assert_eq!(h_a, h_a2);
}

// -----------------------------------------------------------------------------
// #4 mm_spans_blocks: an MM run that spans block boundaries. block_size=8, run starts
// at offset 8, length 20 ⇒ block 1 fully placeholder, block 2 fully placeholder,
// block 3 partial (4 placeholders + 4 reals), block 4 partial.
// -----------------------------------------------------------------------------
#[test]
fn mm_spans_blocks() {
    let block_size: u32 = 8;
    let tokens: Vec<Token> = (0..40).collect();
    let mm = vec![RequestMmObjectInfo {
        mm_hash: 0xCAFE,
        offset: 8,
        length: 20,
    }];
    let mm_request = req(tokens.clone(), None, None, mm);
    let baseline = req(tokens, None, None, vec![]);

    let h_mm = mm_request.positional_lineage_hashes(block_size).unwrap();
    let h_base = baseline.positional_lineage_hashes(block_size).unwrap();
    assert_eq!(h_mm.len(), 5);
    assert_eq!(h_mm.len(), h_base.len());

    // Block 0 is pre-MM ⇒ matches baseline.
    assert_eq!(h_mm[0], h_base[0]);
    // Block 1 onward differs (mm coverage starts at 8).
    for i in 1..h_mm.len() {
        assert_ne!(h_mm[i], h_base[i], "block {i} should diverge from baseline");
    }

    // Block 1 (fully placeholder, run_offsets 0..7) ≠ Block 2 (fully placeholder,
    // run_offsets 8..15) — a multi-block MM run produces distinct block hashes via
    // the run_offset bytes alone.
    let bh = mm_request.block_hashes(block_size).unwrap();
    assert_ne!(
        bh[1], bh[2],
        "fully-placeholder blocks must differ via run_offset"
    );
}

// -----------------------------------------------------------------------------
// #5 mm_full_block: a block that is entirely placeholders has a deterministic
// 16*13=208-byte tagged buffer.
// -----------------------------------------------------------------------------
#[test]
fn mm_full_block() {
    use dynamo_kv_hashing::MM_SLOT_TAG_PLACEHOLDER;
    let block_size: u32 = 16;
    // 16 placeholder slots + 16 real tokens. Block 0 fully placeholder.
    let mut tokens: Vec<Token> = vec![0u32; 16];
    tokens.extend(16..32);
    let mm = vec![RequestMmObjectInfo {
        mm_hash: 0xDEADBEEF_DEADBEEF,
        offset: 0,
        length: 16,
    }];
    let request = req(tokens, None, None, mm);
    let salt = request.salt_hash().unwrap();

    // Manually build the expected 208-byte tagged buffer:
    //   per slot: [tag=PLACEHOLDER (1) | run_offset u32 LE (4) | mm_hash u64 LE (8)] = 13 bytes.
    let mut expected = Vec::with_capacity(208);
    for i in 0..16u32 {
        expected.push(MM_SLOT_TAG_PLACEHOLDER);
        expected.extend_from_slice(&i.to_le_bytes());
        expected.extend_from_slice(&0xDEADBEEF_DEADBEEFu64.to_le_bytes());
    }
    assert_eq!(expected.len(), 16 * 13);
    let expected_block_hash = compute_block_hash(&expected, salt);

    let blocks = request.into_blocks(block_size).unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].block_hash, expected_block_hash);
}

// -----------------------------------------------------------------------------
// Reviewer P2 — empty LoRA name normalizes to None for cache-share parity with the
// existing router behavior at lib/kv-router/src/protocols.rs:84
// (`options.lora_name.filter(|n| !n.is_empty())`). A client that sends "" must share
// cache with a client that sends None.
// -----------------------------------------------------------------------------
#[test]
fn empty_lora_normalizes_to_none() {
    let r_none = req((1..=12).collect(), None, None, vec![]);
    let r_empty = req((1..=12).collect(), Some(""), None, vec![]);
    let r_empty_salt = req((1..=12).collect(), None, Some(""), vec![]);
    assert_eq!(r_none.salt_hash().unwrap(), r_empty.salt_hash().unwrap());
    assert_eq!(
        r_none.salt_hash().unwrap(),
        r_empty_salt.salt_hash().unwrap()
    );
    assert_eq!(
        r_none.positional_lineage_hashes(BS).unwrap(),
        r_empty.positional_lineage_hashes(BS).unwrap()
    );

    // Real LoRA names are still distinct.
    let r_real = req((1..=12).collect(), Some("a"), None, vec![]);
    assert_ne!(r_none.salt_hash().unwrap(), r_real.salt_hash().unwrap());
}

// -----------------------------------------------------------------------------
// #6 partial_tail: trailing partial block is not hashed; n_blocks = total / block_size.
// -----------------------------------------------------------------------------
#[test]
fn partial_tail_not_hashed() {
    // block_size 4, 11 tokens ⇒ 2 complete blocks, 3 trailing.
    let r = req((0..11).collect(), None, None, vec![]);
    let blocks = r.into_blocks(4).unwrap();
    assert_eq!(blocks.len(), 2);

    // With MM placeholders too: block_size 4, 7 tokens (4 real + 3 placeholders) → 1 complete block.
    let mm = vec![RequestMmObjectInfo {
        mm_hash: 1,
        offset: 4,
        length: 3,
    }];
    let r = req(vec![0u32; 7], None, None, mm);
    let blocks = r.into_blocks(4).unwrap();
    assert_eq!(blocks.len(), 1);
}

// -----------------------------------------------------------------------------
// #7 cross_check_tokens: an MM-empty Request matches dynamo_tokens::TokenBlockSequence::new
// field-for-field.
// -----------------------------------------------------------------------------
#[test]
fn cross_check_tokens_zero_mm() {
    let tokens: Vec<Token> = (1..=12).collect();
    let r = req(tokens.clone(), Some("lora-z"), Some("salty"), vec![]);
    let salt: SaltHash = r.salt_hash().unwrap();

    let baseline = TokenBlockSequence::new(Tokens::from(tokens), 4, Some(salt));
    let universal = r.into_blocks(4).unwrap();
    assert_eq!(universal.len(), baseline.blocks().len());
    for (u, b) in universal.iter().zip(baseline.blocks().iter()) {
        assert_eq!(u.position(), b.position());
        assert_eq!(u.block_hash, b.block_hash());
        assert_eq!(u.sequence_hash(), b.sequence_hash());
        assert_eq!(u.plh, b.positional_lineage_hash());
    }
    // Salt is per-request, not per-block.
    assert_eq!(r.salt_hash().unwrap(), salt);
}

// -----------------------------------------------------------------------------
// #8 extension_consistency: split a sequence into prefix + full; the prefix's blocks
// match the full's first N, and the chain extends correctly via PLH alone (no
// out-of-band sequence-hash tracking).
// -----------------------------------------------------------------------------
#[test]
fn extension_consistency() {
    let tokens: Vec<Token> = (1..=20).collect();
    let prefix = req(tokens[..12].to_vec(), Some("ll"), None, vec![]);
    let full = req(tokens, Some("ll"), None, vec![]);

    let h_prefix = prefix.into_blocks(4).unwrap();
    let h_full = full.into_blocks(4).unwrap();
    assert_eq!(h_prefix.len(), 3);
    assert_eq!(h_full.len(), 5);
    for i in 0..h_prefix.len() {
        assert_eq!(h_prefix[i], h_full[i]);
    }

    // PLH self-extension: extending block N-1's PLH by block N's block_hash must
    // reproduce block N's PLH bitwise.
    for i in 1..h_full.len() {
        let extended = h_full[i - 1].plh.extend(h_full[i].block_hash);
        assert_eq!(
            extended,
            h_full[i].plh,
            "PLH::extend should reproduce block {i} from block {}",
            i - 1
        );
    }
}

// -----------------------------------------------------------------------------
// Bonus: Request::new validates mm_info via the same path as TokenBlockSequence.
// -----------------------------------------------------------------------------
#[test]
fn request_new_rejects_invalid_mm_info() {
    let bad = vec![
        RequestMmObjectInfo {
            mm_hash: 1,
            offset: 0,
            length: 5,
        },
        RequestMmObjectInfo {
            mm_hash: 2,
            offset: 4,
            length: 5,
        },
    ];
    let err = Request::builder()
        .tokens(vec![0u32; 10])
        .mm_info(bad)
        .build()
        .unwrap_err();
    assert!(matches!(err, KvHashingError::MmInfo(_)));
}

#[test]
fn request_builder_requires_tokens() {
    let err = Request::builder().build().unwrap_err();
    assert!(matches!(err, KvHashingError::MissingField("tokens")));
}

// Sanity: TokenBlockMmInfo ↔ RequestMmObjectInfo conversions.
#[test]
fn mm_info_conversion_roundtrip() {
    let r = RequestMmObjectInfo {
        mm_hash: 0xAB,
        offset: 1,
        length: 2,
    };
    let t: TokenBlockMmInfo = r.into();
    let back: RequestMmObjectInfo = t.into();
    assert_eq!(r, back);
}

// -----------------------------------------------------------------------------
// Producer's block_hash matches the router's LocalBlockHash for default requests
// -----------------------------------------------------------------------------
//
// The kvbm-consolidator's vLLM/TRT-LLM ingestion path (`lib/kvbm-consolidator/src/tracker.rs`)
// routes through `Request::into_blocks()` to compute block_hash + PLH. The kv-router's
// indexer recomputes `tokens_hash` from token_ids via `compute_block_hash_for_seq`
// (`lib/kv-router/src/protocols.rs:70-125`) using `XXH3_SEED = 1337` as the base seed
// (plus `xxh3_64(lora_name)` when lora is set). For the indexer's lookup to find a
// match, the producer's `salt_hash` for a default request must equal that seed.
#[test]
fn producer_block_hash_matches_router_local_block_hash_default() {
    use dynamo_tokens::compute_hash_v2;

    const ROUTER_XXH3_SEED: u64 = 1337;
    let tokens: Vec<Token> = (1u32..=16).collect();
    let block_size: u32 = 4;

    let producer = req(tokens.clone(), None, None, vec![])
        .block_hashes(block_size)
        .expect("block_hashes");

    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let router: Vec<u64> = token_bytes
        .chunks_exact((block_size as usize) * std::mem::size_of::<u32>())
        .map(|chunk| compute_hash_v2(chunk, ROUTER_XXH3_SEED))
        .collect();

    assert_eq!(
        producer, router,
        "producer block_hash (default request) must equal the router's LocalBlockHash"
    );
}

// Same check with a LoRA adapter set: producer seeds with
// `XXH3_SEED + xxh3_64(lora_name)`, matching `compute_block_hash_for_seq`'s lora path.
#[test]
fn producer_block_hash_matches_router_local_block_hash_lora() {
    use dynamo_tokens::compute_hash_v2;

    const ROUTER_XXH3_SEED: u64 = 1337;
    let lora = "alpha-7b";
    let tokens: Vec<Token> = (1u32..=16).collect();
    let block_size: u32 = 4;

    let producer = req(tokens.clone(), Some(lora), None, vec![])
        .block_hashes(block_size)
        .expect("block_hashes");

    let router_seed = ROUTER_XXH3_SEED.wrapping_add(compute_hash_v2(lora.as_bytes(), 0));
    let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    let router: Vec<u64> = token_bytes
        .chunks_exact((block_size as usize) * std::mem::size_of::<u32>())
        .map(|chunk| compute_hash_v2(chunk, router_seed))
        .collect();

    assert_eq!(
        producer, router,
        "producer block_hash with lora must equal compute_block_hash_for_seq with the same lora"
    );
}

// -----------------------------------------------------------------------------
// Chain step is delegated to dynamo_tokens::compute_next_sequence_hash
// -----------------------------------------------------------------------------
//
// The producer's per-block sequence_hash chain and the kv-router's request-side chain
// must agree, or the positional indexer's chain re-validation (which composes
// `compute_next_seq_hash(prev_seq, local_hash)` against the stored event's `block_hash`)
// drains at position 1+ and the radix-tree silently keeps walking by tokens_hash.
// Both sides now route through `dynamo_tokens::compute_next_sequence_hash`; verify
// the producer's `Request::sequence_hashes()` matches a manual chain built with it.
#[test]
fn request_sequence_hashes_match_canonical_chain() {
    use dynamo_tokens::compute_next_sequence_hash;

    let tokens: Vec<Token> = (1u32..=20).collect();
    let block_size: u32 = 4;

    let request = req(tokens, None, None, vec![]);
    let block_hashes = request.block_hashes(block_size).unwrap();
    let sequence_hashes = request.sequence_hashes(block_size).unwrap();
    assert_eq!(block_hashes.len(), sequence_hashes.len());
    assert!(block_hashes.len() >= 3);

    // First block: sequence_hash == block_hash.
    assert_eq!(sequence_hashes[0], block_hashes[0]);

    // Every subsequent step uses the canonical chain helper.
    for i in 1..block_hashes.len() {
        let expected = compute_next_sequence_hash(sequence_hashes[i - 1], block_hashes[i]);
        assert_eq!(
            sequence_hashes[i], expected,
            "chain step at position {i} must use compute_next_sequence_hash"
        );
    }
}
