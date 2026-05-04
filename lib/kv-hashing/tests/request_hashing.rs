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

const BS: u32 = 16;

// -----------------------------------------------------------------------------
// #1 determinism
// -----------------------------------------------------------------------------
#[test]
fn determinism_same_request_same_hashes() {
    let r1 = req((1..=48).collect(), Some("lora-a"), Some("model-x"), vec![]);
    let r2 = req((1..=48).collect(), Some("lora-a"), Some("model-x"), vec![]);
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
    let base = req((1..=48).collect(), None, None, vec![]);
    let lora_a = req((1..=48).collect(), Some("a"), None, vec![]);
    let lora_b = req((1..=48).collect(), Some("b"), None, vec![]);
    let salty = req((1..=48).collect(), None, Some("salt"), vec![]);

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
    let lora_a_again = req((1..=48).collect(), Some("a"), None, vec![]);
    let h_a2 = lora_a_again.positional_lineage_hashes(BS).unwrap();
    assert_eq!(h_a, h_a2);
}

// -----------------------------------------------------------------------------
// #3 mm_per_block: an MM run inside one block diverges only at that block (and downstream).
// -----------------------------------------------------------------------------
#[test]
fn mm_per_block_diverges_only_at_affected_block() {
    // 4 blocks of size 16 = 64 tokens. MM run inside block 3 (positions [48..56)).
    let tokens: Vec<Token> = (0..64).collect();
    let mm_a = vec![RequestMmObjectInfo {
        mm_hash: 0xAA,
        offset: 48,
        length: 8,
    }];
    let mm_b = vec![RequestMmObjectInfo {
        mm_hash: 0xBB,
        offset: 48,
        length: 8,
    }];
    let r_a = req(tokens.clone(), None, None, mm_a);
    let r_b = req(tokens.clone(), None, None, mm_b);

    let h_a = r_a.positional_lineage_hashes(BS).unwrap();
    let h_b = r_b.positional_lineage_hashes(BS).unwrap();
    assert_eq!(h_a.len(), 4);

    // Blocks 0..3 (covering positions [0..48)) are identical (prefix sharing).
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
            offset: 48,
            length: 8,
        }],
    );
    let h_a2 = r_a2.positional_lineage_hashes(BS).unwrap();
    assert_eq!(h_a, h_a2);
}

// -----------------------------------------------------------------------------
// #4 mm_spans_blocks: an MM run spanning multiple blocks. block_size=16, run starts
// at offset 16, length 40 ⇒ block 1 fully placeholder, block 2 fully placeholder,
// block 3 partial (8 placeholders + 8 reals), blocks 4..= partial-real.
// -----------------------------------------------------------------------------
#[test]
fn mm_spans_blocks() {
    let block_size: u32 = 16;
    let tokens: Vec<Token> = (0..80).collect();
    let mm = vec![RequestMmObjectInfo {
        mm_hash: 0xCAFE,
        offset: 16,
        length: 40,
    }];
    let mm_request = req(tokens.clone(), None, None, mm);
    let baseline = req(tokens, None, None, vec![]);

    let h_mm = mm_request.positional_lineage_hashes(block_size).unwrap();
    let h_base = baseline.positional_lineage_hashes(block_size).unwrap();
    assert_eq!(h_mm.len(), 5);
    assert_eq!(h_mm.len(), h_base.len());

    // Block 0 is pre-MM ⇒ matches baseline.
    assert_eq!(h_mm[0], h_base[0]);
    // Block 1 onward differs (mm coverage starts at 16).
    for i in 1..h_mm.len() {
        assert_ne!(h_mm[i], h_base[i], "block {i} should diverge from baseline");
    }

    // Block 1 (fully placeholder, run_offsets 0..15) ≠ Block 2 (fully placeholder,
    // run_offsets 16..31) — a multi-block MM run produces distinct block hashes via
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

    // Manually build the expected 208-byte tagged buffer:
    //   per slot: [tag=PLACEHOLDER (1) | run_offset u32 LE (4) | mm_hash u64 LE (8)] = 13 bytes.
    let mut expected = Vec::with_capacity(208);
    for i in 0..16u32 {
        expected.push(MM_SLOT_TAG_PLACEHOLDER);
        expected.extend_from_slice(&i.to_le_bytes());
        expected.extend_from_slice(&0xDEADBEEF_DEADBEEFu64.to_le_bytes());
    }
    assert_eq!(expected.len(), 16 * 13);
    // LocalBlockHash is salt-free (xxh3(bytes, 0)); salt enters via PLH::root_with_salt.
    let expected_block_hash = compute_block_hash(&expected);

    let blocks = request.into_blocks(block_size).unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].block_hash, expected_block_hash);
}

// -----------------------------------------------------------------------------
// Empty LoRA name normalizes to None for cache-share parity with the existing router
// behavior at lib/kv-router/src/protocols.rs:84
// (`options.lora_name.filter(|n| !n.is_empty())`). A client that sends "" must share
// cache with a client that sends None.
// -----------------------------------------------------------------------------
#[test]
fn empty_lora_normalizes_to_none() {
    let r_none = req((1..=48).collect(), None, None, vec![]);
    let r_empty = req((1..=48).collect(), Some(""), None, vec![]);
    let r_empty_salt = req((1..=48).collect(), None, Some(""), vec![]);
    assert_eq!(
        r_none.salt_hash(BS).unwrap(),
        r_empty.salt_hash(BS).unwrap()
    );
    assert_eq!(
        r_none.salt_hash(BS).unwrap(),
        r_empty_salt.salt_hash(BS).unwrap()
    );
    assert_eq!(
        r_none.positional_lineage_hashes(BS).unwrap(),
        r_empty.positional_lineage_hashes(BS).unwrap()
    );

    // Real LoRA names are still distinct.
    let r_real = req((1..=48).collect(), Some("a"), None, vec![]);
    assert_ne!(r_none.salt_hash(BS).unwrap(), r_real.salt_hash(BS).unwrap());
}

// -----------------------------------------------------------------------------
// #6 partial_tail: trailing partial block is not hashed; n_blocks = total / block_size.
// -----------------------------------------------------------------------------
#[test]
fn partial_tail_not_hashed() {
    // block_size 16, 35 tokens ⇒ 2 complete blocks, 3 trailing.
    let r = req((0..35).collect(), None, None, vec![]);
    let blocks = r.into_blocks(16).unwrap();
    assert_eq!(blocks.len(), 2);

    // With MM placeholders too: block_size 16, 19 tokens (16 real + 3 placeholders) → 1 complete block.
    let mm = vec![RequestMmObjectInfo {
        mm_hash: 1,
        offset: 16,
        length: 3,
    }];
    let r = req(vec![0u32; 19], None, None, mm);
    let blocks = r.into_blocks(16).unwrap();
    assert_eq!(blocks.len(), 1);
}

// -----------------------------------------------------------------------------
// #7 cross_check_tokens: an MM-empty Request matches dynamo_tokens::TokenBlockSequence::new
// field-for-field.
// -----------------------------------------------------------------------------
#[test]
fn cross_check_tokens_zero_mm() {
    let tokens: Vec<Token> = (1..=48).collect();
    let r = req(tokens.clone(), Some("lora-z"), Some("salty"), vec![]);
    let salt: SaltHash = r.salt_hash(BS).unwrap();

    let baseline = TokenBlockSequence::new(Tokens::from(tokens), BS, Some(salt));
    let universal = r.into_blocks(BS).unwrap();
    assert_eq!(universal.len(), baseline.blocks().len());
    for (u, b) in universal.iter().zip(baseline.blocks().iter()) {
        assert_eq!(u.position() as u64, b.position());
        assert_eq!(u.block_hash, b.block_hash());
        // Compare PLH chains: `UniversalBlock::sequence_hash` projects from PLH
        // (salt-mixed), and `TokenBlock::sequence_hash` now projects the salt-free
        // wire chain consumed by kv-router events. Read the PLH side of the
        // baseline directly to keep the cross-check at the chain-identity layer.
        assert_eq!(
            u.sequence_hash(),
            b.positional_lineage_hash().current_sequence_hash(),
        );
        assert_eq!(u.plh, b.positional_lineage_hash());
    }
    // Salt is per-request, not per-block.
    assert_eq!(r.salt_hash(BS).unwrap(), salt);
}

// -----------------------------------------------------------------------------
// #8 extension_consistency: split a sequence into prefix + full; the prefix's blocks
// match the full's first N, and the chain extends correctly via PLH alone (no
// out-of-band sequence-hash tracking).
// -----------------------------------------------------------------------------
#[test]
fn extension_consistency() {
    let tokens: Vec<Token> = (1..=80).collect();
    let prefix = req(tokens[..48].to_vec(), Some("ll"), None, vec![]);
    let full = req(tokens, Some("ll"), None, vec![]);

    let h_prefix = prefix.into_blocks(BS).unwrap();
    let h_full = full.into_blocks(BS).unwrap();
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
// Cross-block_size isolation: two requests with identical tokens but different
// `block_size` must produce disjoint sequence-level identifiers (PLH). Per-block
// content hashes (`LocalBlockHash`) are now salt-free, so they diverge across
// block_sizes purely because the byte buffers fed into XXH3 have different lengths
// (16-token vs 32-token vs 64-token chunks). At the chain level, `block_size` is
// also mixed into the per-request `SaltHash`, which feeds `PLH::root_with_salt`.
// -----------------------------------------------------------------------------
#[test]
fn cross_block_size_does_not_collide() {
    let tokens: Vec<Token> = (1..=128).collect();
    let r = req(tokens, Some("lora-a"), Some("model-x"), vec![]);

    let bh_16 = r.block_hashes(16).unwrap();
    let bh_32 = r.block_hashes(32).unwrap();
    let bh_64 = r.block_hashes(64).unwrap();

    // Cross-block_size LocalBlockHashes are unique because the token byte buffers
    // hashed by XXH3 differ in length (and content past the prefix).
    let combined: Vec<_> = bh_16
        .iter()
        .chain(bh_32.iter())
        .chain(bh_64.iter())
        .collect();
    let mut sorted = combined.clone();
    sorted.sort_by_key(|b| **b);
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        combined.len(),
        "block hashes must be unique across block_sizes"
    );

    // SaltHash itself varies with block_size, ensuring the chain (PLH) cannot collide
    // across block_sizes even when content prefixes are identical.
    let s16 = r.salt_hash(16).unwrap();
    let s32 = r.salt_hash(32).unwrap();
    let s64 = r.salt_hash(64).unwrap();
    assert_ne!(s16, s32);
    assert_ne!(s16, s64);
    assert_ne!(s32, s64);

    // PLH-level uniqueness: across all block_sizes, every PLH is distinct.
    let plh_16 = r.positional_lineage_hashes(16).unwrap();
    let plh_32 = r.positional_lineage_hashes(32).unwrap();
    let plh_64 = r.positional_lineage_hashes(64).unwrap();
    let combined_plh: Vec<_> = plh_16
        .iter()
        .chain(plh_32.iter())
        .chain(plh_64.iter())
        .collect();
    let mut sorted_plh = combined_plh.clone();
    sorted_plh.sort_by_key(|p| p.current_sequence_hash());
    sorted_plh.dedup_by_key(|p| p.current_sequence_hash());
    assert_eq!(
        sorted_plh.len(),
        combined_plh.len(),
        "PLH sequence hashes must be unique across block_sizes"
    );
}

// -----------------------------------------------------------------------------
// block_size validation: out-of-range / non-power-of-two fails fast at the salt
// boundary rather than producing silently colliding hashes.
// -----------------------------------------------------------------------------
#[test]
fn block_size_validation_rejects_bad_values() {
    let r = req((1..=32).collect(), None, None, vec![]);
    for bad in [0u32, 1, 2, 4, 8, 24, 100, 2048, 1u32 << 16] {
        match r.salt_hash(bad) {
            Err(KvHashingError::InvalidBlockSize(_)) => {}
            other => panic!("expected InvalidBlockSize for {bad}, got {other:?}"),
        }
    }
    // All accepted values round-trip.
    for ok in [16u32, 32, 64, 128, 256, 512, 1024] {
        r.salt_hash(ok).expect("valid block_size must compute");
    }
}

// -----------------------------------------------------------------------------
// UniversalBlock layout sanity: BlockHash (8 bytes) + PLH (24 bytes) + serde-derived
// extras may bump it slightly, but the PLH's struct size is fixed at 24.
// -----------------------------------------------------------------------------
#[test]
fn universal_block_struct_sizes() {
    use dynamo_kv_hashing::PositionalLineageHash;
    assert_eq!(std::mem::size_of::<PositionalLineageHash>(), 24);
}

// -----------------------------------------------------------------------------
// Bonus: Request::builder().build() validates mm_info via the same path as TokenBlockSequence.
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
