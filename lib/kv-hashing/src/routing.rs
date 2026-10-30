// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The router's block and sequence hashes, computed from token ids.
//!
//! Engines publish [`ExternalSequenceBlockHash`](dynamo_router_protocol::block_hash::ExternalSequenceBlockHash)
//! values they computed themselves; the router recomputes the same identities from the request's
//! tokens with these functions so the two sides agree. The seed and the multimodal padding rules
//! here are part of that contract. Nothing in this module reads configuration or performs IO.

use dynamo_router_protocol::block_hash::LocalBlockHash;
use dynamo_router_protocol::mm::BlockExtraInfo;
use dynamo_tokens::{SequenceHash, compute_hash_v2, compute_next_sequence_hash};
use xxhash_rust::xxh3;

use crate::compute_salt_hash;

/// Seed for XXH3 hashing, consistent with indexer.rs
pub const XXH3_SEED: u64 = 1337;

/// Compute the hash of a local block.
pub fn compute_block_hash(data: &[u8]) -> LocalBlockHash {
    LocalBlockHash(compute_hash_v2(data, XXH3_SEED))
}

/// Inputs that change a block's hash beyond its token ids: multimodal metadata, the
/// LoRA adapter, and the cache namespace. All optional; `Default` means plain tokens.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockHashOptions<'a> {
    /// Per-block multimodal metadata, one entry per block, `None` for text-only blocks.
    pub block_mm_infos: Option<&'a [Option<BlockExtraInfo>]>,
    /// LoRA adapter name folded into the salt; `None` or empty means no adapter.
    pub lora_name: Option<&'a str>,
    /// Cache namespace folded into the salt; `None` or empty means the default namespace.
    pub cache_namespace: Option<&'a str>,
    /// Whether the request uses EAGLE speculative decoding, which shifts block windows;
    /// `None` means not specified.
    pub is_eagle: Option<bool>,
}

fn block_hash_seed(options: BlockHashOptions<'_>) -> u64 {
    compute_salt_hash(options.cache_namespace, options.lora_name)
        .expect("string salt derivation is infallible")
}

#[inline]
fn hash_block_no_mm(chunk: &[u32], seed: u64, scratch_bytes: &mut Vec<u8>) -> LocalBlockHash {
    #[cfg(target_endian = "little")]
    {
        let _ = scratch_bytes;
        // SAFETY: `u32` is plain-old-data, and on little-endian targets its in-memory
        // representation matches the `to_le_bytes()` sequence used for hashing.
        let chunk_bytes = unsafe {
            std::slice::from_raw_parts(chunk.as_ptr().cast::<u8>(), std::mem::size_of_val(chunk))
        };
        LocalBlockHash(xxh3::xxh3_64_with_seed(chunk_bytes, seed))
    }

    #[cfg(not(target_endian = "little"))]
    {
        scratch_bytes.clear();
        for &token in chunk {
            scratch_bytes.extend_from_slice(&token.to_le_bytes());
        }
        LocalBlockHash(xxh3::xxh3_64_with_seed(scratch_bytes, seed))
    }
}

/// sglang's `MultimodalItem._compute_pad_value` constants — must track upstream;
/// if they drift, MM routing silently degrades to text-prefix. Pinned by
/// `pad_value_matches_sglang_protocol`.
/// Offset added to a multimodal hash before masking, matching the engine-side padding rule.
pub const MM_PAD_SHIFT_VALUE: u64 = 1_000_000;
/// Mask applied to a shifted multimodal hash to produce a padding token value.
pub const MM_PAD_HASH_MASK: u64 = (1 << 30) - 1;

/// Canonical per-image pad_value from a routing-side `mm_hash`, called by both
/// the frontend and the kv-router so request- and event-side hashes agree.
/// Keeps the low 30 bits only (sglang's limit).
pub fn pad_value_for_mm_hash(mm_hash: u64) -> u32 {
    (MM_PAD_SHIFT_VALUE + (mm_hash & MM_PAD_HASH_MASK)) as u32
}

/// Map a non-empty multimodal identifier to Dynamo's routing hash.
///
/// Preserve vLLM's canonical 64-character hex-digest mapping for compatibility,
/// and hash shorter opaque identifiers emitted by external renderers with XXH3.
pub fn hash_mm_identifier(identifier: &str) -> Option<u64> {
    if identifier.is_empty() {
        return None;
    }
    if identifier.len() == 64 && identifier.chars().all(|c| c.is_ascii_hexdigit()) {
        return u64::from_str_radix(&identifier[..16], 16).ok();
    }
    Some(xxh3::xxh3_64(identifier.as_bytes()))
}

/// Compute the hash for a sequence of tokens, optionally including multimodal metadata,
/// LoRA adapter identity, and cache namespace.
///
/// When multimodal extra info is provided, the mm_hashes are included in the hash computation
/// to ensure that blocks with identical tokens but different multimodal objects produce
/// different hashes.
///
/// When `lora_name` or `cache_namespace` is provided, those request-wide identities are
/// mixed into the XXH3 seed so blocks cached under different adapters or namespaces produce
/// distinct hashes. Empty strings are treated as absent.
pub fn compute_block_hash_for_seq(
    tokens: &[u32],
    kv_block_size: u32,
    options: BlockHashOptions<'_>,
) -> Vec<LocalBlockHash> {
    compute_block_hash_for_seq_with_seed(tokens, kv_block_size, options, block_hash_seed(options))
}

/// Count complete canonical blocks for normal and Eagle token windows.
pub fn complete_block_count(token_count: usize, kv_block_size: u32, is_eagle: bool) -> usize {
    let stride = kv_block_size as usize;
    if stride == 0 {
        return 0;
    }
    if is_eagle {
        token_count.saturating_sub(1) / stride
    } else {
        token_count / stride
    }
}

/// Compute local block hashes with an explicit XXH3 seed while preserving the
/// canonical token, multimodal, and Eagle encodings used by the public hash path.
pub fn compute_block_hash_for_seq_with_seed(
    tokens: &[u32],
    kv_block_size: u32,
    options: BlockHashOptions<'_>,
    seed: u64,
) -> Vec<LocalBlockHash> {
    let estimated_blocks = complete_block_count(
        tokens.len(),
        kv_block_size,
        options.is_eagle.unwrap_or(false),
    );
    let mut hashes = Vec::with_capacity(estimated_blocks);
    for_each_block_hash_for_seq_with_seed(tokens, kv_block_size, options, seed, |hash| {
        hashes.push(hash);
    });
    hashes
}

fn for_each_block_hash_for_seq_with_seed(
    tokens: &[u32],
    kv_block_size: u32,
    options: BlockHashOptions<'_>,
    seed: u64,
    mut visit: impl FnMut(LocalBlockHash),
) {
    if kv_block_size == 0 {
        return;
    }

    let is_eagle_flag = options.is_eagle.unwrap_or(false);
    let stride = kv_block_size as usize;
    let window_size = if is_eagle_flag { stride + 1 } else { stride };
    let mut bytes = Vec::with_capacity(window_size * std::mem::size_of::<u32>());
    let mut mm_hashes = Vec::new();
    let mut block_idx = 0;
    let mut start = 0;

    while start + window_size <= tokens.len() {
        let chunk = &tokens[start..start + window_size];
        if let Some(mm_infos) = options.block_mm_infos
            && let Some(Some(block_mm_info)) = mm_infos.get(block_idx)
        {
            bytes.clear();
            for &token in chunk {
                bytes.extend_from_slice(&token.to_le_bytes());
            }

            mm_hashes.clear();
            mm_hashes.extend(block_mm_info.mm_objects.iter().map(|obj| obj.mm_hash));
            mm_hashes.sort_unstable();

            for &mm_hash in &mm_hashes {
                bytes.extend_from_slice(&mm_hash.to_le_bytes());
            }

            visit(LocalBlockHash(xxh3::xxh3_64_with_seed(&bytes, seed)));
        } else {
            visit(hash_block_no_mm(chunk, seed, &mut bytes));
        }

        start += stride;
        block_idx += 1;
    }
}

/// Compute the next rolling sequence hash from a parent sequence hash and the
/// current block hash. Delegates to [`dynamo_tokens::compute_next_sequence_hash`] — the
/// single source of truth for the chain recurrence shared across kv-router,
/// kvbm-logical, and the universal hashing crate.
#[inline]
pub fn compute_next_seq_hash(
    parent_seq_hash: SequenceHash,
    current_block_hash: LocalBlockHash,
) -> SequenceHash {
    compute_next_sequence_hash(parent_seq_hash, current_block_hash.0)
}

/// Compute rolling sequence hashes for a vector of block hashes.
///
/// - The first block's sequence hash equals its block hash
/// - Subsequent blocks' sequence hash = hash([parent_sequence_hash, current_block_hash], seed)
pub fn compute_seq_hash_for_block(block_hashes: &[LocalBlockHash]) -> Vec<SequenceHash> {
    compute_seq_hash_for_block_with(block_hashes, compute_next_seq_hash)
}

/// Compute rolling sequence hashes directly from canonical token blocks with
/// separate XXH3 block and chain seeds, without materializing block hashes.
pub fn compute_seq_hash_for_tokens_with_seeds(
    tokens: &[u32],
    kv_block_size: u32,
    options: BlockHashOptions<'_>,
    block_seed: u64,
    chain_seed: u64,
) -> Vec<SequenceHash> {
    let estimated_blocks = complete_block_count(
        tokens.len(),
        kv_block_size,
        options.is_eagle.unwrap_or(false),
    );
    let mut sequence_hashes = Vec::with_capacity(estimated_blocks);
    for_each_block_hash_for_seq_with_seed(
        tokens,
        kv_block_size,
        options,
        block_seed,
        |block_hash| {
            let sequence_hash = sequence_hashes
                .last()
                .copied()
                .map_or(block_hash.0, |parent| {
                    compute_next_seq_hash_with_seed(parent, block_hash, chain_seed)
                });
            sequence_hashes.push(sequence_hash);
        },
    );
    sequence_hashes
}

#[inline]
fn compute_next_seq_hash_with_seed(
    parent: SequenceHash,
    block: LocalBlockHash,
    seed: u64,
) -> SequenceHash {
    let mut bytes = [0_u8; 16];
    bytes[..8].copy_from_slice(&parent.to_le_bytes());
    bytes[8..].copy_from_slice(&block.0.to_le_bytes());
    xxh3::xxh3_64_with_seed(&bytes, seed)
}

fn compute_seq_hash_for_block_with(
    block_hashes: &[LocalBlockHash],
    next_hash: impl Fn(SequenceHash, LocalBlockHash) -> SequenceHash,
) -> Vec<SequenceHash> {
    if block_hashes.is_empty() {
        return Vec::new();
    }

    let mut sequence_hashes = Vec::with_capacity(block_hashes.len());
    sequence_hashes.push(block_hashes[0].0);

    for i in 1..block_hashes.len() {
        let parent_seq_hash = sequence_hashes[i - 1];
        sequence_hashes.push(next_hash(parent_seq_hash, block_hashes[i]));
    }

    sequence_hashes
}

/// A container for tokens with lazily computed block and sequence hashes.
///
/// This struct avoids redundant hash computations by caching results:
/// - `get_or_compute_block_hashes()` computes block hashes if not cached
/// - `get_or_compute_seq_hashes()` computes seq hashes if not cached,
///   and will also compute block hashes first if needed (since seq hashes depend on them)
#[derive(Debug, Clone)]
pub struct TokensWithHashes {
    tokens: Vec<u32>,
    block_size: u32,
    block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
    lora_name: Option<String>,
    cache_namespace: Option<String>,
    block_hashes: Option<Vec<LocalBlockHash>>,
    seq_hashes: Option<Vec<SequenceHash>>,
    is_eagle: Option<bool>,
}

impl TokensWithHashes {
    /// Creates a new TokensWithHashes from tokens and block size.
    pub fn new(tokens: Vec<u32>, block_size: u32) -> Self {
        Self {
            tokens,
            block_size,
            block_mm_infos: None,
            lora_name: None,
            cache_namespace: None,
            block_hashes: None,
            seq_hashes: None,
            is_eagle: None,
        }
    }

    /// Adds multimodal extra info for blocks.
    pub fn with_mm_infos(mut self, infos: Vec<Option<BlockExtraInfo>>) -> Self {
        self.block_mm_infos = Some(infos);
        self.invalidate_hashes();
        self
    }

    /// Sets the LoRA adapter name for hash computation.
    pub fn with_lora_name(mut self, name: String) -> Self {
        self.lora_name = Some(name);
        self.invalidate_hashes();
        self
    }

    /// Sets the cache namespace for hash computation.
    pub fn with_cache_namespace(mut self, namespace: String) -> Self {
        self.cache_namespace = Some(namespace);
        self.invalidate_hashes();
        self
    }

    /// Sets Eagle hashing semantics for this token sequence.
    pub fn with_is_eagle(mut self, is_eagle: bool) -> Self {
        self.set_is_eagle(is_eagle);
        self
    }

    /// Updates Eagle hashing semantics and invalidates cached hashes when it changes.
    pub fn set_is_eagle(&mut self, is_eagle: bool) {
        let is_eagle = Some(is_eagle);
        if self.is_eagle == is_eagle {
            return;
        }

        self.is_eagle = is_eagle;
        self.invalidate_hashes();
    }

    fn invalidate_hashes(&mut self) {
        self.block_hashes = None;
        self.seq_hashes = None;
    }

    /// Returns a reference to the tokens.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Returns the number of tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns true if there are no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Returns the block size.
    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Returns the multimodal extra info, if set.
    pub fn block_mm_infos(&self) -> Option<&[Option<BlockExtraInfo>]> {
        self.block_mm_infos.as_deref()
    }

    /// Returns block hashes, computing them if not already cached.
    pub fn get_or_compute_block_hashes(&mut self) -> &[LocalBlockHash] {
        if self.block_hashes.is_none() {
            self.block_hashes = Some(compute_block_hash_for_seq(
                &self.tokens,
                self.block_size,
                BlockHashOptions {
                    block_mm_infos: self.block_mm_infos.as_deref(),
                    lora_name: self.lora_name.as_deref(),
                    cache_namespace: self.cache_namespace.as_deref(),
                    is_eagle: self.is_eagle,
                },
            ));
        }
        self.block_hashes.as_ref().unwrap()
    }

    /// Returns sequence hashes, computing them if not already cached.
    /// This will also compute block hashes if they haven't been computed yet,
    /// since sequence hashes depend on block hashes.
    pub fn get_or_compute_seq_hashes(&mut self) -> &[SequenceHash] {
        if self.seq_hashes.is_none() {
            // Ensure block hashes are computed first
            let block_hashes = self.get_or_compute_block_hashes();
            self.seq_hashes = Some(compute_seq_hash_for_block(block_hashes));
        }
        self.seq_hashes.as_ref().unwrap()
    }

    /// Returns cached block hashes without computing. Returns None if not yet computed.
    pub fn block_hashes(&self) -> Option<&[LocalBlockHash]> {
        self.block_hashes.as_deref()
    }

    /// Returns cached seq hashes without computing. Returns None if not yet computed.
    pub fn seq_hashes(&self) -> Option<&[SequenceHash]> {
        self.seq_hashes.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_router_protocol::mm::BlockMmObjectInfo;
    use rstest::rstest;

    #[test]
    fn pad_value_matches_sglang_protocol() {
        assert_eq!(MM_PAD_SHIFT_VALUE, 1_000_000);
        assert_eq!(MM_PAD_HASH_MASK, (1u64 << 30) - 1);
        assert_eq!(pad_value_for_mm_hash(0), MM_PAD_SHIFT_VALUE as u32);
        let fits = (1u64 << 30) - 1;
        assert_eq!(
            pad_value_for_mm_hash(fits),
            (MM_PAD_SHIFT_VALUE + fits) as u32
        );
        let overflow = (1u64 << 30) | 0xCAFE;
        assert_eq!(
            pad_value_for_mm_hash(overflow),
            (MM_PAD_SHIFT_VALUE + 0xCAFE) as u32,
            "high bits above the 30-bit mask must be discarded"
        );
    }
    #[test]
    fn mm_identifier_hash_preserves_canonical_vllm_digest_mapping() {
        let identifier = "0123456789abcdef".repeat(4);
        assert_eq!(hash_mm_identifier(&identifier), Some(0x0123_4567_89ab_cdef));
    }
    #[test]
    fn mm_identifier_hash_supports_opaque_identifiers() {
        let identifier = "opaque-renderer-image-0";
        assert_eq!(
            hash_mm_identifier(identifier),
            Some(xxh3::xxh3_64(identifier.as_bytes()))
        );
        assert_eq!(hash_mm_identifier(""), None);
    }

    #[rstest]
    #[case(11)]
    #[case(32)]
    #[case(64)]
    fn test_compute_block_hash_for_seq(#[case] kv_block_size: u32) {
        let sequence = (0..kv_block_size).collect::<Vec<u32>>();
        let hashes =
            compute_block_hash_for_seq(&sequence, kv_block_size, BlockHashOptions::default());
        assert_eq!(hashes.len(), 1);

        let sequence = (0..(kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes =
            compute_block_hash_for_seq(&sequence, kv_block_size, BlockHashOptions::default());
        assert_eq!(hashes.len(), 1);

        let sequence = (0..(2 * kv_block_size + 1)).collect::<Vec<u32>>();
        let hashes =
            compute_block_hash_for_seq(&sequence, kv_block_size, BlockHashOptions::default());
        assert_eq!(hashes.len(), 2);
    }
    #[test]
    fn test_compute_next_seq_hash_matches_rolling_hash() {
        let block_hashes = [LocalBlockHash(11), LocalBlockHash(22), LocalBlockHash(33)];
        let seq_hashes = compute_seq_hash_for_block(&block_hashes);

        assert_eq!(
            seq_hashes[1],
            compute_next_seq_hash(seq_hashes[0], block_hashes[1])
        );
        assert_eq!(
            seq_hashes[2],
            compute_next_seq_hash(seq_hashes[1], block_hashes[2])
        );
    }
    #[test]
    fn test_lora_name_produces_different_hash() {
        let tokens: Vec<u32> = (0..4).collect();
        let base = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let lora_a = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                lora_name: Some("adapter-a"),
                ..Default::default()
            },
        );
        let lora_b = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                lora_name: Some("adapter-b"),
                ..Default::default()
            },
        );

        assert_ne!(base[0], lora_a[0]);
        assert_ne!(base[0], lora_b[0]);
        assert_ne!(lora_a[0], lora_b[0]);
    }
    #[test]
    fn test_lora_hash_matches_kv_hashing_contract() {
        let tokens: Vec<u32> = (0..4).collect();
        let lora_name = "adapter-a";
        let actual = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                lora_name: Some(lora_name),
                ..Default::default()
            },
        );
        let token_bytes = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        let salt_hash = compute_salt_hash(None, Some(lora_name)).unwrap();
        let expected = LocalBlockHash(crate::compute_hash_v2(&token_bytes, salt_hash));

        assert_eq!(actual, vec![expected]);
    }
    #[test]
    fn test_lora_name_empty_string_normalized_to_none() {
        let tokens: Vec<u32> = (0..4).collect();
        let base = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let empty = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                lora_name: Some(""),
                ..Default::default()
            },
        );
        assert_eq!(
            base, empty,
            "empty lora_name should be treated as base model"
        );
    }
    #[test]
    fn test_cache_namespace_produces_different_hash() {
        let tokens: Vec<u32> = (0..4).collect();
        let base = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let namespace_a = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                cache_namespace: Some("tenant-a"),
                ..Default::default()
            },
        );
        let namespace_b = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                cache_namespace: Some("tenant-b"),
                ..Default::default()
            },
        );
        let lora_a = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                lora_name: Some("tenant-a"),
                ..Default::default()
            },
        );

        assert_ne!(base[0], namespace_a[0]);
        assert_ne!(base[0], namespace_b[0]);
        assert_ne!(namespace_a[0], namespace_b[0]);
        assert_ne!(
            namespace_a[0], lora_a[0],
            "namespace and lora salts must use independent seed domains"
        );
    }
    #[test]
    fn test_cache_namespace_hash_matches_kv_hashing_contract() {
        let tokens: Vec<u32> = (0..4).collect();
        let cache_namespace = "tenant-a";
        let lora_name = "adapter-a";
        let actual = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                lora_name: Some(lora_name),
                cache_namespace: Some(cache_namespace),
                ..Default::default()
            },
        );
        let token_bytes = tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect::<Vec<_>>();
        let salt_hash = compute_salt_hash(Some(cache_namespace), Some(lora_name)).unwrap();
        let expected = LocalBlockHash(crate::compute_hash_v2(&token_bytes, salt_hash));

        assert_eq!(actual, vec![expected]);
    }
    #[test]
    fn test_cache_namespace_empty_string_normalized_to_none() {
        let tokens: Vec<u32> = (0..4).collect();
        let base = compute_block_hash_for_seq(&tokens, 4, BlockHashOptions::default());
        let empty = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                cache_namespace: Some(""),
                ..Default::default()
            },
        );
        assert_eq!(
            base, empty,
            "empty cache_namespace should be treated as absent"
        );
    }
    #[test]
    fn test_tokens_with_hashes_lora() {
        let tokens: Vec<u32> = (0..8).collect();

        let mut base = TokensWithHashes::new(tokens.clone(), 4);
        let base_hashes = base.get_or_compute_block_hashes().to_vec();

        let mut with_lora =
            TokensWithHashes::new(tokens, 4).with_lora_name("my-adapter".to_string());
        let lora_hashes = with_lora.get_or_compute_block_hashes().to_vec();

        assert_eq!(base_hashes.len(), lora_hashes.len());
        for (b, l) in base_hashes.iter().zip(lora_hashes.iter()) {
            assert_ne!(b, l);
        }
    }
    #[test]
    fn test_tokens_with_hashes_lora_change_recomputes_cached_hashes() {
        let tokens: Vec<u32> = (0..8).collect();
        let mut with_hashes = TokensWithHashes::new(tokens.clone(), 4);
        let base_sequence_hashes = with_hashes.get_or_compute_seq_hashes().to_vec();

        let mut with_hashes = with_hashes.with_lora_name("my-adapter".to_string());
        let actual_block_hashes = with_hashes.get_or_compute_block_hashes().to_vec();
        let actual_sequence_hashes = with_hashes.get_or_compute_seq_hashes().to_vec();
        let expected_block_hashes = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                lora_name: Some("my-adapter"),
                ..Default::default()
            },
        );
        let expected_sequence_hashes = compute_seq_hash_for_block(&expected_block_hashes);

        assert_eq!(actual_block_hashes, expected_block_hashes);
        assert_eq!(actual_sequence_hashes, expected_sequence_hashes);
        assert_ne!(actual_sequence_hashes, base_sequence_hashes);
    }
    #[test]
    fn test_tokens_with_hashes_mm_change_recomputes_cached_hashes() {
        let tokens: Vec<u32> = (0..4).collect();
        let mm_infos = vec![
            Some(BlockExtraInfo {
                mm_objects: vec![BlockMmObjectInfo {
                    mm_hash: 42,
                    offsets: vec![(0, 1)],
                }],
            }),
            None,
        ];
        let mut with_hashes = TokensWithHashes::new(tokens.clone(), 2);
        let text_sequence_hashes = with_hashes.get_or_compute_seq_hashes().to_vec();

        let mut with_hashes = with_hashes.with_mm_infos(mm_infos.clone());
        let actual_block_hashes = with_hashes.get_or_compute_block_hashes().to_vec();
        let actual_sequence_hashes = with_hashes.get_or_compute_seq_hashes().to_vec();
        let expected_block_hashes = compute_block_hash_for_seq(
            &tokens,
            2,
            BlockHashOptions {
                block_mm_infos: Some(&mm_infos),
                ..Default::default()
            },
        );
        let expected_sequence_hashes = compute_seq_hash_for_block(&expected_block_hashes);

        assert_eq!(actual_block_hashes, expected_block_hashes);
        assert_eq!(actual_sequence_hashes, expected_sequence_hashes);
        assert_ne!(actual_sequence_hashes, text_sequence_hashes);
    }
    #[test]
    fn test_tokens_with_hashes_cache_namespace() {
        let tokens: Vec<u32> = (0..8).collect();

        let mut base = TokensWithHashes::new(tokens.clone(), 4);
        let base_hashes = base.get_or_compute_block_hashes().to_vec();

        let mut with_namespace =
            TokensWithHashes::new(tokens, 4).with_cache_namespace("tenant-a".to_string());
        let namespace_hashes = with_namespace.get_or_compute_block_hashes().to_vec();

        assert_eq!(base_hashes.len(), namespace_hashes.len());
        for (base, namespaced) in base_hashes.iter().zip(namespace_hashes.iter()) {
            assert_ne!(base, namespaced);
        }
    }
    #[test]
    fn test_tokens_with_hashes_cache_namespace_change_recomputes_cached_hashes() {
        let tokens: Vec<u32> = (0..8).collect();
        let mut with_hashes = TokensWithHashes::new(tokens.clone(), 4);
        let base_sequence_hashes = with_hashes.get_or_compute_seq_hashes().to_vec();

        let mut with_hashes = with_hashes.with_cache_namespace("tenant-a".to_string());
        let actual_block_hashes = with_hashes.get_or_compute_block_hashes().to_vec();
        let actual_sequence_hashes = with_hashes.get_or_compute_seq_hashes().to_vec();
        let expected_block_hashes = compute_block_hash_for_seq(
            &tokens,
            4,
            BlockHashOptions {
                cache_namespace: Some("tenant-a"),
                ..Default::default()
            },
        );
        let expected_sequence_hashes = compute_seq_hash_for_block(&expected_block_hashes);

        assert_eq!(actual_block_hashes, expected_block_hashes);
        assert_eq!(actual_sequence_hashes, expected_sequence_hashes);
        assert_ne!(actual_sequence_hashes, base_sequence_hashes);
    }
    #[test]
    fn test_compute_block_hash_for_seq_eagle_windows() {
        let tokens: Vec<u32> = (0..6).collect();

        let default_hashes = compute_block_hash_for_seq(&tokens, 2, BlockHashOptions::default());
        let eagle_hashes = compute_block_hash_for_seq(
            &tokens,
            2,
            BlockHashOptions {
                is_eagle: Some(true),
                ..Default::default()
            },
        );
        let expected_first = compute_block_hash_for_seq(
            &[0, 1, 2],
            2,
            BlockHashOptions {
                is_eagle: Some(true),
                ..Default::default()
            },
        );
        let expected_second = compute_block_hash_for_seq(
            &[2, 3, 4],
            2,
            BlockHashOptions {
                is_eagle: Some(true),
                ..Default::default()
            },
        );

        assert_eq!(default_hashes.len(), 3);
        assert_eq!(eagle_hashes.len(), 2);
        assert_eq!(eagle_hashes, vec![expected_first[0], expected_second[0]]);
        assert_ne!(default_hashes[0], eagle_hashes[0]);
    }
    #[test]
    fn test_tokens_with_hashes_set_is_eagle_invalidates_cache() {
        let tokens: Vec<u32> = (0..6).collect();
        let mut with_hashes = TokensWithHashes::new(tokens, 2);

        let default_hashes = with_hashes.get_or_compute_block_hashes().to_vec();
        with_hashes.set_is_eagle(true);
        let eagle_hashes = with_hashes.get_or_compute_block_hashes().to_vec();
        let expected_first = compute_block_hash_for_seq(
            &[0, 1, 2],
            2,
            BlockHashOptions {
                is_eagle: Some(true),
                ..Default::default()
            },
        );
        let expected_second = compute_block_hash_for_seq(
            &[2, 3, 4],
            2,
            BlockHashOptions {
                is_eagle: Some(true),
                ..Default::default()
            },
        );

        assert_eq!(default_hashes.len(), 3);
        assert_eq!(eagle_hashes.len(), 2);
        assert_eq!(eagle_hashes, vec![expected_first[0], expected_second[0]]);
        assert_ne!(default_hashes[0], eagle_hashes[0]);
    }
}
