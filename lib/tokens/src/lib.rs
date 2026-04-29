// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![deny(missing_docs)]

//! Types and utilities for handling sequences of tokens, including block creation and hashing.

use bytemuck::cast_slice;
use derive_getters::Dissolve;
use std::ops::Range;

pub mod blocks;
mod radix;
pub use radix::PositionalRadixTree;

/// Trait for hashes that include position information.
pub trait PositionalHash {
    /// Returns the position associated with the hash.
    fn position(&self) -> u64;
}

/// A token is represented as a 32-bit unsigned integer.
pub type Token = u32;

/// A salt used for hashing, represented as a vector of bytes.
/// This might encode model architecture, weights, PEFT info, etc.
pub type Salt = Vec<u8>;

/// A 64-bit hash of the salt. Computed once per request and used as the seed for
/// every block-hash computation in that request.
///
/// The canonical construction path is [`compute_salt_hash_from_bytes`] (or
/// `dynamo_kv_hashing::Request::salt_hash` at the application layer).
///
/// TODO(universal-hashing): promote back to a `pub struct SaltHash(pub u64)` newtype
/// once the `lib/kvbm-*` call-sites are ready to migrate. Reverted to a type alias
/// here to keep the kvbm-* surface source-compatible with `main` while
/// `lib/{tokens,kv-hashing}` evolve. See `.claude/plans/the-blast-radius-on-radiant-valley.md`.
pub type SaltHash = u64;

/// A 64-bit content-only hash computed from the tokens within a single block (with
/// optional MM frames). **Not** seeded by [`SaltHash`] — request salt mixes in only at
/// [`PositionalLineageHash::root_with_salt`], which keeps `BlockHash` content-addressable
/// across requests (two requests with identical token contents at the same block
/// position share this hash, regardless of LoRA / salt / block_size).
///
/// The canonical construction path is [`compute_block_hash`].
///
/// TODO(universal-hashing): promote back to a `pub struct BlockHash(pub u64)` newtype
/// once the `lib/kvbm-*` call-sites are ready to migrate (same rollout as
/// [`SaltHash`]).
pub type BlockHash = u64;

/// Alias clarifying that this hash is local to a block and contains no request-level
/// state (salt, LoRA, block_size). Identical token contents always produce the same
/// `LocalBlockHash`.
pub type LocalBlockHash = BlockHash;

/// A 64-bit sequence-aware hash. At the root the chain hash is
/// `xxh3([salt_u64, local_block_hash], 0)` (see
/// [`PositionalLineageHash::root_with_salt`]); at every subsequent position it is
/// `xxh3([parent_seq, local_block_hash], 0)`. Salt influences only the root and
/// propagates through `parent_seq` thereafter, so each chain step uses seed `0`.
///
/// Stays a type alias because `SequenceHash` flows through downstream protocol crates
/// (kvbm-engine sessions, kv-router events) where a wider newtype rollout is a
/// separate, larger refactor.
pub type SequenceHash = u64;

/// Computes a hash of the data using the given seed (raw u64).
///
/// Prefer [`compute_block_hash`] / [`compute_salt_hash_from_bytes`] for typed
/// construction; this raw-u64 form is kept for low-level callers.
pub fn compute_hash_v2(data: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh3::xxh3_64_with_seed(data, seed)
}

/// Fixed XXH3 seed used by [`compute_block_hash`]. Shared with
/// `dynamo_kv_router::protocols::XXH3_SEED` so the per-block content hash produced
/// here is byte-identical to the router's `compute_block_hash_for_seq` for the
/// no-MM, no-LoRA path. Both crates must agree on this constant; there is no
/// per-request component.
pub const LOCAL_BLOCK_HASH_SEED: u64 = 1337;

/// Canonical [`BlockHash`] (a [`LocalBlockHash`]) construction: XXH3 over the
/// per-block byte buffer (already encoded by [`compute_block_bytes_with_mm`] or
/// `cast_slice` for the no-MM path) with the fixed [`LOCAL_BLOCK_HASH_SEED`].
///
/// **Salt is not folded in here.** Request-level salt enters the chain only at the
/// root via [`PositionalLineageHash::root_with_salt`]. The fixed seed keeps the
/// per-block content hash request-independent so the same tokens at the same
/// position produce the same `BlockHash` across requests, while remaining
/// compatible with the kv-router's `tokens_hash` indexing.
#[inline]
pub fn compute_block_hash(block_bytes: &[u8]) -> BlockHash {
    compute_hash_v2(block_bytes, LOCAL_BLOCK_HASH_SEED)
}

/// Validate that `block_size` is non-zero.
///
/// Universal-hashing PLH flags can only encode power-of-two block_size in `1..=2^15`,
/// but [`TokenBlockSequence`] constructors are also exercised by `lib/kvbm-*` tests
/// that pass arbitrary `tokens.len()` as block_size — the strict pre-flight
/// validation is therefore enforced inside [`PlhFlags::new`] (the universal-hashing
/// path) rather than at the `TokenBlockSequence` boundary. Here we only catch the
/// truly fatal case: a zero block_size would make `chunks_exact(0)` infinite-loop.
///
/// # Panics
///
/// Panics if `block_size` is zero.
#[inline]
fn assert_valid_block_size(block_size: u32) {
    assert!(block_size > 0, "block_size must be non-zero");
}

/// Canonical [`SaltHash`] construction from a pre-canonicalized salt-payload byte
/// buffer. Application-layer callers should use `dynamo_kv_hashing::Request::salt_hash`
/// which canonicalizes `(salt, lora_name)` first; this function is the low-level path.
#[inline]
pub fn compute_salt_hash_from_bytes(payload: &[u8]) -> SaltHash {
    compute_hash_v2(payload, 0)
}

/// Metadata describing a single multimodal placeholder run within a token sequence.
///
/// A run occupies `length` consecutive slots starting at `offset`. The token IDs at
/// those slot positions are opaque for hashing — the `(mm_hash, run_offset)` pair drives
/// the per-slot bytes during block formation.
///
/// See [`compute_block_bytes_with_mm`] for the byte-encoding rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TokenBlockMmInfo {
    /// Hash identifying the multimodal object (image / audio / etc.).
    pub mm_hash: u64,
    /// Start position of the placeholder run in the full token sequence (zero-based).
    pub offset: usize,
    /// Number of placeholder slots in the run.
    pub length: usize,
}

/// Slot-tag byte distinguishing real-token slots from multimodal placeholder slots in
/// the per-block byte buffer. See [`compute_block_bytes_with_mm`].
pub const MM_SLOT_TAG_TOKEN: u8 = 0x00;
/// Slot-tag byte for multimodal placeholder slots. See [`compute_block_bytes_with_mm`].
pub const MM_SLOT_TAG_PLACEHOLDER: u8 = 0x01;

impl TokenBlockMmInfo {
    /// Returns the exclusive end position of this run, or `None` on `usize` overflow.
    #[inline]
    pub fn checked_end(&self) -> Option<usize> {
        self.offset.checked_add(self.length)
    }

    /// Returns the exclusive end position of this run.
    ///
    /// # Panics
    /// Panics if `offset + length` overflows `usize`. Prefer [`Self::checked_end`] in
    /// validation paths; this helper is for already-validated runs.
    #[inline]
    pub fn end(&self) -> usize {
        self.checked_end()
            .expect("TokenBlockMmInfo::end overflowed usize; run was not validated")
    }

    /// Returns `true` if the given absolute position falls inside this run.
    /// Returns `false` if the run's end overflows `usize` (such a run is invalid; use
    /// [`validate_and_sort_mm_info`] before relying on this method).
    #[inline]
    pub fn covers(&self, position: usize) -> bool {
        match self.checked_end() {
            Some(end) => position >= self.offset && position < end,
            None => false,
        }
    }
}

/// Errors raised while validating [`TokenBlockMmInfo`] inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum MmInfoError {
    /// The run extends past the end of the token sequence.
    #[error(
        "mm_info range starting at {offset} (length {length}) exceeds tokens length {tokens_len}"
    )]
    OutOfBounds {
        /// Run start.
        offset: usize,
        /// Run length.
        length: usize,
        /// Length of the token sequence the run was validated against.
        tokens_len: usize,
    },
    /// `offset + length` overflows `usize`.
    #[error("mm_info range starting at {offset} (length {length}) overflows usize")]
    OffsetOverflow {
        /// Run start.
        offset: usize,
        /// Run length.
        length: usize,
    },
    /// Two runs overlap.
    #[error("mm_info ranges overlap at position {position}")]
    Overlapping {
        /// Position where the overlap begins.
        position: usize,
    },
    /// A run has zero length.
    #[error("mm_info length must be greater than zero")]
    EmptyRun,
}

/// Validates `mm_info` against `tokens_len` and returns a copy sorted by `offset`.
///
/// Validation rules:
/// - Every run must have `length > 0`.
/// - `offset + length` must not overflow `usize`.
/// - Every run's end (`offset + length`) must be `<= tokens_len`.
/// - No two runs may overlap.
pub fn validate_and_sort_mm_info(
    mm_info: &[TokenBlockMmInfo],
    tokens_len: usize,
) -> Result<Vec<TokenBlockMmInfo>, MmInfoError> {
    let mut sorted: Vec<TokenBlockMmInfo> = mm_info.to_vec();
    sorted.sort_by_key(|m| m.offset);
    let mut prev_end = 0usize;
    for m in &sorted {
        if m.length == 0 {
            return Err(MmInfoError::EmptyRun);
        }
        let end = m
            .offset
            .checked_add(m.length)
            .ok_or(MmInfoError::OffsetOverflow {
                offset: m.offset,
                length: m.length,
            })?;
        if end > tokens_len {
            return Err(MmInfoError::OutOfBounds {
                offset: m.offset,
                length: m.length,
                tokens_len,
            });
        }
        if m.offset < prev_end {
            return Err(MmInfoError::Overlapping { position: m.offset });
        }
        prev_end = end;
    }
    Ok(sorted)
}

/// Returns `true` if any run in `mm_runs` overlaps the block `[block_offset, block_offset + len)`.
/// `mm_runs` must be validated and sorted.
fn block_has_mm(block_offset: usize, len: usize, mm_runs: &[TokenBlockMmInfo]) -> bool {
    let block_end = block_offset.saturating_add(len);
    mm_runs
        .iter()
        .any(|m| m.offset < block_end && m.end() > block_offset)
}

/// Builds the byte buffer used to compute a block's [`BlockHash`].
///
/// **Two encodings, picked per-block:**
///
/// 1. **Legacy / zero-MM** — when no run in `mm_runs` overlaps this block, the buffer is
///    `bytemuck::cast_slice(tokens)` (4 bytes per slot, LE u32). This is the same per-slot
///    layout used by every zero-MM [`TokenBlock`] in `dynamo_tokens` and kvbm; combined
///    with the fixed-seed [`compute_block_hash`] it keeps the per-block content hash
///    request-independent.
///
/// 2. **Tagged / MM-affected** — when at least one run overlaps this block, every slot
///    emits a fixed 13-byte frame:
///    - Real-token slot: `[MM_SLOT_TAG_TOKEN | token_id u32 LE | 0u64 LE]`
///      The trailing `0u64` is **frame padding only** — it has no semantic meaning,
///      it is there so the real-token frame matches the placeholder frame's width.
///      Token IDs at placeholder positions are *ignored* by this encoder; whether
///      a slot is a placeholder is determined solely by `mm_runs[run_idx].covers(g)`.
///    - Placeholder slot: `[MM_SLOT_TAG_PLACEHOLDER | run_offset u32 LE | mm_hash u64 LE]`,
///      where `run_offset = (block_offset + s) - run.offset`.
///
///    The 1-byte tag plus the fixed-width frame make the encoding self-delimiting and
///    slot-position-preserving: two MM-affected byte buffers compare equal iff they describe
///    the same `(slot_kind, slot_payload)` sequence at the same slot positions.
///
/// The two encodings have different per-slot widths (4 vs 13), so an all-tokens block can
/// never produce the same byte buffer as an MM-affected block — eliminating cross-encoding
/// collisions.
///
/// `mm_runs` must be validated and sorted (typically the output of
/// [`validate_and_sort_mm_info`]).
pub fn compute_block_bytes_with_mm(
    tokens: &[Token],
    block_offset: usize,
    mm_runs: &[TokenBlockMmInfo],
) -> Vec<u8> {
    // Defense-in-depth: routing of each slot to the placeholder vs real-token branch
    // depends on `mm_runs` being sorted by offset and non-overlapping. The public
    // entry points (`Request::new`, `TokenBlockSequence::new_with_mm`,
    // `split_tokens_with_mm`) enforce this via `validate_and_sort_mm_info`, but this
    // function is also pub so we re-check in debug to catch direct misuse.
    debug_assert!(
        mm_runs.windows(2).all(|w| w[0].end() <= w[1].offset),
        "compute_block_bytes_with_mm: mm_runs must be sorted by offset and non-overlapping (use validate_and_sort_mm_info)",
    );
    debug_assert!(
        mm_runs.iter().all(|r| r.length > 0),
        "compute_block_bytes_with_mm: mm_runs must have non-zero length",
    );

    if !block_has_mm(block_offset, tokens.len(), mm_runs) {
        // Zero-MM-affecting-this-block: use the legacy encoding so the resulting block_hash
        // matches what TokenBlockSequence::new would produce.
        return cast_slice::<Token, u8>(tokens).to_vec();
    }

    const FRAME: usize = 13;
    let mut out: Vec<u8> = Vec::with_capacity(tokens.len() * FRAME);
    let mut run_idx = 0usize;
    // Skip runs that ended at or before this block starts.
    while run_idx < mm_runs.len() && mm_runs[run_idx].end() <= block_offset {
        run_idx += 1;
    }
    for (s, &tok) in tokens.iter().enumerate() {
        let g = block_offset + s;
        // Advance past runs that end at or before g (validated non-overlapping => monotonic).
        while run_idx < mm_runs.len() && mm_runs[run_idx].end() <= g {
            run_idx += 1;
        }
        if run_idx < mm_runs.len() && mm_runs[run_idx].covers(g) {
            let run = &mm_runs[run_idx];
            let run_offset = (g - run.offset) as u32;
            out.push(MM_SLOT_TAG_PLACEHOLDER);
            out.extend_from_slice(&run_offset.to_le_bytes());
            out.extend_from_slice(&run.mm_hash.to_le_bytes());
        } else {
            out.push(MM_SLOT_TAG_TOKEN);
            out.extend_from_slice(&tok.to_le_bytes());
            out.extend_from_slice(&0u64.to_le_bytes());
        }
    }
    out
}

/// A 128-bit positional sequence hash combining traditional sequence hash with positional information.
///
/// Layout:
/// - Lower 64 bits: Traditional SequenceHash
/// - Upper 64 bits: 2-bit mode + position + LocalBlockHash (BlockHash)
///
/// Modes (automatically selected based on position):
/// - Mode 00: 8-bit position (max 255) + 54-bit LBH
/// - Mode 01: 16-bit position (max 65,535) + 46-bit LBH
/// - Mode 10: 24-bit position (max 16,777,215) + 38-bit LBH
/// - Mode 11: 31-bit position (max 2,147,483,647) + 31-bit LBH
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
pub struct PositionalSequenceHash(u128);

impl PositionalSequenceHash {
    /// Creates a new PositionalSequenceHash from components.
    ///
    /// The mode is automatically selected based on the position value to use the minimal
    /// representation that can fit the position.
    pub fn new(sequence_hash: SequenceHash, position: u64, local_block_hash: BlockHash) -> Self {
        let mode = Self::select_mode(position);
        let upper = Self::encode_upper(mode, position, local_block_hash);
        let value = ((upper as u128) << 64) | (sequence_hash as u128);
        PositionalSequenceHash(value)
    }

    /// Returns the sequence hash component (lower 64 bits).
    pub fn sequence_hash(&self) -> SequenceHash {
        (self.0 & 0xFFFF_FFFF_FFFF_FFFF) as u64
    }

    /// Returns the block position.
    pub fn position(&self) -> u64 {
        let (_, position, _) = self.decode_upper();
        position
    }

    /// Returns the local block hash (BlockHash) component.
    pub fn local_block_hash(&self) -> BlockHash {
        let (_, _, lbh) = self.decode_upper();
        lbh
    }

    /// Returns the mode used for encoding (0, 1, 2, or 3).
    pub fn mode(&self) -> u8 {
        let (mode, _, _) = self.decode_upper();
        mode
    }

    /// Returns the inner 128-bit value.
    #[inline(always)]
    pub fn as_u128(&self) -> u128 {
        self.0
    }

    /// Selects the minimal mode that can represent the given position.
    fn select_mode(position: u64) -> u8 {
        if position < (1u64 << 8) {
            0 // Mode 00: 8-bit position
        } else if position < (1u64 << 16) {
            1 // Mode 01: 16-bit position
        } else if position < (1u64 << 24) {
            2 // Mode 10: 24-bit position
        } else if position < (1u64 << 31) {
            3 // Mode 11: 31-bit position
        } else {
            panic!(
                "Position {} exceeds maximum supported value (2^31 - 1)",
                position
            );
        }
    }

    /// Encodes the upper 64 bits from mode, position, and local block hash.
    fn encode_upper(mode: u8, position: u64, local_block_hash: u64) -> u64 {
        let (position_bits, lbh_bits) = match mode {
            0 => (8, 54),  // 2 + 8 + 54 = 64
            1 => (16, 46), // 2 + 16 + 46 = 64
            2 => (24, 38), // 2 + 24 + 38 = 64
            3 => (31, 31), // 2 + 31 + 31 = 64
            _ => unreachable!(
                "Invalid mode {} when encoding PositionalSequenceHash; mode must be 0, 1, 2, or 3",
                mode
            ),
        };

        // Create masks for extracting the relevant bits
        let position_mask = (1u64 << position_bits) - 1;
        let lbh_mask = (1u64 << lbh_bits) - 1;

        // Extract and position components
        let position_part = position & position_mask;
        let lbh_part = local_block_hash & lbh_mask;

        // Combine: [mode (2 bits)][position (X bits)][lbh (R bits)]
        ((mode as u64) << 62) | (position_part << lbh_bits) | lbh_part
    }

    /// Decodes the upper 64 bits into (mode, position, local_block_hash).
    fn decode_upper(&self) -> (u8, u64, u64) {
        let upper = (self.0 >> 64) as u64;

        // Extract mode from top 2 bits
        let mode = (upper >> 62) as u8;

        let (position_bits, lbh_bits) = match mode {
            0 => (8, 54),
            1 => (16, 46),
            2 => (24, 38),
            3 => (31, 31),
            _ => unreachable!(
                "Invalid mode {} in PositionalSequenceHash - value may be corrupted",
                mode
            ),
        };

        // Create masks
        let lbh_mask = (1u64 << lbh_bits) - 1;
        let position_mask = (1u64 << position_bits) - 1;

        // Extract components
        let lbh = upper & lbh_mask;
        let position = (upper >> lbh_bits) & position_mask;

        (mode, position, lbh)
    }
}

impl std::fmt::Debug for PositionalSequenceHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PositionalSequenceHash")
            .field("sequence_hash", &self.sequence_hash())
            .field("local_block_hash", &self.local_block_hash())
            .field("position", &self.position())
            .finish()
    }
}

/// Schema/version tag stamped into every canonically-constructed [`PositionalLineageHash`]
/// (one whose `current` is reachable from `root_with_salt` followed by zero or more `extend`
/// calls).
///
/// `0x00` is reserved as a "zeroed/uninitialized" sentinel so a default-constructed
/// PLH is detectably invalid; current canonical layout uses `0x01`.
pub const PLH_SCHEMA_V1: u8 = 0x01;

/// Schema tag stamped into [`PositionalLineageHash`]es minted via
/// [`PositionalLineageHash::synthetic_unique`]. Distinct from [`PLH_SCHEMA_V1`] so a
/// downstream consumer can tell, from the flags alone, that a PLH was *not* produced by
/// a real chain — useful for invariants ("no synthetic PLH should reach component X")
/// and diagnostics. The high bit (`0x80`) is set so the value is also visually
/// distinct from the V1 tag at a glance.
pub const PLH_SCHEMA_SYNTHETIC: u8 = 0x80;

/// Typed view of [`PositionalLineageHash`]'s 32-bit `flags` slot.
///
/// Bit layout (low → high):
/// - bits  0..=3   `log2_block_size` (4 bits, fits `block_size` up to `2^15`)
/// - bits  4..=11  `schema` (8 bits, [`PLH_SCHEMA_V1`] = `0x01`)
/// - bits 12..=31  `feature_flags` (20 reserved bits)
///
/// Wrapping these bits in a newtype keeps the bit layout out of constructor call sites
/// and makes future feature-flag additions a typed API change rather than a magic-number
/// shift. The on-the-wire encoding is unchanged.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct PlhFlags(u32);

impl PlhFlags {
    /// Builds a flags word from a power-of-two `block_size` (≤ `2^15`), a `schema`
    /// version byte, and 20 reserved feature-flag bits (high bits truncated).
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is zero, not a power of two, or greater than `2^15`.
    #[inline]
    pub fn new(block_size: u32, schema: u8, feature_flags: u32) -> Self {
        assert!(
            block_size > 0 && block_size.is_power_of_two() && block_size <= (1u32 << 15),
            "block_size must be a power of two in 1..=32768, got {block_size}",
        );
        let log2_bs = block_size.trailing_zeros() & 0xF;
        let schema_bits = (schema as u32) << 4;
        let feature_bits = (feature_flags & 0x000F_FFFF) << 12;
        Self(log2_bs | schema_bits | feature_bits)
    }

    /// Builds a flags word with the canonical schema and zero feature bits.
    ///
    /// Tolerates non-power-of-two `block_size` for back-compat with `lib/kvbm-*` tests
    /// that pass arbitrary token-length values; the recorded `log2_block_size` falls
    /// back to the next-lower power of two in that case (the 4-bit field cannot
    /// represent the true value). Universal-hashing call-sites that need strict
    /// validation should call [`Self::new`] directly.
    #[inline]
    pub fn for_block_size(block_size: u32) -> Self {
        let bs = block_size.max(1);
        let pow2 = if bs.is_power_of_two() {
            bs.min(1u32 << 15)
        } else {
            // Round down to the nearest power of two; cap at 2^15.
            let log2_floor = (u32::BITS - 1 - bs.leading_zeros()).min(15);
            1u32 << log2_floor
        };
        Self::new(pow2, PLH_SCHEMA_V1, 0)
    }

    /// Reconstructs a [`PlhFlags`] from a raw 32-bit value (deserialization path).
    /// No validation; callers are responsible for ensuring the bits are meaningful.
    #[inline]
    pub fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the encoded `block_size` decoded from the 4-bit `log2_block_size`.
    #[inline]
    pub fn block_size(self) -> u32 {
        1u32 << (self.0 & 0xF)
    }

    /// Returns the schema/version byte.
    #[inline]
    pub fn schema(self) -> u8 {
        ((self.0 >> 4) & 0xFF) as u8
    }

    /// Returns the 20-bit reserved feature-flags region (right-shifted to bits 0..=19).
    #[inline]
    pub fn feature_flags(self) -> u32 {
        self.0 >> 12
    }

    /// Returns the raw 32-bit value.
    #[inline]
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Returns `true` if the schema field matches the canonical [`PLH_SCHEMA_V1`].
    #[inline]
    pub fn is_canonical_schema(self) -> bool {
        self.schema() == PLH_SCHEMA_V1
    }
}

impl std::fmt::Debug for PlhFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlhFlags")
            .field("block_size", &self.block_size())
            .field("schema", &self.schema())
            .field(
                "feature_flags",
                &format_args!("0x{:05x}", self.feature_flags()),
            )
            .finish()
    }
}

/// A 24-byte positional lineage hash carrying full-width parent and current sequence
/// hashes plus position and per-block metadata.
///
/// Layout (`#[repr(C)]`, naturally aligned):
///
/// ```text
///   off 0:  current  (u64)  — full chain hash for this block
///   off 8:  parent   (u64)  — parent's chain hash (0 at position 0)
///   off 16: position (u32)  — block index in the sequence
///   off 20: flags    (u32)  — packed metadata, low → high:
///             bits  0..=3   log2_block_size  (4 bits)
///             bits  4..=11  schema           (8 bits, [`PLH_SCHEMA_V1`] = 0x01)
///             bits 12..=31  reserved feature flags (20 bits)
/// ```
///
/// `block_size` is required to be a power of two; `log2(block_size)` fits in 4 bits
/// up through `2^15 = 32768`. Production cache-safety enforcement (16..=1024) lives in
/// the salt layer ([`crate`]/`kv-hashing` `compute_salt_hash`); the PLH constructor
/// only enforces the encodability invariant (power of two ≤ 2^15).
///
/// PLH is self-contained for chain extension: a child PLH can be derived from a
/// parent PLH plus the child's [`BlockHash`] alone (see [`Self::extend`]).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub struct PositionalLineageHash {
    current: u64,
    parent: u64,
    position: u32,
    flags: PlhFlags,
}

impl PositionalLineageHash {
    /// Reconstruct a [`PositionalLineageHash`] from its raw fields without validation.
    ///
    /// Intended for deserialization paths and tests that need to inject specific bit
    /// patterns. Callers are responsible for ensuring the resulting struct is meaningful.
    #[inline]
    pub fn from_raw_parts(current: u64, parent: u64, position: u32, flags: u32) -> Self {
        Self {
            current,
            parent,
            position,
            flags: PlhFlags::from_raw(flags),
        }
    }

    /// Creates a root [`PositionalLineageHash`] (position 0) by mixing the request's
    /// [`SaltHash`] into the chain at the root only.
    ///
    /// Formula: `current = xxh3([salt.0, local_block_hash.0], 0)`. Subsequent
    /// [`Self::extend`] calls are unchanged — salt baked into `current` propagates
    /// through every `parent_seq` in the chain. Two sequences with identical tokens
    /// but different salts diverge starting at this root; identical tokens + identical
    /// salt produce identical chains. `LocalBlockHash` itself stays request-independent.
    ///
    /// Pass [`SaltHash`]`(0)` for an unsalted root: that is the *only* canonical
    /// construction for the "no salt" case, matching what
    /// [`crate::TokenBlockSequence::new`] produces when `salt_hash` is `None`. There
    /// is intentionally no `root(local, block_size)` shortcut — it would set
    /// `current = local_block_hash.0`, which silently diverges from
    /// `root_with_salt(local, SaltHash(0), bs)` and would split the chain identity
    /// of two callers that both meant "no salt."
    pub fn root_with_salt(
        local_block_hash: LocalBlockHash,
        salt: SaltHash,
        block_size: u32,
    ) -> Self {
        let current = compute_hash_v2(cast_slice(&[salt, local_block_hash]), 0);
        Self {
            current,
            parent: 0,
            position: 0,
            flags: PlhFlags::for_block_size(block_size),
        }
    }

    /// Mints a non-canonical PLH carrying a randomly-generated `current`,
    /// `parent = 0`, and the [`PLH_SCHEMA_SYNTHETIC`] schema tag in flags
    /// (instead of [`PLH_SCHEMA_V1`]). Two calls produce PLHs that compare
    /// unequal with overwhelming probability.
    ///
    /// **Not produced by any real chain.** This is deliberately *not*
    /// `Self::new` — there is no public path to a PLH with caller-chosen
    /// `current` / `parent` outside of [`Self::from_raw_parts`] (the serde
    /// door). Use [`Self::is_synthetic`] / [`Self::is_canonical_schema`] to
    /// detect.
    ///
    /// Sole legitimate caller today: the mocker's prefix-caching-disabled
    /// path, where the kvbm-logical registry is still consulted for
    /// same-sequence preempt-retry reuse and therefore needs *something*
    /// PLH-shaped to register. This is a transitional API; the durable shape
    /// is to skip registration entirely on that path, at which point this
    /// constructor goes away. See the TODOs in
    /// `lib/mocker/src/common/sequence.rs` and
    /// `lib/mocker/src/kv_manager/kvbm_backend.rs`.
    pub fn synthetic_unique(position: u32, block_size: u32) -> Self {
        // Use thread-local RNG via `rand::random` to keep this dep-free of a
        // direct `rand::Rng` import.
        let current: u64 = rand::random();
        Self {
            current,
            parent: 0,
            position,
            flags: PlhFlags::new(block_size, PLH_SCHEMA_SYNTHETIC, 0),
        }
    }

    /// Returns `true` when this PLH was minted via [`Self::synthetic_unique`]
    /// (its flags carry the [`PLH_SCHEMA_SYNTHETIC`] tag).
    #[inline]
    pub fn is_synthetic(&self) -> bool {
        self.schema() == PLH_SCHEMA_SYNTHETIC
    }

    /// Extends this lineage by one block, producing the child PLH.
    ///
    /// The chain recurrence is `xxh3([parent_seq_u64, child_local_block_hash], 0)`.
    /// Salt does not seed the per-step xxh3 — it was mixed into `current` at the root
    /// (via [`Self::root_with_salt`]) and propagates through every parent thereafter.
    /// The child inherits this PLH's `flags` (and therefore `block_size`).
    ///
    /// # Panics
    ///
    /// Panics if `self.position == u32::MAX`. Silently wrapping to 0 would corrupt
    /// lineage identity — `position` is part of the PLH's identity surface and a
    /// wrap collides chains across positional boundaries.
    pub fn extend(&self, child_local_block_hash: LocalBlockHash) -> Self {
        let parent_seq = self.current;
        let child_seq = compute_hash_v2(cast_slice(&[parent_seq, child_local_block_hash]), 0);
        let next_position = self
            .position
            .checked_add(1)
            .expect("PositionalLineageHash::extend: position overflowed u32::MAX");
        Self {
            current: child_seq,
            parent: parent_seq,
            position: next_position,
            flags: self.flags,
        }
    }

    /// Returns the block position.
    ///
    /// Returns `u64` for source compatibility with the pre-universal-hashing API used
    /// by `lib/kvbm-*`. The internal storage is `u32`; this widens at the boundary.
    /// Callers that want the native width can read [`Self::position_u32`].
    #[inline]
    pub fn position(&self) -> u64 {
        self.position as u64
    }

    /// Returns the block position as `u32`, the native storage width.
    #[inline]
    pub fn position_u32(&self) -> u32 {
        self.position
    }

    /// Deprecated 3-arg constructor preserved for `lib/kvbm-*` source compatibility
    /// with the pre-universal-hashing API. New code should use [`Self::from_raw_parts`]
    /// or [`Self::root_with_salt`] / [`Self::extend`].
    ///
    /// `parent` defaults to `0` when `None` (matching the root convention). The flags
    /// word is built with the canonical schema and a default `block_size` of 16, which
    /// matches what every existing call-site on `main` was implicitly assuming.
    #[deprecated(note = "use PositionalLineageHash::from_raw_parts, root_with_salt, or extend")]
    pub fn new(current: u64, parent: Option<u64>, position: u64) -> Self {
        const DEFAULT_BLOCK_SIZE: u32 = 16;
        Self {
            current,
            parent: parent.unwrap_or(0),
            position: position as u32,
            flags: PlhFlags::for_block_size(DEFAULT_BLOCK_SIZE),
        }
    }

    /// Returns the full 64-bit sequence hash of the current block.
    #[inline]
    pub fn current_sequence_hash(&self) -> SequenceHash {
        self.current
    }

    /// Returns the parent sequence hash, or `None` at the root (position 0 with zero parent).
    #[inline]
    pub fn parent_sequence_hash(&self) -> Option<SequenceHash> {
        if self.position == 0 {
            None
        } else {
            Some(self.parent)
        }
    }

    /// Returns the raw parent slot (0 at the root).
    #[inline]
    pub fn parent_raw(&self) -> u64 {
        self.parent
    }

    /// Returns the typed flags word.
    #[inline]
    pub fn flags(&self) -> PlhFlags {
        self.flags
    }

    /// Returns the block size encoded in `flags` (decoded from the 4-bit `log2_block_size`).
    #[inline]
    pub fn block_size(&self) -> u32 {
        self.flags.block_size()
    }

    /// Returns the schema/version tag encoded in `flags`.
    #[inline]
    pub fn schema(&self) -> u8 {
        self.flags.schema()
    }

    /// Returns the reserved 20-bit feature-flags region of `flags`.
    #[inline]
    pub fn feature_flags(&self) -> u32 {
        self.flags.feature_flags()
    }

    /// Returns the raw 32-bit `flags` word.
    #[inline]
    pub fn flags_raw(&self) -> u32 {
        self.flags.raw()
    }

    /// Returns `true` if the schema field matches the canonical [`PLH_SCHEMA_V1`].
    #[inline]
    pub fn is_canonical_schema(&self) -> bool {
        self.flags.is_canonical_schema()
    }

    // ----- Backwards-compatible shims for callers written against the prior
    // truncated-fragment layout. With the flat 24-byte layout, parent is full width:
    // these now return the full u64 parent hash and the full u64 current hash.
    // Lookup tables keyed by (position, parent_fragment) keep working — the
    // fragment is now the complete hash, which is strictly stronger.

    /// Returns the parent's sequence hash. Now full-width (no truncation).
    ///
    /// At the root this returns 0 (callers should gate with [`Self::position`]
    /// or [`Self::parent_sequence_hash`] when distinguishing root from non-root).
    #[inline]
    pub fn parent_hash_fragment(&self) -> u64 {
        self.parent
    }

    /// Returns this PLH's current sequence hash. The legacy "fragment for child
    /// position" semantics (truncating to a child-mode-specific width) no longer
    /// applies — the parent slot is always full width.
    #[inline]
    pub fn parent_fragment_for_child_position(&self, _child_position: u32) -> u64 {
        self.current
    }

    /// Deprecated alias for the current sequence hash, preserved for `lib/kvbm-*`
    /// source compatibility with the pre-universal-hashing API. Returns
    /// [`Self::current_sequence_hash`] (full u64 — the legacy mode-specific
    /// truncation no longer applies).
    #[deprecated(note = "use current_sequence_hash()")]
    #[inline]
    pub fn current_hash_fragment(&self) -> u64 {
        self.current
    }

    /// Returns a 128-bit fingerprint composed of `(current << 64) | parent`.
    ///
    /// Compatibility shim for callers that previously consumed the packed u128
    /// representation (e.g. frequency-tracker keys). Two PLHs with the same
    /// current+parent collide here, so prefer direct struct equality for
    /// authoritative comparisons.
    #[inline]
    pub fn as_u128(&self) -> u128 {
        ((self.current as u128) << 64) | (self.parent as u128)
    }
}

// `Hash` is implemented manually so only the 64-bit `current` field is fed to the
// hasher: `current` is already a hash. Combined with [`PositionalLineageHash`]'s
// own [`std::hash::Hasher`] impl below, this lets `HashMap<PLH, V,
// BuildHasherDefault<PLH>>` skip a redundant hash pass.
impl std::hash::Hash for PositionalLineageHash {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.current);
    }
}

// Passthrough hasher: `write` panics so any accidental fall-through to byte-by-byte
// hashing (e.g. if the manual `Hash` impl above is changed) is loud, not silent.
impl std::hash::Hasher for PositionalLineageHash {
    fn finish(&self) -> u64 {
        self.current
    }
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("PositionalLineageHash should only call write_u64");
    }
    fn write_u64(&mut self, i: u64) {
        self.current = i;
    }
}

impl PositionalLineageHash {
    fn format_impl(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let current_b58 = bs58::encode(self.current.to_be_bytes()).into_string();
        if self.position == 0 {
            write!(f, "{}:{}", self.position, current_b58)
        } else {
            let parent_b58 = bs58::encode(self.parent.to_be_bytes()).into_string();
            write!(f, "{}:{}:{}", self.position, current_b58, parent_b58)
        }
    }
}

impl std::fmt::Debug for PositionalLineageHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_impl(f)
    }
}

impl std::fmt::Display for PositionalLineageHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format_impl(f)
    }
}

/// A collection of tokens, represented as a `Vec<Token>`.
///
/// Provides convenience methods for conversion and manipulation.
#[derive(Debug, Clone, Dissolve, Default, Eq)]
pub struct Tokens(Vec<Token>);

impl AsRef<[Token]> for Tokens {
    fn as_ref(&self) -> &[Token] {
        &self.0
    }
}

impl std::ops::Deref for Tokens {
    type Target = [Token];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::borrow::Borrow<[Token]> for Tokens {
    fn borrow(&self) -> &[Token] {
        &self.0
    }
}

impl From<Vec<Token>> for Tokens {
    fn from(tokens: Vec<Token>) -> Self {
        Tokens(tokens)
    }
}

impl From<&[Token]> for Tokens {
    fn from(tokens: &[Token]) -> Self {
        Tokens(tokens.to_vec())
    }
}

impl From<Vec<usize>> for Tokens {
    fn from(tokens: Vec<usize>) -> Self {
        Tokens(
            tokens
                .into_iter()
                .map(|t| t.try_into().expect("Token ID exceeds u32::MAX"))
                .collect(),
        )
    }
}

impl From<Vec<i32>> for Tokens {
    /// Converts `Vec<i32>` to `Tokens`, casting each `i32` to `u32`.
    fn from(tokens: Vec<i32>) -> Self {
        Tokens(tokens.into_iter().map(|t| t as u32).collect())
    }
}

impl From<&[i32]> for Tokens {
    /// Converts `&[i32]` to `Tokens`, casting each `i32` to `u32`.
    fn from(tokens: &[i32]) -> Self {
        Tokens(tokens.iter().map(|&t| t as u32).collect())
    }
}

impl From<Tokens> for Vec<Token> {
    fn from(tokens: Tokens) -> Self {
        tokens.0
    }
}

// PartialEq implementations for comparing Tokens with Vec<Token> and &[Token]
// (Generated implementations are usually sufficient, but explicit ones can be clearer)
impl PartialEq<Vec<Token>> for Tokens {
    fn eq(&self, other: &Vec<Token>) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Tokens> for Vec<Token> {
    fn eq(&self, other: &Tokens) -> bool {
        *self == other.0
    }
}

impl PartialEq<[Token]> for Tokens {
    fn eq(&self, other: &[Token]) -> bool {
        self.0.as_slice() == other
    }
}

impl PartialEq<Tokens> for &[Token] {
    fn eq(&self, other: &Tokens) -> bool {
        *self == other.0.as_slice()
    }
}

impl PartialEq for Tokens {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

// Add PartialEq<&[T]> where T: Into<Token> + Copy could be more general,
// but specifically implementing for &[Token] is sufficient for the tests.
impl PartialEq<&[Token]> for Tokens {
    fn eq(&self, other: &&[Token]) -> bool {
        self.0.as_slice() == *other
    }
}

impl Tokens {
    /// Consumes the [`Tokens`] object and creates a [`TokenBlockSequence`].
    ///
    /// The sequence is initialized with the provided tokens, splitting them into blocks
    /// of the specified `block_size` using the given `salt_hash` (or 0 if `None`).
    ///
    /// # Arguments
    ///
    /// * `block_size` - The fixed size for each [`TokenBlock`].
    /// * `salt_hash` - An optional [`SaltHash`] used as the base seed for hashing. Defaults to 0.
    pub fn into_sequence(self, block_size: u32, salt_hash: Option<SaltHash>) -> TokenBlockSequence {
        TokenBlockSequence::new(self, block_size, salt_hash)
    }
}

/// Errors that can occur during [`PartialTokenBlock`] operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum TokenBlockError {
    /// The operation could not be completed because the block is full.
    #[error("TokenBlock is full")]
    Full,

    /// The operation requires a full block, but the block is incomplete.
    #[error("TokenBlock is incomplete")]
    Incomplete,

    /// The operation could not be completed because the block is empty.
    #[error("TokenBlock is empty")]
    Empty,

    /// The operation requires more tokens than are currently in the block.
    #[error("TokenBlock has insufficient tokens")]
    InsufficientTokens,

    /// Multimodal info validation failed.
    #[error(transparent)]
    MmInfo(#[from] MmInfoError),

    /// A mutating operation is not supported on a sequence with multimodal runs.
    #[error("operation is not supported on a TokenBlockSequence with multimodal runs")]
    MmRunsPresent,
}

/// Represents a partially filled block of tokens within a sequence.
///
/// This structure accumulates tokens until it reaches the specified `block_size`,
/// at which point it can be [`commit`](PartialTokenBlock::commit)ted into a full [`TokenBlock`].
///
/// The chain context is carried as the parent's [`PositionalLineageHash`]; at the
/// sequence root `parent_plh` is `None` and the resulting first block's PLH will be
/// constructed via [`PositionalLineageHash::root_with_salt`].
#[derive(Debug, PartialEq)] // No Clone: intended to be unique within a sequence
pub struct PartialTokenBlock {
    tokens: Tokens,
    block_size: u32,
    salt_hash: SaltHash,
    parent_plh: Option<PositionalLineageHash>,
    position: usize, // The position this block will have when committed
}

impl PartialTokenBlock {
    /// Creates the first partial block (root) for a new sequence.
    ///
    /// # Arguments
    ///
    /// * `block_size` - The fixed size for blocks in this sequence.
    /// * `salt_hash` - The [`SaltHash`] for the sequence.
    pub(crate) fn create_sequence_root(block_size: u32, salt_hash: SaltHash) -> Self {
        assert_valid_block_size(block_size);
        Self {
            tokens: Tokens::default(),
            block_size,
            salt_hash,
            parent_plh: None,
            position: 0,
        }
    }

    /// Attempts to push multiple tokens onto the block from a [`Tokens`] object.
    ///
    /// Tokens are added until the block is full or all input tokens are consumed.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The [`Tokens`] to push.
    ///
    /// # Returns
    ///
    /// A new [`Tokens`] object containing any tokens that did not fit,
    /// if all tokens were added, the returned object will be empty.
    pub(crate) fn push_tokens(&mut self, tokens: Tokens) -> Tokens {
        let remaining_space = self.remaining();

        if remaining_space == 0 {
            return tokens; // Block is already full
        }

        if tokens.0.len() <= remaining_space {
            // All tokens fit
            self.tokens.0.extend(tokens.0);
            Tokens::default() // No remaining tokens
        } else {
            // Only some tokens fit
            let (to_add, remaining) = tokens.0.split_at(remaining_space);
            self.tokens.0.extend_from_slice(to_add);
            Tokens(remaining.to_vec()) // Return the leftover tokens
        }
    }

    /// Attempts to remove the last `count` tokens from the block.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of tokens to remove.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the specified number of tokens were successfully removed.
    /// * `Err(TokenBlockError::InsufficientTokens)` - If `count` is greater than the number of tokens in the block.
    pub(crate) fn pop_tokens(&mut self, count: usize) -> Result<(), TokenBlockError> {
        if self.tokens.0.len() < count {
            return Err(TokenBlockError::InsufficientTokens);
        }
        self.tokens.0.truncate(self.tokens.0.len() - count);
        Ok(())
    }

    /// Attempts to commit the current partial block into a full [`TokenBlock`].
    ///
    /// This operation consumes the tokens within the partial block.
    /// After a successful commit, this `PartialTokenBlock` instance is reset
    /// to represent the *next* partial block in the sequence, inheriting the
    /// sequence hash from the block just committed.
    ///
    /// # Returns
    ///
    /// * `Ok(TokenBlock)` - The newly created full [`TokenBlock`].
    /// * `Err(TokenBlockError::Incomplete)` - If the block does not contain exactly `block_size` tokens.
    pub fn commit(&mut self) -> Result<TokenBlock, TokenBlockError> {
        if self.tokens.0.len() != self.block_size as usize {
            // Check for exact size match for committing
            return Err(TokenBlockError::Incomplete);
        }

        // Take ownership of the tokens, leaving the internal tokens empty
        let tokens = std::mem::take(&mut self.tokens);

        let chunk = TokenBlockChunk::new(tokens);
        let block = TokenBlock::from_chunk(chunk, self.parent_plh, self.salt_hash, self.block_size);

        // Reset self to be the next block in the sequence
        self.parent_plh = Some(block.positional_lineage_hash());
        self.position += 1;
        // self.tokens is already empty due to mem::take
        // self.block_size and self.salt_hash remain the same

        Ok(block)
    }

    /// Returns the number of additional tokens required to fill the block.
    pub fn remaining(&self) -> usize {
        // Use saturating_sub to prevent underflow if len somehow exceeds block_size
        (self.block_size as usize).saturating_sub(self.tokens.0.len())
    }

    /// Returns the number of tokens currently in the block.
    pub fn len(&self) -> usize {
        self.tokens.0.len()
    }

    /// Returns `true` if the block contains no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.0.is_empty()
    }

    /// Returns a reference to the tokens currently in the block.
    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    /// Returns the parent block's [`PositionalLineageHash`], or `None` at the
    /// sequence root (no committed predecessor yet).
    pub fn parent_plh(&self) -> Option<PositionalLineageHash> {
        self.parent_plh
    }

    /// Returns the parent block's [`SequenceHash`] projected from `parent_plh`, or
    /// `None` at the sequence root.
    pub fn parent_sequence_hash(&self) -> Option<SequenceHash> {
        self.parent_plh.map(|p| p.current_sequence_hash())
    }
}

// Deref allows treating &PartialTokenBlock like &Tokens for read-only access.
impl std::ops::Deref for PartialTokenBlock {
    type Target = Tokens;

    fn deref(&self) -> &Self::Target {
        &self.tokens
    }
}

/// An intermediate structure holding a chunk of tokens destined to become a [`TokenBlock`].
///
/// Computes the content-only [`LocalBlockHash`] (a salt-free [`BlockHash`]) but does not
/// produce the parent-chained [`SequenceHash`] / [`PositionalLineageHash`]; chunk
/// production can therefore happen independently of chain context (and in parallel).
#[derive(Debug)] // No Clone: temporary intermediate value
struct TokenBlockChunk {
    tokens: Tokens,
    local_block_hash: LocalBlockHash,
}

impl TokenBlockChunk {
    /// Creates a new chunk from [`Tokens`] using the legacy zero-MM byte encoding.
    fn new(tokens: Tokens) -> Self {
        let local_block_hash = compute_block_hash(cast_slice(&tokens));
        Self {
            tokens,
            local_block_hash,
        }
    }

    /// Creates a new chunk from a slice of `&[Token]` using the zero-MM byte encoding.
    fn from_tokens(tokens: &[Token]) -> Self {
        let local_block_hash = compute_block_hash(cast_slice(tokens));
        Self {
            tokens: tokens.into(),
            local_block_hash,
        }
    }

    /// Creates a new chunk from a slice of `&[Token]` using the multimodal-aware byte
    /// encoding from [`compute_block_bytes_with_mm`]. `block_offset` is the global token
    /// index where this block starts; `mm_runs` must be pre-validated and sorted.
    fn from_tokens_with_mm(
        tokens: &[Token],
        block_offset: usize,
        mm_runs: &[TokenBlockMmInfo],
    ) -> Self {
        let bytes = compute_block_bytes_with_mm(tokens, block_offset, mm_runs);
        let local_block_hash = compute_block_hash(&bytes);
        Self {
            tokens: tokens.into(),
            local_block_hash,
        }
    }
}

/// Represents a completed, immutable block of tokens with associated hashes.
///
/// Contains exactly `block_size` tokens. The chain hash, parent linkage, position, and
/// per-block flags are all carried by an embedded [`PositionalLineageHash`]; the
/// content-only [`LocalBlockHash`] and the per-request [`SaltHash`] sit alongside it
/// for projection (the salt is preserved for accessor compatibility but does not
/// participate in `local_block_hash` — it is mixed into the chain only at
/// [`PositionalLineageHash::root_with_salt`]).
#[derive(Debug, Clone, Default, PartialEq)] // PartialEq for tests
pub struct TokenBlock {
    tokens: Tokens,
    salt_hash: SaltHash,
    local_block_hash: LocalBlockHash,
    plh: PositionalLineageHash,
}

impl TokenBlock {
    /// Creates a new [`PartialTokenBlock`] representing the block immediately following this one.
    ///
    /// The new partial block carries this block's PLH as its `parent_plh` so the
    /// chain extends correctly on the next commit.
    pub fn next_block(&self) -> PartialTokenBlock {
        PartialTokenBlock {
            tokens: Tokens::default(),
            block_size: self.tokens.len() as u32,
            salt_hash: self.salt_hash,
            parent_plh: Some(self.plh),
            position: self.position() as usize + 1,
        }
    }

    /// Finalizes a [`TokenBlock`] from a [`TokenBlockChunk`] and the parent's
    /// [`PositionalLineageHash`] (or `None` at the sequence root).
    ///
    /// At the root, `plh = PLH::root_with_salt(local, salt, block_size)`. Subsequent
    /// blocks chain via `parent_plh.extend(local)`.
    fn from_chunk(
        chunk: TokenBlockChunk,
        parent_plh: Option<PositionalLineageHash>,
        salt_hash: SaltHash,
        block_size: u32,
    ) -> Self {
        let plh = match parent_plh {
            Some(parent) => parent.extend(chunk.local_block_hash),
            None => {
                PositionalLineageHash::root_with_salt(chunk.local_block_hash, salt_hash, block_size)
            }
        };
        Self {
            tokens: chunk.tokens,
            salt_hash,
            local_block_hash: chunk.local_block_hash,
            plh,
        }
    }

    /// Returns a reference to the tokens in this block.
    pub fn tokens(&self) -> &Tokens {
        &self.tokens
    }

    /// Returns the salt hash carried by this block (preserved for accessor
    /// compatibility; it does not participate in the per-block content hash).
    pub fn salt_hash(&self) -> SaltHash {
        self.salt_hash
    }

    /// Returns the content-only [`LocalBlockHash`] for this block.
    pub fn block_hash(&self) -> LocalBlockHash {
        self.local_block_hash
    }

    /// Returns the parent-chained sequence hash for this block (projected from PLH).
    pub fn sequence_hash(&self) -> SequenceHash {
        self.plh.current_sequence_hash()
    }

    /// Returns the sequence hash of the preceding block, if any (projected from PLH).
    pub fn parent_sequence_hash(&self) -> Option<SequenceHash> {
        self.plh.parent_sequence_hash()
    }

    /// Returns the number of tokens in the block.
    pub fn block_size(&self) -> usize {
        self.tokens.0.len()
    }

    /// Returns the positional lineage hash for this block.
    pub fn positional_lineage_hash(&self) -> PositionalLineageHash {
        self.plh
    }

    /// Returns the position of this block in the sequence.
    pub fn position(&self) -> u64 {
        self.plh.position() as u64
    }
}

impl PositionalHash for PositionalSequenceHash {
    fn position(&self) -> u64 {
        self.position()
    }
}

impl PositionalHash for PositionalLineageHash {
    fn position(&self) -> u64 {
        self.position() as u64
    }
}

/// Represents a sequence of tokens, segmented into fixed-size, hashed blocks.
///
/// This structure manages a series of completed [`TokenBlock`]s and one
/// [`PartialTokenBlock`] for accumulating incoming tokens.
/// It provides methods for appending tokens (`append`, `extend`), removing tokens
/// (`pop`, `truncate`, `unwind`), and accessing sequence information.
///
/// Hashing incorporates an initial [`SaltHash`] at the chain root to ensure uniqueness
/// across different contexts (e.g., different models, PEFTs). Per-block content hashes
/// stay request-independent.
///
/// Key Hashes:
/// - [`LocalBlockHash`] (`BlockHash`): salt-free hash of tokens within a single block.
/// - [`SequenceHash`]: at the root, `xxh3([salt, local_block_hash], 0)`; at every
///   subsequent position, `xxh3([parent_sequence_hash, local_block_hash], 0)`.
/// - [`PositionalLineageHash`]: the canonical chain identity carrying `current`,
///   `parent`, position, and per-block flags.
#[derive(Debug, PartialEq)]
pub struct TokenBlockSequence {
    blocks: Vec<TokenBlock>,
    current_block: PartialTokenBlock,
    salt_hash: SaltHash,
    block_size: usize,
    /// Validated, sorted multimodal runs covering committed and partial slots.
    /// Empty for sequences built via the zero-MM constructors; populated by
    /// [`TokenBlockSequence::new_with_mm`] and the streaming MM helpers.
    mm_runs: Vec<TokenBlockMmInfo>,
}

impl TokenBlockSequence {
    /// Creates a new [`TokenBlockSequence`] from an initial set of tokens.
    ///
    /// The tokens are split into blocks of `block_size`. Any remaining tokens
    /// form the initial `current_block`.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The initial [`Tokens`] for the sequence.
    /// * `block_size` - The fixed size for each [`TokenBlock`]. Must be greater than 0.
    /// * `salt_hash` - An optional [`SaltHash`]. Defaults to 0 if `None`.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is not a power of two in `1..=32768`.
    /// todo(maybe): deprecate and use a builder pattern instead
    pub fn new(tokens: Tokens, block_size: u32, salt_hash: Option<SaltHash>) -> Self {
        assert_valid_block_size(block_size);
        // todo: enforce block_size is a power of 2
        // todo: generate our core SaltHash from block_size, schema_version, and optional user_salt_hash
        let salt_hash = salt_hash.unwrap_or_default();
        let (blocks, current_block) = Self::split_tokens(&tokens, block_size, salt_hash);

        Self {
            blocks,
            current_block,
            salt_hash,
            block_size: block_size as usize,
            mm_runs: Vec::new(),
        }
    }

    /// Extends the sequence with the given tokens, potentially completing multiple blocks.
    ///
    /// This method processes all tokens from the input [`Tokens`] object.
    /// If adding tokens causes one or more blocks to become full, they are committed
    /// and added to the internal list of completed blocks.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The [`Tokens`] object containing the tokens to extend the sequence with.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(Range<usize>))` - The range of indices in the `blocks` vector corresponding
    ///   to the blocks completed during this `extend` operation.
    /// * `Ok(None)` - If no blocks were completed.
    /// * `Err(TokenBlockError)` - If an internal error occurs during commit.
    pub fn extend(&mut self, tokens: Tokens) -> Result<Option<Range<usize>>, TokenBlockError> {
        let start_block_index = self.blocks.len();
        let mut tokens_to_append = tokens;

        while !tokens_to_append.is_empty() {
            let remaining_in_current = self.current_block.remaining();

            if remaining_in_current == 0 {
                // Current block is full, commit it first.
                let new_block = self.commit_current()?;
                self.blocks.push(new_block);
                // Continue loop to add tokens to the *new* current_block.
            }

            // Push as many tokens as possible into the current (potentially new) block.
            let available_tokens = tokens_to_append;
            tokens_to_append = self.current_block.push_tokens(available_tokens);

            // Check if the current block *became* full after pushing tokens.
            if self.current_block.remaining() == 0 {
                // If it became full AND there are still more tokens to append,
                // commit it now so the next loop iteration starts with a fresh block.
                let new_block = self.commit_current()?;
                self.blocks.push(new_block);
            }
        }

        let end_block_index = self.blocks.len();
        if start_block_index == end_block_index {
            Ok(None) // No blocks were completed.
        } else {
            Ok(Some(start_block_index..end_block_index))
        }
    }

    /// Commits the current partial block.
    ///
    /// Routes through the MM-aware byte encoding when [`Self::mm_runs`] is non-empty;
    /// otherwise behaves identically to [`PartialTokenBlock::commit`].
    fn commit_current(&mut self) -> Result<TokenBlock, TokenBlockError> {
        if self.mm_runs.is_empty() {
            return self.current_block.commit();
        }
        // MM-aware path: compute LocalBlockHash from the substituted byte buffer.
        if self.current_block.tokens.0.len() != self.current_block.block_size as usize {
            return Err(TokenBlockError::Incomplete);
        }
        let block_offset = self.blocks.len() * (self.current_block.block_size as usize);
        let tokens = std::mem::take(&mut self.current_block.tokens);
        let chunk = TokenBlockChunk::from_tokens_with_mm(&tokens, block_offset, &self.mm_runs);
        // `from_tokens_with_mm` clones tokens via `tokens.into()`; reuse the original
        // owned `Tokens` for the block to avoid the extra allocation.
        let chunk = TokenBlockChunk {
            tokens,
            local_block_hash: chunk.local_block_hash,
        };
        let block = TokenBlock::from_chunk(
            chunk,
            self.current_block.parent_plh,
            self.current_block.salt_hash,
            self.current_block.block_size,
        );
        self.current_block.parent_plh = Some(block.positional_lineage_hash());
        self.current_block.position += 1;
        Ok(block)
    }

    /// Appends a single token to the sequence.
    ///
    /// If adding this token completes the current partial block, the block is committed,
    /// and the index of the newly completed block is returned.
    ///
    /// This method is equivalent to calling [`extend`] with a single-token [`Tokens`] object.
    ///
    /// # Arguments
    ///
    /// * `token` - The [`Token`] to append.
    ///
    /// # Returns
    ///
    /// * `Ok(Some(usize))` - The index of the block that was just completed.
    /// * `Ok(None)` - No block was completed by adding this token.
    /// * `Err(TokenBlockError)` - If an internal error occurs during processing.
    pub fn append(&mut self, token: Token) -> Result<Option<usize>, TokenBlockError> {
        // Create a single-token Tokens object
        let tokens = Tokens::from(vec![token]);

        // Call extend
        let range_option = self.extend(tokens)?;

        // Convert the range to Option<usize>
        match range_option {
            None => Ok(None),
            Some(range) => {
                // Since we only added one token, the range can only be empty or have one element.
                // If it's not empty, it must be `n..(n+1)`.
                assert_eq!(
                    range.len(),
                    1,
                    "Appending a single token completed more than one block, which should be impossible."
                );
                Ok(Some(range.start))
            }
        }
    }

    /// Shortens the sequence, keeping the first `len` tokens and removing the rest.
    ///
    /// If `len` is greater than the sequence's current length, this has no effect.
    ///
    /// This operation is analogous to `Vec::truncate`.
    /// It may involve removing tokens from the current partial block, removing entire
    /// completed blocks, and adjusting the current partial block
    /// to reflect the new end of the sequence.
    ///
    /// # Arguments
    ///
    /// * `len` - The number of tokens to keep.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the sequence was successfully truncated.
    /// * `Err(TokenBlockError::InsufficientTokens)` - This error should ideally not occur if `len`
    ///   is correctly checked against `total_tokens`, but the underlying `pop_tokens` might return it.
    pub fn truncate(&mut self, len: usize) -> Result<(), TokenBlockError> {
        if !self.mm_runs.is_empty() {
            return Err(TokenBlockError::MmRunsPresent);
        }
        let current_total_len = self.total_tokens();
        if len >= current_total_len {
            return Ok(()); // Nothing to truncate
        }

        let n = current_total_len - len; // Number of tokens to remove

        // This inner block handles the actual removal logic based on `n` tokens to remove.
        {
            let current_len = self.current_block.len();
            // Avoid division by zero if block_size is somehow 0 (though asserted in new)
            let block_size = self.current_block.block_size.max(1);

            if n <= current_len {
                // Only need to pop from the current partial block
                self.current_block.pop_tokens(n)?;
            } else {
                // Need to pop from full blocks as well
                let tokens_to_pop_from_blocks = n - current_len;

                // Calculate how many blocks are affected (including the one partially popped)
                let num_blocks_to_affect = tokens_to_pop_from_blocks.div_ceil(block_size as usize);

                // Check if we need to pop more blocks than available (should be prevented by initial len check)
                if num_blocks_to_affect > self.blocks.len() {
                    // This indicates an inconsistency between total_tokens() and internal state.
                    debug_assert!(
                        false,
                        "Truncate calculation error: trying to pop too many blocks."
                    );
                    return Err(TokenBlockError::InsufficientTokens);
                }

                // Determine the index of the block that will be the source for the new partial block
                let source_block_index = self.blocks.len() - num_blocks_to_affect;

                // Calculate how many tokens to keep from that source block
                let num_full_blocks_completely_popped = num_blocks_to_affect - 1;
                let num_tokens_to_pop_from_source_block = tokens_to_pop_from_blocks
                    - num_full_blocks_completely_popped * block_size as usize;
                let num_tokens_to_keep_in_new_partial =
                    (block_size as usize).saturating_sub(num_tokens_to_pop_from_source_block);

                // Get the tokens for the new partial block
                let new_partial_tokens = if num_tokens_to_keep_in_new_partial > 0 {
                    self.blocks[source_block_index].tokens().as_ref()
                        [..num_tokens_to_keep_in_new_partial]
                        .to_vec()
                } else {
                    Vec::new()
                };

                // Truncate the blocks vector to remove popped blocks
                self.blocks.truncate(source_block_index);

                // Update the current_block state
                self.current_block.tokens = Tokens(new_partial_tokens);
                // Reattach to the *new* last block's PLH, or detach to root if no blocks remain.
                self.current_block.parent_plh =
                    self.blocks.last().map(|b| b.positional_lineage_hash());
                // Update position to match the number of complete blocks
                self.current_block.position = self.blocks.len();
                // salt_hash and block_size remain the same for current_block
            }
        }
        Ok(())
    }

    /// Removes the last `count` tokens from the sequence.
    ///
    /// This is a convenience method that calculates the required length and calls [`truncate`].
    ///
    /// # Arguments
    ///
    /// * `count` - The number of tokens to remove from the end.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the tokens were successfully removed.
    /// * `Err(TokenBlockError::InsufficientTokens)` - If `count` is greater than or equal to
    ///   the total number of tokens in the sequence.
    pub fn unwind(&mut self, count: usize) -> Result<(), TokenBlockError> {
        let current_total_len = self.total_tokens();
        if count > current_total_len {
            // Allow count == current_total_len, which truncates to 0.
            return Err(TokenBlockError::InsufficientTokens);
        }

        // number of tokens remaining in the sequence after undoing the given count
        let len = current_total_len - count;
        self.truncate(len)
    }

    /// Resets the sequence to the initial state.
    ///
    /// Clears any accumulated multimodal runs; after `reset` the sequence behaves
    /// identically to a freshly-constructed zero-MM sequence with the same `salt_hash`
    /// and `block_size`.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.current_block =
            PartialTokenBlock::create_sequence_root(self.block_size as u32, self.salt_hash);
        self.mm_runs.clear();
    }

    /// Removes the last token from the sequence and returns it, or [`None`] if it is empty.
    ///
    /// This operation is analogous to `Vec::pop`.
    ///
    /// # Returns
    ///
    /// * `Some(Token)` - The last token, if the sequence was not empty.
    /// * `None` - If the sequence was empty.
    ///
    /// # Panics
    ///
    /// Panics if the sequence has accumulated multimodal runs (see [`Self::try_pop`] for a
    /// non-panicking variant). `pop` returns `Option<Token>` and cannot signal
    /// "operation unsupported on MM sequence" through its return type without breaking the
    /// `Vec::pop` analogy and silently lying to callers.
    pub fn pop(&mut self) -> Option<Token> {
        if !self.mm_runs.is_empty() {
            panic!(
                "TokenBlockSequence::pop is not supported on a sequence with multimodal runs; \
                 use try_pop or reset before pop"
            );
        }
        let current_total_len = self.total_tokens();
        if current_total_len == 0 {
            return None;
        }

        // Determine the last token. It must be in the current_block if current_block is not empty.
        // If current_block is empty, it must be the last token of the last full block.
        let last_token = if !self.current_block.tokens.is_empty() {
            // Last token is in the partial block
            *self
                .current_block
                .tokens
                .last()
                .expect("Current block checked for non-empty")
        } else {
            // Current block is empty, sequence is not. Must be in the last full block.
            let last_block = self
                .blocks
                .last()
                .expect("Sequence is not empty but has no blocks and empty current block?");
            *last_block
                .tokens()
                .last()
                .expect("Last block cannot be empty")
        };

        // Truncate the sequence by one element.
        // We expect this to succeed since we know the length > 0.
        match self.truncate(current_total_len - 1) {
            Ok(_) => Some(last_token),
            Err(_) => {
                // This should be logically impossible if total_tokens() and truncate() are correct.
                // Panic in debug, return None in release as a fallback, though it indicates a bug.
                debug_assert!(
                    false,
                    "truncate failed unexpectedly after checking length in pop"
                );
                None
            }
        }
    }

    /// Non-panicking variant of [`Self::pop`].
    ///
    /// Returns:
    /// - `Ok(Some(token))` when the sequence had at least one token and pop succeeded.
    /// - `Ok(None)` when the sequence was empty.
    /// - `Err(TokenBlockError::MmRunsPresent)` when the sequence has multimodal runs.
    pub fn try_pop(&mut self) -> Result<Option<Token>, TokenBlockError> {
        if !self.mm_runs.is_empty() {
            return Err(TokenBlockError::MmRunsPresent);
        }
        Ok(self.pop())
    }

    /// Returns a slice containing all the completed [`TokenBlock`]s in the sequence.
    pub fn blocks(&self) -> &[TokenBlock] {
        &self.blocks
    }

    /// Returns a reference to the last completed [`TokenBlock`] in the sequence, if any.
    pub fn last_complete_block(&self) -> Option<&TokenBlock> {
        self.blocks.last()
    }

    /// Returns a reference to the current [`PartialTokenBlock`] where new tokens are added.
    pub fn current_block(&self) -> &PartialTokenBlock {
        &self.current_block
    }

    /// Consumes the sequence and returns its parts: a `Vec` of completed blocks and the final partial block.
    pub fn into_parts(self) -> (Vec<TokenBlock>, PartialTokenBlock) {
        (self.blocks, self.current_block)
    }

    /// Returns the block size used for this sequence.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the [`SaltHash`] used for this sequence.
    pub fn salt_hash(&self) -> SaltHash {
        self.salt_hash
    }

    /// Returns the total number of tokens in the sequence (sum of tokens in all completed blocks
    /// plus tokens in the current partial block).
    pub fn total_tokens(&self) -> usize {
        let block_size = self.current_block.block_size as usize;
        (self.blocks.len() * block_size) + self.current_block.len()
    }

    /// Extract the token with the range
    pub fn tokens_at(&self, range: Range<usize>) -> Tokens {
        let total = self.total_tokens();

        // Validate range - return empty tokens for invalid ranges
        if range.start > range.end || range.end > total {
            return Tokens::default();
        }

        // Handle empty range
        if range.is_empty() {
            return Tokens::default();
        }

        let mut result = Vec::with_capacity(range.len());

        for i in range {
            if i < self.blocks.len() * self.block_size {
                // Token is in a completed block
                let block_index = i / self.block_size;
                let token_index = i % self.block_size;
                result.push(self.blocks[block_index].tokens()[token_index]);
            } else {
                // Token is in the current partial block
                let current_block_index = i - (self.blocks.len() * self.block_size);
                result.push(self.current_block.tokens()[current_block_index]);
            }
        }

        Tokens::from(result)
    }

    /// Splits a [`Tokens`] object into a vector of completed blocks and a final partial block.
    ///
    /// This is primarily used internally by [`TokenBlockSequence::new`] but can be used externally.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The [`Tokens`] to split.
    /// * `block_size` - The size of each block.
    /// * `salt_hash` - The [`SaltHash`] to use for hashing.
    ///
    /// # Returns
    ///
    /// A tuple containing `(Vec<TokenBlock>, PartialTokenBlock)`.
    ///
    /// # Panics
    ///
    /// Panics if `block_size` is not a power of two in `1..=32768`.
    pub fn split_tokens(
        tokens: &[Token],
        block_size: u32,
        salt_hash: SaltHash,
    ) -> (Vec<TokenBlock>, PartialTokenBlock) {
        assert_valid_block_size(block_size);
        let chunks: Vec<TokenBlockChunk> = tokens
            .as_ref()
            .chunks_exact(block_size as usize)
            .map(TokenBlockChunk::from_tokens)
            .collect();

        let (result_blocks, last_plh) = Self::chain_chunks(chunks, salt_hash, block_size);

        let remainder = tokens
            .as_ref()
            .chunks_exact(block_size as usize)
            .remainder();

        let current_block = PartialTokenBlock {
            tokens: remainder.into(),
            block_size,
            salt_hash,
            parent_plh: last_plh,
            position: result_blocks.len(),
        };

        (result_blocks, current_block)
    }

    /// Shared chunk-chaining helper used by both the zero-MM and MM-aware splitters.
    ///
    /// Walks `chunks` in order, threading a [`PositionalLineageHash`] through each
    /// step (`root_with_salt` for the very first chunk, `extend` for every chunk
    /// after). Returns the produced blocks and the final block's PLH (or `None` for
    /// an empty input — the next commit would be the sequence root).
    fn chain_chunks(
        chunks: Vec<TokenBlockChunk>,
        salt_hash: SaltHash,
        block_size: u32,
    ) -> (Vec<TokenBlock>, Option<PositionalLineageHash>) {
        let mut result_blocks = Vec::with_capacity(chunks.len());
        let mut parent_plh: Option<PositionalLineageHash> = None;
        for chunk in chunks {
            let new_block = TokenBlock::from_chunk(chunk, parent_plh, salt_hash, block_size);
            parent_plh = Some(new_block.positional_lineage_hash());
            result_blocks.push(new_block);
        }
        (result_blocks, parent_plh)
    }

    /// Creates a new [`TokenBlockSequence`] from a slice of tokens.
    ///
    /// The tokens are split into blocks of `block_size`. Any remaining tokens
    /// form the initial `current_block`.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The slice of tokens to create the sequence from.
    /// * `block_size` - The size of each block.
    /// * `salt_hash` - The [`SaltHash`] to use for hashing.
    pub fn from_slice(tokens: &[Token], block_size: u32, salt_hash: Option<SaltHash>) -> Self {
        assert_valid_block_size(block_size);
        let salt_hash = salt_hash.unwrap_or_default();
        let (blocks, current_block) = Self::split_tokens(tokens, block_size, salt_hash);

        Self {
            blocks,
            current_block,
            salt_hash,
            block_size: block_size as usize,
            mm_runs: Vec::new(),
        }
    }

    /// Creates a [`TokenBlockSequence`] with multimodal placeholder runs.
    ///
    /// `mm_info` is validated and sorted via [`validate_and_sort_mm_info`]. Each block's
    /// [`BlockHash`] is computed using the byte encoding documented on
    /// [`compute_block_bytes_with_mm`]: token slots emit 4 bytes (LE u32), placeholder slots
    /// emit 12 bytes (LE u64 mm_hash + LE u32 run_offset).
    ///
    /// Returns an error if `mm_info` is invalid (overlap, out of bounds, zero-length run).
    pub fn new_with_mm(
        tokens: Tokens,
        mm_info: &[TokenBlockMmInfo],
        block_size: u32,
        salt_hash: Option<SaltHash>,
    ) -> Result<Self, TokenBlockError> {
        assert_valid_block_size(block_size);
        let salt_hash = salt_hash.unwrap_or_default();
        let validated =
            validate_and_sort_mm_info(mm_info, tokens.len()).map_err(TokenBlockError::MmInfo)?;
        let (blocks, current_block) =
            Self::split_tokens_with_mm(&tokens, &validated, block_size, salt_hash);
        Ok(Self {
            blocks,
            current_block,
            salt_hash,
            block_size: block_size as usize,
            mm_runs: validated,
        })
    }

    /// MM-aware variant of [`Self::split_tokens`].
    ///
    /// `mm_runs` must be pre-validated and sorted (e.g., via [`validate_and_sort_mm_info`]).
    pub fn split_tokens_with_mm(
        tokens: &[Token],
        mm_runs: &[TokenBlockMmInfo],
        block_size: u32,
        salt_hash: SaltHash,
    ) -> (Vec<TokenBlock>, PartialTokenBlock) {
        assert_valid_block_size(block_size);
        let bs = block_size as usize;
        let n_complete = tokens.len() / bs;
        let chunks: Vec<TokenBlockChunk> = (0..n_complete)
            .map(|i| {
                let block_offset = i * bs;
                let block_tokens = &tokens[block_offset..block_offset + bs];
                TokenBlockChunk::from_tokens_with_mm(block_tokens, block_offset, mm_runs)
            })
            .collect();

        let (result_blocks, last_plh) = Self::chain_chunks(chunks, salt_hash, block_size);

        let remainder = &tokens[n_complete * bs..];
        let current_block = PartialTokenBlock {
            tokens: remainder.into(),
            block_size,
            salt_hash,
            parent_plh: last_plh,
            position: n_complete,
        };
        (result_blocks, current_block)
    }

    /// Returns the validated, sorted multimodal runs accumulated by this sequence.
    pub fn mm_runs(&self) -> &[TokenBlockMmInfo] {
        &self.mm_runs
    }

    /// Appends a single real token to the sequence.
    ///
    /// Equivalent to [`Self::append`] but named for symmetry with [`Self::push_mm_run`].
    pub fn push_token(&mut self, token: Token) -> Result<Option<usize>, TokenBlockError> {
        self.append(token)
    }

    /// Appends a multimodal placeholder run of `length` slots all tagged with `mm_hash`.
    ///
    /// The placeholder run starts at the current end of the sequence (`total_tokens()` before
    /// the call). The token IDs at placeholder slot positions are filled with zero sentinels;
    /// hashing uses the `(mm_hash, run_offset)` pair instead of those token bytes.
    ///
    /// Returns the range of fully-committed block indices completed during the call (if any).
    pub fn push_mm_run(
        &mut self,
        mm_hash: u64,
        length: usize,
    ) -> Result<Option<Range<usize>>, TokenBlockError> {
        if length == 0 {
            return Err(TokenBlockError::MmInfo(MmInfoError::EmptyRun));
        }
        let offset = self.total_tokens();
        self.mm_runs.push(TokenBlockMmInfo {
            mm_hash,
            offset,
            length,
        });
        // The token values at placeholder slots are opaque for hashing; use 0 sentinels.
        let placeholders = Tokens::from(vec![0u32; length]);
        self.extend(placeholders)
    }

    /// Batch-validated extension: appends `tokens` (with embedded multimodal runs in `mm_info`)
    /// to the sequence in a single validated step.
    ///
    /// `mm_info` offsets are **relative to the start of `tokens`** (not the existing sequence).
    /// The function validates the chunk, then translates to absolute offsets and applies the
    /// updates atomically: it errors before mutating any state if `mm_info` is invalid.
    ///
    /// Real-token regions and placeholder runs are interleaved per the chunk's layout.
    pub fn extend_with_mm(
        &mut self,
        tokens: &[Token],
        mm_info: &[TokenBlockMmInfo],
    ) -> Result<Option<Range<usize>>, TokenBlockError> {
        let validated =
            validate_and_sort_mm_info(mm_info, tokens.len()).map_err(TokenBlockError::MmInfo)?;
        let start_block = self.blocks.len();
        let mut cursor = 0usize;
        for run in &validated {
            if run.offset > cursor {
                let real = Tokens::from(tokens[cursor..run.offset].to_vec());
                self.extend(real)?;
            }
            self.push_mm_run(run.mm_hash, run.length)?;
            cursor = run.offset + run.length;
        }
        if cursor < tokens.len() {
            let real = Tokens::from(tokens[cursor..].to_vec());
            self.extend(real)?;
        }
        let end_block = self.blocks.len();
        if start_block == end_block {
            Ok(None)
        } else {
            Ok(Some(start_block..end_block))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice;

    // Helper to create a sequence for testing
    fn create_test_sequence(
        initial_tokens: &[Token],
        block_size: u32,
        salt_hash: Option<SaltHash>,
    ) -> TokenBlockSequence {
        TokenBlockSequence::new(Tokens::from(initial_tokens), block_size, salt_hash)
    }

    const TEST_SALT_HASH: SaltHash = 1337;
    const TEST_BLOCK_SIZE: u32 = 4;

    // Per-block content hashes (no salt). xxh3(tokens, 0).
    fn hash_1_4() -> BlockHash {
        compute_block_hash(cast_slice(&[1u32, 2, 3, 4]))
    }
    fn hash_5_8() -> BlockHash {
        compute_block_hash(cast_slice(&[5u32, 6, 7, 8]))
    }
    fn hash_9_12() -> BlockHash {
        compute_block_hash(cast_slice(&[9u32, 10, 11, 12]))
    }

    // Sequence hashes derived via PLH::root_with_salt + extend.
    fn seq_hash_1_4() -> SequenceHash {
        PositionalLineageHash::root_with_salt(hash_1_4(), TEST_SALT_HASH, TEST_BLOCK_SIZE)
            .current_sequence_hash()
    }
    fn seq_hash_5_8() -> SequenceHash {
        PositionalLineageHash::root_with_salt(hash_1_4(), TEST_SALT_HASH, TEST_BLOCK_SIZE)
            .extend(hash_5_8())
            .current_sequence_hash()
    }
    fn seq_hash_9_12() -> SequenceHash {
        PositionalLineageHash::root_with_salt(hash_1_4(), TEST_SALT_HASH, TEST_BLOCK_SIZE)
            .extend(hash_5_8())
            .extend(hash_9_12())
            .current_sequence_hash()
    }

    impl PartialTokenBlock {
        /// Attempts to push a single token onto the block.
        ///
        /// # Arguments
        ///
        /// * `token` - The [`Token`] to push.
        ///
        /// # Returns
        ///
        /// * `Ok(())` - If the token was successfully added.
        /// * `Err(TokenBlockError::Full)` - If the block already contains `block_size` tokens.
        pub fn push_token(&mut self, token: Token) -> Result<(), TokenBlockError> {
            if self.tokens.0.len() >= self.block_size as usize {
                return Err(TokenBlockError::Full);
            }
            self.tokens.0.push(token);
            Ok(())
        }

        /// Attempts to remove the last token from the block.
        ///
        /// # Returns
        ///
        /// * `Ok(())` - If a token was successfully removed.
        /// * `Err(TokenBlockError::Empty)` - If the block was already empty.
        pub fn pop_token(&mut self) -> Result<(), TokenBlockError> {
            if self.tokens.0.is_empty() {
                return Err(TokenBlockError::Empty);
            }
            self.tokens.0.pop();
            Ok(())
        }
    }

    #[test]
    fn test_validate_hash_chain() {
        // Local block hashes: salt-free xxh3(tokens, 0).
        let lbh1 = compute_block_hash(cast_slice(&[1u32, 2, 3, 4]));
        let lbh2 = compute_block_hash(cast_slice(&[5u32, 6, 7, 8]));
        let lbh3 = compute_block_hash(cast_slice(&[9u32, 10, 11, 12]));
        assert_eq!(lbh1, hash_1_4());
        assert_eq!(lbh2, hash_5_8());
        assert_eq!(lbh3, hash_9_12());

        // Root: current = xxh3([salt, lbh1], 0). Subsequent steps: xxh3([parent, lbh], 0).
        let expected_root = compute_hash_v2(cast_slice(&[TEST_SALT_HASH, lbh1]), 0);
        assert_eq!(seq_hash_1_4(), expected_root);

        let expected_seq2 = compute_hash_v2(cast_slice(&[seq_hash_1_4(), lbh2]), 0);
        assert_eq!(seq_hash_5_8(), expected_seq2);

        let expected_seq3 = compute_hash_v2(cast_slice(&[seq_hash_5_8(), lbh3]), 0);
        assert_eq!(seq_hash_9_12(), expected_seq3);

        // Different salt diverges the entire chain starting at the root.
        let alt: SaltHash = 0xDEAD_BEEF;
        let alt_root = PositionalLineageHash::root_with_salt(lbh1, alt, TEST_BLOCK_SIZE);
        assert_ne!(alt_root.current_sequence_hash(), seq_hash_1_4());
    }

    #[test]
    fn test_positional_sequence_hash_encoding_decoding() {
        // Test Mode 0: position fits in 8 bits (< 256)
        let seq_hash_0 = 0x1234567890ABCDEF;
        let position_0 = 100;
        let lbh_0 = 0xFEDCBA9876543210;
        let psh_0 = PositionalSequenceHash::new(seq_hash_0, position_0, lbh_0);

        assert_eq!(psh_0.mode(), 0, "Position 100 should use mode 0");
        assert_eq!(psh_0.sequence_hash(), seq_hash_0);
        assert_eq!(psh_0.position(), position_0);
        // LBH is truncated to 54 bits in mode 0
        assert_eq!(
            psh_0.local_block_hash(),
            lbh_0 & ((1u64 << 54) - 1),
            "LBH should be truncated to 54 bits"
        );

        // Test Mode 1: position fits in 16 bits (256 <= pos < 65536)
        let position_1 = 1000;
        let psh_1 = PositionalSequenceHash::new(seq_hash_0, position_1, lbh_0);

        assert_eq!(psh_1.mode(), 1, "Position 1000 should use mode 1");
        assert_eq!(psh_1.sequence_hash(), seq_hash_0);
        assert_eq!(psh_1.position(), position_1);
        // LBH is truncated to 46 bits in mode 1
        assert_eq!(
            psh_1.local_block_hash(),
            lbh_0 & ((1u64 << 46) - 1),
            "LBH should be truncated to 46 bits"
        );

        // Test Mode 2: position fits in 24 bits (65536 <= pos < 16777216)
        let position_2 = 100_000;
        let psh_2 = PositionalSequenceHash::new(seq_hash_0, position_2, lbh_0);

        assert_eq!(psh_2.mode(), 2, "Position 100,000 should use mode 2");
        assert_eq!(psh_2.sequence_hash(), seq_hash_0);
        assert_eq!(psh_2.position(), position_2);
        // LBH is truncated to 38 bits in mode 2
        assert_eq!(
            psh_2.local_block_hash(),
            lbh_0 & ((1u64 << 38) - 1),
            "LBH should be truncated to 38 bits"
        );

        // Test Mode 3: position fits in 31 bits (16777216 <= pos < 2^31)
        let position_3 = 20_000_000;
        let psh_3 = PositionalSequenceHash::new(seq_hash_0, position_3, lbh_0);

        assert_eq!(psh_3.mode(), 3, "Position 20,000,000 should use mode 3");
        assert_eq!(psh_3.sequence_hash(), seq_hash_0);
        assert_eq!(psh_3.position(), position_3);
        // LBH is truncated to 31 bits in mode 3
        assert_eq!(
            psh_3.local_block_hash(),
            lbh_0 & ((1u64 << 31) - 1),
            "LBH should be truncated to 31 bits"
        );

        // Test edge case: position at boundary
        let position_255 = 255;
        let psh_255 = PositionalSequenceHash::new(seq_hash_0, position_255, lbh_0);
        assert_eq!(psh_255.mode(), 0, "Position 255 should use mode 0");
        assert_eq!(psh_255.position(), position_255);

        let position_256 = 256;
        let psh_256 = PositionalSequenceHash::new(seq_hash_0, position_256, lbh_0);
        assert_eq!(psh_256.mode(), 1, "Position 256 should use mode 1");
        assert_eq!(psh_256.position(), position_256);
    }

    #[test]
    fn test_positional_lineage_hash_flat_layout() {
        const BS: u32 = 16;
        let current = 0x1234567890ABCDEFu64;
        let parent = 0xFEDCBA9876543210u64;

        // Round-trip basic fields.
        let plh = PositionalLineageHash::from_raw_parts(
            current,
            parent,
            100,
            PlhFlags::for_block_size(BS).raw(),
        );
        assert_eq!(plh.position(), 100);
        assert_eq!(plh.current_sequence_hash(), current);
        assert_eq!(plh.parent_sequence_hash(), Some(parent));
        assert_eq!(plh.parent_raw(), parent);
        assert_eq!(plh.block_size(), BS);
        assert_eq!(plh.schema(), PLH_SCHEMA_V1);

        // Parent is full-width — no truncation, regardless of position.
        for pos in [0u32, 100, 1_000, 65_536, 1_000_000, 1u32 << 24, 1u32 << 30] {
            let plh = PositionalLineageHash::from_raw_parts(
                current,
                parent,
                pos,
                PlhFlags::for_block_size(BS).raw(),
            );
            assert_eq!(
                plh.parent_raw(),
                parent,
                "parent must stay full width at pos {pos}"
            );
            assert_eq!(plh.current_sequence_hash(), current);
            assert_eq!(plh.position_u32(), pos);
        }

        // Root: parent_sequence_hash is None at position 0.
        let root = PositionalLineageHash::from_raw_parts(
            current,
            0,
            0,
            PlhFlags::for_block_size(BS).raw(),
        );
        assert_eq!(root.position(), 0);
        assert_eq!(root.parent_sequence_hash(), None);
        assert_eq!(root.parent_raw(), 0);
        assert_eq!(root.current_sequence_hash(), current);
    }

    #[test]
    fn test_positional_lineage_hash_block_size_round_trip() {
        let current = 0xAAAA_BBBB_CCCC_DDDDu64;
        for bs in [16u32, 32, 64, 128, 256, 512, 1024] {
            let plh = PositionalLineageHash::from_raw_parts(
                current,
                0,
                0,
                PlhFlags::for_block_size(bs).raw(),
            );
            assert_eq!(plh.block_size(), bs, "round trip for bs={bs}");
            // Round-trip via from_raw_parts preserves block_size and schema.
            let decoded = PositionalLineageHash::from_raw_parts(
                plh.current_sequence_hash(),
                plh.parent_raw(),
                plh.position_u32(),
                plh.flags_raw(),
            );
            assert_eq!(decoded, plh);
            assert_eq!(decoded.block_size(), bs);
            assert_eq!(decoded.schema(), PLH_SCHEMA_V1);
        }
    }

    #[test]
    #[should_panic(expected = "block_size must be a power of two")]
    fn test_positional_lineage_hash_block_size_must_be_power_of_two() {
        let _ = PlhFlags::new(24, PLH_SCHEMA_V1, 0);
    }

    #[test]
    #[should_panic(expected = "block_size must be a power of two")]
    fn test_positional_lineage_hash_block_size_zero_rejected() {
        let _ = PlhFlags::new(0, PLH_SCHEMA_V1, 0);
    }

    #[test]
    #[should_panic(expected = "block_size must be a power of two")]
    fn test_positional_lineage_hash_block_size_too_large_rejected() {
        // 2^16 cannot be encoded in 4-bit log2 field.
        let _ = PlhFlags::new(1u32 << 16, PLH_SCHEMA_V1, 0);
    }

    #[test]
    fn test_positional_lineage_hash_position_no_panic_past_legacy_limit() {
        // The old layout panicked at 2^24; the flat layout must accept any u32.
        let plh = PositionalLineageHash::from_raw_parts(
            0xDEAD_BEEF,
            0xBADC0FFEE,
            (1u32 << 24) + 1,
            PlhFlags::for_block_size(16).raw(),
        );
        assert_eq!(plh.position_u32(), (1u32 << 24) + 1);
        let plh = PositionalLineageHash::from_raw_parts(
            0xDEAD_BEEF,
            0xBADC0FFEE,
            u32::MAX,
            PlhFlags::for_block_size(16).raw(),
        );
        assert_eq!(plh.position_u32(), u32::MAX);
    }

    #[test]
    fn test_positional_lineage_hash_extend() {
        // PLH must be self-extending: a chain built from PLH::root_with_salt + extend
        // should be bitwise identical to the chain produced by full TokenBlock construction.
        const BS: u32 = 4;
        let salt: SaltHash = 1337;
        let lbh: [LocalBlockHash; 3] = [
            compute_block_hash(cast_slice(&[1u32, 2, 3, 4])),
            compute_block_hash(cast_slice(&[5u32, 6, 7, 8])),
            compute_block_hash(cast_slice(&[9u32, 10, 11, 12])),
        ];

        // Direct construction via TokenBlock — chains by passing parent_plh.
        let blk0 =
            TokenBlock::from_chunk(TokenBlockChunk::from_tokens(&[1, 2, 3, 4]), None, salt, BS);
        let blk1 = TokenBlock::from_chunk(
            TokenBlockChunk::from_tokens(&[5, 6, 7, 8]),
            Some(blk0.positional_lineage_hash()),
            salt,
            BS,
        );
        let blk2 = TokenBlock::from_chunk(
            TokenBlockChunk::from_tokens(&[9, 10, 11, 12]),
            Some(blk1.positional_lineage_hash()),
            salt,
            BS,
        );

        // Self-extending construction via PLH::root_with_salt + extend.
        let plh0 = PositionalLineageHash::root_with_salt(lbh[0], salt, BS);
        let plh1 = plh0.extend(lbh[1]);
        let plh2 = plh1.extend(lbh[2]);

        assert_eq!(plh0, blk0.positional_lineage_hash());
        assert_eq!(plh1, blk1.positional_lineage_hash());
        assert_eq!(plh2, blk2.positional_lineage_hash());

        // Root recurrence: current = xxh3([salt, local_block_hash], 0).
        assert_eq!(
            plh0.current_sequence_hash(),
            compute_hash_v2(cast_slice(&[salt, lbh[0]]), 0),
        );
        // Step recurrence: current = xxh3([parent, child_local_block_hash], 0).
        assert_eq!(
            plh1.current_sequence_hash(),
            compute_hash_v2(cast_slice(&[plh0.current_sequence_hash(), lbh[1]]), 0),
        );

        // Block size and flags propagate through extend.
        assert_eq!(plh1.block_size(), BS);
        assert_eq!(plh2.flags_raw(), plh0.flags_raw());

        // Salt isolation: changing salt diverges the chain starting at the root, but
        // LocalBlockHash itself is unchanged (content-addressable).
        let alt_salt: SaltHash = 4242;
        let alt_plh0 = PositionalLineageHash::root_with_salt(lbh[0], alt_salt, BS);
        let alt_plh1 = alt_plh0.extend(lbh[1]);
        assert_ne!(alt_plh0, plh0);
        assert_ne!(alt_plh1, plh1);

        // Same tokens + same salt converge.
        let again_plh0 = PositionalLineageHash::root_with_salt(lbh[0], salt, BS);
        let again_plh1 = again_plh0.extend(lbh[1]);
        assert_eq!(again_plh0, plh0);
        assert_eq!(again_plh1, plh1);
    }

    #[test]
    fn test_positional_lineage_hash_passthrough_hasher() {
        use std::collections::HashMap;
        use std::hash::{BuildHasherDefault, Hasher};

        type PlhMap<V> =
            HashMap<PositionalLineageHash, V, BuildHasherDefault<PositionalLineageHash>>;
        let mut m: PlhMap<u32> = PlhMap::default();
        let plh = PositionalLineageHash::from_raw_parts(
            0xFEED_FACE_CAFE_BEEF,
            0,
            0,
            PlhFlags::for_block_size(16).raw(),
        );
        m.insert(plh, 42);
        assert_eq!(m.get(&plh), Some(&42));

        // Hasher::finish reflects the most recent write_u64.
        let mut h = PositionalLineageHash::default();
        std::hash::Hasher::write_u64(&mut h, 0x1122_3344_5566_7788);
        assert_eq!(h.finish(), 0x1122_3344_5566_7788);
    }

    #[test]
    #[should_panic(expected = "should only call write_u64")]
    fn test_positional_lineage_hash_hasher_write_panics() {
        let mut h = PositionalLineageHash::default();
        std::hash::Hasher::write(&mut h, &[0u8, 1, 2, 3]);
    }

    #[test]
    fn test_positional_lineage_hash_size_and_align() {
        assert_eq!(std::mem::size_of::<PositionalLineageHash>(), 24);
        assert_eq!(std::mem::align_of::<PositionalLineageHash>(), 8);
    }

    #[test]
    fn test_tokens_from() {
        let vec_u32: Vec<u32> = vec![1, 2, 3];
        let tokens_u32: Tokens = vec_u32.clone().into();
        assert_eq!(tokens_u32.0, vec_u32);

        let slice_u32: &[u32] = &[4, 5];
        let tokens_slice_u32: Tokens = slice_u32.into();
        assert_eq!(tokens_slice_u32.0, vec![4, 5]);

        let vec_i32: Vec<i32> = vec![-1, 0, 1]; // Note: -1 becomes large u32
        let tokens_i32: Tokens = vec_i32.into();
        assert_eq!(tokens_i32.0, vec![u32::MAX, 0, 1]);

        let slice_i32: &[i32] = &[100, 200];
        let tokens_slice_i32: Tokens = slice_i32.into();
        assert_eq!(tokens_slice_i32.0, vec![100, 200]);

        let into_vec: Vec<u32> = tokens_slice_i32.into();
        assert_eq!(into_vec, vec![100, 200]);
    }

    #[test]
    fn test_tokens_equality() {
        let tokens = Tokens::from(vec![1, 2, 3]);
        assert_eq!(tokens, vec![1, 2, 3]);
        assert_eq!(vec![1, 2, 3], tokens);
        assert_eq!(tokens, &[1, 2, 3][..]);
        assert_eq!(&[1, 2, 3][..], tokens);
        assert_eq!(tokens, Tokens::from(vec![1, 2, 3]));
        assert_ne!(tokens, Tokens::from(vec![1, 2, 4]));
    }

    #[test]
    fn test_tokens_deref_asref() {
        let tokens = Tokens::from(vec![10, 20, 30]);

        // Deref to &[Token]
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[1], 20);
        let slice: &[Token] = &tokens;
        assert_eq!(slice, &[10, 20, 30]);

        // AsRef<[Token]>
        let as_ref_slice: &[Token] = tokens.as_ref();
        assert_eq!(as_ref_slice, &[10, 20, 30]);

        // Borrow<[Token]>
        let borrowed_slice: &[Token] = std::borrow::Borrow::borrow(&tokens);
        assert_eq!(borrowed_slice, &[10, 20, 30]);
    }

    #[test]
    fn test_tokens_into_sequence() {
        let tokens = Tokens::from(vec![1, 2, 3, 4, 5]);
        let seq = tokens.into_sequence(4, Some(TEST_SALT_HASH));
        assert_eq!(seq.blocks().len(), 1);
        assert_eq!(seq.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq.current_block().tokens().as_ref(), &[5]);
        assert_eq!(seq.salt_hash(), TEST_SALT_HASH);
    }

    #[test]
    fn test_partial_block_ops() {
        let mut partial = PartialTokenBlock::create_sequence_root(4, TEST_SALT_HASH);
        assert_eq!(partial.len(), 0);
        assert_eq!(partial.remaining(), 4);
        assert!(partial.is_empty());

        // Push tokens
        assert!(partial.push_token(1).is_ok());
        assert_eq!(partial.len(), 1);
        assert_eq!(partial.remaining(), 3);
        let remaining = partial.push_tokens(Tokens::from(vec![2, 3, 4, 5]));
        assert_eq!(partial.len(), 4);
        assert_eq!(partial.remaining(), 0);
        assert_eq!(remaining.as_ref(), &[5]); // Token 5 didn't fit
        assert_eq!(partial.tokens().as_ref(), &[1, 2, 3, 4]);

        // Push when full
        assert_eq!(partial.push_token(6), Err(TokenBlockError::Full));
        let remaining_full = partial.push_tokens(Tokens::from(vec![6]));
        assert_eq!(remaining_full.as_ref(), &[6]);

        // Pop tokens
        assert!(partial.pop_token().is_ok());
        assert_eq!(partial.len(), 3);
        assert_eq!(partial.tokens().as_ref(), &[1, 2, 3]);
        assert!(partial.pop_tokens(3).is_ok());
        assert!(partial.is_empty());

        // Pop when empty
        assert_eq!(partial.pop_token(), Err(TokenBlockError::Empty));
        assert_eq!(
            partial.pop_tokens(1),
            Err(TokenBlockError::InsufficientTokens)
        );

        // Commit incomplete
        assert!(partial.push_token(10).is_ok());
        assert_eq!(partial.commit(), Err(TokenBlockError::Incomplete));

        // Commit complete
        assert!(partial.push_token(11).is_ok());
        assert!(partial.push_token(12).is_ok());
        assert!(partial.push_token(13).is_ok());
        assert_eq!(partial.len(), 4);
        let commit_result = partial.commit();
        assert!(commit_result.is_ok());
        let committed_block = commit_result.unwrap();
        assert_eq!(committed_block.tokens().as_ref(), &[10, 11, 12, 13]);

        // Check state after commit (partial block is now the next one)
        assert!(partial.is_empty());
        assert_eq!(
            partial.parent_sequence_hash(),
            Some(committed_block.sequence_hash())
        );
        assert_eq!(partial.block_size, 4);
    }

    #[test]
    fn test_token_block_creation_and_hashes() {
        let salt = TEST_SALT_HASH;
        let tokens1 = Tokens::from(vec![1, 2, 3, 4]);
        let chunk1 = TokenBlockChunk::new(tokens1.clone());
        let block1 = TokenBlock::from_chunk(chunk1, None, salt, 4);

        assert_eq!(block1.tokens(), &tokens1);
        assert_eq!(block1.salt_hash(), salt);
        assert_eq!(block1.parent_sequence_hash(), None);
        assert_eq!(block1.block_hash(), hash_1_4());
        assert_eq!(block1.sequence_hash(), seq_hash_1_4());
        assert_eq!(block1.position(), 0);

        let plh1 = block1.positional_lineage_hash();
        assert_eq!(plh1.position(), 0);
        assert_eq!(plh1.parent_hash_fragment(), 0); // Root has no parent
        assert_eq!(plh1.current_sequence_hash(), seq_hash_1_4());

        let tokens2 = Tokens::from(vec![5, 6, 7, 8]);
        // Wrong parent: build block2 as if it were the root. seq_hash differs from the
        // truly-extended block 5..8.
        let chunk2_root = TokenBlockChunk::new(tokens2.clone());
        let block2_wrong = TokenBlock::from_chunk(chunk2_root, None, salt, 4);
        assert_ne!(block2_wrong.sequence_hash(), seq_hash_5_8());

        let chunk2_correct = TokenBlockChunk::new(tokens2.clone());
        let block2_correct = TokenBlock::from_chunk(
            chunk2_correct,
            Some(block1.positional_lineage_hash()),
            salt,
            4,
        );

        assert_eq!(block2_correct.tokens(), &tokens2);
        assert_eq!(block2_correct.salt_hash(), salt);
        assert_eq!(
            block2_correct.parent_sequence_hash(),
            Some(block1.sequence_hash())
        );
        assert_eq!(block2_correct.block_hash(), hash_5_8());
        assert_eq!(block2_correct.sequence_hash(), seq_hash_5_8());
        assert_eq!(block2_correct.position(), 1);

        let plh2 = block2_correct.positional_lineage_hash();
        assert_eq!(plh2.position(), 1);
        assert_eq!(plh2.parent_sequence_hash(), Some(seq_hash_1_4()));
        assert_eq!(plh2.parent_raw(), seq_hash_1_4());
        assert_eq!(plh2.current_sequence_hash(), seq_hash_5_8());
        assert_eq!(plh2.block_size(), 4);
    }

    #[test]
    fn test_new_sequence() {
        // Empty initial tokens
        let seq_empty = create_test_sequence(&[], 4, Some(TEST_SALT_HASH));
        assert!(seq_empty.blocks().is_empty());
        assert!(seq_empty.current_block().is_empty());
        assert_eq!(seq_empty.total_tokens(), 0);
        assert_eq!(seq_empty.salt_hash(), TEST_SALT_HASH);
        assert_eq!(seq_empty.current_block().parent_sequence_hash(), None);

        // Less than one block
        let seq_partial = create_test_sequence(&[1, 2], 4, Some(TEST_SALT_HASH));
        assert!(seq_partial.blocks().is_empty());
        assert_eq!(seq_partial.current_block().tokens().as_ref(), &[1, 2]);
        assert_eq!(seq_partial.total_tokens(), 2);
        assert_eq!(seq_partial.current_block().parent_sequence_hash(), None);

        // Exactly one block
        let seq_one_block = create_test_sequence(&[1, 2, 3, 4], 4, Some(TEST_SALT_HASH));
        assert_eq!(seq_one_block.blocks().len(), 1);
        assert!(seq_one_block.current_block().is_empty());
        assert_eq!(seq_one_block.total_tokens(), 4);
        assert_eq!(seq_one_block.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq_one_block.blocks[0].sequence_hash(), seq_hash_1_4());
        assert_eq!(
            seq_one_block.current_block().parent_sequence_hash(),
            Some(seq_hash_1_4())
        );

        // More than one block
        let seq_multi = create_test_sequence(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 4, Some(TEST_SALT_HASH));
        assert_eq!(seq_multi.blocks().len(), 2);
        assert_eq!(seq_multi.current_block().tokens().as_ref(), &[9]);
        assert_eq!(seq_multi.total_tokens(), 9);
        assert_eq!(seq_multi.blocks[0].sequence_hash(), seq_hash_1_4());
        assert_eq!(seq_multi.blocks[1].sequence_hash(), seq_hash_5_8());
        assert_eq!(
            seq_multi.current_block().parent_sequence_hash(),
            Some(seq_hash_5_8())
        );

        // Test tokens_at across blocks and partial block
        assert_eq!(seq_multi.tokens_at(0..4).as_ref(), &[1, 2, 3, 4]); // First complete block
        assert_eq!(seq_multi.tokens_at(4..8).as_ref(), &[5, 6, 7, 8]); // Second complete block
        assert_eq!(seq_multi.tokens_at(8..9).as_ref(), &[9]); // Current partial block
        assert_eq!(seq_multi.tokens_at(2..6).as_ref(), &[3, 4, 5, 6]); // Spanning blocks
        assert_eq!(seq_multi.tokens_at(6..9).as_ref(), &[7, 8, 9]); // Spanning to partial
        assert_eq!(seq_multi.tokens_at(5..5).as_ref(), &[0u32; 0]); // Empty range
        assert_eq!(seq_multi.tokens_at(10..15).as_ref(), &[0u32; 0]); // Out of bounds

        // No salt hash. LocalBlockHash is content-only and salt-independent, so
        // block_hash matches the salted-sequence's block_hash byte-for-byte. Salt
        // diverges the *sequence_hash* (chain) at the root only.
        let seq_no_salt = create_test_sequence(&[1, 2, 3, 4, 5], 4, None);
        assert_eq!(seq_no_salt.salt_hash(), 0);
        assert_eq!(seq_no_salt.blocks().len(), 1);
        assert_eq!(seq_no_salt.blocks[0].block_hash(), hash_1_4());
        assert_ne!(seq_no_salt.blocks[0].sequence_hash(), seq_hash_1_4());
        assert_eq!(seq_no_salt.current_block().tokens().as_ref(), &[5]);
    }

    #[test]
    #[should_panic]
    fn test_new_sequence_zero_block_size() {
        let _ = create_test_sequence(&[1], 0, None);
    }

    #[test]
    fn test_append_single_token() {
        let mut sequence =
            create_test_sequence(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4, Some(TEST_SALT_HASH));
        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().tokens.len(), 2);
        assert_eq!(sequence.current_block().tokens, vec![9, 10]);
        assert_eq!(
            sequence.current_block().parent_sequence_hash(),
            Some(seq_hash_5_8())
        );

        // Append token 11 - should not complete a block
        let completed_idx = sequence.append(11).unwrap();
        assert_eq!(completed_idx, None);
        assert_eq!(sequence.blocks().len(), 2);
        assert_eq!(sequence.current_block().tokens.as_ref(), &[9, 10, 11]);

        // Append token 12 - should complete block 2 (index 2)
        // This will also commit block 2
        let completed_idx = sequence.append(12).unwrap();
        assert_eq!(completed_idx, Some(2));
        assert_eq!(sequence.blocks().len(), 3);
        assert_eq!(sequence.current_block.tokens.as_ref(), &[0u32; 0]);
        assert_eq!(sequence.current_block.remaining(), 4);
        assert_eq!(
            sequence.current_block().parent_sequence_hash(),
            Some(seq_hash_9_12())
        ); // Still linked to block 1

        // Append token 13 - should not complete a block
        let completed_idx_13 = sequence.append(13).unwrap();
        assert_eq!(completed_idx_13, None);
        assert_eq!(sequence.blocks().len(), 3);
        assert_eq!(sequence.blocks[2].tokens().as_ref(), &[9, 10, 11, 12]);
        assert_eq!(sequence.blocks[2].sequence_hash(), seq_hash_9_12());
        assert_eq!(sequence.current_block.tokens.as_ref(), &[13]); // New current block has 13
        assert_eq!(sequence.current_block.remaining(), 3);
        assert_eq!(
            sequence.current_block.parent_sequence_hash(),
            Some(seq_hash_9_12())
        ); // Linked to new block 2
    }

    #[test]
    fn test_extend() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);

        // Case 1: Extend less than block size
        let mut seq1 = create_test_sequence(&[], block_size, salt_hash);
        let tokens1 = Tokens::from(vec![1, 2]);
        let completed1 = seq1.extend(tokens1).unwrap();
        assert_eq!(completed1, None); // No blocks completed
        assert_eq!(seq1.blocks.len(), 0);
        assert_eq!(seq1.current_block.tokens.as_ref(), &[1, 2]);
        assert_eq!(seq1.current_block.remaining(), 2);
        assert_eq!(seq1.current_block.parent_sequence_hash(), None); // Still the root block

        // Case 2: Extend exactly block size
        let mut seq2 = create_test_sequence(&[], block_size, salt_hash);
        let tokens2 = Tokens::from(vec![1, 2, 3, 4]);
        let completed2 = seq2.extend(tokens2).unwrap();
        assert_eq!(completed2, Some(0..1));
        assert_eq!(seq2.blocks.len(), 1);
        assert_eq!(seq2.current_block.tokens.as_ref(), &[0u32; 0]); // Current block is empty
        assert_eq!(seq2.current_block.remaining(), 4);
        assert_eq!(
            seq2.current_block.parent_sequence_hash(),
            Some(seq_hash_1_4())
        ); // Still the root block

        // Case 3: Extend more than block size, less than two blocks
        let mut seq3 = create_test_sequence(&[], block_size, salt_hash);
        let tokens3 = Tokens::from(vec![1, 2, 3, 4, 5, 6]);
        let completed3 = seq3.extend(tokens3).unwrap();
        assert_eq!(completed3, Some(0..1)); // Block at index 0 completed
        assert_eq!(seq3.blocks.len(), 1);
        assert_eq!(seq3.current_block.tokens.as_ref(), &[5, 6]); // Partial block has remainder
        assert_eq!(seq3.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(
            seq3.current_block.parent_sequence_hash(),
            Some(seq_hash_1_4())
        );
        assert_eq!(seq3.current_block.remaining(), 2);

        // Case 4: Extend exactly two blocks
        let mut seq4 = create_test_sequence(&[], block_size, salt_hash);
        let tokens4 = Tokens::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let completed4 = seq4.extend(tokens4).unwrap();
        assert_eq!(completed4, Some(0..2)); // Only block 0 is committed
        assert_eq!(seq4.blocks.len(), 2); // Only 1 block committed
        assert_eq!(seq4.current_block.tokens.as_ref(), &[0u32; 0]);
        assert_eq!(seq4.current_block.remaining(), 4);
        assert_eq!(seq4.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq4.blocks[0].sequence_hash(), seq_hash_1_4());
        assert_eq!(
            seq4.current_block.parent_sequence_hash(),
            Some(seq_hash_5_8())
        ); // Parent is the first block

        // Case 5: Extend multiple times, completing blocks across calls
        let mut seq5 = create_test_sequence(&[], block_size, salt_hash);
        let tokens5a = Tokens::from(vec![1, 2]);
        let completed5a = seq5.extend(tokens5a).unwrap();
        assert_eq!(completed5a, None);
        assert_eq!(seq5.blocks.len(), 0);
        assert_eq!(seq5.current_block.tokens.as_ref(), &[1, 2]);

        let tokens5b = Tokens::from(vec![3, 4, 5]);
        let completed5b = seq5.extend(tokens5b).unwrap();
        assert_eq!(completed5b, Some(0..1)); // Block at index 0 completed
        assert_eq!(seq5.blocks.len(), 1);
        assert_eq!(seq5.current_block.tokens.as_ref(), &[5]);
        assert_eq!(seq5.blocks[0].tokens().as_ref(), &[1, 2, 3, 4]);
        assert_eq!(
            seq5.current_block.parent_sequence_hash(),
            Some(seq_hash_1_4())
        );
        assert_eq!(seq5.current_block.remaining(), 3);

        let tokens5c = Tokens::from(vec![6, 7, 8, 9, 10]);
        let completed5c = seq5.extend(tokens5c).unwrap();
        assert_eq!(completed5c, Some(1..2)); // Block at index 1 completed
        assert_eq!(seq5.blocks.len(), 2);
        assert_eq!(seq5.current_block.tokens.as_ref(), &[9, 10]);
        assert_eq!(seq5.blocks[1].tokens().as_ref(), &[5, 6, 7, 8]);
        assert_eq!(
            seq5.current_block.parent_sequence_hash(),
            Some(seq_hash_5_8())
        );
        assert_eq!(seq5.current_block.remaining(), 2);

        // Case 6: Extend empty tokens
        let mut seq6 = create_test_sequence(&[1], block_size, salt_hash);
        let completed6 = seq6.extend(Tokens::default()).unwrap();
        assert_eq!(completed6, None);
        assert_eq!(seq6.blocks.len(), 0);
        assert_eq!(seq6.current_block.tokens.as_ref(), &[1]);
        assert_eq!(seq6.total_tokens(), 1);

        // Case 7: Extend fills current exactly, no remainder
        let mut seq7 = create_test_sequence(&[1, 2], block_size, salt_hash);
        let tokens7 = Tokens::from(vec![3, 4]);
        let completed7 = seq7.extend(tokens7).unwrap();
        assert_eq!(completed7, Some(0..1)); // Block is full but not committed yet
        assert_eq!(seq7.blocks.len(), 1);
        assert_eq!(seq7.current_block.tokens.as_ref(), &[0u32; 0]); // Current block is full
        assert_eq!(seq7.current_block.remaining(), 4);
        assert_eq!(seq7.total_tokens(), 4);
        assert_eq!(
            seq7.current_block.parent_sequence_hash(),
            Some(seq_hash_1_4())
        ); // Still the root block

        // Test tokens_at extraction
        assert_eq!(seq7.tokens_at(0..2).as_ref(), &[1, 2]);
        assert_eq!(seq7.tokens_at(1..3).as_ref(), &[2, 3]);
        assert_eq!(seq7.tokens_at(0..4).as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq7.tokens_at(2..2).as_ref(), &[0u32; 0]); // Empty range
    }

    #[test]
    fn test_truncate() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);
        let initial_tokens = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 10 tokens

        // Case 1: Truncate within current block (len 9)
        let mut seq1 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq1.truncate(9).is_ok());
        assert_eq!(seq1.total_tokens(), 9);
        assert_eq!(seq1.blocks().len(), 2);
        assert_eq!(seq1.current_block().tokens.as_ref(), &[9]);
        assert_eq!(
            seq1.current_block().parent_sequence_hash(),
            Some(seq_hash_5_8())
        );

        // Case 2: Truncate to exact block boundary (len 8)
        let mut seq2 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq2.truncate(8).is_ok());
        assert_eq!(seq2.total_tokens(), 8);
        assert_eq!(seq2.blocks().len(), 2);
        assert!(seq2.current_block().tokens.is_empty());
        assert_eq!(
            seq2.current_block().parent_sequence_hash(),
            Some(seq_hash_5_8())
        );

        // Case 3: Truncate into last full block (len 7)
        let mut seq3 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq3.truncate(7).is_ok());
        assert_eq!(seq3.total_tokens(), 7);
        assert_eq!(seq3.blocks().len(), 1); // Block [5,6,7,8] removed conceptually
        assert_eq!(seq3.current_block().tokens.as_ref(), &[5, 6, 7]); // Kept 3 from [5,6,7,8]
        assert_eq!(
            seq3.current_block().parent_sequence_hash(),
            Some(seq_hash_1_4())
        ); // Parent is hash of [1,2,3,4]
        assert_eq!(seq3.blocks()[0].tokens().as_ref(), &[1, 2, 3, 4]);

        // Case 4: Truncate removing full block(s) exactly (len 4)
        let mut seq4 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq4.truncate(4).is_ok());
        assert_eq!(seq4.total_tokens(), 4);
        assert_eq!(seq4.blocks().len(), 1); // Block [5,6,7,8] removed
        assert!(seq4.current_block().tokens.is_empty()); // New partial based on block [1,2,3,4]
        assert_eq!(
            seq4.current_block().parent_sequence_hash(),
            Some(seq_hash_1_4())
        );
        assert_eq!(seq4.blocks()[0].tokens().as_ref(), &[1, 2, 3, 4]);

        // Case 5: Truncate into first block (len 3)
        let mut seq5 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq5.truncate(3).is_ok());
        assert_eq!(seq5.total_tokens(), 3);
        assert!(seq5.blocks().is_empty()); // Both blocks removed conceptually
        assert_eq!(seq5.current_block().tokens.as_ref(), &[1, 2, 3]); // Kept 3 from [1,2,3,4]
        assert_eq!(seq5.current_block().parent_sequence_hash(), None); // No parent

        // Case 6: Truncate to zero length (len 0)
        let mut seq6 = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq6.truncate(0).is_ok());
        assert_eq!(seq6.total_tokens(), 0);
        assert!(seq6.blocks().is_empty());
        assert!(seq6.current_block().tokens.is_empty());
        assert_eq!(seq6.current_block().parent_sequence_hash(), None);

        // Case 7: Truncate to length greater than current (len 11)
        let mut seq7 = create_test_sequence(initial_tokens, block_size, salt_hash);
        let original_state = (seq7.blocks.clone(), seq7.current_block.tokens.clone()); // Clone for state check
        assert!(seq7.truncate(11).is_ok()); // Should have no effect
        assert_eq!(seq7.total_tokens(), 10);
        assert_eq!(seq7.blocks, original_state.0);
        assert_eq!(seq7.current_block.tokens, original_state.1);

        // Case 8: Truncate to current length (len 10)
        let mut seq8 = create_test_sequence(initial_tokens, block_size, salt_hash);
        let original_state = (seq8.blocks.clone(), seq8.current_block.tokens.clone());
        assert!(seq8.truncate(10).is_ok());
        assert_eq!(seq8.total_tokens(), 10);
        assert_eq!(seq8.blocks, original_state.0);
        assert_eq!(seq8.current_block.tokens, original_state.1);

        // Case 9: Truncate an empty sequence to 0
        let mut seq9 = create_test_sequence(&[], block_size, salt_hash);
        assert!(seq9.truncate(0).is_ok());
        assert_eq!(seq9.total_tokens(), 0);
        assert!(seq9.blocks().is_empty());
        assert!(seq9.current_block().tokens.is_empty());

        // Case 10: Truncate on exact block boundary when current is empty (len 4)
        let tokens10 = &[1, 2, 3, 4, 5, 6, 7, 8]; // 8 tokens
        let mut seq10 = create_test_sequence(tokens10, block_size, salt_hash);
        assert_eq!(seq10.total_tokens(), 8);
        assert!(seq10.current_block().is_empty());
        assert!(seq10.truncate(4).is_ok()); // Remove block [5, 6, 7, 8]
        assert_eq!(seq10.total_tokens(), 4);
        assert_eq!(seq10.blocks().len(), 1);
        assert!(seq10.current_block().tokens.is_empty());
        assert_eq!(
            seq10.current_block().parent_sequence_hash(),
            Some(seq_hash_1_4())
        );

        // Case 11: Truncate into first block when current is empty (len 3)
        let tokens11 = &[1, 2, 3, 4, 5, 6, 7, 8]; // 8 tokens
        let mut seq11 = create_test_sequence(tokens11, block_size, salt_hash);
        assert!(seq11.truncate(3).is_ok()); // Pop block [5,6,7,8] + 1 from [1,2,3,4]
        assert_eq!(seq11.total_tokens(), 3);
        assert!(seq11.blocks().is_empty());
        assert_eq!(seq11.current_block().tokens.as_ref(), &[1, 2, 3]); // Kept 3 from [1,2,3,4]
        assert_eq!(seq11.current_block().parent_sequence_hash(), None);
    }

    #[test]
    fn test_unwind() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);
        let initial_tokens = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 10 tokens

        // Unwind 0
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(0).is_ok());
        assert_eq!(seq.total_tokens(), 10);

        // Unwind 1
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(1).is_ok());
        assert_eq!(seq.total_tokens(), 9);
        assert_eq!(seq.current_block.tokens.as_ref(), &[9]);

        // Unwind 3 (crosses boundary)
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(3).is_ok());
        assert_eq!(seq.total_tokens(), 7);
        assert_eq!(seq.blocks.len(), 1);
        assert_eq!(seq.current_block.tokens.as_ref(), &[5, 6, 7]);

        // Unwind all (10)
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert!(seq.unwind(10).is_ok());
        assert_eq!(seq.total_tokens(), 0);
        assert!(seq.blocks.is_empty());
        assert!(seq.current_block.is_empty());

        // Unwind more than available (11)
        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);
        assert_eq!(seq.unwind(11), Err(TokenBlockError::InsufficientTokens));
        assert_eq!(seq.total_tokens(), 10); // State unchanged

        // Unwind from empty
        let mut seq_empty = create_test_sequence(&[], block_size, salt_hash);
        assert_eq!(
            seq_empty.unwind(1),
            Err(TokenBlockError::InsufficientTokens)
        );
    }

    #[test]
    fn test_pop() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);
        let initial_tokens = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // 10 tokens

        let mut seq = create_test_sequence(initial_tokens, block_size, salt_hash);

        // Pop 10
        assert_eq!(seq.pop(), Some(10));
        assert_eq!(seq.total_tokens(), 9);
        assert_eq!(seq.current_block.tokens.as_ref(), &[9]);
        assert_eq!(seq.blocks.len(), 2);

        // Pop 9
        assert_eq!(seq.pop(), Some(9));
        assert_eq!(seq.total_tokens(), 8);
        assert!(seq.current_block.is_empty());
        assert_eq!(seq.blocks.len(), 2);
        assert_eq!(
            seq.current_block.parent_sequence_hash(),
            Some(seq_hash_5_8())
        );

        // Pop 8 (crosses boundary)
        assert_eq!(seq.pop(), Some(8));
        assert_eq!(seq.total_tokens(), 7);
        assert_eq!(seq.current_block.tokens.as_ref(), &[5, 6, 7]);
        assert_eq!(seq.blocks.len(), 1);
        assert_eq!(
            seq.current_block.parent_sequence_hash(),
            Some(seq_hash_1_4())
        );

        // Pop remaining partial (7, 6, 5)
        assert_eq!(seq.pop(), Some(7));
        assert_eq!(seq.pop(), Some(6));
        assert_eq!(seq.pop(), Some(5));
        assert_eq!(seq.total_tokens(), 4);
        assert!(seq.current_block.is_empty());
        assert_eq!(seq.blocks.len(), 1);
        assert_eq!(
            seq.current_block.parent_sequence_hash(),
            Some(seq_hash_1_4())
        );

        // Pop 4 (crosses boundary)
        assert_eq!(seq.pop(), Some(4));
        assert_eq!(seq.total_tokens(), 3);
        assert_eq!(seq.current_block.tokens.as_ref(), &[1, 2, 3]);
        assert!(seq.blocks.is_empty());
        assert_eq!(seq.current_block.parent_sequence_hash(), None);

        // Pop 3, 2, 1
        assert_eq!(seq.pop(), Some(3));
        assert_eq!(seq.pop(), Some(2));
        assert_eq!(seq.pop(), Some(1));
        assert_eq!(seq.total_tokens(), 0);
        assert!(seq.current_block.is_empty());
        assert!(seq.blocks.is_empty());

        // Pop from empty
        assert_eq!(seq.pop(), None);
        assert_eq!(seq.total_tokens(), 0);
    }

    #[test]
    fn test_total_tokens() {
        let block_size = 4;
        let salt_hash = Some(TEST_SALT_HASH);

        let mut seq = create_test_sequence(&[], block_size, salt_hash);
        assert_eq!(seq.total_tokens(), 0);

        seq.extend(Tokens::from(vec![1, 2, 3])).unwrap();
        assert_eq!(seq.total_tokens(), 3);

        seq.append(4).unwrap(); // Completes block 0
        assert_eq!(seq.total_tokens(), 4);

        seq.extend(Tokens::from(vec![5, 6, 7, 8, 9])).unwrap(); // Completes block 1, partial [9]
        assert_eq!(seq.total_tokens(), 9);

        seq.pop().unwrap(); // Removes 9
        assert_eq!(seq.total_tokens(), 8);

        seq.truncate(5).unwrap(); // Keep [1..=5]
        assert_eq!(seq.total_tokens(), 5);

        seq.unwind(3).unwrap(); // Drop 3 → keep [1, 2]
        assert_eq!(seq.total_tokens(), 2);
    }

    #[test]
    fn test_push_tokens_partial_block() {
        let mut partial = PartialTokenBlock::create_sequence_root(4, 1337);

        let tokens = Tokens(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

        let remaining = partial.push_tokens(tokens);
        assert_eq!(partial.tokens.len(), 4);
        assert_eq!(remaining.len(), 6);
    }

    // ========== Additional tests for coverage improvement ==========

    // === PositionalRadixTree Tests ===

    #[test]
    fn test_positional_radix_tree_basic_operations() {
        use crate::PositionalRadixTree;

        // Test new() and is_empty()
        let tree: PositionalRadixTree<String> = PositionalRadixTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);

        // Test default()
        let tree2: PositionalRadixTree<i32> = PositionalRadixTree::default();
        assert!(tree2.is_empty());

        // Test prefix() and insertion
        let psh1 = PositionalSequenceHash::new(0x1234, 0, 0xABCD);
        let psh2 = PositionalSequenceHash::new(0x5678, 0, 0xEF01);
        let psh3 = PositionalSequenceHash::new(0x9ABC, 1, 0x2345);

        tree.prefix(&psh1).insert(psh1, "value1".to_string());
        assert!(!tree.is_empty());
        assert_eq!(tree.len(), 1);

        tree.prefix(&psh2).insert(psh2, "value2".to_string());
        assert_eq!(tree.len(), 2);

        tree.prefix(&psh3).insert(psh3, "value3".to_string());
        assert_eq!(tree.len(), 3);

        // Test retrieval
        assert_eq!(
            tree.prefix(&psh1).get(&psh1).map(|v| v.clone()),
            Some("value1".to_string())
        );
    }

    #[test]
    fn test_positional_radix_tree_with_lineage_hash() {
        use crate::PositionalRadixTree;

        // Test generic usage with PositionalLineageHash
        let tree: PositionalRadixTree<u32, PositionalLineageHash> = PositionalRadixTree::new();
        assert!(tree.is_empty());

        let plh1 =
            PositionalLineageHash::from_raw_parts(0x1234, 0, 0, PlhFlags::for_block_size(16).raw());
        let plh2 = PositionalLineageHash::from_raw_parts(
            0x5678,
            0x1234,
            1,
            PlhFlags::for_block_size(16).raw(),
        );

        tree.prefix(&plh1).insert(plh1, 100);
        tree.prefix(&plh2).insert(plh2, 200);

        assert_eq!(tree.len(), 2);
        assert_eq!(tree.prefix(&plh1).get(&plh1).map(|v| *v), Some(100));
        assert_eq!(tree.prefix(&plh2).get(&plh2).map(|v| *v), Some(200));
    }

    #[test]
    fn test_positional_radix_tree_position_lookup() {
        use crate::PositionalRadixTree;

        let tree: PositionalRadixTree<String> = PositionalRadixTree::new();

        // Insert at different positions
        let psh0 = PositionalSequenceHash::new(0x1111, 0, 0xAAAA);
        let psh1 = PositionalSequenceHash::new(0x2222, 1, 0xBBBB);
        let psh2 = PositionalSequenceHash::new(0x3333, 2, 0xCCCC);

        tree.prefix(&psh0).insert(psh0, "pos0".to_string());
        tree.prefix(&psh1).insert(psh1, "pos1".to_string());
        tree.prefix(&psh2).insert(psh2, "pos2".to_string());

        // Test position() method
        assert!(tree.position(0).is_some());
        assert!(tree.position(1).is_some());
        assert!(tree.position(2).is_some());
        assert!(tree.position(3).is_none()); // No entries at position 3

        // Verify position lookup returns correct submap
        let pos0_map = tree.position(0).unwrap();
        assert_eq!(pos0_map.len(), 1);
    }

    // === PositionalSequenceHash Additional Tests ===

    #[test]
    fn test_positional_sequence_hash_mode_2_and_3() {
        // Mode 2: position fits in 24 bits (65536 <= pos < 16777216)
        let position_mode2 = 100_000u64;
        let seq_hash = 0x1234567890ABCDEF;
        let block_hash = 0xFEDCBA9876543210;

        let psh_mode2 = PositionalSequenceHash::new(seq_hash, position_mode2, block_hash);
        assert_eq!(psh_mode2.mode(), 2, "Position 100,000 should use mode 2");
        assert_eq!(psh_mode2.position(), position_mode2);
        assert_eq!(psh_mode2.sequence_hash(), seq_hash);
        // Local block hash truncated to 38 bits in mode 2
        assert_eq!(
            psh_mode2.local_block_hash(),
            block_hash & ((1u64 << 38) - 1)
        );

        // Mode 3: position fits in 31 bits (16777216 <= pos < 2147483648)
        let position_mode3 = 100_000_000u64;
        let psh_mode3 = PositionalSequenceHash::new(seq_hash, position_mode3, block_hash);
        assert_eq!(
            psh_mode3.mode(),
            3,
            "Position 100,000,000 should use mode 3"
        );
        assert_eq!(psh_mode3.position(), position_mode3);
        assert_eq!(psh_mode3.sequence_hash(), seq_hash);
        // Local block hash truncated to 31 bits in mode 3
        assert_eq!(
            psh_mode3.local_block_hash(),
            block_hash & ((1u64 << 31) - 1)
        );
    }

    #[test]
    fn test_positional_sequence_hash_as_u128() {
        let psh = PositionalSequenceHash::new(0x1234, 100, 0xABCD);
        let raw = psh.as_u128();

        // Verify we can reconstruct from raw value
        assert_eq!(raw & 0xFFFF_FFFF_FFFF_FFFF, 0x1234);
        assert!(raw > 0); // Non-zero

        // Create another and compare
        let psh2 = PositionalSequenceHash::new(0x1234, 100, 0xABCD);
        assert_eq!(psh.as_u128(), psh2.as_u128());
    }

    #[test]
    fn test_positional_sequence_hash_debug() {
        let psh = PositionalSequenceHash::new(0x1234567890ABCDEF, 42, 0xFEDCBA98);
        let debug_str = format!("{:?}", psh);

        // Debug should contain field names and values
        assert!(debug_str.contains("PositionalSequenceHash"));
        assert!(debug_str.contains("sequence_hash"));
        assert!(debug_str.contains("local_block_hash"));
        assert!(debug_str.contains("position"));
    }

    // === PositionalLineageHash Additional Tests ===

    #[test]
    fn test_positional_lineage_hash_debug_and_display() {
        // Test position 0 (no parent shown)
        let plh_root = PositionalLineageHash::from_raw_parts(
            0x123456789ABCDEF0,
            0,
            0,
            PlhFlags::for_block_size(16).raw(),
        );
        let debug_root = format!("{:?}", plh_root);
        let display_root = format!("{}", plh_root);

        // Debug and Display should show position 0
        assert!(debug_root.starts_with("0:"));
        assert!(display_root.starts_with("0:"));
        // Position 0 should not show parent
        assert_eq!(debug_root.matches(':').count(), 1);
        assert_eq!(display_root.matches(':').count(), 1);

        // Test position > 0 (parent shown)
        let plh_child = PositionalLineageHash::from_raw_parts(
            0xABCDEF0123456789,
            0x123456789ABCDEF0,
            5,
            PlhFlags::for_block_size(16).raw(),
        );
        let debug_child = format!("{:?}", plh_child);
        let display_child = format!("{}", plh_child);

        // Should show position:current:parent
        assert!(debug_child.starts_with("5:"));
        assert!(display_child.starts_with("5:"));
        // Position > 0 should show parent (3 parts)
        assert_eq!(debug_child.matches(':').count(), 2);
        assert_eq!(display_child.matches(':').count(), 2);
    }

    #[test]
    fn test_positional_lineage_hash_as_u128() {
        let plh = PositionalLineageHash::from_raw_parts(
            0x1234,
            0x5678,
            10,
            PlhFlags::for_block_size(16).raw(),
        );
        let raw = plh.as_u128();
        assert_eq!(raw, ((0x1234u128) << 64) | 0x5678u128);

        // Same current+parent → same as_u128 (the shim is current+parent only).
        let plh2 = PositionalLineageHash::from_raw_parts(
            0x1234,
            0x5678,
            10,
            PlhFlags::for_block_size(16).raw(),
        );
        assert_eq!(plh.as_u128(), plh2.as_u128());

        // Differing current → different as_u128.
        let plh3 = PositionalLineageHash::from_raw_parts(
            0xAAAA,
            0x5678,
            10,
            PlhFlags::for_block_size(16).raw(),
        );
        assert_ne!(plh.as_u128(), plh3.as_u128());
    }

    // === Tokens From Impls ===

    #[test]
    fn test_tokens_from_vec_usize() {
        let usize_vec: Vec<usize> = vec![1, 2, 3, 4, 5];
        let tokens = Tokens::from(usize_vec);

        assert_eq!(tokens.as_ref(), &[1u32, 2, 3, 4, 5]);
        assert_eq!(tokens.len(), 5);
    }

    #[test]
    fn test_tokens_partial_eq_slice_ref() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4]);
        let slice: &[Token] = &[1, 2, 3, 4];

        // Test PartialEq<&[Token]> for Tokens
        assert!(tokens == slice);

        let different_slice: &[Token] = &[1, 2, 3, 5];
        assert!(tokens != different_slice);
    }

    // === TokenBlock Accessors ===

    #[test]
    fn test_token_block_accessors() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        let block = &seq.blocks()[0];

        assert_eq!(block.block_size(), 4);

        let plh = block.positional_lineage_hash();
        assert_eq!(plh.position(), 0);
        assert_eq!(plh.parent_sequence_hash(), None); // Root has no parent
        assert_eq!(plh.block_size(), 4);
    }

    #[test]
    fn test_positional_hash_trait_impls() {
        use crate::PositionalHash;

        // Test PositionalHash for PositionalSequenceHash
        let psh = PositionalSequenceHash::new(0x1234, 42, 0xABCD);
        assert_eq!(PositionalHash::position(&psh), 42);

        // Test PositionalHash for PositionalLineageHash
        let plh = PositionalLineageHash::from_raw_parts(
            0x1234,
            0,
            99,
            PlhFlags::for_block_size(16).raw(),
        );
        assert_eq!(PositionalHash::position(&plh), 99);
    }

    // === TokenBlockSequence Edge Cases ===

    #[test]
    fn test_sequence_pop_from_full_block() {
        // Test pop when current partial block is empty (must pop from full block)
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
        let mut seq = TokenBlockSequence::new(tokens, 4, Some(TEST_SALT_HASH));

        // Current block should be empty, all tokens in completed blocks
        assert!(seq.current_block().is_empty());
        assert_eq!(seq.blocks().len(), 2);
        assert_eq!(seq.total_tokens(), 8);

        // Pop should remove from last full block
        let popped = seq.pop();
        assert_eq!(popped, Some(8));
        assert_eq!(seq.total_tokens(), 7);
        assert_eq!(seq.blocks().len(), 1);
        assert_eq!(seq.current_block().tokens.as_ref(), &[5, 6, 7]);
    }

    #[test]
    #[allow(clippy::reversed_empty_ranges)] // so we can explicitly test invalid ranges
    fn test_sequence_tokens_at_edge_cases() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(TEST_SALT_HASH));

        // Start > end (invalid range
        assert!(seq.tokens_at(3..2).is_empty());

        // End > total (out of bounds)
        assert!(seq.tokens_at(0..10).is_empty());

        // Valid edge case: exact boundaries
        assert_eq!(seq.tokens_at(0..4).as_ref(), &[1, 2, 3, 4]);
        assert_eq!(seq.tokens_at(4..5).as_ref(), &[5]);
    }

    #[test]
    fn test_sequence_next_block() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        let block = &seq.blocks()[0];
        let next_partial = block.next_block();

        // next_block should create a partial block linked to this block
        assert!(next_partial.is_empty());
        assert_eq!(next_partial.remaining(), 4);
        assert_eq!(
            next_partial.parent_sequence_hash(),
            Some(block.sequence_hash())
        );
        assert_eq!(next_partial.position, 1);
    }

    #[test]
    fn test_sequence_reset() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9]);
        let mut seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        assert_eq!(seq.blocks().len(), 2);
        assert_eq!(seq.total_tokens(), 9);

        seq.reset();

        assert!(seq.blocks().is_empty());
        assert!(seq.current_block().is_empty());
        assert_eq!(seq.total_tokens(), 0);
        assert_eq!(seq.current_block().parent_sequence_hash(), None);
    }

    #[test]
    fn test_sequence_into_parts() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));

        let (blocks, partial) = seq.into_parts();

        assert_eq!(blocks.len(), 1);
        assert_eq!(partial.tokens.as_ref(), &[5]);
    }

    #[test]
    fn test_sequence_last_complete_block() {
        // Empty sequence
        let seq_empty = TokenBlockSequence::new(Tokens::default(), 4, None);
        assert!(seq_empty.last_complete_block().is_none());

        // With blocks
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8]);
        let seq = TokenBlockSequence::new(tokens, 4, Some(1337));
        let last = seq.last_complete_block();
        assert!(last.is_some());
        assert_eq!(last.unwrap().tokens().as_ref(), &[5, 6, 7, 8]);
    }

    // ----------------------------------------------------------------------------------------
    // Multimodal block-formation tests (#10–14 in the kv-hashing plan).
    // ----------------------------------------------------------------------------------------

    /// #10: a sequence built via `new_with_mm` with empty mm_info must equal one built via `new`.
    #[test]
    fn tokens_mm_zero_mm_equivalence() {
        let tokens = Tokens::from(vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9]);
        let baseline = TokenBlockSequence::new(tokens.clone(), 4, Some(TEST_SALT_HASH));
        let mm = TokenBlockSequence::new_with_mm(tokens, &[], 4, Some(TEST_SALT_HASH))
            .expect("validation should pass for empty mm_info");

        assert_eq!(mm.blocks().len(), baseline.blocks().len());
        for (a, b) in mm.blocks().iter().zip(baseline.blocks().iter()) {
            assert_eq!(a.salt_hash(), b.salt_hash());
            assert_eq!(a.block_hash(), b.block_hash());
            assert_eq!(a.sequence_hash(), b.sequence_hash());
            assert_eq!(a.parent_sequence_hash(), b.parent_sequence_hash());
            assert_eq!(a.positional_lineage_hash(), b.positional_lineage_hash());
        }
        assert!(mm.mm_runs().is_empty());
    }

    /// #11: byte layout — verify the MM-aware buffer matches the documented
    /// 13-bytes-per-slot tagged framing, and that block_hash is XXH3 over that exact buffer.
    #[test]
    fn tokens_mm_byte_layout() {
        // Block 0: tokens [t0..t3], placeholder run [4..6) with mm_hash=0xAA, then t6, t7.
        // block_size = 8 ⇒ block_offset = 0, run covers slots 4..6. The block is MM-affected
        // ⇒ tagged 13-byte frames apply to *every* slot.
        let tokens = Tokens::from(vec![100u32, 101, 102, 103, 0, 0, 106, 107]);
        let mm = vec![TokenBlockMmInfo {
            mm_hash: 0xAAu64,
            offset: 4,
            length: 2,
        }];
        let salt = TEST_SALT_HASH;

        // Build expected bytes manually: each slot is 13 bytes.
        let mut expected = Vec::new();
        for &t in &[100u32, 101, 102, 103] {
            expected.push(MM_SLOT_TAG_TOKEN);
            expected.extend_from_slice(&t.to_le_bytes());
            expected.extend_from_slice(&0u64.to_le_bytes());
        }
        for run_off in 0u32..2 {
            expected.push(MM_SLOT_TAG_PLACEHOLDER);
            expected.extend_from_slice(&run_off.to_le_bytes());
            expected.extend_from_slice(&0xAAu64.to_le_bytes());
        }
        for &t in &[106u32, 107] {
            expected.push(MM_SLOT_TAG_TOKEN);
            expected.extend_from_slice(&t.to_le_bytes());
            expected.extend_from_slice(&0u64.to_le_bytes());
        }
        assert_eq!(expected.len(), 8 * 13);

        // Validate helper output matches.
        let helper_bytes = compute_block_bytes_with_mm(&tokens, 0, &mm);
        assert_eq!(helper_bytes, expected, "MM-aware byte buffer mismatch");

        // Validate LocalBlockHash equals XXH3 over the expected buffer with seed=0
        // (salt enters via PLH::root_with_salt, not the per-block content hash).
        let expected_block_hash = compute_block_hash(&expected);
        let seq = TokenBlockSequence::new_with_mm(tokens, &mm, 8, Some(salt)).unwrap();
        assert_eq!(seq.blocks().len(), 1);
        assert_eq!(seq.blocks()[0].block_hash(), expected_block_hash);
    }

    /// #11b — collision regression. Reviewer P1: with the original 4/12 mixed encoding,
    /// `block_size=2` blocks `[MM(slot 0), token]` and `[token, MM(slot 1)]` could produce
    /// identical byte streams under chosen `mm_hash`/token values. With tagged 13-byte
    /// framing they MUST differ.
    #[test]
    fn tokens_mm_no_position_collision() {
        let salt = TEST_SALT_HASH;
        // Layout A: block_size=2, MM at slot 0, token at slot 1.
        let tokens_a = Tokens::from(vec![0u32, 0xAB]);
        let mm_a = vec![TokenBlockMmInfo {
            mm_hash: 0x1122_3344_5566_7788,
            offset: 0,
            length: 1,
        }];
        // Layout B: block_size=2, token at slot 0, MM at slot 1.
        let tokens_b = Tokens::from(vec![0xAB, 0u32]);
        let mm_b = vec![TokenBlockMmInfo {
            mm_hash: 0x1122_3344_5566_7788,
            offset: 1,
            length: 1,
        }];

        let bytes_a = compute_block_bytes_with_mm(&tokens_a, 0, &mm_a);
        let bytes_b = compute_block_bytes_with_mm(&tokens_b, 0, &mm_b);
        assert_ne!(
            bytes_a, bytes_b,
            "tagged framing must distinguish slot kinds at different positions"
        );

        let seq_a = TokenBlockSequence::new_with_mm(tokens_a, &mm_a, 2, Some(salt)).unwrap();
        let seq_b = TokenBlockSequence::new_with_mm(tokens_b, &mm_b, 2, Some(salt)).unwrap();
        assert_ne!(
            seq_a.blocks()[0].block_hash(),
            seq_b.blocks()[0].block_hash()
        );
    }

    /// #11c — per-block legacy fallback. A block with no overlapping MM run uses the legacy
    /// 4-byte-per-slot encoding so its `block_hash` matches the existing zero-MM path. Block 0
    /// of an MM-bearing sequence (run starts in block 1) must equal block 0 of a no-MM sequence.
    #[test]
    fn tokens_mm_legacy_fallback_per_block() {
        let block_size: u32 = 4;
        let salt = Some(TEST_SALT_HASH);
        let raw = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        // MM run covers only block 1 (positions [4..7)).
        let mm = vec![TokenBlockMmInfo {
            mm_hash: 0xAB,
            offset: 4,
            length: 3,
        }];
        let seq_mm =
            TokenBlockSequence::new_with_mm(Tokens::from(raw.clone()), &mm, block_size, salt)
                .unwrap();
        let seq_plain = TokenBlockSequence::new(Tokens::from(raw), block_size, salt);

        // Block 0 untouched by MM ⇒ identical hashes.
        assert_eq!(
            seq_mm.blocks()[0].block_hash(),
            seq_plain.blocks()[0].block_hash()
        );
        assert_eq!(
            seq_mm.blocks()[0].sequence_hash(),
            seq_plain.blocks()[0].sequence_hash()
        );
        // Block 1 IS MM-affected ⇒ hashes diverge.
        assert_ne!(
            seq_mm.blocks()[1].block_hash(),
            seq_plain.blocks()[1].block_hash()
        );
    }

    /// #11d — `offset + length` overflow is rejected as a dedicated error variant rather
    /// than panicking or silently wrapping.
    #[test]
    fn tokens_mm_validation_overflow() {
        let bad = vec![TokenBlockMmInfo {
            mm_hash: 1,
            offset: usize::MAX - 2,
            length: 10,
        }];
        let err = validate_and_sort_mm_info(&bad, usize::MAX).expect_err("must reject overflow");
        assert!(matches!(err, MmInfoError::OffsetOverflow { .. }));
    }

    /// #12: building incrementally via push_token / push_mm_run yields the same sequence
    /// as the batch `new_with_mm` constructor.
    #[test]
    fn tokens_mm_streaming_equals_batch() {
        // Layout: [t,t,t,(MM=0xAA len=4),t,t] over block_size=4 ⇒ 2 blocks + partial.
        let tokens = Tokens::from(vec![1u32, 2, 3, 0, 0, 0, 0, 6, 7]);
        let mm = vec![TokenBlockMmInfo {
            mm_hash: 0xAAu64,
            offset: 3,
            length: 4,
        }];
        let salt = Some(TEST_SALT_HASH);
        let batch = TokenBlockSequence::new_with_mm(tokens, &mm, 4, salt).unwrap();

        let mut streamed = TokenBlockSequence::new(Tokens::default(), 4, salt);
        streamed.push_token(1).unwrap();
        streamed.push_token(2).unwrap();
        streamed.push_token(3).unwrap();
        streamed.push_mm_run(0xAAu64, 4).unwrap();
        streamed.push_token(6).unwrap();
        streamed.push_token(7).unwrap();

        assert_eq!(streamed.blocks().len(), batch.blocks().len());
        for (a, b) in streamed.blocks().iter().zip(batch.blocks().iter()) {
            assert_eq!(a.block_hash(), b.block_hash(), "block_hash mismatch");
            assert_eq!(a.sequence_hash(), b.sequence_hash(), "seq_hash mismatch");
            assert_eq!(
                a.positional_lineage_hash(),
                b.positional_lineage_hash(),
                "PLH mismatch"
            );
        }
        assert_eq!(streamed.mm_runs(), batch.mm_runs());
    }

    /// #13: a multi-block MM run produces distinct block_hashes for blocks fully covered by
    /// the run (run_offset increases monotonically), and shares prefix hashes with another
    /// request whose run starts at the same global offset.
    #[test]
    fn tokens_mm_multi_block_run() {
        let block_size: u32 = 8;
        let bs = block_size as usize;
        // Run of length 2*bs + k = 20 starting at the boundary of block 0 ⇒ spans blocks 0,1,2.
        // Blocks 0 and 1 are fully placeholders; block 2 starts as 4 placeholders + 4 reals.
        let mut tokens_a: Vec<Token> = vec![0u32; 2 * bs]; // blocks 0, 1 (placeholders)
        tokens_a.extend_from_slice(&[0u32, 0, 0, 0, 100, 101, 102, 103]); // block 2
        let tokens_a = Tokens::from(tokens_a);
        let mm = vec![TokenBlockMmInfo {
            mm_hash: 0xCAFEBABEu64,
            offset: 0,
            length: 20,
        }];
        let seq_a = TokenBlockSequence::new_with_mm(
            tokens_a.clone(),
            &mm,
            block_size,
            Some(TEST_SALT_HASH),
        )
        .unwrap();
        assert_eq!(seq_a.blocks().len(), 3);

        // Block 0 and block 1 are *both* fully placeholder, but with different run_offsets
        // (0..7 vs 8..15) → block_hashes must differ.
        let bh0 = seq_a.blocks()[0].block_hash();
        let bh1 = seq_a.blocks()[1].block_hash();
        assert_ne!(
            bh0, bh1,
            "fully-placeholder blocks at different run_offsets must hash differently"
        );

        // Same image at the same global starting position in another request must share blocks.
        let seq_b =
            TokenBlockSequence::new_with_mm(tokens_a, &mm, block_size, Some(TEST_SALT_HASH))
                .unwrap();
        assert_eq!(
            seq_a.blocks()[0].block_hash(),
            seq_b.blocks()[0].block_hash()
        );
        assert_eq!(
            seq_a.blocks()[1].block_hash(),
            seq_b.blocks()[1].block_hash()
        );
        assert_eq!(
            seq_a.blocks()[2].block_hash(),
            seq_b.blocks()[2].block_hash()
        );

        // A different mm_hash at the same position must diverge starting at block 0.
        let mm_diff = vec![TokenBlockMmInfo {
            mm_hash: 0xDEADBEEFu64,
            offset: 0,
            length: 20,
        }];
        let mut tokens_c: Vec<Token> = vec![0u32; 2 * bs];
        tokens_c.extend_from_slice(&[0u32, 0, 0, 0, 100, 101, 102, 103]);
        let seq_c = TokenBlockSequence::new_with_mm(
            Tokens::from(tokens_c),
            &mm_diff,
            block_size,
            Some(TEST_SALT_HASH),
        )
        .unwrap();
        assert_ne!(
            seq_a.blocks()[0].block_hash(),
            seq_c.blocks()[0].block_hash()
        );
    }

    /// #14: mm_info validation rejects overlap, out-of-bounds, and zero-length runs.
    #[test]
    fn tokens_mm_validation() {
        let tokens = Tokens::from(vec![0u32; 32]);
        // Overlap.
        let overlap = vec![
            TokenBlockMmInfo {
                mm_hash: 1,
                offset: 0,
                length: 5,
            },
            TokenBlockMmInfo {
                mm_hash: 2,
                offset: 4,
                length: 5,
            },
        ];
        let err = TokenBlockSequence::new_with_mm(tokens.clone(), &overlap, 4, None).unwrap_err();
        assert!(matches!(
            err,
            TokenBlockError::MmInfo(MmInfoError::Overlapping { .. })
        ));

        // Out-of-bounds.
        let oob = vec![TokenBlockMmInfo {
            mm_hash: 1,
            offset: 30,
            length: 10,
        }];
        let err = TokenBlockSequence::new_with_mm(tokens.clone(), &oob, 4, None).unwrap_err();
        assert!(matches!(
            err,
            TokenBlockError::MmInfo(MmInfoError::OutOfBounds { .. })
        ));

        // Zero-length run.
        let empty = vec![TokenBlockMmInfo {
            mm_hash: 1,
            offset: 0,
            length: 0,
        }];
        let err = TokenBlockSequence::new_with_mm(tokens, &empty, 4, None).unwrap_err();
        assert!(matches!(
            err,
            TokenBlockError::MmInfo(MmInfoError::EmptyRun)
        ));

        // push_mm_run with length 0.
        let mut seq = TokenBlockSequence::new(Tokens::default(), 4, None);
        let err = seq.push_mm_run(0xAB, 0).unwrap_err();
        assert!(matches!(
            err,
            TokenBlockError::MmInfo(MmInfoError::EmptyRun)
        ));

        // truncate / pop / unwind blocked once mm_runs is non-empty.
        let mut seq = TokenBlockSequence::new(Tokens::from(vec![1u32, 2, 3]), 4, None);
        seq.push_mm_run(0xAB, 2).unwrap();
        assert!(matches!(
            seq.truncate(0).unwrap_err(),
            TokenBlockError::MmRunsPresent
        ));
        assert!(matches!(
            seq.unwind(1).unwrap_err(),
            TokenBlockError::MmRunsPresent
        ));
    }
}
