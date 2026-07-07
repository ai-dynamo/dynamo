// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Encode CKF1 snapshots and deltas in a form that can be checked, chunked, and
//! applied without copying whole filters into temporary buffers.

use super::filter::{CuckooFilter, SLOTS};
use super::pages::BucketPages;

const SNAP_MAGIC: [u8; 4] = *b"CKF1";
const SNAP_VERSION: u16 = 1;
pub const SNAP_HEADER_LEN: usize = 48;

/// Typed snapshot failures let the consumer distinguish corruption, mismatch,
/// and incomplete assembly during recovery.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SnapshotError {
    #[error("truncated header")]
    TruncatedHeader,
    #[error("bad magic")]
    BadMagic,
    #[error("version mismatch")]
    VersionMismatch { expected: u16, actual: u16 },
    #[error("incompatible filter params")]
    IncompatibleFilterParams,
    #[error("num_buckets not pow2")]
    NumBucketsNotPowerOfTwo { num_buckets: u64 },
    #[error("not a snapshot chunk")]
    NotSnapshotChunk,
    #[error("chunk checksum mismatch (corruption)")]
    ChunkChecksumMismatch,
    #[error("truncated chunk")]
    TruncatedChunk,
    #[error("chunk slot bytes misaligned")]
    MisalignedChunkSlots,
    #[error("chunk index out of range")]
    ChunkIndexOutOfRange { index: u32, count: u32 },
    #[error("chunk bucket range out of bounds")]
    ChunkBucketRangeOutOfBounds {
        offset: u64,
        count: u64,
        num_buckets: u64,
    },
    #[error("snapshot chunk without a leading first chunk")]
    MissingFirstChunk,
    #[error("snapshot chunk sequence mismatch")]
    ChunkSequenceMismatch,
    #[error("snapshot chunks do not cover the filter")]
    IncompleteCoverage,
    #[error("incomplete snapshot chunk sequence")]
    IncompleteSequence,
}

/// Typed delta failures let the consumer resync only when the delta contract
/// really broke.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeltaError {
    #[error(transparent)]
    Frame(#[from] SnapshotError),
    #[error("not a delta")]
    NotDelta,
    #[error("no base snapshot for DC")]
    MissingBaseSnapshot,
    #[error("delta shape/seed mismatch — full snapshot required")]
    ShapeOrSeedMismatch,
    #[error("delta checksum mismatch")]
    ChecksumMismatch,
    #[error("truncated delta")]
    Truncated,
    #[error("delta base_epoch gap — full snapshot required")]
    BaseEpochGap { expected: u64, actual: u64 },
    #[error("delta new_epoch is not the base epoch plus one")]
    NewEpochOutOfOrder { base: u64, new: u64 },
    #[error("delta entry count mismatch")]
    EntryCountMismatch { count: usize, actual_bytes: usize },
    #[error("delta bucket index out of range")]
    BucketIndexOutOfRange { index: usize, num_buckets: usize },
    #[error("delta repeats bucket index")]
    RepeatedBucketIndex { index: usize },
    #[error("delta bucket indices are not strictly increasing")]
    BucketOrder { previous: usize, current: usize },
}
/// Keep chunk bodies around 4 MiB so they stay under message caps and keep
/// consumer work bounded.
const CHUNK_BUCKETS: usize = (4 * 1024 * 1024) / (SLOTS * 2);
pub(super) const CHUNK_BODY_PREFIX: usize = 16;

struct SnapFlags;
impl SnapFlags {
    const DELTA: u16 = 1;
    const CHUNK: u16 = 2;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotMeta {
    pub dc_worker_id: u64,
    pub filter_epoch: u64,
    pub num_buckets: usize,
    pub seed: u64,
}

struct ParsedHeader {
    flags: u16,
    seed: u64,
    num_buckets: u64,
    dc_worker_id: u64,
    epoch: u64,
    body_checksum: u32,
}

#[allow(clippy::too_many_arguments)]
fn fill_header(
    out: &mut [u8],
    flags: u16,
    seed: u64,
    num_buckets: u64,
    dc_worker_id: u64,
    epoch: u64,
    body_checksum: u32,
) {
    debug_assert_eq!(out.len(), SNAP_HEADER_LEN);
    out[0..4].copy_from_slice(&SNAP_MAGIC);
    out[4..6].copy_from_slice(&SNAP_VERSION.to_le_bytes());
    out[6..8].copy_from_slice(&flags.to_le_bytes());
    out[8] = 16;
    out[9] = SLOTS as u8;
    out[10] = 0;
    out[11] = 0;
    out[12..20].copy_from_slice(&seed.to_le_bytes());
    out[20..28].copy_from_slice(&num_buckets.to_le_bytes());
    out[28..36].copy_from_slice(&dc_worker_id.to_le_bytes());
    out[36..44].copy_from_slice(&epoch.to_le_bytes());
    out[44..48].copy_from_slice(&body_checksum.to_le_bytes());
}

fn parse_header(bytes: &[u8]) -> Result<ParsedHeader, SnapshotError> {
    if bytes.len() < SNAP_HEADER_LEN {
        return Err(SnapshotError::TruncatedHeader);
    }
    if bytes[0..4] != SNAP_MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let rd16 = |o: usize| u16::from_le_bytes([bytes[o], bytes[o + 1]]);
    let rd32 = |o: usize| u32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    let rd64 = |o: usize| u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
    let version = rd16(4);
    if version != SNAP_VERSION {
        return Err(SnapshotError::VersionMismatch {
            expected: SNAP_VERSION,
            actual: version,
        });
    }
    if bytes[8] != 16 || bytes[9] != SLOTS as u8 || bytes[10] != 0 || bytes[11] != 0 {
        return Err(SnapshotError::IncompatibleFilterParams);
    }
    let num_buckets = rd64(20);
    if !(num_buckets as usize).is_power_of_two() {
        return Err(SnapshotError::NumBucketsNotPowerOfTwo { num_buckets });
    }
    Ok(ParsedHeader {
        flags: rd16(6),
        seed: rd64(12),
        num_buckets,
        dc_worker_id: rd64(28),
        epoch: rd64(36),
        body_checksum: rd32(44),
    })
}

/// Use a fast checksum for corruption detection; transport security already
/// comes from mTLS.
fn body_checksum(body: &[u8]) -> u32 {
    xxhash_rust::xxh3::xxh3_64(body) as u32
}

/// Append slots in little-endian form without flattening the page store first.
#[inline]
fn slots_to_le_bytes(out: &mut Vec<u8>, slots: &[u16]) {
    #[cfg(target_endian = "little")]
    out.extend_from_slice(bytemuck::cast_slice(slots));
    #[cfg(target_endian = "big")]
    for &fp in slots {
        out.extend_from_slice(&fp.to_le_bytes());
    }
}

/// Hold a point-in-time view by sharing untouched pages and copying only dirty
/// pages on mutation.
pub struct SnapshotState {
    pub(super) buckets: BucketPages,
    pub(super) num_buckets: usize,
    pub(super) seed: u64,
    pub(super) dc_worker_id: u64,
    pub(super) epoch: u64,
}

impl SnapshotState {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    fn chunk_count_with(&self, per: usize) -> usize {
        self.num_buckets.div_ceil(per.max(1))
    }

    /// Serialize self-contained chunks so the receiver never needs a whole
    /// snapshot buffer.
    pub fn chunks(&self) -> impl Iterator<Item = Vec<u8>> + '_ {
        self.chunks_with(CHUNK_BUCKETS)
    }

    /// Use a custom chunk size when tests or callers want to stress assembly.
    pub fn chunks_with(&self, per: usize) -> impl Iterator<Item = Vec<u8>> + '_ {
        let per = per.max(1);
        let count = self.chunk_count_with(per);
        (0..count).map(move |index| self.encode_chunk(index, count, per))
    }

    /// Byte length of each chunk `chunks()` would emit, derived from the layout
    /// so a publisher can account update bytes without paying the serialization.
    pub fn chunk_lens(&self) -> impl Iterator<Item = usize> + '_ {
        let per = CHUNK_BUCKETS;
        let count = self.chunk_count_with(per);
        (0..count).map(move |index| {
            let offset = index * per;
            let buckets = per.min(self.num_buckets - offset);
            SNAP_HEADER_LEN + CHUNK_BODY_PREFIX + buckets * SLOTS * 2
        })
    }

    fn encode_chunk(&self, index: usize, count: usize, per: usize) -> Vec<u8> {
        let offset = index * per;
        let buckets = per.min(self.num_buckets - offset);
        let lo = offset * SLOTS;
        let hi = lo + buckets * SLOTS;

        let body_len = CHUNK_BODY_PREFIX + (hi - lo) * 2;
        let mut out = Vec::with_capacity(SNAP_HEADER_LEN + body_len);
        out.resize(SNAP_HEADER_LEN, 0);
        out.extend_from_slice(&(index as u32).to_le_bytes());
        out.extend_from_slice(&(count as u32).to_le_bytes());
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        self.buckets.append_le_bytes(&mut out, lo, hi);
        let checksum = body_checksum(&out[SNAP_HEADER_LEN..]);
        fill_header(
            &mut out[..SNAP_HEADER_LEN],
            SnapFlags::CHUNK,
            self.seed,
            self.num_buckets as u64,
            self.dc_worker_id,
            self.epoch,
            checksum,
        );
        out
    }
}

/// Parsed view of one snapshot chunk frame.
#[derive(Debug)]
pub(super) struct ChunkInfo<'a> {
    meta: SnapshotMeta,
    chunk_index: u32,
    chunk_count: u32,
    bucket_offset: u64,
    slots: &'a [u8],
}

impl ChunkInfo<'_> {
    fn bucket_count(&self) -> usize {
        self.slots.len() / (SLOTS * 2)
    }
}

/// Check the frame tag cheaply when callers only need to branch on delta vs
/// non-delta.
pub fn is_delta(bytes: &[u8]) -> bool {
    frame_flags(bytes) == Some(SnapFlags::DELTA)
}

/// Check whether the frame tag denotes a snapshot chunk.
pub fn is_chunk(bytes: &[u8]) -> bool {
    frame_flags(bytes) == Some(SnapFlags::CHUNK)
}

fn frame_flags(bytes: &[u8]) -> Option<u16> {
    (bytes.len() >= 8 && bytes[0..4] == SNAP_MAGIC)
        .then(|| u16::from_le_bytes([bytes[6], bytes[7]]))
}

pub(super) fn parse_chunk(bytes: &[u8]) -> Result<ChunkInfo<'_>, SnapshotError> {
    let h = parse_header(bytes)?;
    if h.flags != SnapFlags::CHUNK {
        return Err(SnapshotError::NotSnapshotChunk);
    }
    let body = &bytes[SNAP_HEADER_LEN..];
    if body_checksum(body) != h.body_checksum {
        return Err(SnapshotError::ChunkChecksumMismatch);
    }
    if body.len() < CHUNK_BODY_PREFIX {
        return Err(SnapshotError::TruncatedChunk);
    }
    let chunk_index = u32::from_le_bytes(body[0..4].try_into().unwrap());
    let chunk_count = u32::from_le_bytes(body[4..8].try_into().unwrap());
    let bucket_offset = u64::from_le_bytes(body[8..16].try_into().unwrap());
    let slots = &body[CHUNK_BODY_PREFIX..];
    if !slots.len().is_multiple_of(SLOTS * 2) {
        return Err(SnapshotError::MisalignedChunkSlots);
    }
    if chunk_count == 0 || chunk_index >= chunk_count {
        return Err(SnapshotError::ChunkIndexOutOfRange {
            index: chunk_index,
            count: chunk_count,
        });
    }
    let bucket_count = (slots.len() / (SLOTS * 2)) as u64;
    if bucket_offset.saturating_add(bucket_count) > h.num_buckets {
        return Err(SnapshotError::ChunkBucketRangeOutOfBounds {
            offset: bucket_offset,
            count: bucket_count,
            num_buckets: h.num_buckets,
        });
    }
    Ok(ChunkInfo {
        meta: SnapshotMeta {
            dc_worker_id: h.dc_worker_id,
            filter_epoch: h.epoch,
            num_buckets: h.num_buckets as usize,
            seed: h.seed,
        },
        chunk_index,
        chunk_count,
        bucket_offset,
        slots,
    })
}

/// Reassemble chunked snapshots directly into a pre-allocated filter so the
/// receiver never needs an intermediate whole-snapshot buffer.
#[derive(Default)]
pub struct SnapshotAssembler {
    inner: Option<Assembling>,
}

struct Assembling {
    filter: CuckooFilter,
    meta: SnapshotMeta,
    chunk_count: u32,
    next_chunk: u32,
    next_offset: u64,
    len: usize,
}

impl SnapshotAssembler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn in_progress(&self) -> bool {
        self.inner.is_some()
    }

    /// Apply one chunk frame and drop partial state on any error.
    pub fn push(
        &mut self,
        bytes: &[u8],
    ) -> Result<Option<(CuckooFilter, SnapshotMeta)>, SnapshotError> {
        let result = self.push_impl(bytes);
        if result.is_err() {
            self.inner = None;
        }
        result
    }

    fn push_impl(
        &mut self,
        bytes: &[u8],
    ) -> Result<Option<(CuckooFilter, SnapshotMeta)>, SnapshotError> {
        let chunk = parse_chunk(bytes)?;
        if chunk.chunk_index == 0 {
            self.inner = Some(Assembling {
                filter: CuckooFilter::with_num_buckets(chunk.meta.num_buckets, chunk.meta.seed),
                meta: chunk.meta,
                chunk_count: chunk.chunk_count,
                next_chunk: 0,
                next_offset: 0,
                len: 0,
            });
        }
        let mut assembling = match self.inner.take() {
            Some(assembling) => assembling,
            None => return Err(SnapshotError::MissingFirstChunk),
        };
        if chunk.chunk_index != assembling.next_chunk
            || chunk.chunk_count != assembling.chunk_count
            || chunk.bucket_offset != assembling.next_offset
            || chunk.meta != assembling.meta
        {
            return Err(SnapshotError::ChunkSequenceMismatch);
        }
        let lo = chunk.bucket_offset as usize * SLOTS;
        assembling.filter.buckets.write_le_bytes(lo, chunk.slots);
        assembling.len += chunk
            .slots
            .chunks_exact(2)
            .filter(|bytes| bytes[0] != 0 || bytes[1] != 0)
            .count();
        assembling.next_chunk += 1;
        assembling.next_offset += chunk.bucket_count() as u64;

        if assembling.next_chunk == assembling.chunk_count {
            if assembling.next_offset != assembling.meta.num_buckets as u64 {
                return Err(SnapshotError::IncompleteCoverage);
            }
            assembling.filter.len = assembling.len;
            return Ok(Some((assembling.filter, assembling.meta)));
        }
        self.inner = Some(assembling);
        Ok(None)
    }
}

/// Convenience wrapper for callers that already have all chunk bytes.
pub fn assemble_chunks<I, B>(chunks: I) -> Result<(CuckooFilter, SnapshotMeta), SnapshotError>
where
    I: IntoIterator<Item = B>,
    B: AsRef<[u8]>,
{
    let mut assembler = SnapshotAssembler::new();
    let mut complete = None;
    for chunk in chunks {
        if complete.is_some() {
            return Err(SnapshotError::ChunkSequenceMismatch);
        }
        complete = assembler.push(chunk.as_ref())?;
    }
    complete.ok_or(SnapshotError::IncompleteSequence)
}

#[derive(Debug, Clone)]
pub struct DeltaInfo {
    pub dc_worker_id: u64,
    pub base_epoch: u64,
    pub new_epoch: u64,
    pub entries: Vec<DeltaEntry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaEntry {
    pub bucket: usize,
    pub slots: [u16; SLOTS],
}

/// Build an incremental delta from the buckets that actually changed; a shape
/// mismatch must fall back to a full snapshot.
///
/// `base` holds the values as of the last successful publish (see
/// [`SnapshotProducer`](super::producer::SnapshotProducer)'s `last_shipped`)
/// so a bucket that was touched but net-reverted to its shipped value is
/// dropped instead of costing wire bytes.
pub(super) fn build_delta_for_buckets(
    base: &BucketPages,
    new: &CuckooFilter,
    dc_worker_id: u64,
    base_epoch: u64,
    new_epoch: u64,
    buckets: impl ExactSizeIterator<Item = usize>,
) -> Option<Vec<u8>> {
    let nb = new.num_buckets();
    if base.len() != nb * SLOTS {
        return None;
    }
    // Serialize straight into the final frame — the payload can approach
    // MAX_DELTA_BYTES, so staging it through intermediate buffers would
    // multiply both memcpy and peak memory. Count and checksum are only known
    // after the scan, so they are backfilled.
    let mut out: Vec<u8> =
        Vec::with_capacity(SNAP_HEADER_LEN + 12 + buckets.len() * (4 + SLOTS * 2));
    out.resize(SNAP_HEADER_LEN, 0);
    out.extend_from_slice(&base_epoch.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    let count_offset = out.len() - 4;
    let mut count = 0u32;
    for b in buckets {
        if b >= nb {
            return None;
        }
        let lo = b * SLOTS;
        let new_slots = new.bucket_slots(b);
        let base_slots: [u16; SLOTS] = base.read_array(lo);
        if new_slots != base_slots {
            out.extend_from_slice(&(b as u32).to_le_bytes());
            slots_to_le_bytes(&mut out, &new_slots);
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    out[count_offset..count_offset + 4].copy_from_slice(&count.to_le_bytes());
    let cks = body_checksum(&out[SNAP_HEADER_LEN..]);
    fill_header(
        &mut out[..SNAP_HEADER_LEN],
        SnapFlags::DELTA,
        new.seed,
        nb as u64,
        dc_worker_id,
        new_epoch,
        cks,
    );
    Some(out)
}

/// Apply a delta in place and force a resync when the frame contract or epoch
/// no longer matches.
pub fn decode_delta(
    filter: &CuckooFilter,
    current_epoch: u64,
    bytes: &[u8],
) -> Result<DeltaInfo, DeltaError> {
    let h = parse_header(bytes)?;
    if h.flags != SnapFlags::DELTA {
        return Err(DeltaError::NotDelta);
    }
    if h.num_buckets as usize != filter.num_buckets() || h.seed != filter.seed {
        return Err(DeltaError::ShapeOrSeedMismatch);
    }
    let body = &bytes[SNAP_HEADER_LEN..];
    if body_checksum(body) != h.body_checksum {
        return Err(DeltaError::ChecksumMismatch);
    }
    if body.len() < 12 {
        return Err(DeltaError::Truncated);
    }
    let base_epoch = u64::from_le_bytes(body[0..8].try_into().unwrap());
    if base_epoch != current_epoch {
        return Err(DeltaError::BaseEpochGap {
            expected: current_epoch,
            actual: base_epoch,
        });
    }
    if base_epoch.checked_add(1) != Some(h.epoch) {
        return Err(DeltaError::NewEpochOutOfOrder {
            base: base_epoch,
            new: h.epoch,
        });
    }
    let count = u32::from_le_bytes(body[8..12].try_into().unwrap()) as usize;
    let entry_len = 4 + SLOTS * 2;
    let entries = &body[12..];
    if entries.len() != count * entry_len {
        return Err(DeltaError::EntryCountMismatch {
            count,
            actual_bytes: entries.len(),
        });
    }
    let mut decoded = Vec::with_capacity(count);
    let mut previous = None;
    for e in entries.chunks_exact(entry_len) {
        let b = u32::from_le_bytes(e[0..4].try_into().unwrap()) as usize;
        if b >= filter.num_buckets() {
            return Err(DeltaError::BucketIndexOutOfRange {
                index: b,
                num_buckets: filter.num_buckets(),
            });
        }
        if let Some(previous) = previous {
            if previous == b {
                return Err(DeltaError::RepeatedBucketIndex { index: b });
            }
            if previous > b {
                return Err(DeltaError::BucketOrder {
                    previous,
                    current: b,
                });
            }
        }
        previous = Some(b);
        let mut new_slots = [0; SLOTS];
        for (s, slot) in new_slots.iter_mut().enumerate() {
            let off = 4 + s * 2;
            let fp = u16::from_le_bytes([e[off], e[off + 1]]);
            *slot = fp;
        }
        decoded.push(DeltaEntry {
            bucket: b,
            slots: new_slots,
        });
    }
    Ok(DeltaInfo {
        dc_worker_id: h.dc_worker_id,
        base_epoch,
        new_epoch: h.epoch,
        entries: decoded,
    })
}

pub fn apply_decoded_delta(filter: &mut CuckooFilter, delta: &DeltaInfo) {
    let mut len = filter.len;
    for entry in &delta.entries {
        let old_slots = filter.bucket_slots(entry.bucket);
        let old_count = old_slots.iter().filter(|&&fp| fp != 0).count();
        let new_count = entry.slots.iter().filter(|&&fp| fp != 0).count();
        filter.set_bucket_slots(entry.bucket, &entry.slots);
        len = len + new_count - old_count;
    }
    filter.len = len;
}

pub fn apply_delta(
    filter: &mut CuckooFilter,
    current_epoch: u64,
    bytes: &[u8],
) -> Result<DeltaInfo, DeltaError> {
    let delta = decode_delta(filter, current_epoch, bytes)?;
    apply_decoded_delta(filter, &delta);
    Ok(delta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::cuckoo::{DEFAULT_FILTER_SEED, Publish, SnapshotProducer};

    fn filter_and_delta() -> (CuckooFilter, u64, Vec<u8>) {
        let mut producer = SnapshotProducer::new(7, 64, DEFAULT_FILTER_SEED);
        assert!(producer.insert(11));
        let snapshot = producer.full_snapshot();
        let (filter, meta) = assemble_chunks(snapshot.chunks_with(4)).unwrap();
        assert!(producer.insert(22));
        let Publish::Delta(delta) = producer.publish() else {
            panic!("expected delta")
        };
        (filter, meta.filter_epoch, delta)
    }

    fn refresh_checksum(frame: &mut [u8]) {
        let checksum = body_checksum(&frame[SNAP_HEADER_LEN..]);
        frame[44..48].copy_from_slice(&checksum.to_le_bytes());
    }

    #[test]
    fn rejected_checksum_leaves_filter_unchanged() {
        let (mut filter, epoch, mut delta) = filter_and_delta();
        let before = filter.to_raw_buckets();
        *delta.last_mut().unwrap() ^= 1;
        assert!(matches!(
            apply_delta(&mut filter, epoch, &delta),
            Err(DeltaError::ChecksumMismatch)
        ));
        assert_eq!(filter.to_raw_buckets(), before);
    }

    #[test]
    fn duplicate_bucket_is_rejected_before_mutation() {
        let (mut filter, epoch, mut delta) = filter_and_delta();
        let before = filter.to_raw_buckets();
        let entry_len = 4 + SLOTS * 2;
        let entry = delta[SNAP_HEADER_LEN + 12..SNAP_HEADER_LEN + 12 + entry_len].to_vec();
        delta.extend_from_slice(&entry);
        delta[SNAP_HEADER_LEN + 8..SNAP_HEADER_LEN + 12].copy_from_slice(&2u32.to_le_bytes());
        let checksum = body_checksum(&delta[SNAP_HEADER_LEN..]);
        delta[44..48].copy_from_slice(&checksum.to_le_bytes());
        assert!(matches!(
            apply_delta(&mut filter, epoch, &delta),
            Err(DeltaError::RepeatedBucketIndex { .. })
        ));
        assert_eq!(filter.to_raw_buckets(), before);
    }

    #[test]
    fn skipped_epoch_is_rejected() {
        let (mut filter, epoch, delta) = filter_and_delta();
        assert!(matches!(
            apply_delta(&mut filter, epoch + 1, &delta),
            Err(DeltaError::BaseEpochGap { .. })
        ));
    }

    #[test]
    fn shape_seed_length_and_bucket_bounds_are_validated_before_mutation() {
        let (filter, epoch, delta) = filter_and_delta();
        let original = filter.to_raw_buckets();

        let mut wrong_seed = delta.clone();
        wrong_seed[12..20].copy_from_slice(&(filter.seed() ^ 1).to_le_bytes());
        let mut candidate = filter.clone();
        assert!(matches!(
            apply_delta(&mut candidate, epoch, &wrong_seed),
            Err(DeltaError::ShapeOrSeedMismatch)
        ));
        assert_eq!(candidate.to_raw_buckets(), original);

        let mut wrong_shape = delta.clone();
        wrong_shape[20..28].copy_from_slice(&((filter.num_buckets() * 2) as u64).to_le_bytes());
        let mut candidate = filter.clone();
        assert!(matches!(
            apply_delta(&mut candidate, epoch, &wrong_shape),
            Err(DeltaError::ShapeOrSeedMismatch)
        ));
        assert_eq!(candidate.to_raw_buckets(), original);

        let mut truncated = delta.clone();
        truncated.pop();
        refresh_checksum(&mut truncated);
        let mut candidate = filter.clone();
        assert!(matches!(
            apply_delta(&mut candidate, epoch, &truncated),
            Err(DeltaError::EntryCountMismatch { .. })
        ));
        assert_eq!(candidate.to_raw_buckets(), original);

        let mut out_of_range = delta;
        out_of_range[SNAP_HEADER_LEN + 12..SNAP_HEADER_LEN + 16]
            .copy_from_slice(&(filter.num_buckets() as u32).to_le_bytes());
        refresh_checksum(&mut out_of_range);
        let mut candidate = filter;
        assert!(matches!(
            apply_delta(&mut candidate, epoch, &out_of_range),
            Err(DeltaError::BucketIndexOutOfRange { .. })
        ));
        assert_eq!(candidate.to_raw_buckets(), original);
    }

    #[test]
    fn snapshot_assembler_rejects_truncated_and_out_of_order_chunks() {
        let mut producer = SnapshotProducer::new(7, 64, DEFAULT_FILTER_SEED);
        assert!(producer.insert(11));
        let mut chunks: Vec<_> = producer.full_snapshot().chunks_with(1).collect();
        assert!(chunks.len() > 1);

        let mut truncated = chunks[0].clone();
        truncated.pop();
        refresh_checksum(&mut truncated);
        assert!(matches!(
            assemble_chunks([truncated]),
            Err(SnapshotError::MisalignedChunkSlots)
                | Err(SnapshotError::IncompleteCoverage)
                | Err(SnapshotError::IncompleteSequence)
        ));

        let mut with_trailing_chunk = chunks.clone();
        with_trailing_chunk.push(chunks[0].clone());
        assert!(matches!(
            assemble_chunks(with_trailing_chunk),
            Err(SnapshotError::ChunkSequenceMismatch)
        ));

        chunks.swap(0, 1);
        assert!(matches!(
            assemble_chunks(chunks),
            Err(SnapshotError::MissingFirstChunk)
        ));
    }
}
