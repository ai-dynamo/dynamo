// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CBI1 encoding for absolute Cuckoo Filter bucket images.
//!
//! Deltas carry one publication's absolute bucket words. Snapshots carry the
//! complete lane in bounded, ordered chunks. Identity and stream sequencing
//! remain in the protobuf envelope; the frame repeats the CKF format and DC
//! dimension so consumers can reject drift before applying bytes.

use xxhash_rust::xxh3;

pub const IMAGES_MAGIC: [u8; 4] = *b"CBI1";
pub const IMAGES_WIRE_VERSION: u16 = 1;
pub const IMAGES_HEADER_LEN: usize = 48;
pub const SNAPSHOT_CHUNK_BUCKETS: usize = 512 * 1024;
pub const MAX_BUCKET_COUNT: usize = 1 << 24;
pub const IMAGES_MAX_FRAME_BYTES: usize = IMAGES_HEADER_LEN + 16 + SNAPSHOT_CHUNK_BUCKETS * 8;

const DELTA_IMAGE_BYTES: usize = 12;
const DELTA_BODY_PREFIX: usize = 12;
const FLAG_SNAPSHOT_CHUNK: u16 = 1;
const FLAG_DELTA: u16 = 2;

pub const FORMAT_VERSION: u16 = 1;
pub const FINGERPRINT_BITS: u8 = 16;
pub const SLOTS_PER_BUCKET: u8 = 4;

pub const fn max_delta_images() -> usize {
    (IMAGES_MAX_FRAME_BYTES - IMAGES_HEADER_LEN - DELTA_BODY_PREFIX) / DELTA_IMAGE_BYTES
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FilterFormat {
    pub seed: u64,
    pub bucket_count: usize,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum FormatError {
    #[error("unsupported CKF format version {0} (expected {FORMAT_VERSION})")]
    Version(u16),
    #[error("unsupported fingerprint width {0} (expected {FINGERPRINT_BITS})")]
    FingerprintBits(u8),
    #[error("unsupported slots per bucket {0} (expected {SLOTS_PER_BUCKET})")]
    SlotsPerBucket(u8),
    #[error("bucket count {0} is not a power of two in 2..={MAX_BUCKET_COUNT}")]
    BucketCount(usize),
    #[error("format mismatch: expected {expected:?}, received seed {seed:#x} buckets {buckets}")]
    Mismatch {
        expected: FilterFormat,
        seed: u64,
        buckets: usize,
    },
}

impl FilterFormat {
    pub fn new(seed: u64, bucket_count: usize) -> Result<Self, FormatError> {
        if !bucket_count.is_power_of_two() || !(2..=MAX_BUCKET_COUNT).contains(&bucket_count) {
            return Err(FormatError::BucketCount(bucket_count));
        }
        Ok(Self { seed, bucket_count })
    }

    fn validate(
        self,
        version: u16,
        seed: u64,
        bucket_count: usize,
        fingerprint_bits: u8,
        slots_per_bucket: u8,
    ) -> Result<(), FormatError> {
        if version != FORMAT_VERSION {
            return Err(FormatError::Version(version));
        }
        if fingerprint_bits != FINGERPRINT_BITS {
            return Err(FormatError::FingerprintBits(fingerprint_bits));
        }
        if slots_per_bucket != SLOTS_PER_BUCKET {
            return Err(FormatError::SlotsPerBucket(slots_per_bucket));
        }
        if seed != self.seed || bucket_count != self.bucket_count {
            return Err(FormatError::Mismatch {
                expected: self,
                seed,
                buckets: bucket_count,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BucketImage {
    pub bucket: u32,
    pub value: u64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ImagesWireError {
    #[error("CBI1 frame has {actual} bytes, exceeding the maximum {maximum}")]
    FrameTooLarge { actual: usize, maximum: usize },
    #[error("delta has {actual} images, exceeding the CBI1 maximum {maximum}")]
    DeltaImageCount { actual: usize, maximum: usize },
    #[error("snapshot chunk has {actual} words, exceeding the CBI1 maximum {maximum}")]
    SnapshotChunkWordCount { actual: usize, maximum: usize },
    #[error("bucket {bucket} is outside the declared {bucket_count}-bucket lane")]
    BucketIndex { bucket: u32, bucket_count: usize },
    #[error("snapshot has {actual} buckets, format declares {expected}")]
    SnapshotBucketCount { expected: usize, actual: usize },
    #[error("frame shorter than the CBI1 header")]
    Truncated,
    #[error("bad CBI1 magic")]
    Magic,
    #[error("unsupported CBI1 wire version {0}")]
    WireVersion(u16),
    #[error("unknown CBI1 frame flags {0:#06x}")]
    Flags(u16),
    #[error("CBI1 body checksum mismatch")]
    Checksum,
    #[error("CBI1 frame body is malformed")]
    Malformed,
    #[error(transparent)]
    Format(#[from] FormatError),
    #[error("delta base epoch {received} does not extend current epoch {current}")]
    EpochGap { current: u64, received: u64 },
    #[error("snapshot chunk sequence violation")]
    ChunkSequence,
    #[error("snapshot chunks do not cover the lane")]
    IncompleteCoverage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImagesHeader {
    pub dc_id: u64,
    pub epoch: u64,
    pub seed: u64,
    pub bucket_count: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ImagesFrame {
    SnapshotChunk {
        header: ImagesHeader,
        chunk_index: u32,
        chunk_count: u32,
        bucket_offset: u64,
        words: Vec<u64>,
    },
    Delta {
        header: ImagesHeader,
        base_epoch: u64,
        images: Vec<BucketImage>,
    },
}

impl ImagesFrame {
    pub const fn header(&self) -> ImagesHeader {
        match self {
            Self::SnapshotChunk { header, .. } | Self::Delta { header, .. } => *header,
        }
    }
}

pub fn encode_delta(
    format: FilterFormat,
    dc_id: u64,
    base_epoch: u64,
    epoch: u64,
    images: &[BucketImage],
) -> Result<Vec<u8>, ImagesWireError> {
    let maximum = max_delta_images();
    if images.len() > maximum {
        return Err(ImagesWireError::DeltaImageCount {
            actual: images.len(),
            maximum,
        });
    }
    if let Some(image) = images
        .iter()
        .find(|image| u64::from(image.bucket) >= format.bucket_count as u64)
    {
        return Err(ImagesWireError::BucketIndex {
            bucket: image.bucket,
            bucket_count: format.bucket_count,
        });
    }

    let mut out = Vec::with_capacity(IMAGES_HEADER_LEN + DELTA_BODY_PREFIX + images.len() * 12);
    write_header(&mut out, FLAG_DELTA, format, dc_id, epoch);
    out.extend_from_slice(&base_epoch.to_le_bytes());
    out.extend_from_slice(&(images.len() as u32).to_le_bytes());
    for image in images {
        out.extend_from_slice(&image.bucket.to_le_bytes());
        out.extend_from_slice(&image.value.to_le_bytes());
    }
    patch_checksum(&mut out);
    Ok(out)
}

pub fn encode_snapshot_chunks(
    format: FilterFormat,
    dc_id: u64,
    epoch: u64,
    words: &[u64],
) -> Result<Vec<Vec<u8>>, ImagesWireError> {
    if words.len() != format.bucket_count {
        return Err(ImagesWireError::SnapshotBucketCount {
            expected: format.bucket_count,
            actual: words.len(),
        });
    }
    let chunk_count = u32::try_from(words.len().div_ceil(SNAPSHOT_CHUNK_BUCKETS))
        .map_err(|_| ImagesWireError::Malformed)?;
    let mut frames = Vec::with_capacity(chunk_count as usize);
    for (chunk_index, chunk) in words.chunks(SNAPSHOT_CHUNK_BUCKETS).enumerate() {
        frames.push(encode_snapshot_chunk(
            format,
            dc_id,
            epoch,
            chunk_index,
            chunk_count,
            chunk,
        ));
    }
    Ok(frames)
}

pub(crate) fn encode_snapshot_chunk(
    format: FilterFormat,
    dc_id: u64,
    epoch: u64,
    chunk_index: usize,
    chunk_count: u32,
    chunk: &[u64],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(IMAGES_HEADER_LEN + 16 + chunk.len() * 8);
    write_header(&mut out, FLAG_SNAPSHOT_CHUNK, format, dc_id, epoch);
    out.extend_from_slice(&(chunk_index as u32).to_le_bytes());
    out.extend_from_slice(&chunk_count.to_le_bytes());
    out.extend_from_slice(&((chunk_index * SNAPSHOT_CHUNK_BUCKETS) as u64).to_le_bytes());
    for word in chunk {
        out.extend_from_slice(&word.to_le_bytes());
    }
    patch_checksum(&mut out);
    out
}

fn write_header(out: &mut Vec<u8>, flags: u16, format: FilterFormat, dc_id: u64, epoch: u64) {
    out.extend_from_slice(&IMAGES_MAGIC);
    out.extend_from_slice(&IMAGES_WIRE_VERSION.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());
    out.push(FINGERPRINT_BITS);
    out.push(SLOTS_PER_BUCKET);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&format.seed.to_le_bytes());
    out.extend_from_slice(&(format.bucket_count as u64).to_le_bytes());
    out.extend_from_slice(&dc_id.to_le_bytes());
    out.extend_from_slice(&epoch.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
}

fn patch_checksum(frame: &mut [u8]) {
    let checksum = xxh3::xxh3_64(&frame[IMAGES_HEADER_LEN..]) as u32;
    frame[IMAGES_HEADER_LEN - 4..IMAGES_HEADER_LEN].copy_from_slice(&checksum.to_le_bytes());
}

fn read_u16(bytes: &[u8], at: usize) -> u16 {
    let mut value = [0; 2];
    value.copy_from_slice(&bytes[at..at + 2]);
    u16::from_le_bytes(value)
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let mut value = [0; 4];
    value.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(value)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut value = [0; 8];
    value.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(value)
}

pub fn decode(expected: FilterFormat, bytes: &[u8]) -> Result<ImagesFrame, ImagesWireError> {
    if bytes.len() < IMAGES_HEADER_LEN {
        return Err(ImagesWireError::Truncated);
    }
    if bytes.len() > IMAGES_MAX_FRAME_BYTES {
        return Err(ImagesWireError::FrameTooLarge {
            actual: bytes.len(),
            maximum: IMAGES_MAX_FRAME_BYTES,
        });
    }
    if bytes[0..4] != IMAGES_MAGIC {
        return Err(ImagesWireError::Magic);
    }
    let wire_version = read_u16(bytes, 4);
    if wire_version != IMAGES_WIRE_VERSION {
        return Err(ImagesWireError::WireVersion(wire_version));
    }
    let flags = read_u16(bytes, 6);
    let fingerprint_bits = bytes[8];
    let slots_per_bucket = bytes[9];
    let format_version = read_u16(bytes, 10);
    let seed = read_u64(bytes, 12);
    let bucket_count = read_u64(bytes, 20);
    let dc_id = read_u64(bytes, 28);
    let epoch = read_u64(bytes, 36);
    let checksum = read_u32(bytes, IMAGES_HEADER_LEN - 4);
    let body = &bytes[IMAGES_HEADER_LEN..];
    if xxh3::xxh3_64(body) as u32 != checksum {
        return Err(ImagesWireError::Checksum);
    }
    expected.validate(
        format_version,
        seed,
        usize::try_from(bucket_count).map_err(|_| ImagesWireError::Malformed)?,
        fingerprint_bits,
        slots_per_bucket,
    )?;
    let header = ImagesHeader {
        dc_id,
        epoch,
        seed,
        bucket_count,
    };

    match flags {
        FLAG_DELTA => decode_delta(header, body),
        FLAG_SNAPSHOT_CHUNK => decode_snapshot_chunk(header, body),
        other => Err(ImagesWireError::Flags(other)),
    }
}

fn decode_delta(header: ImagesHeader, body: &[u8]) -> Result<ImagesFrame, ImagesWireError> {
    if body.len() < DELTA_BODY_PREFIX {
        return Err(ImagesWireError::Malformed);
    }
    let base_epoch = read_u64(body, 0);
    let count = usize::try_from(read_u32(body, 8)).map_err(|_| ImagesWireError::Malformed)?;
    let maximum = max_delta_images();
    if count > maximum {
        return Err(ImagesWireError::DeltaImageCount {
            actual: count,
            maximum,
        });
    }
    let expected_len = count
        .checked_mul(DELTA_IMAGE_BYTES)
        .and_then(|bytes| DELTA_BODY_PREFIX.checked_add(bytes))
        .ok_or(ImagesWireError::Malformed)?;
    if body.len() != expected_len {
        return Err(ImagesWireError::Malformed);
    }
    let mut images = Vec::with_capacity(count);
    for index in 0..count {
        let at = DELTA_BODY_PREFIX + index * DELTA_IMAGE_BYTES;
        let bucket = read_u32(body, at);
        if u64::from(bucket) >= header.bucket_count {
            return Err(ImagesWireError::Malformed);
        }
        images.push(BucketImage {
            bucket,
            value: read_u64(body, at + 4),
        });
    }
    Ok(ImagesFrame::Delta {
        header,
        base_epoch,
        images,
    })
}

fn decode_snapshot_chunk(
    header: ImagesHeader,
    body: &[u8],
) -> Result<ImagesFrame, ImagesWireError> {
    if body.len() < 16 {
        return Err(ImagesWireError::Malformed);
    }
    let chunk_index = read_u32(body, 0);
    let chunk_count = read_u32(body, 4);
    if chunk_count == 0 || chunk_index >= chunk_count {
        return Err(ImagesWireError::Malformed);
    }
    let bucket_offset = read_u64(body, 8);
    let words_bytes = &body[16..];
    if !words_bytes.len().is_multiple_of(8) {
        return Err(ImagesWireError::Malformed);
    }
    let word_count = words_bytes.len() / 8;
    if word_count > SNAPSHOT_CHUNK_BUCKETS {
        return Err(ImagesWireError::SnapshotChunkWordCount {
            actual: word_count,
            maximum: SNAPSHOT_CHUNK_BUCKETS,
        });
    }
    let words = words_bytes
        .chunks_exact(8)
        .map(|chunk| {
            let mut value = [0; 8];
            value.copy_from_slice(chunk);
            u64::from_le_bytes(value)
        })
        .collect::<Vec<_>>();
    let end_bucket = bucket_offset
        .checked_add(u64::try_from(words.len()).map_err(|_| ImagesWireError::Malformed)?)
        .ok_or(ImagesWireError::Malformed)?;
    if end_bucket > header.bucket_count {
        return Err(ImagesWireError::Malformed);
    }
    Ok(ImagesFrame::SnapshotChunk {
        header,
        chunk_index,
        chunk_count,
        bucket_offset,
        words,
    })
}

pub struct SnapshotAssembly {
    epoch: u64,
    chunk_count: u32,
    next_chunk: u32,
    next_bucket: u64,
    bucket_count: u64,
    images: Vec<BucketImage>,
}

impl SnapshotAssembly {
    pub fn new(format: FilterFormat) -> Self {
        Self {
            epoch: 0,
            chunk_count: 0,
            next_chunk: 0,
            next_bucket: 0,
            bucket_count: format.bucket_count as u64,
            images: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.epoch = 0;
        self.chunk_count = 0;
        self.next_chunk = 0;
        self.next_bucket = 0;
        self.images.clear();
    }

    pub fn absorb(
        &mut self,
        frame: &ImagesFrame,
    ) -> Result<Option<(u64, Vec<BucketImage>)>, ImagesWireError> {
        let ImagesFrame::SnapshotChunk {
            header,
            chunk_index,
            chunk_count,
            bucket_offset,
            words,
        } = frame
        else {
            return Err(ImagesWireError::ChunkSequence);
        };
        if *chunk_count == 0 || *chunk_index >= *chunk_count {
            self.reset();
            return Err(ImagesWireError::ChunkSequence);
        }
        if *chunk_index == 0 {
            self.reset();
            self.epoch = header.epoch;
            self.chunk_count = *chunk_count;
        }
        if *chunk_index != self.next_chunk
            || *chunk_count != self.chunk_count
            || header.epoch != self.epoch
            || header.bucket_count != self.bucket_count
            || *bucket_offset != self.next_bucket
        {
            self.reset();
            return Err(ImagesWireError::ChunkSequence);
        }
        for (offset, word) in words.iter().copied().enumerate() {
            if word != 0 {
                let bucket = bucket_offset
                    .checked_add(u64::try_from(offset).map_err(|_| ImagesWireError::Malformed)?)
                    .ok_or(ImagesWireError::Malformed)?;
                self.images.push(BucketImage {
                    bucket: u32::try_from(bucket).map_err(|_| ImagesWireError::Malformed)?,
                    value: word,
                });
            }
        }
        self.next_bucket = self
            .next_bucket
            .checked_add(u64::try_from(words.len()).map_err(|_| ImagesWireError::Malformed)?)
            .ok_or(ImagesWireError::Malformed)?;
        self.next_chunk = self
            .next_chunk
            .checked_add(1)
            .ok_or(ImagesWireError::Malformed)?;
        if self.next_chunk != self.chunk_count {
            return Ok(None);
        }
        if self.next_bucket != self.bucket_count {
            self.reset();
            return Err(ImagesWireError::IncompleteCoverage);
        }
        let epoch = self.epoch;
        let images = std::mem::take(&mut self.images);
        self.reset();
        Ok(Some((epoch, images)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format() -> FilterFormat {
        FilterFormat::new(0x5EED, 1 << 10).expect("valid fixture")
    }

    #[test]
    fn delta_round_trips_absolute_images() {
        let images = vec![
            BucketImage {
                bucket: 3,
                value: 0xAAAA_BBBB_CCCC_DDDD,
            },
            BucketImage {
                bucket: 1000,
                value: 42,
            },
        ];
        let frame = decode(
            format(),
            &encode_delta(format(), 7, 4, 5, &images).expect("encode"),
        )
        .expect("decode");
        assert_eq!(
            frame,
            ImagesFrame::Delta {
                header: ImagesHeader {
                    dc_id: 7,
                    epoch: 5,
                    seed: format().seed,
                    bucket_count: format().bucket_count as u64,
                },
                base_epoch: 4,
                images,
            }
        );
    }

    #[test]
    fn encoder_bounds_delta_and_snapshot_input() {
        let maximum = max_delta_images();
        let oversized = vec![
            BucketImage {
                bucket: 0,
                value: 0
            };
            maximum + 1
        ];
        assert_eq!(
            encode_delta(format(), 7, 0, 1, &oversized),
            Err(ImagesWireError::DeltaImageCount {
                actual: maximum + 1,
                maximum,
            })
        );
        assert_eq!(
            encode_delta(
                format(),
                7,
                0,
                1,
                &[BucketImage {
                    bucket: format().bucket_count as u32,
                    value: 1,
                }],
            ),
            Err(ImagesWireError::BucketIndex {
                bucket: format().bucket_count as u32,
                bucket_count: format().bucket_count,
            })
        );
        assert_eq!(
            encode_snapshot_chunks(format(), 7, 1, &[0]),
            Err(ImagesWireError::SnapshotBucketCount {
                expected: format().bucket_count,
                actual: 1,
            })
        );
    }

    #[test]
    fn decoder_bounds_frames_before_allocating_images() {
        assert_eq!(
            decode(format(), &vec![0; IMAGES_MAX_FRAME_BYTES + 1]),
            Err(ImagesWireError::FrameTooLarge {
                actual: IMAGES_MAX_FRAME_BYTES + 1,
                maximum: IMAGES_MAX_FRAME_BYTES,
            })
        );

        let delta_count = max_delta_images() + 1;
        let mut delta_body =
            Vec::with_capacity(DELTA_BODY_PREFIX + delta_count * DELTA_IMAGE_BYTES);
        delta_body.extend_from_slice(&0u64.to_le_bytes());
        delta_body.extend_from_slice(&(delta_count as u32).to_le_bytes());
        delta_body.resize(DELTA_BODY_PREFIX + delta_count * DELTA_IMAGE_BYTES, 0);
        assert_eq!(
            decode_delta(
                ImagesHeader {
                    dc_id: 7,
                    epoch: 1,
                    seed: format().seed,
                    bucket_count: format().bucket_count as u64,
                },
                &delta_body,
            ),
            Err(ImagesWireError::DeltaImageCount {
                actual: delta_count,
                maximum: max_delta_images(),
            })
        );

        let word_count = SNAPSHOT_CHUNK_BUCKETS + 1;
        let mut snapshot_body = vec![0; 16 + word_count * 8];
        snapshot_body[4..8].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            decode_snapshot_chunk(
                ImagesHeader {
                    dc_id: 7,
                    epoch: 1,
                    seed: format().seed,
                    bucket_count: MAX_BUCKET_COUNT as u64,
                },
                &snapshot_body,
            ),
            Err(ImagesWireError::SnapshotChunkWordCount {
                actual: word_count,
                maximum: SNAPSHOT_CHUNK_BUCKETS,
            })
        );
    }

    #[test]
    fn malformed_corrupt_and_drifted_frames_fail_closed() {
        assert_eq!(
            decode(format(), &[0; IMAGES_HEADER_LEN - 1]),
            Err(ImagesWireError::Truncated)
        );

        let mut corrupt = encode_delta(format(), 7, 0, 1, &[]).expect("encode");
        corrupt.push(1);
        assert_eq!(decode(format(), &corrupt), Err(ImagesWireError::Checksum));

        let bytes = encode_delta(format(), 7, 0, 1, &[]).expect("encode");
        let other = FilterFormat::new(format().seed ^ 1, format().bucket_count).expect("format");
        assert!(matches!(
            decode(other, &bytes),
            Err(ImagesWireError::Format(FormatError::Mismatch { .. }))
        ));

        let mut flags = bytes;
        flags[6..8].copy_from_slice(&0x8000u16.to_le_bytes());
        assert_eq!(
            decode(format(), &flags),
            Err(ImagesWireError::Flags(0x8000))
        );

        let mut format_version =
            encode_delta(format(), 7, 0, 1, &[]).expect("encode version fixture");
        format_version[10..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        assert_eq!(
            decode(format(), &format_version),
            Err(ImagesWireError::Format(FormatError::Version(
                FORMAT_VERSION + 1
            )))
        );
    }

    #[test]
    fn snapshot_chunks_rebuild_exact_lane_and_reject_reordering() {
        let format = FilterFormat::new(0x5EED, 1 << 20).expect("format");
        let mut words = vec![0; format.bucket_count];
        words[0] = 11;
        words[SNAPSHOT_CHUNK_BUCKETS + 1] = 22;
        words[format.bucket_count - 1] = 33;
        let encoded = encode_snapshot_chunks(format, 9, 6, &words).expect("encode snapshot");
        assert_eq!(encoded.len(), 2);

        let decoded = encoded
            .iter()
            .map(|frame| decode(format, frame).expect("decode chunk"))
            .collect::<Vec<_>>();
        let mut reordered = SnapshotAssembly::new(format);
        assert_eq!(
            reordered.absorb(&decoded[1]),
            Err(ImagesWireError::ChunkSequence)
        );

        let mut assembly = SnapshotAssembly::new(format);
        let mut complete = None;
        for frame in &decoded {
            if let Some(snapshot) = assembly.absorb(frame).expect("ordered chunk") {
                complete = Some(snapshot);
            }
        }
        let (epoch, images) = complete.expect("complete snapshot");
        assert_eq!(epoch, 6);
        let mut rebuilt = vec![0; format.bucket_count];
        for image in images {
            rebuilt[image.bucket as usize] = image.value;
        }
        assert_eq!(rebuilt, words);
    }
}
