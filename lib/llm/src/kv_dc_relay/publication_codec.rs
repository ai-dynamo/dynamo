// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use dynamo_kv_router::indexer::cuckoo::{DcCkfDelta, DcCkfFormatIdentity, ProducerIdentity};

use super::protocol::wire::images;
use super::publication_hub::HubSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PublicationFrameKind {
    SnapshotChunk,
    Delta,
    Heartbeat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PublicationFrame {
    pub(crate) identity: ProducerIdentity,
    pub(crate) base_sequence: u64,
    pub(crate) sequence: u64,
    pub(crate) kind: PublicationFrameKind,
    pub(crate) payload: Bytes,
}

impl PublicationFrame {
    pub(crate) fn queued_bytes(&self) -> usize {
        // Account for the encoded payload plus a conservative protobuf/gRPC envelope budget.
        self.payload.len().saturating_add(256)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum Cbi1AdapterError {
    #[error(transparent)]
    Format(#[from] images::FormatError),
    #[error(transparent)]
    Wire(#[from] images::ImagesWireError),
    #[error(
        "unsupported CKF format identity: version={format_version}, fingerprint_bits={fingerprint_bits}, slots_per_bucket={slots_per_bucket}"
    )]
    UnsupportedFormatIdentity {
        format_version: u16,
        fingerprint_bits: u8,
        slots_per_bucket: u8,
    },
    #[error("snapshot has {actual} buckets, format declares {expected}")]
    SnapshotBucketCount { expected: usize, actual: usize },
    #[error("delta bucket index {0} exceeds the CBI1 u32 address space")]
    BucketIndexOverflow(usize),
}

pub(crate) struct Cbi1SnapshotFrames {
    snapshot: HubSnapshot,
    format: images::FilterFormat,
    next_chunk: usize,
    chunk_count: usize,
}

impl Iterator for Cbi1SnapshotFrames {
    type Item = PublicationFrame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_chunk == self.chunk_count {
            return None;
        }
        let chunk_index = self.next_chunk;
        self.next_chunk += 1;
        let start = chunk_index * images::SNAPSHOT_CHUNK_BUCKETS;
        let end = (start + images::SNAPSHOT_CHUNK_BUCKETS).min(self.snapshot.buckets().len());
        let identity = self.snapshot.identity();
        let sequence = self.snapshot.sequence();
        Some(PublicationFrame {
            identity,
            base_sequence: sequence,
            sequence,
            kind: PublicationFrameKind::SnapshotChunk,
            payload: images::encode_snapshot_chunk(
                self.format,
                identity.dc_id().get(),
                sequence,
                chunk_index,
                self.chunk_count as u32,
                &self.snapshot.buckets()[start..end],
            )
            .into(),
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.chunk_count - self.next_chunk;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Cbi1SnapshotFrames {}

pub(crate) fn encode_snapshot(
    snapshot: HubSnapshot,
) -> Result<Cbi1SnapshotFrames, Cbi1AdapterError> {
    let format = filter_format(snapshot.identity().format())?;
    if snapshot.buckets().len() != format.bucket_count {
        return Err(Cbi1AdapterError::SnapshotBucketCount {
            expected: format.bucket_count,
            actual: snapshot.buckets().len(),
        });
    }
    Ok(Cbi1SnapshotFrames {
        chunk_count: snapshot
            .buckets()
            .len()
            .div_ceil(images::SNAPSHOT_CHUNK_BUCKETS),
        snapshot,
        format,
        next_chunk: 0,
    })
}

pub(crate) fn encode_delta(delta: &DcCkfDelta) -> Result<PublicationFrame, Cbi1AdapterError> {
    let identity = delta.identity();
    let format = filter_format(identity.format())?;
    let mut bucket_images = Vec::with_capacity(delta.images().len());
    for image in delta.images() {
        bucket_images.push(images::BucketImage {
            bucket: u32::try_from(image.bucket())
                .map_err(|_| Cbi1AdapterError::BucketIndexOverflow(image.bucket()))?,
            value: image.value(),
        });
    }
    Ok(PublicationFrame {
        identity,
        base_sequence: delta.base_sequence(),
        sequence: delta.sequence(),
        kind: PublicationFrameKind::Delta,
        payload: images::encode_delta(
            format,
            identity.dc_id().get(),
            delta.base_sequence(),
            delta.sequence(),
            &bucket_images,
        )?
        .into(),
    })
}

pub(crate) fn encode_heartbeat(identity: ProducerIdentity, sequence: u64) -> PublicationFrame {
    PublicationFrame {
        identity,
        base_sequence: sequence,
        sequence,
        kind: PublicationFrameKind::Heartbeat,
        payload: Bytes::new(),
    }
}

fn filter_format(format: DcCkfFormatIdentity) -> Result<images::FilterFormat, Cbi1AdapterError> {
    if format.format_version() != images::FORMAT_VERSION
        || format.fingerprint_bits() != images::FINGERPRINT_BITS
        || format.slots_per_bucket() != images::SLOTS_PER_BUCKET
    {
        return Err(Cbi1AdapterError::UnsupportedFormatIdentity {
            format_version: format.format_version(),
            fingerprint_bits: format.fingerprint_bits(),
            slots_per_bucket: format.slots_per_bucket(),
        });
    }
    Ok(images::FilterFormat::new(
        format.seed(),
        format.bucket_count(),
    )?)
}
