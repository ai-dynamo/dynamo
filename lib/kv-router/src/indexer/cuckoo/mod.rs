// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental Relay-published Cuckoo-filter indexers.
//!
//! The filter, producer, and CKF1 wire format preserve the implementation from
//! Kaonael/dynamo PR #4, contributed as reference material for DEP #11225. The
//! router-side frame consumer and transposed read layout are evaluation code:
//! native filters remain authoritative and the bucket-major table is derived.

mod filter;
mod frame_indexer;
mod overlap;
mod pages;
mod producer;
mod snapshot;
mod transposed;

pub use filter::{CuckooFilter, DEFAULT_FILTER_SEED, SLOTS};
pub use frame_indexer::{
    CuckooConsumerSession, CuckooDcConfig, CuckooFrameEnvelope, CuckooFrameIndexer,
    CuckooFrameMetadata, CuckooIndexerConfig, CuckooIndexerMode, CuckooPipelineErrors,
    CuckooPublication, CuckooQueueMetrics, CuckooStatsSnapshot,
};
pub use overlap::{OVERLAP_VERIFY_WINDOW, Probe, overlap_depth_searched, probes_for};
pub use producer::{Publish, SnapshotProducer};
pub use snapshot::{
    DeltaEntry, DeltaError, DeltaInfo, SnapshotAssembler, SnapshotError, SnapshotMeta,
    apply_decoded_delta, apply_delta, assemble_chunks, decode_delta, is_chunk, is_delta,
};
pub use transposed::{MaskLookup, TransposedTable};
