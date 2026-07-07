// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;

use clap::ValueEnum;
use dynamo_kv_router::LocalBlockHash;
pub use dynamo_kv_router::indexer::cuckoo::CuckooPublication as CkfPublication;
use dynamo_kv_router::protocols::RouterEvent;
use serde::{Deserialize, Serialize};

pub const CORPUS_VERSION: u32 = 1;
pub const DEFAULT_DCS: usize = 16;
pub const DEFAULT_EVENT_THREADS: usize = 8;
pub const DEFAULT_QUERY_CONCURRENCY: usize = 16;
pub const DEFAULT_BALLAST_MEMBERSHIPS: usize = 10_000_000;
pub const DEFAULT_BALLAST_DEPTH: usize = 128;
pub const DEFAULT_TRACE_SHA256: &str =
    "b434f1816a707f4bac697235588184ebc374c9907cb981bb65fb0643471fe711";
pub fn relay_instance_id(dc: usize) -> u64 {
    0xC0C0_0000_0000_0000 | dc as u64
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum BackendKind {
    Crtc,
    CkfNative,
    CkfTransposed,
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Crtc => f.write_str("crtc"),
            Self::CkfNative => f.write_str("ckf-native"),
            Self::CkfTransposed => f.write_str("ckf-transposed"),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BallastSpec {
    pub memberships_per_dc: usize,
    pub prefix_depth: usize,
    pub seed: u64,
    pub namespace_tag: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FilterShape {
    pub seed: u64,
    pub buckets: usize,
    pub slots: usize,
    pub fingerprint_bits: u8,
    pub expected_load: f64,
    pub maximum_live_count: usize,
    pub insertion_failures: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct LogicalTotals {
    pub requests: u64,
    pub events: u64,
    pub request_blocks: u64,
    pub event_blocks: u64,
}

impl LogicalTotals {
    pub fn mixed_ops(&self) -> u64 {
        self.requests + self.events
    }

    pub fn block_ops(&self) -> u64 {
        self.request_blocks + self.event_blocks
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CorpusHeader {
    pub version: u32,
    pub created_by_git_sha: String,
    pub codec_id: String,
    pub hash_scheme_id: String,
    pub filter_shape_sha256: String,
    pub trace_sha256: String,
    pub trace_duplication_factor: usize,
    pub trace_length_factor: usize,
    pub trace_partition_seed: u64,
    pub block_size: u32,
    pub num_dcs: usize,
    pub event_threads: usize,
    pub query_concurrency: usize,
    pub source_span_us: u64,
    pub ballast: BallastSpec,
    pub filter_shapes: Vec<FilterShape>,
    pub totals: LogicalTotals,
    pub prefix_closure_violations: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum CorpusOperation {
    Request {
        local_hashes: Vec<LocalBlockHash>,
    },
    Update {
        logical_event_id: u64,
        dc: u16,
        event: RouterEvent,
        publication: CkfPublication,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CorpusEntry {
    pub timestamp_us: u64,
    pub stable_order: u64,
    pub operation: CorpusOperation,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PublisherStats {
    pub bytes: u64,
    pub full_bytes: u64,
    pub delta_bytes: u64,
    pub full: u64,
    pub delta: u64,
    pub unchanged: u64,
    pub dirty_buckets: u64,
    pub missing_removals: u64,
    pub filter_removal_failures: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PublisherTiming {
    pub generation_ns: u64,
    pub full_generation_ns: u64,
    pub delta_generation_ns: u64,
    pub unchanged_generation_ns: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PreparedCorpus {
    pub header: CorpusHeader,
    pub bootstrap_chunks: Vec<Vec<std::sync::Arc<[u8]>>>,
    pub entries: Vec<CorpusEntry>,
    pub final_trace_resident: Vec<Vec<u64>>,
    pub accuracy_samples: Vec<AccuracySample>,
    pub publisher: PublisherStats,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AccuracySample {
    pub local_hashes: Vec<LocalBlockHash>,
    pub exact_depths: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CorpusManifest {
    pub corpus_path: String,
    pub corpus_sha256: String,
    pub corpus_bytes: u64,
    pub resident_path: String,
    pub resident_sha256: String,
    pub resident_bytes: u64,
    pub header: CorpusHeader,
    pub publisher: PublisherStats,
    pub publisher_timing: PublisherTiming,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ResidentImage {
    pub header: CorpusHeader,
    pub bootstrap_chunks: Vec<Vec<std::sync::Arc<[u8]>>>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Percentiles {
    pub p50_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct QueueMetrics {
    pub at_stop: u64,
    pub maximum_depth: u64,
    pub drain_ms: f64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct AccuracyMetrics {
    pub checked_results: u64,
    pub inflated: u64,
    pub maximum_inflation: u32,
    pub under_reported: u64,
    pub full_map_mismatches: u64,
    pub wrong_best_dc: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PipelineErrors {
    pub insertion: u64,
    pub removal: u64,
    pub decode: u64,
    pub application: u64,
    pub epoch: u64,
    pub desynchronization: u64,
}

impl PipelineErrors {
    pub fn total(&self) -> u64 {
        self.insertion
            + self.removal
            + self.decode
            + self.application
            + self.epoch
            + self.desynchronization
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrialResult {
    pub schema_version: u32,
    pub measured_code_sha: String,
    pub corpus_sha256: String,
    pub backend: BackendKind,
    pub repetition: usize,
    pub phase: String,
    pub replay_window_ms: f64,
    pub nominal_offered_mixed_ops_s: f64,
    pub nominal_offered_block_ops_s: f64,
    pub actual_issue_mixed_ops_s: f64,
    pub achieved_mixed_ops_s: f64,
    pub achieved_block_ops_s: f64,
    pub achieved_over_offered: f64,
    pub total_elapsed_ms: f64,
    pub generator_limited: bool,
    pub kept_up: bool,
    pub issue_lag: Percentiles,
    pub query_queue_wait: Percentiles,
    pub lookup_service: Percentiles,
    pub scheduled_to_completion: Percentiles,
    pub update_scheduled_to_enqueue: Percentiles,
    pub update_enqueue_to_applied: Percentiles,
    pub update_scheduled_to_applied: Percentiles,
    pub query_queue: QueueMetrics,
    pub update_queue: QueueMetrics,
    pub crtc_raw_events: u64,
    pub crtc_raw_blocks: u64,
    pub ckf_frames: u64,
    pub ckf_dirty_buckets: u64,
    pub ckf_bytes: u64,
    pub ckf_apply_mib_s: f64,
    pub ckf_full_apply_mib_s: f64,
    pub ckf_delta_apply_mib_s: f64,
    pub full_publications: u64,
    pub delta_publications: u64,
    pub unchanged_publications: u64,
    pub generation_conflicts: u64,
    pub native_fallbacks: u64,
    pub repeated_fallbacks: u64,
    pub errors: PipelineErrors,
    pub accuracy: AccuracyMetrics,
    pub rss_bytes: u64,
    pub pss_bytes: Option<u64>,
    pub uss_bytes: Option<u64>,
}

pub fn percentile_summary(mut ns: Vec<u64>) -> Percentiles {
    if ns.is_empty() {
        return Percentiles::default();
    }
    ns.sort_unstable();
    let at = |pct: usize| ns[(ns.len().saturating_sub(1) * pct).div_ceil(100)] as f64 / 1000.0;
    Percentiles {
        p50_us: at(50),
        p99_us: at(99),
        max_us: ns[ns.len() - 1] as f64 / 1000.0,
    }
}
