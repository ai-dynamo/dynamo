// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

pub trait StagingValue {
    fn is_empty(&self) -> bool;
}

impl<T> StagingValue for Vec<T> {
    fn is_empty(&self) -> bool {
        Vec::is_empty(self)
    }
}

/// Per-tier staging and cache tracking. Shared by host, disk, and G4.
#[derive(Debug)]
pub struct TierState<T> {
    pub staging: Option<T>,
    pub tokens_cached: usize,
}

impl<T> TierState<T> {
    pub fn new() -> Self {
        Self {
            staging: None,
            tokens_cached: 0,
        }
    }

    pub fn reset(&mut self) {
        self.staging = None;
        self.tokens_cached = 0;
    }

    pub fn take_staged(&mut self) -> Option<T> {
        self.staging.take()
    }

    pub fn cached_blocks(&self, block_size: usize) -> usize {
        self.tokens_cached.div_ceil(block_size)
    }

    pub fn clear_staging(&mut self, request_id: &str, label: &str) {
        if self.staging.is_some() {
            tracing::warn!(request_id, "Preemption while {} staged", label);
            self.staging.take();
        }
    }

    pub fn record_cached_tokens(&mut self, n: usize) {
        self.tokens_cached = n;
    }
}

impl<T: StagingValue> TierState<T> {
    pub fn stage_non_empty(&mut self, value: T) {
        self.staging = if !value.is_empty() { Some(value) } else { None };
    }
}

impl<T> Default for TierState<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// G4 (object storage) tier state. Extends TierState<Vec<u64>> with
/// async lookup, retry tracking, and hash invalidation state.
#[derive(Debug)]
pub struct G4State<P> {
    pub tier: TierState<Vec<u64>>,
    pub skip_on_retry: bool,
    pub retry_count: u32,
    pub attempted_hashes: Option<Vec<(u64, u32)>>,
    pub onboarding_started_at: Option<Instant>,
    pub pending_lookup: Option<P>,
}

impl<P> G4State<P> {
    pub fn new() -> Self {
        Self {
            tier: TierState::new(),
            skip_on_retry: false,
            retry_count: 0,
            attempted_hashes: None,
            onboarding_started_at: None,
            pending_lookup: None,
        }
    }

    pub fn reset(&mut self) {
        self.tier.reset();
        self.skip_on_retry = false;
        self.retry_count = 0;
        self.attempted_hashes = None;
        self.onboarding_started_at = None;
        self.pending_lookup = None;
    }

    pub fn clear_all(&mut self, request_id: &str) {
        self.tier.clear_staging(request_id, "g4 hashes");
        if self.pending_lookup.is_some() {
            tracing::warn!(
                target: "kvbm-g4",
                request_id,
                "preemption while async G4 lookup pending"
            );
            self.pending_lookup.take();
        }
    }

    pub fn has_pending_lookup(&self) -> bool {
        self.pending_lookup.is_some()
    }

    pub fn prepare_onboard(&mut self, hashes: &[u64]) {
        self.attempted_hashes = Some(
            hashes
                .iter()
                .enumerate()
                .map(|(pos, &hash)| (hash, pos as u32))
                .collect(),
        );
        self.onboarding_started_at = Some(Instant::now());
    }
}

impl<P> Default for G4State<P> {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply an operation across all storage tiers (host, disk, g4).
#[macro_export]
macro_rules! all_tiers {
    (clear_staging $self:expr) => {
        $self.host.clear_staging(&$self.request_id, "host blocks");
        $self.disk.clear_staging(&$self.request_id, "disk blocks");
        $self.g4.clear_all(&$self.request_id);
    };
    (reset $self:expr) => {
        $self.host.reset();
        $self.disk.reset();
        $self.g4.reset();
    };
    (cache_stats $self:expr, $block_size:expr) => {{
        (
            $self.host.cached_blocks($block_size),
            $self.disk.cached_blocks($block_size),
            $self.g4.tier.cached_blocks($block_size),
        )
    }};
}

/// Onboard staged blocks from a local tier (host or disk).
/// G4 uses a different onboard path and stays explicit.
#[macro_export]
macro_rules! onboard_local_tier {
    ($self:expr, $tier:ident, $storage:ty) => {
        if let Some(blocks) = $self.$tier.take_staged() {
            let n = blocks.len();
            let dst = $self.prepare_onboard_dst(n);
            let src = Box::new(AnyImmutableBlocks::<$storage, _, _>::new(blocks));
            $self.onboard_blocks(src, dst)?;
            $self.evaluated_blocks += n;
        }
    };
}

/// Onboard staged blocks from G4 (remote/object storage).
/// Handles hash tracking, token block extraction, and the remote onboard path.
#[macro_export]
macro_rules! onboard_remote_tier {
    ($self:expr) => {
        if let Some(g4_hashes) = $self.g4.tier.take_staged() {
            let n = g4_hashes.len();
            let dst = $self.prepare_onboard_dst(n);
            let token_blocks = $self.sequence.blocks()
                [$self.evaluated_blocks..$self.evaluated_blocks + n]
                .to_vec();
            $self.g4.prepare_onboard(&g4_hashes);
            $self.onboard_from_g4(g4_hashes, dst, token_blocks)?;
            $self.evaluated_blocks += n;
        }
    };
}

/// Record remote transfer metrics, branching on onboard vs offload direction.
#[macro_export]
macro_rules! record_remote_metrics {
    ($metrics:expr, $is_onboard:expr, $num_blocks:expr, $bytes:expr,
     $status:expr, $backend:expr, $elapsed:expr) => {
        if $is_onboard {
            match $status {
                "success" => {
                    $metrics.onboard_blocks_r2d.inc_by($num_blocks as u64);
                    $metrics.record_remote_onboard_bytes($bytes);
                }
                _ => $metrics.record_remote_read_failure($num_blocks as u64),
            }
        } else {
            match $status {
                "success" => {
                    $metrics.offload_blocks_d2r.inc_by($num_blocks as u64);
                    $metrics.record_remote_offload_bytes($bytes);
                }
                _ => $metrics.record_remote_write_failure($num_blocks as u64),
            }
        }
        $metrics.record_remote_transfer_latency(
            if $is_onboard { "onboard" } else { "offload" },
            $status,
            $backend,
            $elapsed,
        );
    };
}

/// Acquire a mutex-guarded slot from the slot manager.
#[macro_export]
macro_rules! lock_slot {
    ($self:expr, $request_id:expr => $binding:ident) => {
        let shared_slot = $self.slot_manager().get_slot($request_id)?;
        let mut $binding = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
    };
}

/// Take pending ops from a slot, count immediate ops, and create metadata slot entry.
#[macro_export]
macro_rules! flush_slot_to_metadata {
    ($slot:expr, $md:expr, $request_id:expr) => {{
        if let Some(pending_ops) = $slot.take_pending_operations() {
            let num_immediate = pending_ops
                .iter()
                .filter(|op| {
                    op.request_type
                        == $crate::block_manager::connector::protocol::RequestType::Immediate
                })
                .count() as u64;
            $md.create_slot($request_id.clone(), num_immediate);
            $md.add_operations(pending_ops);
        } else {
            $md.create_slot($request_id.clone(), 0);
        }
    }};
}
