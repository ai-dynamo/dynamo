// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CRTC reclamation and edge-capacity flags shared by the router benches.

use dynamo_kv_router::concurrent_radix_tree_compressed::ReclaimConfig;

/// Starting point for CRTC reclamation and edge-capacity settings.
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum CrtcReclaimPreset {
    /// Volume-triggered stale-leaf sweeps, capped leaf slack, exact split prefixes.
    Default,
    /// Timer-only sweeps, `Vec` doubling, split prefixes keep their capacity.
    Legacy,
}

/// CRTC reclamation and edge-capacity knobs, applied over `--crtc-reclaim`.
#[derive(clap::Args, Clone, Debug)]
pub struct CrtcReclaimArgs {
    #[clap(long, value_enum, default_value = "default")]
    crtc_reclaim: CrtcReclaimPreset,

    /// Schedule stale-leaf sweeps on dead-block volume, besides the five-minute timer.
    #[clap(long)]
    crtc_volume_sweep: Option<bool>,

    /// Dead blocks below which no volume-triggered sweep runs.
    #[clap(long)]
    crtc_dead_floor: Option<u64>,

    /// Minimum milliseconds between a sweep's end and the next volume-triggered sweep.
    #[clap(long)]
    crtc_min_gap_ms: Option<u64>,

    /// Leaf append slack divisor (slack is `need / divisor`); 0 keeps `Vec` doubling.
    #[clap(long)]
    crtc_leaf_slack_divisor: Option<u32>,

    /// Minimum leaf append slack in blocks.
    #[clap(long)]
    crtc_leaf_min_slack: Option<u32>,

    /// Copy split prefixes into exact-size allocations.
    #[clap(long)]
    crtc_exact_split_prefix: Option<bool>,
}

impl CrtcReclaimArgs {
    pub fn config(&self) -> ReclaimConfig {
        let mut config = match self.crtc_reclaim {
            CrtcReclaimPreset::Default => ReclaimConfig::default(),
            CrtcReclaimPreset::Legacy => ReclaimConfig::legacy(),
        };
        if let Some(volume_sweep) = self.crtc_volume_sweep {
            config.volume_sweep = volume_sweep;
        }
        if let Some(dead_floor) = self.crtc_dead_floor {
            config.dead_floor = dead_floor;
        }
        if let Some(min_gap_ms) = self.crtc_min_gap_ms {
            config.min_gap = std::time::Duration::from_millis(min_gap_ms);
        }
        if let Some(divisor) = self.crtc_leaf_slack_divisor {
            config.leaf_slack_divisor = divisor;
        }
        if let Some(min_slack) = self.crtc_leaf_min_slack {
            config.leaf_min_slack = min_slack;
        }
        if let Some(exact) = self.crtc_exact_split_prefix {
            config.exact_split_prefix = exact;
        }
        config
    }
}
