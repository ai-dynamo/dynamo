// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod block_tracker;
mod compressed_path_arena;
pub mod multi_worker;
mod prefill_tracker;
mod prompt_membership_trie;
mod prompt_registry;
mod replica_sync;
mod request_maps;
pub mod single;
pub mod topology;

pub use multi_worker::*;
pub use prefill_tracker::PrefillTokenDeltas;
pub use prompt_registry::{PotentialLoadMaps, WorkerLoadProjection};
pub use single::*;

pub(super) fn estimate_physical_active_blocks(
    prompt_units: f64,
    output_blocks: f64,
    prompt_stride: usize,
) -> usize {
    (prompt_units * prompt_stride as f64 + output_blocks).round() as usize
}
