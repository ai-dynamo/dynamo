// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-exports of the kernel's Mooncake overlap summary for the indexer HTTP server.

pub use dynamo_scheduler::overlap_summary::{
    MooncakeOverlapSummary, build_mooncake_overlap_summaries,
};
