// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Test helpers shared with the kernel.

pub use dynamo_scheduler::test_utils::*;
// Pre-split paths kept through 1.9 and removed in 1.10; the named list is the inventory in compat.rs.
#[allow(deprecated)]
pub use crate::compat::test_utils::{
    assert_exact_scores, assert_no_scores, assert_overlap_scores_eq, assert_score,
    flush_and_settle, query_scores, snapshot_tree,
};
