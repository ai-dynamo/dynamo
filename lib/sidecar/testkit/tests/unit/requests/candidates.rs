// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{pb, top_n_candidates};

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn full_vocabulary_logprobs_select_all_candidates() {
        let candidates = top_n_candidates(u32::MAX).expect("map full vocabulary");
        assert_eq!(
            candidates.select,
            Some(pb::candidate_tokens::Select::All(true))
        );
    }
}
