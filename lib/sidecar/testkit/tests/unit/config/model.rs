// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::adapter;

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn model_identity_and_limits_are_preserved() {
        let config = adapter::engine_config();
        assert_eq!(config.model, "model-source");
        assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
        let llm = config.llm.expect("LLM registration");
        assert_eq!(llm.context_length, Some(8192));
        assert_eq!(llm.kv_cache_block_size, Some(16));
        assert_eq!(llm.max_num_seqs, Some(128));
        assert_eq!(llm.max_num_batched_tokens, Some(2048));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn missing_optional_limits_are_not_invented() {
        let llm = adapter::without_optional_limits().llm.expect("LLM registration");
        assert_eq!(
            (
                llm.context_length,
                llm.kv_cache_block_size,
                llm.max_num_seqs,
                llm.max_num_batched_tokens,
            ),
            (None, None, None, None)
        );
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn logical_block_size_and_per_rank_capacity_are_registered() {
        let llm = adapter::logical_block_size_and_capacity()
            .llm
            .expect("LLM registration");
        assert_eq!(llm.kv_cache_block_size, Some(64));
        assert_eq!(llm.total_kv_blocks, Some(2048));
        assert_eq!(llm.data_parallel_size, Some(2));
        assert_eq!(llm.data_parallel_start_rank, Some(0));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn missing_model_identity_is_rejected() {
        assert!(adapter::without_model_identity().is_err());
    }
}
