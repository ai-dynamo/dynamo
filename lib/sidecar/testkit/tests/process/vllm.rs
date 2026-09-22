// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

use dynamo_backend_common::PreprocessedRequest;
use dynamo_llm::model_card::ModelDeploymentCard;

use crate::process::{ProcessFixture, sidecar_command};
use crate::support::vllm::Fixture;

impl ProcessFixture for Fixture {
    fn endpoint(&self) -> String {
        self.server.endpoint()
    }

    fn command() -> Command {
        let mut command = sidecar_command("dynamo-vllm-sidecar", "DYNAMO_VLLM_SIDECAR");
        command.env_remove("VLLM_HTTP_ENDPOINT");
        command
    }

    fn configure_request(request: &mut PreprocessedRequest) {
        request.sampling_options.temperature = Some(0.125);
        request.sampling_options.presence_penalty = Some(0.25);
        request.sampling_options.frequency_penalty = Some(0.75);
        request.output_options.logprobs = Some(2);
        request.output_options.prompt_logprobs = Some(1);
    }

    fn assert_registration(card: &ModelDeploymentCard) {
        assert_eq!(card.kv_cache_block_size, 4);
        assert_eq!(card.runtime_config.total_kv_blocks, Some(4096));
        assert_eq!(card.runtime_config.max_num_seqs, Some(64));
        assert_eq!(card.runtime_config.max_num_batched_tokens, Some(1024));
        assert_eq!(card.runtime_config.data_parallel_start_rank, 0);
        assert_eq!(card.runtime_config.data_parallel_size, 1);
        assert!(card.runtime_config.tool_call_parser.is_none());
        assert!(card.runtime_config.reasoning_parser.is_none());
        assert_eq!(card.effective_context_length(), 4096);
    }
}
