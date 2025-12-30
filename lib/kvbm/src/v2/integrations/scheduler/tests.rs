// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for the scheduler module.

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::v2::integrations::common::Request;

    // =========================================================================
    // Request Tests
    // =========================================================================

    mod request_tests {
        use super::*;

        #[test]
        fn test_request_creation() {
            let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
            let request = Request::new("test-1", tokens.clone(), None, None, Some(100));

            assert_eq!(request.request_id, "test-1");
            assert_eq!(request.tokens.len(), 5);
            assert!(request.lora_name.is_none());
            assert_eq!(request.max_tokens, Some(100));
        }

        #[test]
        fn test_request_with_lora() {
            let tokens: Vec<u32> = vec![1, 2, 3];
            let request = Request::new(
                "test-2",
                tokens,
                Some("my-lora".to_string()),
                None,
                None,
            );

            assert_eq!(request.lora_name, Some("my-lora".to_string()));
        }

        #[test]
        fn test_request_salt_hash_differs() {
            let tokens: Vec<u32> = vec![1, 2, 3];
            let request1 = Request::new("test", tokens.clone(), None, Some("salt1".to_string()), None);
            let request2 = Request::new("test", tokens.clone(), None, Some("salt2".to_string()), None);

            // Different salts should produce different hashes
            assert_ne!(request1.salt_hash, request2.salt_hash);
        }
    }

    // =========================================================================
    // RequestStatus Tests
    // =========================================================================

    mod status_tests {
        use super::*;

        #[test]
        fn test_status_is_finished() {
            assert!(!RequestStatus::Waiting.is_finished());
            assert!(!RequestStatus::Running.is_finished());
            assert!(!RequestStatus::Preempted.is_finished());
            assert!(RequestStatus::FinishedStopped.is_finished());
            assert!(RequestStatus::FinishedAborted.is_finished());
            assert!(RequestStatus::FinishedLengthCapped.is_finished());
        }

        #[test]
        fn test_status_can_schedule() {
            assert!(RequestStatus::Waiting.can_schedule());
            assert!(!RequestStatus::Running.can_schedule());
            assert!(RequestStatus::Preempted.can_schedule());
            assert!(!RequestStatus::FinishedStopped.can_schedule());
        }
    }

    // =========================================================================
    // SchedulerRequest Tests
    // =========================================================================

    mod scheduler_request_tests {
        use super::*;

        fn create_test_request(id: &str, num_tokens: usize) -> Request {
            let tokens: Vec<u32> = (0..num_tokens as u32).collect();
            Request::new(id, tokens, None, None, Some(100))
        }

        #[test]
        fn test_scheduler_request_creation() {
            let request = create_test_request("req-1", 50);
            let sched_req = SchedulerRequest::new(request);

            assert_eq!(sched_req.request_id(), "req-1");
            assert_eq!(sched_req.prompt_len(), 50);
            assert_eq!(sched_req.status, RequestStatus::Waiting);
            assert_eq!(sched_req.num_computed_tokens, 0);
            assert_eq!(sched_req.num_output_tokens, 0);
            assert!(sched_req.block_state.is_empty());
        }

        #[test]
        fn test_scheduler_request_total_tokens() {
            let request = create_test_request("req-1", 50);
            let mut sched_req = SchedulerRequest::new(request);

            assert_eq!(sched_req.total_tokens(), 50);

            sched_req.add_output_tokens(10);
            assert_eq!(sched_req.total_tokens(), 60);
        }

        #[test]
        fn test_scheduler_request_blocks_needed() {
            let request = create_test_request("req-1", 50);
            let sched_req = SchedulerRequest::new(request);

            // With block size 16: ceil(50/16) = 4 blocks needed
            assert_eq!(sched_req.num_blocks_required(16), 4);

            // With block size 32: ceil(50/32) = 2 blocks needed
            assert_eq!(sched_req.num_blocks_required(32), 2);
        }

        // Note: The test_scheduler_request_new_blocks_needed test was removed because
        // it relied on the old `add_blocks(Vec<BlockId>)` API. The new RAII-based
        // block management requires actual MutableBlock<G1> objects from a BlockManager,
        // which can't be easily created in unit tests without full infrastructure.
        // Block state tests are covered by integration tests.

        #[test]
        fn test_scheduler_request_lifecycle() {
            let request = create_test_request("req-1", 50);
            let mut sched_req = SchedulerRequest::new(request);

            // Initial state
            assert_eq!(sched_req.status, RequestStatus::Waiting);

            // Start running
            sched_req.start_running();
            assert_eq!(sched_req.status, RequestStatus::Running);

            // Preempt (without adding blocks - block management requires RAII blocks)
            sched_req.preempt();
            assert_eq!(sched_req.status, RequestStatus::Preempted);
            assert!(sched_req.block_state.is_empty());
            assert_eq!(sched_req.num_computed_tokens, 0);

            // Resume
            sched_req.resume();
            assert_eq!(sched_req.status, RequestStatus::Waiting);
            assert!(sched_req.resumed_from_preemption);

            // Finish
            sched_req.finish(RequestStatus::FinishedStopped);
            assert_eq!(sched_req.status, RequestStatus::FinishedStopped);
        }

        #[test]
        fn test_scheduler_request_at_max_tokens() {
            let request = create_test_request("req-1", 50);
            let mut sched_req = SchedulerRequest::new(request);

            assert!(!sched_req.is_at_max_tokens());

            // Add tokens up to max
            sched_req.add_output_tokens(100);
            assert!(sched_req.is_at_max_tokens());
        }
    }

    // =========================================================================
    // Queue Tests
    // =========================================================================

    mod queue_tests {
        use super::*;

        fn create_test_sched_request(id: &str) -> SchedulerRequest {
            let tokens: Vec<u32> = vec![1, 2, 3, 4];
            let request = Request::new(id, tokens, None, None, None);
            SchedulerRequest::new(request)
        }

        #[test]
        fn test_waiting_queue_basic() {
            let mut queue = WaitingQueue::new();

            assert!(queue.is_empty());
            assert_eq!(queue.len(), 0);

            queue.push_back(create_test_sched_request("req-1"));
            queue.push_back(create_test_sched_request("req-2"));

            assert_eq!(queue.len(), 2);

            let req = queue.pop_front().unwrap();
            assert_eq!(req.request_id(), "req-1");

            assert_eq!(queue.len(), 1);
        }

        #[test]
        fn test_waiting_queue_priority() {
            let mut queue = WaitingQueue::new();

            queue.push_back(create_test_sched_request("req-1"));
            queue.push_back(create_test_sched_request("req-2"));

            // Push to front (preempted request priority)
            queue.push_front(create_test_sched_request("req-priority"));

            let req = queue.pop_front().unwrap();
            assert_eq!(req.request_id(), "req-priority");
        }

        #[test]
        fn test_waiting_queue_remove() {
            let mut queue = WaitingQueue::new();

            queue.push_back(create_test_sched_request("req-1"));
            queue.push_back(create_test_sched_request("req-2"));
            queue.push_back(create_test_sched_request("req-3"));

            let removed = queue.remove("req-2");
            assert!(removed.is_some());
            assert_eq!(removed.unwrap().request_id(), "req-2");
            assert_eq!(queue.len(), 2);
        }

        #[test]
        fn test_running_requests_basic() {
            let mut running = RunningRequests::new();

            assert!(running.is_empty());

            let req = create_test_sched_request("req-1");
            running.insert(req);

            assert!(!running.is_empty());
            assert_eq!(running.len(), 1);
            assert!(running.contains("req-1"));

            let req = running.get("req-1").unwrap();
            assert_eq!(req.status, RequestStatus::Running);
        }

        #[test]
        fn test_running_requests_remove() {
            let mut running = RunningRequests::new();

            running.insert(create_test_sched_request("req-1"));
            running.insert(create_test_sched_request("req-2"));

            let removed = running.remove("req-1");
            assert!(removed.is_some());
            assert_eq!(running.len(), 1);
            assert!(!running.contains("req-1"));
            assert!(running.contains("req-2"));
        }
    }

    // =========================================================================
    // Policy Tests
    // =========================================================================

    mod policy_tests {
        use super::*;

        fn create_test_sched_request(id: &str, num_tokens: usize) -> SchedulerRequest {
            let tokens: Vec<u32> = (0..num_tokens as u32).collect();
            let request = Request::new(id, tokens, None, None, None);
            SchedulerRequest::new(request)
        }

        #[test]
        fn test_fcfs_policy_select_next() {
            let policy = FCFSPolicy::new(10);

            let req1 = create_test_sched_request("req-1", 32);
            let req2 = create_test_sched_request("req-2", 32);
            let waiting: Vec<&SchedulerRequest> = vec![&req1, &req2];

            // Should select first request when resources available
            let selected = policy.select_next(&waiting, 0, 100, 16);
            assert_eq!(selected, Some(0));
        }

        #[test]
        fn test_fcfs_policy_max_seqs() {
            let policy = FCFSPolicy::new(2);

            let req1 = create_test_sched_request("req-1", 16);
            let waiting: Vec<&SchedulerRequest> = vec![&req1];

            // Should not schedule when at max seqs
            let selected = policy.select_next(&waiting, 2, 100, 16);
            assert!(selected.is_none());
        }

        #[test]
        fn test_fcfs_policy_not_enough_blocks() {
            let policy = FCFSPolicy::new(10);

            let req1 = create_test_sched_request("req-1", 64); // Needs 4 blocks
            let waiting: Vec<&SchedulerRequest> = vec![&req1];

            // Only 2 blocks available - should not schedule
            let selected = policy.select_next(&waiting, 0, 2, 16);
            assert!(selected.is_none());
        }

        #[test]
        fn test_fcfs_policy_select_victim() {
            let policy = FCFSPolicy::new(10);

            let mut req1 = create_test_sched_request("req-1", 32);
            let mut req2 = create_test_sched_request("req-2", 32);

            // req1 has more computed tokens
            req1.update_computed_tokens(20);
            req2.update_computed_tokens(5);

            let running: Vec<&SchedulerRequest> = vec![&req1, &req2];

            // Should select req2 as victim (fewer computed tokens)
            let victim = policy.select_victim(&running, 1);
            assert_eq!(victim, Some("req-2"));
        }
    }

    // =========================================================================
    // Config Tests
    // =========================================================================

    mod config_tests {
        use super::*;

        #[test]
        fn test_scheduler_config_default() {
            let config = SchedulerConfig::default();

            assert_eq!(config.max_num_batched_tokens, 8192);
            assert_eq!(config.max_num_seqs, 256);
            assert_eq!(config.block_size, 16);
            assert!(!config.enable_prefix_caching);
            assert!(!config.enable_chunked_prefill);
        }

        #[test]
        fn test_scheduler_config_custom() {
            let config = SchedulerConfig::new(4096, 128, 32);

            assert_eq!(config.max_num_batched_tokens, 4096);
            assert_eq!(config.max_num_seqs, 128);
            assert_eq!(config.block_size, 32);
        }

        #[test]
        fn test_scheduler_config_builder() {
            let config = SchedulerConfig::builder()
                .max_num_batched_tokens(8192)
                .max_num_seqs(256)
                .block_size(16)
                .enable_prefix_caching(true)
                .enable_chunked_prefill(true)
                .max_prefill_chunk_size(512)
                .build()
                .expect("Should build config");

            assert!(config.enable_prefix_caching);
            assert!(config.enable_chunked_prefill);
            assert_eq!(config.max_prefill_chunk_size, Some(512));
        }

        #[test]
        fn test_scheduler_config_builder_defaults() {
            let config = SchedulerConfig::builder()
                .build()
                .expect("Should build with defaults");

            assert_eq!(config.max_num_batched_tokens, 8192);
            assert_eq!(config.max_num_seqs, 256);
            assert_eq!(config.block_size, 16);
            assert!(!config.enable_prefix_caching);
            assert!(!config.enable_chunked_prefill);
            assert_eq!(config.max_prefill_chunk_size, None);
        }
    }

    // =========================================================================
    // AllocatedBlocks Tests
    // =========================================================================

    mod allocated_blocks_tests {
        use crate::v2::integrations::scheduler::kv_cache::AllocatedBlocks;

        #[test]
        fn test_allocated_blocks_new() {
            let blocks = AllocatedBlocks::new();
            assert!(blocks.is_empty());
            assert_eq!(blocks.len(), 0);
        }

        #[test]
        fn test_allocated_blocks_extend() {
            let mut blocks = AllocatedBlocks::new();
            blocks.extend(vec![1, 2, 3]);

            assert_eq!(blocks.len(), 3);
            assert_eq!(blocks.block_ids, vec![1, 2, 3]);

            blocks.extend(vec![4, 5]);
            assert_eq!(blocks.len(), 5);
        }

        #[test]
        fn test_allocated_blocks_clear() {
            let mut blocks = AllocatedBlocks::new();
            blocks.extend(vec![1, 2, 3]);

            blocks.clear();
            assert!(blocks.is_empty());
        }
    }

    // =========================================================================
    // SchedulerOutput Tests
    // =========================================================================

    mod output_tests {
        use crate::v2::integrations::common::SchedulerOutput;
        use std::collections::HashMap;

        #[test]
        fn test_scheduler_output_new() {
            let output = SchedulerOutput::new(1);

            assert_eq!(output.iteration, 1);
            assert!(output.scheduled_new_reqs.is_empty());
            assert!(output.scheduled_cached_reqs.is_empty());
            assert_eq!(output.total_num_scheduled_tokens(), 0);
        }

        #[test]
        fn test_scheduler_output_add_new_request() {
            let mut output = SchedulerOutput::new(1);

            output.add_new_request(
                "req-1".to_string(),
                vec![1, 2, 3, 4],
                vec![0, 1],
                0,
            );

            assert_eq!(output.scheduled_new_reqs.len(), 1);
            assert_eq!(output.scheduled_new_reqs[0].req_id, "req-1");
            assert_eq!(output.scheduled_new_reqs[0].block_ids, vec![0, 1]);
        }

        #[test]
        fn test_scheduler_output_add_cached_request() {
            let mut output = SchedulerOutput::new(1);

            output.add_cached_request(
                "req-1".to_string(),
                false,
                vec![5],
                None,
                vec![2],
                10,
                1,
            );

            assert_eq!(output.scheduled_cached_reqs.len(), 1);
            assert!(!output.scheduled_cached_reqs[0].resumed);
        }

        #[test]
        fn test_scheduler_output_set_scheduled_tokens() {
            let mut output = SchedulerOutput::new(1);

            let mut tokens = HashMap::new();
            tokens.insert("req-1".to_string(), 100);
            tokens.insert("req-2".to_string(), 50);

            output.set_num_scheduled_tokens(tokens);

            assert_eq!(output.total_num_scheduled_tokens(), 150);
            assert_eq!(output.num_scheduled_tokens("req-1"), Some(100));
            assert_eq!(output.num_scheduled_tokens("req-2"), Some(50));
            assert_eq!(output.num_scheduled_tokens("req-3"), None);
        }
    }
}

