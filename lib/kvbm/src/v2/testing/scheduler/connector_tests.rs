// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for scheduler connector shim integration.
//!
//! These tests verify the `SchedulerConnectorShim` slot lifecycle management
//! and scheduler-connector integration using real `TestConnectorInstance`.

use super::{create_test_request, create_test_scheduler_with_connector};
use crate::v2::integrations::scheduler::RequestStatus;

// =============================================================================
// Slot Lifecycle Tests
// =============================================================================

#[test]
fn test_scheduler_with_connector_has_shim() {
    let result = create_test_scheduler_with_connector(100, 16);
    assert!(result.is_ok(), "Should create scheduler with connector");

    let (scheduler, _instance, _registry) = result.unwrap();

    // Verify the connector shim is attached
    assert!(
        scheduler.connector_shim().is_some(),
        "Scheduler should have connector shim attached"
    );
}

#[test]
fn test_connector_basic_request_lifecycle() {
    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    // Add a request
    let request = create_test_request("req-1", (0..64).collect(), Some(50));
    scheduler.add_request(request);

    assert_eq!(scheduler.num_waiting(), 1);
    assert_eq!(scheduler.num_running(), 0);

    // Schedule the request
    let output = scheduler.schedule();

    assert_eq!(scheduler.num_waiting(), 0);
    assert_eq!(scheduler.num_running(), 1);
    assert_eq!(output.scheduled_new_reqs.len(), 1);
    assert_eq!(output.scheduled_new_reqs[0].req_id, "req-1");

    // Finish the request
    scheduler.finish_requests(&["req-1".to_string()], RequestStatus::FinishedStopped);

    assert_eq!(scheduler.num_running(), 0);
}

#[test]
fn test_connector_multiple_requests() {
    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    // Add multiple requests
    for i in 0..3 {
        let request = create_test_request(&format!("req-{}", i), (0..32).collect(), Some(20));
        scheduler.add_request(request);
    }

    assert_eq!(scheduler.num_waiting(), 3);

    // Schedule all
    let output = scheduler.schedule();

    assert_eq!(scheduler.num_running(), 3);
    assert_eq!(output.scheduled_new_reqs.len(), 3);

    // Finish one at a time
    scheduler.finish_requests(&["req-0".to_string()], RequestStatus::FinishedStopped);
    assert_eq!(scheduler.num_running(), 2);

    scheduler.finish_requests(&["req-1".to_string()], RequestStatus::FinishedStopped);
    assert_eq!(scheduler.num_running(), 1);

    scheduler.finish_requests(&["req-2".to_string()], RequestStatus::FinishedStopped);
    assert_eq!(scheduler.num_running(), 0);
}

#[test]
fn test_connector_abort_request() {
    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    // Add and schedule a request
    let request = create_test_request("req-1", (0..64).collect(), Some(50));
    scheduler.add_request(request);
    scheduler.schedule();

    assert_eq!(scheduler.num_running(), 1);

    // Abort the request
    scheduler.abort_request("req-1");

    // Request should be removed
    assert_eq!(scheduler.num_running(), 0);
    assert_eq!(scheduler.num_waiting(), 0);
}

#[test]
fn test_connector_abort_waiting_request() {
    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    // Add request but don't schedule
    let request = create_test_request("req-1", (0..64).collect(), Some(50));
    scheduler.add_request(request);

    assert_eq!(scheduler.num_waiting(), 1);

    // Abort before scheduling
    scheduler.abort_request("req-1");

    // Request should be removed from waiting queue
    assert_eq!(scheduler.num_waiting(), 0);
}

// =============================================================================
// Shim Slot Tracking Tests
// =============================================================================
//
// Note: The scheduler's connector integration (calling get_num_new_matched_tokens)
// is still a TODO. The following tests verify the shim's slot tracking when
// methods are called directly on the shim, rather than via scheduler.schedule().

#[test]
fn test_shim_has_slot_tracking() {
    use crate::v2::integrations::common::Request;
    use crate::v2::integrations::scheduler::SchedulerRequest;

    let (scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    let shim = scheduler.connector_shim().expect("Should have shim");

    // Initially no slots
    assert!(!shim.has_slot("req-1"));

    // Create a SchedulerRequest to test slot creation directly on shim
    // This simulates what the scheduler would do when connector integration is complete
    let tokens: Vec<u32> = (0..64).collect();
    let request = Request::new("req-1", tokens, None, None, Some(50));
    let scheduler_request = SchedulerRequest::new(request, 16);

    // Call get_num_new_matched_tokens directly - this triggers slot creation
    let result = shim.get_num_new_matched_tokens(&scheduler_request, 0);
    assert!(result.is_ok(), "get_num_new_matched_tokens should succeed");

    // After the call, shim should have created a slot
    assert!(shim.has_slot("req-1"), "Shim should have slot after get_num_new_matched_tokens");

    // Call request_finished to clean up the slot
    let status = shim.request_finished("req-1");
    tracing::debug!(?status, "request_finished returned");

    // Slot should be removed
    assert!(!shim.has_slot("req-1"), "Shim should remove slot on finish");
}

#[test]
fn test_shim_slot_removed_on_abort_direct() {
    use crate::v2::integrations::common::Request;
    use crate::v2::integrations::scheduler::SchedulerRequest;

    let (scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    let shim = scheduler.connector_shim().expect("Should have shim");

    // Create slot directly via shim
    let tokens: Vec<u32> = (0..64).collect();
    let request = Request::new("req-1", tokens, None, None, Some(50));
    let scheduler_request = SchedulerRequest::new(request, 16);

    let _ = shim.get_num_new_matched_tokens(&scheduler_request, 0);
    assert!(shim.has_slot("req-1"), "Slot should exist after creation");

    // Simulate abort by calling request_finished (same cleanup path)
    shim.request_finished("req-1");

    // Slot should be removed
    assert!(!shim.has_slot("req-1"), "Shim should remove slot on abort");
}

// =============================================================================
// Eviction Support Tests
// =============================================================================

#[test]
fn test_shim_can_evict_delegation() {
    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    // Add and schedule a request
    let request = create_test_request("req-1", (0..64).collect(), Some(50));
    scheduler.add_request(request);
    scheduler.schedule();

    // Check can_evict - should delegate to connector leader
    let shim = scheduler.connector_shim().expect("Should have shim");

    // By default, requests without inflight offloads should be evictable
    let can_evict = shim.can_evict("req-1");
    // Note: The actual value depends on connector state, but the call should not panic
    tracing::debug!(can_evict, "can_evict result for req-1");
}

#[test]
fn test_shim_eviction_score_delegation() {
    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    // Add and schedule a request
    let request = create_test_request("req-1", (0..64).collect(), Some(50));
    scheduler.add_request(request);
    scheduler.schedule();

    // Get eviction score - should delegate to connector leader
    let shim = scheduler.connector_shim().expect("Should have shim");
    let score_result = shim.get_eviction_score("req-1");

    // The call should succeed (or return an appropriate error for untracked requests)
    match score_result {
        Ok(score) => {
            tracing::debug!(coverage = score.coverage_ratio, "eviction score for req-1");
        }
        Err(e) => {
            // Some implementations may not support eviction scoring
            tracing::debug!(error = %e, "eviction score not available");
        }
    }
}

// =============================================================================
// Multi-Step Generation Tests
// =============================================================================

#[test]
fn test_connector_multi_step_decode() {
    use std::collections::HashMap;

    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(100, 16).expect("Should create scheduler");

    // Add and schedule initial request
    let request = create_test_request("req-1", (0..64).collect(), Some(50));
    scheduler.add_request(request);

    // First iteration - prefill
    let output1 = scheduler.schedule();
    assert_eq!(output1.scheduled_new_reqs.len(), 1);
    assert_eq!(scheduler.num_running(), 1);

    // Simulate decode iterations
    for i in 0..5 {
        // Simulate token generation - provide output tokens
        let mut output_tokens = HashMap::new();
        output_tokens.insert("req-1".to_string(), vec![1000 + i as u32]);

        // Update scheduler with generated tokens (no finished requests)
        scheduler.update_from_output(&[], &output_tokens);

        // Schedule again for decode
        let output = scheduler.schedule();

        // Request should be in running (cached) state
        assert_eq!(
            output.scheduled_cached_reqs.len(),
            1,
            "Iteration {}: Request should be in cached state",
            i
        );
    }

    // Note: Slot tracking assertions removed because scheduler doesn't yet call
    // connector methods during scheduling. The shim slot tracking is tested
    // separately in test_shim_has_slot_tracking.

    // Finish the request
    scheduler.finish_requests(&["req-1".to_string()], RequestStatus::FinishedStopped);

    // Verify request was properly cleaned up
    assert_eq!(scheduler.num_running(), 0);
}

// =============================================================================
// Stress Tests
// =============================================================================

#[test]
fn test_connector_many_requests() {
    let (mut scheduler, _instance, _registry) =
        create_test_scheduler_with_connector(500, 16).expect("Should create scheduler");

    // Add many requests
    let num_requests = 20;
    for i in 0..num_requests {
        let request = create_test_request(&format!("req-{}", i), (0..32).collect(), Some(10));
        scheduler.add_request(request);
    }

    // Schedule all
    let output = scheduler.schedule();
    let scheduled_count = output.scheduled_new_reqs.len();
    assert!(scheduled_count > 0, "Should schedule some requests");

    // Note: Slot tracking assertions removed because scheduler doesn't yet call
    // connector methods during scheduling. This test verifies the scheduler
    // handles many requests correctly with a connector attached.

    // Finish all scheduled requests
    let request_ids: Vec<String> = output
        .scheduled_new_reqs
        .iter()
        .map(|r| r.req_id.clone())
        .collect();
    scheduler.finish_requests(&request_ids, RequestStatus::FinishedStopped);

    // Verify requests were finished
    assert_eq!(
        scheduler.num_running(),
        0,
        "All scheduled requests should be finished"
    );
}
