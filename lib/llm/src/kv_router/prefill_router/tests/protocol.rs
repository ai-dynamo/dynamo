// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[test]
fn client_cancellation_does_not_cancel_prefill_context() {
    for kill in [false, true] {
        let request = query_only_request();
        let mut source = Context::new(());
        source.insert_metadata("traceparent", "test-trace");
        let prefill = independent_prefill_context(request, &source).unwrap();
        assert_eq!(prefill.id(), source.id());
        assert_eq!(prefill.metadata(), source.metadata());

        if kill {
            source.context().kill();
        } else {
            source.context().stop_generating();
        }
        assert!(source.context().is_stopped());
        assert!(!prefill.context().is_stopped());

        // Prefill still honors its own worker cancellation/shutdown.
        prefill.context().kill();
        assert!(prefill.context().is_killed());
    }
}

#[test]
fn prefill_dispatch_errors_distinguish_cancellation_from_worker_failure() {
    use dynamo_runtime::error::DynamoError;

    let cancelled: anyhow::Error = DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message("client disconnected")
        .build()
        .into();
    let at_capacity: anyhow::Error = DynamoError::builder()
        .error_type(ErrorType::ResourceExhausted)
        .message("worker at capacity")
        .build()
        .into();
    let failed = anyhow::anyhow!("transport failed");

    assert_eq!(
        classify_prefill_dispatch_error(cancelled.as_ref()),
        PrefillDispatchErrorClass::Cancelled
    );
    assert_eq!(
        classify_prefill_dispatch_error(&PrefillError::Cancelled),
        PrefillDispatchErrorClass::Cancelled
    );
    assert_eq!(
        classify_prefill_dispatch_error(at_capacity.as_ref()),
        PrefillDispatchErrorClass::AtCapacity
    );
    assert_eq!(
        classify_prefill_dispatch_error(failed.as_ref()),
        PrefillDispatchErrorClass::Failed
    );
}

#[tokio::test]
async fn precomputed_bootstrap_observes_initial_and_late_prefill_failures() {
    for after_handoff in [false, true] {
        let router = conditional_router();
        let tracker = Arc::new(RequestTracker::new());
        let permit = tracker.set_phase(RequestPhase::Prefill).await;
        let mut outputs = Vec::new();
        if after_handoff {
            outputs.push(Annotated::from_data(LLMEngineOutput {
                disaggregated_params: Some(Box::new(serde_json::json!({
                    "bootstrap_host": "prefill", "bootstrap_port": 1, "bootstrap_room": 7,
                }))),
                ..Default::default()
            }));
        }
        outputs.push(Annotated::from_error("prefill connection lost"));
        let prefill =
            ResponseStream::new(Box::pin(stream::iter(outputs)), Context::new(()).context());
        let task = router.spawn_prefill_task(prefill, Some(tracker.clone()), permit);
        // Decode can enter its phase even though prefill is still observed.
        let _decode_permit = tracker.set_phase(RequestPhase::Decode).await;
        let error = task
            .forward_decode(
                std::future::pending(),
                Context::new(()).context(),
                PrefillNeed::None,
                Context::new(()).context(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("prefill connection lost"));
    }
}

/// The decode leg must carry the cleanup obligation, on both outcomes that
/// reach decode routing. Without it the routing host treats a disconnected
/// decode request as ordinary traffic and drops it, which is the leak this
/// whole path exists to prevent — and nothing else in the Rust suite
/// notices, because every routing-host test builds its own request.
#[test]
fn decode_leg_carries_the_staged_kv_cleanup_obligation() {
    let bootstrap = PrefillOutcome::Bootstrap {
        bootstrap_info: BootstrapInfo {
            bootstrap_host: "prefill-0".to_string(),
            bootstrap_port: 4242,
            bootstrap_room: 7,
            handoff_id: None,
        },
        worker_id: 11,
        prefill_dp_rank: Some(3),
    };
    let completed = PrefillOutcome::Completed {
        result: PrefillResult {
            disaggregated_params: serde_json::json!({}),
            prompt_tokens_details: None,
        },
        worker_id: 12,
        worker_link: None,
    };

    for (label, outcome, expected_worker, expected_dp_rank) in [
        ("bootstrap", bootstrap, 11, Some(3)),
        ("completed", completed, 12, None),
    ] {
        let decode = into_decode_request(query_only_request(), outcome);
        assert_eq!(
            decode.routing.as_ref().and_then(|r| r.prefill_dp_rank),
            expected_dp_rank,
            "{label}: the bootstrap leg must tell decode which prefill DP rank owns the KV"
        );
        assert!(
            decode.staged_kv_cleanup,
            "{label}: prefill staged KV for one decode worker, so the leg must be marked for cleanup dispatch"
        );
        assert_eq!(
            decode.routing.as_ref().and_then(|r| r.prefill_worker_id),
            Some(expected_worker),
            "{label}: the decode leg must record which prefill worker holds the blocks"
        );
    }
}

#[test]
fn decode_router_override_disables_overlap_and_prefill_tracking() {
    let override_config = build_decode_router_override(
        Some(RouterConfigOverride {
            overlap_score_credit: Some(0.5),
            router_temperature: Some(0.7),
            ..Default::default()
        }),
        false,
    );

    assert_eq!(override_config.overlap_score_credit, Some(0.0));
    assert_eq!(override_config.assume_kv_reuse, Some(false));
    assert_eq!(override_config.track_prefill_tokens, Some(false));
    assert_eq!(override_config.router_temperature, Some(0.7));
}

#[test]
fn terminal_response_strips_disaggregated_params() {
    let output = Annotated::from_data(LLMEngineOutput {
        token_ids: vec![2],
        finish_reason: Some(FinishReason::EoS),
        disaggregated_params: Some(Box::new(serde_json::json!({
            "ctx_request_id": null,
            "request_type": "context_only",
        }))),
        ..Default::default()
    });

    let output = strip_terminal_disaggregated_params(output);
    let data = output
        .data
        .expect("terminal response should retain its data");
    assert_eq!(data.token_ids, vec![2]);
    assert_eq!(data.finish_reason, Some(FinishReason::EoS));
    assert!(data.disaggregated_params.is_none());
}

#[test]
fn decode_router_override_inherits_base_overlap_when_conditional_disagg_allows_it() {
    let override_config = build_decode_router_override(None, true);

    assert_eq!(override_config.overlap_score_credit, None);
    assert_eq!(override_config.assume_kv_reuse, Some(false));
    assert_eq!(override_config.track_prefill_tokens, Some(false));
}

#[test]
fn decode_router_override_preserves_request_overlap_when_conditional_disagg_allows_it() {
    let override_config = build_decode_router_override(
        Some(RouterConfigOverride {
            overlap_score_credit: Some(0.25),
            router_temperature: Some(0.7),
            ..Default::default()
        }),
        true,
    );

    assert_eq!(override_config.overlap_score_credit, Some(0.25));
    assert_eq!(override_config.assume_kv_reuse, Some(false));
    assert_eq!(override_config.track_prefill_tokens, Some(false));
    assert_eq!(override_config.router_temperature, Some(0.7));
}

#[test]
fn bootstrap_room_falls_back_when_dp_unavailable() {
    assert_eq!(compute_bootstrap_room(None, None, 12345), 12345);
    assert_eq!(compute_bootstrap_room(Some(3), None, 12345), 12345);
    assert_eq!(compute_bootstrap_room(None, Some(8), 12345), 12345);
    assert_eq!(compute_bootstrap_room(Some(0), Some(0), 12345), 12345);
}

#[test]
fn bootstrap_room_respects_modulo_and_cap() {
    let random_rooms = [0u64, 1, 49, 1_000_000, 1u64 << 62, MAX_ROOM - 1, MAX_ROOM];
    for size in [3u32, 7, 48, 49, 128] {
        for rank in [0u32, 1, size / 2, size - 1] {
            for random_room in random_rooms {
                let room = compute_bootstrap_room(Some(rank), Some(size), random_room);
                assert!(room <= MAX_ROOM);
                assert_eq!(room % size as u64, rank as u64);
            }
        }
    }
}

#[test]
fn bootstrap_room_is_deterministic_in_random_input() {
    let room_a = compute_bootstrap_room(Some(7), Some(48), 123_456_789);
    let room_b = compute_bootstrap_room(Some(7), Some(48), 123_456_789);
    assert_eq!(room_a, room_b);
    assert_eq!(room_a % 48, 7);
}

#[test]
fn merge_decode_topology_constraints_creates_and_preserves_constraints() {
    for (mut request, expect_user_constraints) in [
        (request_with_constraints(None), false),
        (
            request_with_constraints(Some(RoutingConstraints {
                required_taints: HashSet::from(["user.required".to_string()]),
                preferred_taints: HashMap::from([("user.preferred".to_string(), 0.25)]),
            })),
            true,
        ),
    ] {
        merge_decode_topology_constraints(
            &mut request,
            RoutingConstraints {
                required_taints: HashSet::from(["dynamo.topology/zone=us-east-1a".to_string()]),
                preferred_taints: HashMap::from([(
                    "dynamo.topology/rack=rack-7".to_string(),
                    0.85,
                )]),
            },
        );

        let constraints = request
            .routing
            .as_ref()
            .and_then(|routing| routing.routing_constraints.as_ref())
            .unwrap();
        assert!(
            constraints
                .required_taints
                .contains("dynamo.topology/zone=us-east-1a")
        );
        assert_eq!(
            constraints.preferred_taints["dynamo.topology/rack=rack-7"],
            0.85
        );

        if expect_user_constraints {
            assert!(constraints.required_taints.contains("user.required"));
            assert_eq!(constraints.preferred_taints["user.preferred"], 0.25);
        }
    }
}

#[test]
fn extract_bootstrap_info_parses_valid_params() {
    let params = serde_json::json!({
        "bootstrap_host": "10.0.0.5",
        "bootstrap_port": 12345,
        "bootstrap_room": 987654321u64,
        // extra fields (e.g. worker_id) must be ignored
        "worker_id": {"prefill_worker_id": 7},
    });
    let info = extract_bootstrap_info(&params).expect("valid params should parse");
    assert_eq!(info.bootstrap_host, "10.0.0.5");
    assert_eq!(info.bootstrap_port, 12345);
    assert_eq!(info.bootstrap_room, 987654321);
}

#[test]
fn extract_bootstrap_info_none_when_field_missing() {
    // Missing bootstrap_room -> not the bootstrap path (falls through to Completed).
    let missing_room = serde_json::json!({
        "bootstrap_host": "10.0.0.5",
        "bootstrap_port": 12345,
    });
    assert!(extract_bootstrap_info(&missing_room).is_none());
    // An aggregated / vLLM completed prefill carries no bootstrap fields.
    assert!(extract_bootstrap_info(&serde_json::json!({})).is_none());
}

#[test]
fn extract_bootstrap_info_rejects_out_of_range_port() {
    // bootstrap_port must fit in u16 -> reject rather than silently truncating.
    let params = serde_json::json!({
        "bootstrap_host": "h",
        "bootstrap_port": 70000,
        "bootstrap_room": 1,
    });
    assert!(extract_bootstrap_info(&params).is_none());
}
