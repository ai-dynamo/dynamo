// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The prefill/decode legs and the KV handoff codec between them.

use super::*;

/// The prefill worker marks its request `context_only` and caps generation at
/// the single token the context phase produces.
#[test]
fn prefill_request_is_marked_context_only() {
    let mut req = request();
    req.stop_conditions.max_tokens = Some(128);
    req.stop_conditions.min_tokens = Some(8);
    req.output_options.logprobs = Some(1);
    let proto = build_generate_request(&req, "req", "model", None, DisaggregationMode::Prefill)
        .expect("build prefill request");

    assert!(
        is_context_only(&proto),
        "prefill must set extra.request_type"
    );
    let stopping = proto.stopping.expect("stopping options");
    assert_eq!(stopping.max_tokens, Some(1));
    // The minimum masks EOS rather than extending generation, so it must
    // survive: without it the context phase can sample EOS on its single token
    // and terminate with `Stop` instead of the `PrefillReady` handoff.
    assert_eq!(stopping.min_tokens, Some(8));
    // The prefill worker surfaces no tokens, but it must still compute
    // logprobs: the first generated token comes from the context phase and its
    // logprob only reaches the decode worker through the handoff.
    assert_eq!(
        proto
            .response
            .expect("response options")
            .return_output_logprobs,
        Some(true)
    );
    assert!(proto.kv.is_none(), "prefill carries no session to replay");
}

/// `PrefillReady` is the prefill worker's terminal chunk: no tokens, and the
/// handoff the decode worker will replay.
#[tokio::test]
async fn prefill_ready_is_the_terminal_handoff() {
    let server = FakeServer::start(FakeTrtllm::default()).await;
    let engine = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    engine.start(0).await.expect("start");

    let outputs = collect(&engine, request()).await;
    let terminal = outputs
        .iter()
        .find(|output| output.finish_reason.is_some())
        .expect("a terminal chunk");
    // The frontend's prefill router chains into decode only for `Length`; any
    // other terminal reason is returned to the caller as a finished request.
    assert_eq!(terminal.finish_reason, Some(FinishReason::Length));

    assert!(
        outputs.iter().all(|output| output.token_ids.is_empty()),
        "a prefill worker must not stream tokens to the client"
    );
    let handoff = terminal
        .disaggregated_params
        .as_ref()
        .expect("terminal carries the prefill handoff");
    assert_eq!(handoff["session_id"], json!("12345"));
    assert_eq!(handoff["transfer_backend"], json!("NIXL"));
    assert_eq!(handoff["endpoints"][0]["port"], json!(5601));
    assert_eq!(handoff["attributes"]["first_gen_tokens"], json!([42]));
}

/// The decode worker replays the prefill handoff verbatim in `kv.session`.
#[tokio::test]
async fn decode_request_replays_the_prefill_session() {
    let server = FakeServer::start(FakeTrtllm::default()).await;

    // Phase 1: prefill produces the handoff.
    let prefill = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Prefill);
    prefill.start(0).await.expect("start prefill");
    let handoff = collect(&prefill, request())
        .await
        .into_iter()
        .find_map(|output| output.disaggregated_params)
        .expect("prefill handoff");

    // Phase 2: decode replays it.
    let decode = engine_in_mode(&server.endpoint, 1, DisaggregationMode::Decode);
    decode.start(0).await.expect("start decode");
    let mut req = request();
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: handoff,
        prompt_tokens_details: None,
    });
    let outputs = collect(&decode, req).await;
    assert!(
        outputs.iter().any(|output| !output.token_ids.is_empty()),
        "the decode worker streams the completion"
    );

    let session = server.service.requests.lock().await[1]
        .kv
        .as_ref()
        .and_then(|kv| kv.session.as_ref())
        .expect("decode request carries kv.session")
        .clone();
    assert_eq!(
        session,
        fake_session(),
        "the handoff must round-trip intact"
    );
}

/// Conditional disaggregation dispatches straight to a decode worker with no
/// handoff, expecting it to run the context phase itself. Rejecting that would
/// fail every bypassed request.
#[test]
fn decode_without_a_prefill_result_runs_the_whole_request() {
    let proto = build_generate_request(
        &request(),
        "req",
        "model",
        limits(4096),
        DisaggregationMode::Decode,
    )
    .expect("a decode request without a handoff is the bypass path");
    assert!(
        proto.kv.and_then(|kv| kv.session).is_none(),
        "no session should be replayed when none was handed off"
    );
    assert!(
        proto.extra.is_none(),
        "a bypassed request is not context_only"
    );
}

/// A handoff on a non-decode worker means the frontend routed the request to
/// the wrong role - fail loudly rather than silently prefill it again.
#[test]
fn prefill_result_on_a_non_decode_worker_is_rejected() {
    let mut req = request();
    req.prefill_result = Some(dynamo_backend_common::PrefillResult {
        disaggregated_params: json!({"session_id": "1"}),
        prompt_tokens_details: None,
    });
    for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Prefill] {
        let error = build_generate_request(&req, "req", "model", None, mode)
            .expect_err("handoff must be rejected");
        assert!(
            error
                .to_string()
                .contains("must be routed to a decode worker"),
            "unexpected error for {mode}: {error}"
        );
    }
}

/// An aggregated worker never asks for a context handoff, so receiving one is
/// protocol drift rather than a silent no-op.
#[test]
fn unexpected_prefill_ready_on_an_aggregated_worker_is_rejected() {
    let mut state = ResponseState::new(&request(), AGG);
    let error = state
        .convert(pb::GenerateResponse {
            request_id: "req".to_string(),
            event: Some(pb::generate_response::Event::PrefillReady(
                pb::PrefillReady {
                    kv_session: Some(fake_session()),
                },
            )),
            usage: None,
        })
        .expect_err("prefill_ready must be rejected");
    assert!(
        error.to_string().contains("not running in prefill mode"),
        "unexpected error: {error}"
    );
}

/// The OpenEngine servicer suppresses `finished` on a context request only
/// when it already sent a `PrefillReady`; otherwise -- a stop condition during
/// the one-token context phase, or a cancellation before transmission -- it
/// deliberately emits the real terminal, because there is no decode leg to
/// answer the caller. Context tokens are held back on the normal path so the
/// decode leg's replay cannot duplicate them, which makes this terminal the
/// only place they can surface.
#[test]
fn a_prefill_terminal_without_a_handoff_carries_its_tokens() {
    let mut state = ResponseState::new(&request(), DisaggregationMode::Prefill);
    let token = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::Token(pb::TokenOutput {
            output_index: Some(0),
            tokens: vec![pb::TokenInfo {
                token_id: 99,
                logprob: Some(-0.5),
                rank: Some(1),
                candidates: vec![pb::LogProb {
                    token_id: 100,
                    logprob: -1.5,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        })),
        usage: None,
    };
    assert!(
        state.convert(token).expect("token converts").is_none(),
        "a context token is held back, not streamed"
    );

    let response = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Stop as i32,
                message: String::new(),
                stop_match: None,
            },
        )),
        usage: None,
    };
    let terminal = state
        .convert(response)
        .expect("a handoff-less prefill terminal is serviceable")
        .expect("it yields a terminal");
    assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
    assert_eq!(
        terminal.token_ids,
        [99],
        "the caller gets what the context phase produced"
    );
    // The held token is the whole answer here, so dropping to bare IDs would
    // silently strip logprobs a caller asked for and cannot get anywhere else.
    assert_eq!(terminal.log_probs.as_deref(), Some(&[-0.5][..]));
    let candidates = terminal.top_logprobs.expect("candidates");
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0][0].token_id, 100);
}

/// `PrefillReady` is the final response for a context request, so its usage is
/// authoritative -- the frontend forwards `cached_tokens` from it to the decode
/// leg and cannot reconstruct it.
#[test]
fn prefill_ready_keeps_the_engines_reported_usage() {
    let mut state = ResponseState::new(&request(), DisaggregationMode::Prefill);
    let response = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::PrefillReady(
            pb::PrefillReady {
                kv_session: Some(fake_session()),
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens: 11,
            completion_tokens: 0,
            total_tokens: 11,
            cached_prompt_tokens: Some(7),
            reasoning_tokens: None,
        }),
    };
    let output = state
        .convert(response)
        .expect("prefill_ready converts")
        .expect("prefill_ready yields a terminal");
    let usage = output.completion_usage.expect("usage is set");
    assert_eq!(usage.prompt_tokens, 11);
    assert_eq!(
        usage
            .prompt_tokens_details
            .and_then(|details| details.cached_tokens),
        Some(7),
        "cached prompt tokens must survive to the decode leg"
    );
}

/// A context request that ends at its one-token budget never transmitted its
/// KV, so there is no decode leg to finish the completion. Returning that
/// single token would silently truncate the answer, and passing the `Length`
/// terminal through makes the frontend's prefill router report a missing
/// handoff instead of the real cause.
#[test]
fn a_prefill_terminal_that_ran_out_of_budget_is_rejected() {
    let mut state = ResponseState::new(&request(), DisaggregationMode::Prefill);
    let response = pb::GenerateResponse {
        request_id: "req".to_string(),
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Length as i32,
                ..Default::default()
            },
        )),
        usage: None,
    };
    let error = state
        .convert(response)
        .expect_err("a budget-exhausted context request has no answer to return");
    assert!(
        error.to_string().contains("without a kv_session handoff"),
        "the error must name the missing handoff: {error}"
    );
}

/// The handoff codec is symmetric: the decode worker requires exactly what the
/// prefill worker wrote. A field lost in transit must fail by name rather than
/// decode into a plausible session. `attributes` matters most -- the server
/// reads the session's location and rank out of it, and accepts an endpoint in
/// place of an `opaque_state`, so a handoff that lost it would pass the
/// server's own guard and resume a session with no opaque state at rank 0.
#[test]
fn a_handoff_missing_a_field_is_rejected() {
    let mut session = fake_session();
    session.dp_rank = 3;
    let encoded = crate::disagg::session_to_json(session).expect("encode");

    let decoded = crate::disagg::session_from_json(&encoded).expect("round trip");
    assert_eq!(decoded.dp_rank, 3, "the rank must survive the round trip");

    for field in [
        "attributes",
        "dp_rank",
        "endpoints",
        "transfer_backend",
        "session_id",
    ] {
        let mut mangled = encoded.clone();
        mangled
            .as_object_mut()
            .expect("handoff object")
            .remove(field)
            .expect("field is present");
        let error = crate::disagg::session_from_json(&mangled)
            .expect_err("a handoff missing a field is not usable");
        assert!(
            error.to_string().contains(field),
            "the error must name {field}: {error}"
        );
    }
}

/// A newer prefill worker may add fields this decode worker has never heard of.
/// Rejecting them would fail every new-prefill/old-decode request during a
/// rolling upgrade, as a non-migratable 400, with no version to negotiate on.
#[test]
fn a_handoff_carrying_an_unknown_field_still_decodes() {
    let encoded = crate::disagg::session_to_json(fake_session()).expect("encode");
    let mut newer = encoded.clone();
    newer
        .as_object_mut()
        .expect("handoff object")
        .insert("schedule_style".to_string(), serde_json::json!(2));

    crate::disagg::session_from_json(&newer)
        .expect("a handoff from a newer peer must still decode");
}

/// The prefill leg refuses to emit a handoff the decode leg could not resolve,
/// so the failure names the prefill worker rather than surfacing one hop later.
#[test]
fn a_prefill_ready_without_attributes_is_rejected() {
    let mut session = fake_session();
    session.attributes_struct = None;

    let error = crate::disagg::session_to_json(session)
        .expect_err("a session with no attributes cannot locate the context worker");
    assert!(
        error.to_string().contains("attributes_struct"),
        "the error must name the missing field: {error}"
    );
}
