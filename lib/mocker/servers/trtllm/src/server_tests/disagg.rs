// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The prefill and decode roles, and the handoff they exchange.

use super::*;

#[tokio::test]
async fn role_validation_rejects_mismatched_disaggregation_payloads() {
    let aggregated = service();
    let mut context_only = request("agg-ctx", 4);
    context_only.extra = Some(context_only_extra());
    assert_eq!(
        generate_error(&aggregated, context_only).await.code(),
        Code::FailedPrecondition
    );

    let prefill = TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            ..config()
        },
        admitting_args(),
    )
    .unwrap();
    assert_eq!(
        generate_error(&prefill, request("pf-plain", 4))
            .await
            .code(),
        Code::FailedPrecondition
    );

    let decode = TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Decode,
            ..config()
        },
        admitting_args(),
    )
    .unwrap();
    assert_eq!(
        generate_error(&decode, request("dc-plain", 4)).await.code(),
        Code::FailedPrecondition
    );
}

#[tokio::test]
async fn prefill_stream_ends_with_prefill_ready_and_no_finished() {
    let prefill = TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            ..config()
        },
        admitting_args(),
    )
    .unwrap();
    let mut request = request("pf-1", 1);
    request.extra = Some(context_only_extra());
    let responses = drain(&prefill, request).await.unwrap();
    let events = events(&responses);

    assert!(
        !events
            .iter()
            .any(|event| matches!(event, pb::generate_response::Event::Finished(_)))
    );
    let ready = events
        .iter()
        .filter_map(|event| match event {
            pb::generate_response::Event::PrefillReady(ready) => Some(ready),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ready.len(), 1);
    let session = ready[0].kv_session.as_ref().unwrap();
    assert!(session.session_id.starts_with(handoff::SESSION_PREFIX));
    assert_eq!(session.transfer_backend, handoff::TRANSFER_BACKEND);
    assert_eq!(session.endpoints.len(), 1);
    assert!(matches!(
        events.last().unwrap(),
        pb::generate_response::Event::PrefillReady(_)
    ));
}

/// Each case mutates one leg of the handoff the way a lossy relay would. All of
/// them must be caught, or the round trip proves nothing.
#[tokio::test]
async fn decode_rejects_a_handoff_the_sidecar_mangled() {
    let session = prefill_session("pf-mangle").await;
    let decode = decode_service();

    let without = |key: &str| {
        let mut mutated = session.clone();
        let mut attributes = mutated.attributes_struct.clone().unwrap();
        attributes.fields.remove(key);
        mutated.attributes_struct = Some(attributes);
        mutated
    };
    let with_attribute = |key: &str, value: Kind| {
        let mut mutated = session.clone();
        let mut attributes = mutated.attributes_struct.clone().unwrap();
        attributes
            .fields
            .insert(key.to_string(), Value { kind: Some(value) });
        mutated.attributes_struct = Some(attributes);
        mutated
    };

    let mut no_attributes = session.clone();
    no_attributes.attributes_struct = None;
    let mut bad_backend = session.clone();
    bad_backend.transfer_backend = "NIXL".to_string();
    let mut no_endpoints = session.clone();
    no_endpoints.endpoints.clear();
    let mut lost_port = session.clone();
    lost_port.endpoints[0].port = 0;
    let mut bad_rank = session.clone();
    bad_rank.dp_rank = 3;

    let mutations = [
        ("dropped attribute", without(handoff::ATTR_REQUEST_ID)),
        ("dropped attributes", no_attributes),
        (
            "rounded fractional",
            with_attribute(handoff::ATTR_TTFT_MS, Kind::NumberValue(12.0)),
        ),
        (
            "flattened list",
            with_attribute(handoff::ATTR_FIRST_GEN_TOKENS, Kind::NumberValue(7.0)),
        ),
        ("defaulted string", bad_backend),
        ("dropped repeated", no_endpoints),
        ("defaulted number", lost_port),
        ("altered scalar", bad_rank),
    ];

    for (label, mutated) in mutations {
        let mut decode_request = request("dc-mangle", 4);
        decode_request.kv = Some(pb::KvOptions {
            session: Some(mutated),
            ..Default::default()
        });
        let error = generate_error(&decode, decode_request).await;
        assert_eq!(error.code(), Code::InvalidArgument, "mutation '{label}'");
    }

    // The untouched session still works, so the mutations above are what fail.
    let mut decode_request = request("dc-ok", 4);
    decode_request.kv = Some(pb::KvOptions {
        session: Some(session),
        ..Default::default()
    });
    assert!(drain(&decode, decode_request).await.is_ok());
}

/// A context request that does not ask for exactly one token is a client bug the
/// prefill role must surface rather than silently normalize.
#[tokio::test]
async fn prefill_requires_a_single_token_budget() {
    let prefill = prefill_service();
    let mut oversized = request("pf-budget", 8);
    oversized.extra = Some(context_only_extra());
    let error = generate_error(&prefill, oversized).await;
    assert_eq!(error.code(), Code::InvalidArgument);
    assert!(error.message().contains("exactly one token"), "{error}");
}

/// The decode leg replays the context phase's first token, so the two legs'
/// accounting matches a real engine's instead of inventing a fresh stream.
#[tokio::test]
async fn decode_replays_the_prefill_first_token() {
    let session = prefill_session("pf-replay").await;
    let handed_off = handoff::first_gen_token(&session).unwrap();

    let decode = decode_service();
    let mut decode_request = request("dc-replay", 3);
    decode_request.kv = Some(pb::KvOptions {
        session: Some(session),
        ..Default::default()
    });
    let responses = drain(&decode, decode_request).await.unwrap();
    let first = events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::Token(token) => Some(token.tokens[0].token_id),
            _ => None,
        })
        .unwrap();
    assert_eq!(first, handed_off);
}

/// `PrefillReady` is the context request's terminal event, so it carries the
/// engine's usage: the decode leg cannot reconstruct the context phase's
/// cache-hit count, and a real server reports it here.
#[tokio::test]
async fn prefill_ready_carries_the_context_phases_usage() {
    let prefill = prefill_service();
    let mut prefill_request = request("req-prefill-usage", 1);
    prefill_request.extra = Some(context_only_extra());
    let responses = drain(&prefill, prefill_request).await.unwrap();

    let usage = responses
        .iter()
        .find(|response| {
            matches!(
                response.event,
                Some(pb::generate_response::Event::PrefillReady(_))
            )
        })
        .expect("a context request ends with PrefillReady")
        .usage
        .as_ref()
        .expect("PrefillReady must carry usage");
    assert_eq!(usage.prompt_tokens, 4);
    assert_eq!(usage.completion_tokens, 1);
}

/// The decode leg replays the context phase's first token, so that token's
/// logprob exists only if the context phase computed one. A real server carries
/// it in the handoff as `first_gen_log_probs` and drops it otherwise, and the
/// sidecar fails any request whose delta token lacks a requested logprob -- so
/// the mocker has to lose it in the same place, or that failure mode is
/// unreachable from a test.
#[tokio::test]
async fn the_first_tokens_logprob_survives_only_if_the_context_phase_computed_it() {
    let logprobs = || {
        Some(pb::ResponseOptions {
            return_output_logprobs: Some(true),
            output_candidates: Some(pb::CandidateTokenSelection {
                selection: Some(pb::candidate_token_selection::Selection::TopN(1)),
            }),
            ..Default::default()
        })
    };

    let missing_logprobs = |responses: &[pb::GenerateResponse]| {
        events(responses)
            .iter()
            .filter_map(|event| match event {
                pb::generate_response::Event::Token(token) => Some(token),
                _ => None,
            })
            .flat_map(|token| token.tokens.iter())
            .filter(|info| info.logprob.is_none())
            .count()
    };

    let decode_after = |session: pb::KvSessionRef, request_id: &str| {
        let mut decode_request = request(request_id, 4);
        decode_request.response = logprobs();
        decode_request.kv = Some(pb::KvOptions {
            session: Some(session),
            ..Default::default()
        });
        decode_request
    };

    // Context phase asked for logprobs: the handoff carries the first one.
    let prefill = prefill_service();
    let mut context = request("pf-lp", 1);
    context.extra = Some(context_only_extra());
    context.response = logprobs();
    let responses = drain(&prefill, context).await.unwrap();
    let session = session_of(&responses);
    let responses = drain(&decode_service(), decode_after(session, "dc-lp"))
        .await
        .unwrap();
    assert_eq!(missing_logprobs(&responses), 0);

    // Context phase did not: the replayed token has no logprob to report.
    let mut context = request("pf-nolp", 1);
    context.extra = Some(context_only_extra());
    let responses = drain(&prefill, context).await.unwrap();
    let session = session_of(&responses);
    let responses = drain(&decode_service(), decode_after(session, "dc-nolp"))
        .await
        .unwrap();
    assert_eq!(
        missing_logprobs(&responses),
        1,
        "only the replayed first token can be missing its logprob"
    );
}
