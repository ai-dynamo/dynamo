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

    let prefill = prefill_service();
    assert_eq!(
        generate_error(&prefill, request("pf-plain", 4))
            .await
            .code(),
        Code::FailedPrecondition
    );

    // A decode server rejects the *prefill* shape, not the plain one: see
    // `a_decode_server_runs_a_request_that_bypassed_prefill` below.
    let decode = decode_service();
    let mut decode_ctx = request("dc-ctx", 4);
    decode_ctx.extra = Some(context_only_extra());
    assert_eq!(
        generate_error(&decode, decode_ctx).await.code(),
        Code::FailedPrecondition
    );
}

/// Conditional disaggregation dispatches straight to a decode worker with no
/// handoff, expecting it to run the context phase itself. The sidecar builds
/// exactly that request, so a decode server that required a session would
/// reject every bypassed request rather than serving it.
#[tokio::test]
async fn a_decode_server_runs_a_request_that_bypassed_prefill() {
    let decode = decode_service();

    let responses = drain(&decode, request("dc-plain", 4))
        .await
        .expect("a decode server must serve a request that bypassed prefill");

    assert!(
        events(&responses)
            .iter()
            .any(|event| matches!(event, pb::generate_response::Event::Finished(_))),
        "the bypassed request must run to a normal terminal"
    );
    assert!(
        !events(&responses)
            .iter()
            .any(|event| matches!(event, pb::generate_response::Event::Error(_))),
        "a bypassed request is not an error"
    );
}

#[tokio::test]
async fn prefill_stream_ends_with_prefill_ready_and_no_finished() {
    let prefill = prefill_service();
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

fn decode_service() -> TrtllmMockerService {
    TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Decode,
            ..config()
        },
        admitting_args(),
    )
    .unwrap()
}

/// The handoff's logprob is replayed, not recomputed. Regenerating it from the
/// token id would agree with the handoff on every honest run and hide a
/// corrupted one, so overwrite the value in the session and require the decode
/// leg to report exactly what it was handed.
#[tokio::test]
async fn the_replayed_logprob_is_the_one_the_handoff_carried() {
    use prost_types::{ListValue, Value, value::Kind};

    let prefill = prefill_service();
    let mut context = request("pf-corrupt", 1);
    context.extra = Some(context_only_extra());
    context.response = Some(pb::ResponseOptions {
        return_output_logprobs: Some(true),
        ..Default::default()
    });
    let responses = drain(&prefill, context).await.unwrap();
    let mut session = session_of(&responses);

    // A value no `selected_logprob` can produce: it is negative in the same
    // range but not a multiple of 0.1.
    const HANDED_OFF: f64 = -0.4242;
    let attributes = session
        .attributes_struct
        .as_mut()
        .expect("the prefill handoff carries attributes");
    attributes.fields.insert(
        super::super::handoff::ATTR_FIRST_GEN_LOG_PROBS.to_string(),
        Value {
            kind: Some(Kind::ListValue(ListValue {
                values: vec![Value {
                    kind: Some(Kind::NumberValue(HANDED_OFF)),
                }],
            })),
        },
    );

    let responses = drain(&decode_service(), decode_after(session, "dc-corrupt"))
        .await
        .unwrap();
    let first = events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::Token(token) => Some(token),
            _ => None,
        })
        .expect("the decode leg streams the replayed token");
    assert_eq!(
        first.tokens[0].logprob,
        Some(HANDED_OFF),
        "position 0 must report the handed-off value, not a regenerated one"
    );
}

/// Presence is decided by output position, not by token id. The replayed token
/// can be sampled again later in the same stream, and those occurrences are the
/// decode engine's own work -- they must keep their logprob even when the
/// context phase computed none for position 0.
#[tokio::test]
async fn a_token_repeated_after_the_replay_keeps_its_own_logprob() {
    let prefill = prefill_service();
    let mut context = request("pf-repeat", 1);
    context.extra = Some(context_only_extra());
    let responses = drain(&prefill, context).await.unwrap();
    let session = session_of(&responses);

    // No logprobs requested on the context phase, so position 0 has none.
    let responses = drain(&decode_service(), decode_after(session, "dc-repeat"))
        .await
        .unwrap();
    assert_eq!(
        missing_logprobs(&responses),
        1,
        "exactly one hole, at the replayed position"
    );

    let tokens: Vec<_> = events(&responses)
        .into_iter()
        .filter_map(|event| match event {
            pb::generate_response::Event::Token(token) => Some(token),
            _ => None,
        })
        .collect();
    assert!(
        tokens.len() > 1,
        "this test needs more than the replayed token"
    );
    for (position, token) in tokens.iter().enumerate().skip(1) {
        assert!(
            token.tokens[0].logprob.is_some(),
            "position {position} is this engine's own token and must carry a logprob"
        );
    }
}

/// Prompt and output candidates are configured separately. Reading the output
/// setting for the prompt stream returns nothing whenever only prompt
/// candidates were asked for, while the server advertises support for them.
#[tokio::test]
async fn prompt_candidates_are_honoured_without_output_candidates() {
    let service = service();
    let mut req = request("prompt-cands", 2);
    req.response = Some(pb::ResponseOptions {
        return_prompt_logprobs: Some(true),
        prompt_candidates: Some(pb::CandidateTokenSelection {
            selection: Some(pb::candidate_token_selection::Selection::TopN(3)),
        }),
        ..Default::default()
    });

    let responses = drain(&service, req).await.unwrap();
    let prompt = events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::Prompt(prompt) => Some(prompt),
            _ => None,
        })
        .expect("prompt logprobs were requested");
    assert!(
        prompt
            .tokens
            .iter()
            .all(|token| token.candidates.len() == 3),
        "each prompt token must carry the three requested candidates"
    );
}

/// How many streamed token infos carry no logprob.
fn missing_logprobs(responses: &[pb::GenerateResponse]) -> usize {
    events(responses)
        .iter()
        .filter_map(|event| match event {
            pb::generate_response::Event::Token(token) => Some(token),
            _ => None,
        })
        .flat_map(|token| token.tokens.iter())
        .filter(|info| info.logprob.is_none())
        .count()
}

/// A decode request that replays `session` and asks for output logprobs.
fn decode_after(session: pb::KvSessionRef, request_id: &str) -> pb::GenerateRequest {
    let mut decode_request = request(request_id, 4);
    decode_request.response = Some(pb::ResponseOptions {
        return_output_logprobs: Some(true),
        output_candidates: Some(pb::CandidateTokenSelection {
            selection: Some(pb::candidate_token_selection::Selection::TopN(1)),
        }),
        ..Default::default()
    });
    decode_request.kv = Some(pb::KvOptions {
        session: Some(session),
        ..Default::default()
    });
    decode_request
}
