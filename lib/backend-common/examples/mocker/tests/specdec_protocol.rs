// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{EndpointId, WorkerWithDpRank};
use dynamo_mocker_backend::specdec::PROTOCOL;
use dynamo_mocker_backend::specdec::protocol::{
    Cleanup, CleanupAck, Complete, DraftIdentity, Envelope, ErrorCode, ErrorPayload, FailureState,
    Heartbeat, HeartbeatAck, Hello, HelloAck, MAX_FRAME_BYTES, MAX_OUTPUT_TOKENS, Message,
    Proposal, SequenceValidator, Start, StartAck, proposal_digest,
};
use uuid::Uuid;

fn identity() -> DraftIdentity {
    DraftIdentity {
        endpoint: EndpointId::from("specdec/draft/generate"),
        worker: WorkerWithDpRank::new(17, 0),
        draft_incarnation_id: 23,
        protocol: PROTOCOL.to_string(),
        address: "tcp://127.0.0.1:5560".to_string(),
        orphan_cleanup_timeout_ms: 1_000,
    }
}

fn round_trip(sequence: u64, message: Message) {
    let envelope = Envelope::new(Uuid::from_u128(1), sequence, message);
    let encoded = envelope.encode().unwrap();
    assert_eq!(Envelope::decode(&encoded).unwrap(), envelope);
}

#[test]
fn every_message_type_round_trips_as_versioned_json() {
    let digest = proposal_digest(&[1, 2]);
    let messages = [
        Message::Hello(Hello {
            expected: identity(),
        }),
        Message::HelloAck(HelloAck {
            identity: identity(),
        }),
        Message::Start(Start {
            prompt_token_ids: vec![11, 12],
            max_output_tokens: 2,
        }),
        Message::StartAck(StartAck::default()),
        Message::Heartbeat(Heartbeat::default()),
        Message::HeartbeatAck(HeartbeatAck::default()),
        Message::Cleanup(Cleanup::default()),
        Message::CleanupAck(CleanupAck::default()),
        Message::Proposal(Proposal {
            token_ids: vec![1, 2],
        }),
        Message::Complete(Complete {
            final_sequence: 9,
            proposal_digest: digest,
        }),
        Message::Error(ErrorPayload::new(
            ErrorCode::QueueFull,
            FailureState::NotStarted,
        )),
    ];
    for (sequence, message) in messages.into_iter().enumerate() {
        round_trip(sequence as u64, message);
    }
}

#[test]
fn version_size_token_and_sequence_limits_fail_closed() {
    let mut wrong_version = Envelope::new(
        Uuid::from_u128(2),
        0,
        Message::Heartbeat(Heartbeat::default()),
    );
    wrong_version.protocol_version += 1;
    let wire = serde_json::to_vec(&wrong_version).unwrap();
    assert_eq!(
        Envelope::decode(&wire).unwrap_err().to_string(),
        "unsupported protocol version"
    );
    assert_eq!(
        Envelope::decode(&vec![b' '; MAX_FRAME_BYTES + 1])
            .unwrap_err()
            .to_string(),
        "frame exceeds protocol limit"
    );

    let empty_prompt = Envelope::new(
        Uuid::from_u128(3),
        0,
        Message::Start(Start {
            prompt_token_ids: Vec::new(),
            max_output_tokens: 1,
        }),
    );
    assert!(empty_prompt.encode().is_err());
    let too_many_output = Envelope::new(
        Uuid::from_u128(4),
        0,
        Message::Start(Start {
            prompt_token_ids: vec![1],
            max_output_tokens: MAX_OUTPUT_TOKENS + 1,
        }),
    );
    assert!(too_many_output.encode().is_err());

    let mut sequences = SequenceValidator::starting_at(1);
    sequences.observe(1).unwrap();
    assert!(sequences.observe(1).is_err());
    assert!(sequences.observe(3).is_err());
    assert_eq!(sequences.next(), 2);
}

#[test]
fn malformed_identity_and_noncanonical_error_text_are_rejected_without_echoing_secrets() {
    let mut malformed = identity();
    malformed.address = "tcp://secret.example:5560\npassword=hunter2".to_string();
    let hello = Envelope::new(
        Uuid::from_u128(5),
        0,
        Message::HelloAck(HelloAck {
            identity: malformed,
        }),
    );
    let error = hello.encode().unwrap_err().to_string();
    assert!(!error.contains("secret.example"));
    assert!(!error.contains("hunter2"));

    let injected = Envelope::new(
        Uuid::from_u128(6),
        0,
        Message::Error(ErrorPayload {
            code: ErrorCode::Internal,
            state: FailureState::Ambiguous,
            message: "failed at tcp://user:password@secret.example".to_string(),
        }),
    );
    let error = injected.encode().unwrap_err().to_string();
    assert!(!error.contains("secret.example"));
    assert!(!error.contains("password"));
}
