// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_mocker::common::protocols::{DirectRequest, MockEngineArgs};
use dynamo_mocker::replay::{
    ReplayAdvanceSnapshot, ReplayCheckpointKind, ReplaySession, ReplaySessionConfig,
};
use uuid::Uuid;

#[test]
fn public_replay_session_api_is_usable_from_an_external_crate() {
    fn assert_send<T: Send>() {}
    assert_send::<ReplaySession>();

    let args = MockEngineArgs::builder()
        .block_size(4)
        .num_gpu_blocks(64)
        .max_num_batched_tokens(Some(8))
        .max_num_seqs(Some(2))
        .enable_prefix_caching(true)
        .enable_chunked_prefill(true)
        .build()
        .unwrap();
    let request = DirectRequest {
        tokens: vec![1; 8],
        max_output_tokens: 2,
        output_token_ids: Some(vec![7, 8]),
        uuid: Some(Uuid::from_u128(1)),
        arrival_timestamp_ms: Some(0.0),
        ..Default::default()
    };
    let mut session = ReplaySession::new(ReplaySessionConfig::new(args), vec![request]).unwrap();
    session.advance_to(0.001).unwrap();
    let telemetry = session.telemetry_since(0.0).unwrap();
    assert_eq!(telemetry.cursor_ms, 0.001);
    assert_eq!(telemetry.metrics.avg_isl_tokens, Some(8.0));
    assert_eq!(telemetry.metrics.avg_requested_osl_tokens, Some(2.0));
    let traffic = session.traffic_telemetry_since(0.0).unwrap();
    assert_eq!(traffic.cursor_ms, telemetry.cursor_ms);
    assert_eq!(traffic.metrics.avg_isl_tokens, Some(8.0));
    assert_eq!(traffic.metrics.avg_requested_osl_tokens, Some(2.0));
    assert_eq!(traffic.metrics.p95_ttft_ms, None);
    assert_eq!(traffic.metrics.p95_itl_ms, None);
    assert_eq!(traffic.metrics.p95_e2e_ms, None);

    let checkpoint = session.checkpoint().unwrap();
    assert_eq!(checkpoint.kind(), ReplayCheckpointKind::DeepRuntimeMemento);
    let restored = checkpoint.restore().unwrap();
    assert_eq!(session.state().unwrap(), restored.state().unwrap());

    let sampled = session.advance_sampled(1.0, 0.1, Some(0.5)).unwrap();
    assert_eq!(sampled.final_state.cursor_ms, 1.0);
    assert!(
        sampled
            .observations
            .last()
            .and_then(|observation| observation.telemetry.as_ref())
            .is_some()
    );

    let snapshot: ReplayAdvanceSnapshot = session.advance_to_with_telemetry(2.0, 1.0).unwrap();
    assert_eq!(snapshot.state.cursor_ms, 2.0);
    assert_eq!(snapshot.telemetry.metrics.window_start_ms, 1.0);
}
