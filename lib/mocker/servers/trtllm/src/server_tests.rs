// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_mocker::common::protocols::MockEngineArgsBuilder;
use dynamo_trtllm_sidecar::disagg::context_only_extra;
use futures::StreamExt;
use pb::control_server::Control;
use pb::inference_server::Inference;
use prost_types::{Value, value::Kind};
use tonic::Code;

use super::*;

// The tests themselves live in `server_tests/`, grouped by the surface they
// cover; this file holds only the fixtures they share.
#[path = "server_tests/control.rs"]
mod control;
#[path = "server_tests/disagg.rs"]
mod disagg;
#[path = "server_tests/serving.rs"]
mod serving;

fn admitting_args() -> MockEngineArgs {
    MockEngineArgsBuilder::default()
        .engine_type(EngineType::Trtllm)
        .num_gpu_blocks(4_096usize)
        .block_size(4usize)
        .speedup_ratio(0.0)
        .build()
        .unwrap()
}

fn config() -> MockerServerConfig {
    MockerServerConfig {
        context_length: 1_024,
        ..Default::default()
    }
}

fn service() -> TrtllmMockerService {
    TrtllmMockerService::new(config(), admitting_args()).unwrap()
}

/// Neither the service nor the response stream implements `Debug`, so
/// `unwrap_err` is unavailable on these results.
fn construction_error(config: MockerServerConfig, args: MockEngineArgs) -> String {
    match TrtllmMockerService::new(config, args) {
        Ok(_) => panic!("expected the constructor to fail"),
        Err(error) => error.to_string(),
    }
}

async fn generate_error(service: &TrtllmMockerService, request: pb::GenerateRequest) -> Status {
    match service.generate(Request::new(request)).await {
        Ok(_) => panic!("expected Generate to fail"),
        Err(status) => status,
    }
}

fn request(request_id: &str, max_tokens: u32) -> pb::GenerateRequest {
    pb::GenerateRequest {
        request_id: request_id.to_string(),
        model: "mocker-model".to_string(),
        input: Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: vec![1, 2, 3, 4],
        })),
        stopping: Some(pb::StoppingOptions {
            max_tokens: Some(max_tokens),
            ..Default::default()
        }),
        ..Default::default()
    }
}

async fn drain(
    service: &TrtllmMockerService,
    request: pb::GenerateRequest,
) -> Result<Vec<pb::GenerateResponse>, Status> {
    let mut stream = service.generate(Request::new(request)).await?.into_inner();
    let mut responses = Vec::new();
    while let Some(item) = stream.next().await {
        responses.push(item?);
    }
    Ok(responses)
}

fn prefill_service() -> TrtllmMockerService {
    TrtllmMockerService::new(
        MockerServerConfig {
            mode: ServerMode::Prefill,
            ..config()
        },
        admitting_args(),
    )
    .unwrap()
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

async fn prefill_session(request_id: &str) -> pb::KvSessionRef {
    let prefill = prefill_service();
    let mut prefill_request = request(request_id, 1);
    prefill_request.extra = Some(context_only_extra());
    let responses = drain(&prefill, prefill_request).await.unwrap();
    events(&responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::PrefillReady(ready) => ready.kv_session.clone(),
            _ => None,
        })
        .expect("prefill must emit a session")
}

fn session_of(responses: &[pb::GenerateResponse]) -> pb::KvSessionRef {
    events(responses)
        .into_iter()
        .find_map(|event| match event {
            pb::generate_response::Event::PrefillReady(ready) => ready.kv_session.clone(),
            _ => None,
        })
        .expect("a context request must emit a session")
}

fn events(responses: &[pb::GenerateResponse]) -> Vec<&pb::generate_response::Event> {
    responses
        .iter()
        .map(|response| {
            response
                .event
                .as_ref()
                .expect("every response must carry an event")
        })
        .collect()
}
