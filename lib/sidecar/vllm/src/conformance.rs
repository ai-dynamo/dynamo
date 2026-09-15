// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::BackendError;
use dynamo_sidecar_common::testing::{ScriptedClient, SidecarFixture};

use super::*;
use crate::proto as pb;

struct Vllm;

impl SidecarFixture for Vllm {
    type Engine = VllmSidecarEngine;
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateResponse;

    fn engine(client: Option<ScriptedClient<Self::Request, Self::Response>>) -> Self::Engine {
        let engine = VllmSidecarEngine::new(
            GrpcEndpoint::parse("http://unused:1", "test").unwrap(),
            ConfiguredModel {
                source: "test-model".into(),
            },
            DisaggregationMode::Aggregated,
            GrpcTransportConfig::default(),
        );
        if let Some(client) = client {
            assert!(engine.client.set(VllmClient::Scripted(client)).is_ok());
        }
        engine
    }

    fn first_token() -> Self::Response {
        delta(vec![42], None)
    }

    fn successful_responses() -> Vec<Self::Response> {
        vec![
            pb::GenerateResponse::default(),
            Self::first_token(),
            delta(vec![43, 44], None),
            delta(
                vec![],
                Some(pb::FinishInfo {
                    finish_reason: pb::finish_info::FinishReason::Length.into(),
                    num_output_tokens: 3,
                    ..Default::default()
                }),
            ),
        ]
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.request_id
    }
    fn eof_error() -> BackendError {
        BackendError::Unknown
    }
}

fn delta(token_ids: Vec<u32>, finish_info: Option<pb::FinishInfo>) -> pb::GenerateResponse {
    pb::GenerateResponse {
        outputs: Some(pb::SequenceOutput {
            num_tokens: token_ids.len() as u32,
            token_ids,
            finish_info,
            ..Default::default()
        }),
        ..Default::default()
    }
}

dynamo_sidecar_common::sidecar_contract_tests!(Vllm);
