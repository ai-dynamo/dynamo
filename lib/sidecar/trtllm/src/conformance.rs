// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::BackendError;
use dynamo_sidecar_common::testing::{ScriptedClient, SidecarFixture};

use super::*;
use crate::proto as pb;

struct Trtllm;

impl SidecarFixture for Trtllm {
    type Engine = TrtllmSidecarEngine;
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateResponse;

    fn engine(client: Option<ScriptedClient<Self::Request, Self::Response>>) -> Self::Engine {
        let engine = TrtllmSidecarEngine::new(
            GrpcEndpoint::parse("http://unused:1", "test").unwrap(),
            GrpcTransportConfig::default(),
            ConfiguredModel {
                source: "test-model".into(),
                context_length: None,
            },
        );
        if let Some(client) = client {
            assert!(engine.client.set(TrtllmClient::Scripted(client)).is_ok());
        }
        engine
    }

    fn first_token() -> Self::Response {
        chunk(vec![42], 1)
    }

    fn successful_responses() -> Vec<Self::Response> {
        vec![
            chunk(vec![], 0),
            Self::first_token(),
            chunk(vec![43, 44], 3),
            chunk(vec![], 0),
            pb::GenerateResponse {
                response: Some(pb::generate_response::Response::Complete(
                    pb::GenerateComplete {
                        output_token_ids: vec![42, 43, 44],
                        finish_reason: "length".into(),
                        prompt_tokens: 3,
                        completion_tokens: 0,
                        ..Default::default()
                    },
                )),
                ..Default::default()
            },
        ]
    }

    fn request_id(request: &Self::Request) -> &str {
        &request.request_id
    }
    fn eof_error() -> BackendError {
        BackendError::Unknown
    }
}

fn chunk(token_ids: Vec<u32>, completion_tokens: u32) -> pb::GenerateResponse {
    pb::GenerateResponse {
        response: Some(pb::generate_response::Response::Chunk(
            pb::GenerateStreamChunk {
                token_ids,
                prompt_tokens: 3,
                completion_tokens,
                ..Default::default()
            },
        )),
        ..Default::default()
    }
}

dynamo_sidecar_common::sidecar_contract_tests!(Trtllm);
