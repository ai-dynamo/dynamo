// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::BackendError;
use dynamo_sidecar_common::testing::{ScriptedClient, SidecarFixture};

use super::*;

struct Sglang;

impl SidecarFixture for Sglang {
    type Engine = SglangSidecarEngine;
    type Request = pb::GenerateRequest;
    type Response = pb::GenerateResponse;

    fn engine(scripted: Option<ScriptedClient<Self::Request, Self::Response>>) -> Self::Engine {
        let mut engine = SglangSidecarEngine::new(
            "http://unused:1",
            TransportConfig::default(),
            DisaggregationMode::Aggregated,
            None,
            None,
        );
        engine.scripted = scripted;
        engine
    }

    fn first_token() -> Self::Response {
        cumulative(vec![42])
    }

    fn successful_responses() -> Vec<Self::Response> {
        vec![
            cumulative(vec![]),
            Self::first_token(),
            cumulative(vec![42, 43]),
            cumulative(vec![42, 43]),
            cumulative(vec![42]),
            cumulative(vec![42, 43, 44]),
            pb::GenerateResponse {
                output_ids: vec![42, 43, 44],
                finished: true,
                meta_info: HashMap::from([
                    ("finish_reason".into(), r#"{"type":"length"}"#.into()),
                    ("prompt_tokens".into(), "3".into()),
                    ("completion_tokens".into(), "3".into()),
                ]),
            },
        ]
    }

    fn request_id(request: &Self::Request) -> &str {
        request.rid.as_deref().unwrap()
    }
    fn eof_error() -> BackendError {
        BackendError::EngineShutdown
    }
}

fn cumulative(output_ids: Vec<i32>) -> pb::GenerateResponse {
    pb::GenerateResponse {
        output_ids,
        ..Default::default()
    }
}

dynamo_sidecar_common::sidecar_contract_tests!(Sglang);
