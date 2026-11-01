// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn vllm_format_preserves_binding_terminal_status_and_wire_shape() {
    use dynamo_llm::protocols::common::FinishReason;
    for (finish, missing_logprobs, status) in [
        (FinishReason::Length, false, StatusCode::OK),
        (FinishReason::Cancelled, false, CANCELLED),
        (
            FinishReason::Length,
            true,
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
        (
            FinishReason::Error("private engine detail".into()),
            false,
            StatusCode::INTERNAL_SERVER_ERROR,
        ),
    ] {
        let format: SharedFormat = Arc::new(dynamo_vllm_generate::VllmFormat);
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        headers.insert("x-request-id", "header-id".parse().unwrap());
        headers.insert("x-data-parallel-rank", "3".parse().unwrap());
        let body = serde_json::json!({
            "token_ids": [1], "sampling_params": if missing_logprobs {
                serde_json::json!({"logprobs": 1})
            } else { serde_json::json!({}) },
            "request_id": "body-id", "priority": -7,
        })
        .to_string();
        let opened = format
            .open(ClientRequest {
                body: body.as_bytes(),
                headers: &headers,
                default_request_id: "default-id",
            })
            .unwrap();
        assert_eq!(opened.request_id, "header-id");
        let bound = (opened.bind)(&dynamo_sglang_generate::testkit::worker()).unwrap();
        let hints = bound.request.routing.as_ref().unwrap();
        assert_eq!(hints.dp_rank, Some(3));
        assert_eq!(hints.priority, Some(7));
        let state = service_v2::HttpService::builder()
            .build()
            .unwrap()
            .state_clone();
        let output = LLMEngineOutput {
            token_ids: vec![2, 3],
            finish_reason: (!missing_logprobs).then_some(finish),
            ..Default::default()
        };
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            respond(
                state,
                format,
                Arc::new(ReplayEngine(
                    vec![Annotated::from_data(output)],
                    status != StatusCode::OK,
                )),
                serde_json::from_str(&body).unwrap(),
                headers,
            ),
        )
        .await
        .expect("invalid output must fail before the worker stream ends");
        assert_eq!(response.status(), status);
        let body = json(response).await;
        if status == StatusCode::OK {
            assert_eq!(body["request_id"], "header-id");
            assert_eq!(body["choices"][0]["token_ids"], serde_json::json!([2, 3]));
            assert_eq!(body["choices"][0]["finish_reason"], "length");
        } else {
            assert_eq!(body["error"]["code"], status.as_u16());
            assert!(!body.to_string().contains("private engine detail"));
        }
    }
}
