// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end ext_proc protocol test with a mock EndpointPicker.
//!
//! Verifies the most critical path: a chat completion request flows through the
//! ext_proc server and produces correct routing headers, endpoint metadata, and
//! nvext.token_data injection — without needing Dynamo runtime, NATS, or K8s.

use std::sync::Arc;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_stream::wrappers::ReceiverStream;

use dynamo_ext_proc::picker::{EndpointPicker, Endpoint, PickError, PickResult, RequestInfo};
use dynamo_ext_proc::proto::envoy::config::core::v3::{HeaderMap, HeaderValue};
use dynamo_ext_proc::proto::envoy::service::ext_proc::v3::{
    self as ext_proc,
    external_processor_client::ExternalProcessorClient,
    processing_response, ProcessingRequest,
};
use dynamo_ext_proc::ExtProcServer;

// ---------------------------------------------------------------------------
// MockPicker
// ---------------------------------------------------------------------------

struct MockPicker {
    result: PickResult,
}

#[tonic::async_trait]
impl EndpointPicker for MockPicker {
    async fn pick(
        &self,
        _req: &RequestInfo,
        _endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        Ok(self.result.clone())
    }
}

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_ext_proc_disaggregated_routing() {
    let pick_result = PickResult {
        endpoint: "10.0.1.100:8000".to_string(),
        fallbacks: vec![],
        headers: vec![
            ("x-worker-instance-id".to_string(), "2173882273627495".to_string()),
            ("x-dp-rank".to_string(), "0".to_string()),
            ("x-dynamo-routing-mode".to_string(), "disaggregated".to_string()),
            ("x-prefill-instance-id".to_string(), "966999679619852".to_string()),
            ("x-prefill-dp-rank".to_string(), "0".to_string()),
        ],
        token_ids: Some(vec![1, 2, 3, 4, 5]),
    };

    let picker = Arc::new(MockPicker { result: pick_result });
    let server = ExtProcServer::new(picker);

    // Bind to a random port
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    // Start the server
    let svc = server.into_service();
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(svc)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Give server a moment to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Connect client
    let mut client = ExternalProcessorClient::connect(format!("http://{addr}"))
        .await
        .unwrap();

    // Build the request stream
    let (tx, rx) = tokio::sync::mpsc::channel::<ProcessingRequest>(10);
    let request_stream = ReceiverStream::new(rx);

    let mut response_stream = client
        .process(request_stream)
        .await
        .unwrap()
        .into_inner();

    // Send RequestHeaders (eos=false)
    tx.send(ProcessingRequest {
        request: Some(ext_proc::processing_request::Request::RequestHeaders(
            ext_proc::HttpHeaders {
                headers: Some(HeaderMap {
                    headers: vec![
                        HeaderValue {
                            key: "content-type".to_string(),
                            raw_value: b"application/json".to_vec(),
                            ..Default::default()
                        },
                    ],
                }),
                end_of_stream: false,
            },
        )),
        ..Default::default()
    })
    .await
    .unwrap();

    // Send RequestBody (eos=true)
    let body = br#"{"model":"test-model","messages":[{"role":"user","content":"hello"}]}"#;
    tx.send(ProcessingRequest {
        request: Some(ext_proc::processing_request::Request::RequestBody(
            ext_proc::HttpBody {
                body: body.to_vec(),
                end_of_stream: true,
                ..Default::default()
            },
        )),
        ..Default::default()
    })
    .await
    .unwrap();

    // Collect responses with timeout
    let mut responses = Vec::new();
    let deadline = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(resp) = tokio_stream::StreamExt::next(&mut response_stream).await {
            responses.push(resp.unwrap());
            if responses.len() >= 2 {
                break;
            }
        }
    });
    deadline.await.expect("Timed out waiting for responses");

    assert!(
        responses.len() >= 2,
        "Expected at least 2 responses (header + body), got {}",
        responses.len()
    );

    // -----------------------------------------------------------------------
    // Assert on first response: RequestHeaders
    // -----------------------------------------------------------------------
    let header_resp = &responses[0];

    let req_headers = match &header_resp.response {
        Some(processing_response::Response::RequestHeaders(h)) => h,
        other => panic!("Expected RequestHeaders response, got: {other:?}"),
    };

    let common = req_headers.response.as_ref().expect("missing CommonResponse");
    assert!(common.clear_route_cache, "clear_route_cache should be true");

    let header_mutation = common.header_mutation.as_ref().expect("missing HeaderMutation");
    let set_headers: std::collections::HashMap<String, String> = header_mutation
        .set_headers
        .iter()
        .filter_map(|h| {
            h.header.as_ref().map(|hv| {
                (
                    hv.key.clone(),
                    String::from_utf8_lossy(&hv.raw_value).to_string(),
                )
            })
        })
        .collect();

    assert_eq!(
        set_headers.get("x-gateway-destination-endpoint"),
        Some(&"10.0.1.100:8000".to_string()),
        "destination endpoint header"
    );
    assert_eq!(
        set_headers.get("x-worker-instance-id"),
        Some(&"2173882273627495".to_string()),
        "decode worker ID (decimal)"
    );
    assert_eq!(
        set_headers.get("x-dynamo-routing-mode"),
        Some(&"disaggregated".to_string()),
        "routing mode"
    );
    assert_eq!(
        set_headers.get("x-prefill-instance-id"),
        Some(&"966999679619852".to_string()),
        "prefill worker ID (decimal)"
    );
    assert_eq!(
        set_headers.get("x-dp-rank"),
        Some(&"0".to_string()),
        "decode dp_rank"
    );
    assert_eq!(
        set_headers.get("x-prefill-dp-rank"),
        Some(&"0".to_string()),
        "prefill dp_rank"
    );

    // Check dynamic_metadata has envoy.lb with the endpoint
    let dm = header_resp
        .dynamic_metadata
        .as_ref()
        .expect("missing dynamic_metadata");
    let envoy_lb = dm
        .fields
        .get("envoy.lb")
        .expect("missing envoy.lb in dynamic_metadata");
    if let Some(prost_types::value::Kind::StructValue(inner)) = &envoy_lb.kind {
        let dest = inner
            .fields
            .get("x-gateway-destination-endpoint")
            .expect("missing x-gateway-destination-endpoint in envoy.lb");
        if let Some(prost_types::value::Kind::StringValue(v)) = &dest.kind {
            assert_eq!(v, "10.0.1.100:8000", "dynamic_metadata endpoint");
        } else {
            panic!("x-gateway-destination-endpoint is not a string");
        }
    } else {
        panic!("envoy.lb is not a struct");
    }

    // -----------------------------------------------------------------------
    // Assert on second response: RequestBody with nvext.token_data
    // -----------------------------------------------------------------------
    let body_resp = &responses[1];

    let req_body = match &body_resp.response {
        Some(processing_response::Response::RequestBody(b)) => b,
        other => panic!("Expected RequestBody response, got: {other:?}"),
    };

    let body_common = req_body.response.as_ref().expect("missing body CommonResponse");
    let body_mutation = body_common.body_mutation.as_ref().expect("missing BodyMutation");

    let streamed = match &body_mutation.mutation {
        Some(ext_proc::body_mutation::Mutation::StreamedResponse(s)) => s,
        other => panic!("Expected StreamedResponse body mutation, got: {other:?}"),
    };

    assert!(
        streamed.end_of_stream,
        "body response should have end_of_stream=true"
    );

    // Parse the forwarded body and check nvext.token_data
    let forwarded: serde_json::Value =
        serde_json::from_slice(&streamed.body).expect("forwarded body is not valid JSON");

    let nvext = forwarded
        .get("nvext")
        .expect("forwarded body missing 'nvext' field");
    let token_data = nvext
        .get("token_data")
        .expect("nvext missing 'token_data' field");
    let tokens: Vec<u32> = token_data
        .as_array()
        .expect("token_data is not an array")
        .iter()
        .map(|v| v.as_u64().expect("token not a number") as u32)
        .collect();

    assert_eq!(tokens, vec![1, 2, 3, 4, 5], "nvext.token_data");

    // Verify model field is preserved
    let model = forwarded
        .get("model")
        .and_then(|v| v.as_str())
        .expect("model field missing");
    assert_eq!(model, "test-model", "model field preserved");
}
