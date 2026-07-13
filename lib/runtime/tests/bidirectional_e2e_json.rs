// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-isolated coverage for the explicit JSON request-plane fallback.
//! The codec is globally cached, so this must remain in its own integration
//! test binary rather than sharing a process with the default-codec test.

use std::sync::Arc;

use anyhow::Error;
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use dynamo_runtime::pipeline::network::{
    EncodedResponseFrame, IngressRequestDecoder, IngressResponseEncoder, NetworkStreamWrapper,
    RequestPlanePayloadCodec,
    egress::push_router::{PushRouter, RouterMode},
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider, DataStream},
    error::DynamoError,
    metrics::MetricsHierarchy,
    pipeline::{
        ManyIn, ManyOut, PipelineError, RequestStream, ResponseStream, context::Context,
        network::Ingress,
    },
    protocols::maybe_error::MaybeError,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct EchoResponse {
    value: Option<u64>,
    #[serde(default)]
    error: Option<DynamoError>,
}

impl MaybeError for EchoResponse {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        Self {
            value: None,
            error: Some(DynamoError::from(
                Box::new(err) as Box<dyn std::error::Error + 'static>
            )),
        }
    }

    fn err(&self) -> Option<DynamoError> {
        self.error.clone()
    }
}

struct EchoEngine;

#[async_trait]
impl AsyncEngine<ManyIn<u64>, ManyOut<EchoResponse>, Error> for EchoEngine {
    async fn generate(&self, input: ManyIn<u64>) -> Result<ManyOut<EchoResponse>, Error> {
        let ctx = input.context();
        let (request_stream, _ctx_unit) = input.into_parts();
        let inner = request_stream
            .take()
            .expect("RequestStream::take called twice on EchoEngine input");
        let mapped = futures::StreamExt::map(inner, |value| EchoResponse {
            value: Some(value),
            error: None,
        });
        let stream: DataStream<EchoResponse> = Box::pin(mapped);
        Ok(ResponseStream::new(stream, ctx))
    }
}

#[derive(Debug)]
struct NonSerdeResponse;

struct FailingResponseEngine;

#[async_trait]
impl AsyncEngine<ManyIn<u64>, ManyOut<NonSerdeResponse>, Error> for FailingResponseEngine {
    async fn generate(&self, input: ManyIn<u64>) -> Result<ManyOut<NonSerdeResponse>, Error> {
        let ctx = input.context();
        let (request_stream, _ctx_unit) = input.into_parts();
        let inner = request_stream
            .take()
            .expect("RequestStream::take called twice on FailingResponseEngine input");
        let stream: DataStream<NonSerdeResponse> =
            Box::pin(futures::StreamExt::map(inner, |_| NonSerdeResponse));
        Ok(ResponseStream::new(stream, ctx))
    }
}

#[derive(Debug)]
struct FailingResponseAdapter;

impl IngressRequestDecoder<u64> for FailingResponseAdapter {
    async fn decode_request(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        bytes: Bytes,
    ) -> Result<u64, PipelineError> {
        payload_codec.decode(&bytes).map_err(|error| {
            PipelineError::DeserializationError(format!(
                "failed decoding test request as {}: {error}",
                payload_codec.name()
            ))
        })
    }
}

impl IngressResponseEncoder<NonSerdeResponse> for FailingResponseAdapter {
    async fn encode_response(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        response: Option<NonSerdeResponse>,
        complete_final: bool,
    ) -> Result<EncodedResponseFrame, PipelineError> {
        if response.is_some() {
            return Err(PipelineError::SerializationError(
                "intentional response encoding failure".to_string(),
            ));
        }
        assert!(
            complete_final,
            "only a clean terminal frame has no response"
        );
        let bytes = payload_codec
            .encode(&NetworkStreamWrapper::<EchoResponse> {
                data: None,
                complete_final: true,
            })
            .map_err(|error| PipelineError::SerializationError(error.to_string()))?;
        Ok(EncodedResponseFrame {
            bytes: bytes.into(),
            is_error: false,
            stop_stream: false,
        })
    }
}

#[tokio::test]
async fn bidirectional_end_to_end_echo_with_explicit_json_codec() {
    // This test binary owns the process and sets the value before the first
    // request initializes the request-plane codec cache.
    unsafe {
        std::env::set_var("DYN_REQUEST_PLANE_CODEC", "json");
    }
    assert_eq!(
        RequestPlanePayloadCodec::configured(),
        RequestPlanePayloadCodec::Json
    );

    let rt = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let ns = drt.namespace("test_bidi_e2e_json".to_string()).unwrap();
    let component = ns.component("echo_component".to_string()).unwrap();
    let endpoint = component.endpoint("echo_endpoint".to_string());

    let ingress = Ingress::for_engine(Arc::new(EchoEngine)).unwrap();
    let endpoint_for_server = endpoint.clone();
    tokio::spawn(async move {
        let _ = endpoint_for_server
            .endpoint_builder()
            .handler(ingress)
            .start()
            .await;
    });

    let client = endpoint.client().await.unwrap();
    client.wait_for_instances().await.unwrap();

    let router = PushRouter::<u64, EchoResponse>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
    let input: ManyIn<u64> = Context::new(RequestStream::new(Box::pin(tokio_stream::iter(vec![
        10u64, 20, 30,
    ]))));
    let response_stream = router.generate(input).await.unwrap();
    let responses: Vec<EchoResponse> = futures::StreamExt::collect(response_stream).await;

    assert_eq!(
        responses
            .iter()
            .filter_map(|response| response.value)
            .collect::<Vec<_>>(),
        vec![10u64, 20, 30]
    );

    let failing_endpoint = component.endpoint("failing_endpoint".to_string());
    let failing_ingress =
        Ingress::for_engine_with_adapter(Arc::new(FailingResponseEngine), FailingResponseAdapter)
            .unwrap();
    let failing_endpoint_for_server = failing_endpoint.clone();
    tokio::spawn(async move {
        let _ = failing_endpoint_for_server
            .endpoint_builder()
            .handler(failing_ingress)
            .start()
            .await;
    });

    let failing_client = failing_endpoint.client().await.unwrap();
    failing_client.wait_for_instances().await.unwrap();
    let failing_router =
        PushRouter::<u64, EchoResponse>::from_client(failing_client, RouterMode::RoundRobin)
            .await
            .unwrap();
    let failing_input: ManyIn<u64> =
        Context::new(RequestStream::new(Box::pin(tokio_stream::iter(vec![
            99u64,
        ]))));
    let failure_stream = failing_router.generate(failing_input).await.unwrap();
    let failure_responses: Vec<EchoResponse> = futures::StreamExt::collect(failure_stream).await;
    let error = failure_responses
        .into_iter()
        .find_map(|response| response.error)
        .expect("encoding failure must not be reported as a clean terminal frame");
    assert!(
        error
            .to_string()
            .contains("Stream ended before generation completed"),
        "unexpected client error: {error}"
    );

    let metrics = failing_endpoint.metrics().prometheus_expfmt().unwrap();
    let serialization_error = metrics
        .lines()
        .find(|line| {
            !line.starts_with('#')
                && line.contains("errors_total")
                && line.contains("error_type=\"serialization\"")
        })
        .expect("serialization error metric must be present");
    assert!(
        serialization_error.ends_with(" 1"),
        "unexpected serialization error metric: {serialization_error}"
    );
    rt.shutdown();
}
