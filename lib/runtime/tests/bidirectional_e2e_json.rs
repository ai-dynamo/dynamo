// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-isolated coverage for the explicit JSON request-plane fallback.
//! The codec is globally cached, so this must remain in its own integration
//! test binary rather than sharing a process with the default-codec test.

use std::sync::Arc;

use anyhow::Error;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use dynamo_runtime::pipeline::network::{
    RequestPlanePayloadCodec,
    egress::push_router::{PushRouter, RouterMode},
};
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContextProvider, DataStream},
    error::DynamoError,
    pipeline::{
        ManyIn, ManyOut, RequestStream, ResponseStream, context::Context, network::Ingress,
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
    rt.shutdown();
}
