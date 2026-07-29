// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end bidirectional round trip over the full Rust stack using a
//! *load-aware* router mode (`LeastLoaded`). The sibling `bidirectional_e2e`
//! test only covers `RoundRobin`, which never touches occupancy tracking; this
//! test exercises the path wired up in DGH-941: `select_load_aware_worker` →
//! `bidirectional_dispatch` → permit-tracked stream.
//!
//! It asserts both halves of that path:
//!   1. dispatch — the load-aware router actually routes a `ManyIn` stream to a
//!      real worker and the frames echo back, and
//!   2. occupancy release — the worker is charged one in-flight request while
//!      the response stream is live, and released back to zero once the stream
//!      is drained and dropped.
//!
//! Lives in its own `tests/` binary so the process-global TCP server's accept
//! loop runs for the whole test process (same rationale as `bidirectional_e2e`).

use std::sync::Arc;

use anyhow::Error;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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

use dynamo_runtime::pipeline::network::egress::push_router::{PushRouter, RouterMode};

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
        let mapped = futures::StreamExt::map(inner, |v| EchoResponse {
            value: Some(v),
            error: None,
        });
        let stream: DataStream<EchoResponse> = Box::pin(mapped);
        Ok(ResponseStream::new(stream, ctx))
    }
}

#[tokio::test]
async fn bidirectional_least_loaded_dispatches_and_releases_occupancy() {
    let rt = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let ns = drt
        .namespace("test_bidi_load_aware_e2e".to_string())
        .unwrap();
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
    let instances = client.wait_for_instances().await.unwrap();
    let worker_id = instances[0].id();

    let router =
        PushRouter::<u64, EchoResponse>::from_client(client.clone(), RouterMode::LeastLoaded)
            .await
            .unwrap();

    // No request in flight yet.
    assert_eq!(
        router.occupancy_load(worker_id),
        Some(0),
        "worker should start with zero in-flight requests"
    );

    let input: ManyIn<u64> = Context::new(RequestStream::new(Box::pin(tokio_stream::iter(vec![
        1u64, 2, 3,
    ]))));
    let response_stream = router.generate(input).await.unwrap();

    // The load-aware permit must charge the selected worker while the response
    // stream is live (before it is drained/dropped).
    assert_eq!(
        router.occupancy_load(worker_id),
        Some(1),
        "load-aware bidirectional dispatch must charge one in-flight request to the worker"
    );

    let responses: Vec<EchoResponse> = futures::StreamExt::collect(response_stream).await;
    let values: Vec<u64> = responses.iter().filter_map(|r| r.value).collect();
    assert_eq!(
        values,
        vec![1u64, 2, 3],
        "echo engine should reflect each input frame back over the load-aware path; got {responses:?}"
    );

    // Draining (and dropping) the stream must release the occupancy permit.
    assert_eq!(
        router.occupancy_load(worker_id),
        Some(0),
        "occupancy must be released back to zero once the response stream ends"
    );

    rt.shutdown();
}
