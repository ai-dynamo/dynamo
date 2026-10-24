// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One test runtime keeps the process-global TCP accept loop alive for all cases.
use anyhow::Error;
use async_trait::async_trait;
use bytes::Bytes;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    distributed::DistributedConfig,
    engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, DataStream},
    error::DynamoError,
    pipeline::{
        ManyIn, ManyOut, RequestStream, ResponseStream,
        context::Context,
        error::PipelineError,
        network::{
            EncodedResponseFrame, Ingress, IngressRequestDecoder, IngressResponseEncoder,
            RequestPlanePayloadCodec, SerdeIngressPayloadAdapter,
            egress::push_router::{PushRouter, RouterMode},
        },
    },
    protocols::maybe_error::MaybeError,
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::sync::{Notify, oneshot};

#[derive(Debug)]
struct DropNotice(Option<oneshot::Sender<()>>);
impl Drop for DropNotice {
    fn drop(&mut self) {
        if let Some(tx) = self.0.take() {
            let _ = tx.send(());
        }
    }
}
#[derive(Debug, Serialize, Deserialize)]
struct Input {
    value: u64,
    #[serde(skip)]
    drop_notice: Option<DropNotice>,
}
#[derive(Debug, Serialize, Deserialize)]
struct Output {
    error: Option<DynamoError>,
}
impl MaybeError for Output {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        Self {
            error: Some(DynamoError::from(
                Box::new(err) as Box<dyn std::error::Error + 'static>
            )),
        }
    }
    fn err(&self) -> Option<DynamoError> {
        self.error.clone()
    }
}
struct Adapter {
    ninth_decoded: Mutex<Option<oneshot::Sender<()>>>,
    pending_dropped: Mutex<Option<oneshot::Sender<()>>>,
}
impl IngressRequestDecoder<Input> for Adapter {
    async fn decode_request(
        &self,
        codec: RequestPlanePayloadCodec,
        bytes: Bytes,
    ) -> Result<Input, PipelineError> {
        let mut item: Input = SerdeIngressPayloadAdapter
            .decode_request(codec, bytes)
            .await?;
        if item.value == 8 {
            item.drop_notice = Some(DropNotice(self.pending_dropped.lock().unwrap().take()));
            // The engine retains its input without reading: the first eight
            // items fill the queue, so forwarding this ninth item must wait.
            self.ninth_decoded
                .lock()
                .unwrap()
                .take()
                .unwrap()
                .send(())
                .unwrap();
        }
        Ok(item)
    }
}
impl IngressResponseEncoder<Output> for Adapter {
    fn encode_response(
        &self,
        codec: RequestPlanePayloadCodec,
        response: Option<Output>,
        complete_final: bool,
    ) -> impl std::future::Future<Output = Result<EncodedResponseFrame, PipelineError>> + Send {
        SerdeIngressPayloadAdapter.encode_response(codec, response, complete_final)
    }
}
struct HoldingEngine {
    input: Arc<Mutex<Option<DataStream<Input>>>>,
    context_tx: Mutex<Option<oneshot::Sender<Arc<dyn AsyncEngineContext>>>>,
    response_release: Arc<Notify>,
    response_dropped: Mutex<Option<oneshot::Sender<()>>>,
}
#[async_trait]
impl AsyncEngine<ManyIn<Input>, ManyOut<Output>, Error> for HoldingEngine {
    async fn generate(&self, input: ManyIn<Input>) -> Result<ManyOut<Output>, Error> {
        let ctx = input.context();
        let (input, _) = input.into_parts();
        *self.input.lock().unwrap() = input.take();
        self.context_tx
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(ctx.clone())
            .ok()
            .unwrap();
        let release = self.response_release.clone();
        let dropped = self.response_dropped.lock().unwrap().take();
        let stream = async_stream::stream! {
            let _notice = DropNotice(dropped);
            yield Output { error: None };
            release.notified().await;
        };
        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}
async fn run_case(drt: &DistributedRuntime, mode: &str) -> bool {
    let (decoded_tx, decoded_rx) = oneshot::channel();
    let (pending_tx, mut pending_rx) = oneshot::channel();
    let (context_tx, context_rx) = oneshot::channel();
    let (response_tx, response_rx) = oneshot::channel();
    let held = Arc::new(Mutex::new(None));
    let release = Arc::new(Notify::new());
    let engine = Arc::new(HoldingEngine {
        input: held.clone(),
        context_tx: Mutex::new(Some(context_tx)),
        response_release: release.clone(),
        response_dropped: Mutex::new(Some(response_tx)),
    });
    let adapter = Adapter {
        ninth_decoded: Mutex::new(Some(decoded_tx)),
        pending_dropped: Mutex::new(Some(pending_tx)),
    };
    let ingress = Ingress::for_engine_with_adapter(engine, adapter).unwrap();
    let endpoint = drt
        .namespace(format!("input_cancel_{mode}"))
        .unwrap()
        .component("holding_engine".to_string())
        .unwrap()
        .endpoint("test".to_string());
    let server = endpoint.clone();
    tokio::spawn(async move {
        server
            .endpoint_builder()
            .handler(ingress)
            .start()
            .await
            .unwrap();
    });
    let client = endpoint.client().await.unwrap();
    client.wait_for_instances().await.unwrap();
    let router = PushRouter::<Input, Output>::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
    let frames = (0..9)
        .map(|value| Input {
            value,
            drop_notice: None,
        })
        .collect::<Vec<_>>();
    let request = Context::new(RequestStream::new(Box::pin(tokio_stream::iter(frames))));
    let client_ctx = request.context();
    let mut response = router.generate(request).await.unwrap();
    assert!(response.next().await.unwrap().error.is_none());
    let worker_ctx = context_rx.await.unwrap();
    tokio::time::timeout(Duration::from_secs(2), decoded_rx)
        .await
        .unwrap()
        .unwrap();
    let released_before_owner;
    if mode == "health" {
        let input = held.lock().unwrap().take().unwrap();
        let values = tokio::time::timeout(
            Duration::from_secs(2),
            input.map(|item| item.value).collect::<Vec<_>>(),
        )
        .await
        .unwrap();
        assert_eq!(values, (0..9).collect::<Vec<_>>());
        pending_rx.await.unwrap();
        assert!(!worker_ctx.is_stopped());
        released_before_owner = true;
    } else {
        if mode == "kill" {
            client_ctx.kill();
        } else {
            client_ctx.stop();
        }
        tokio::time::timeout(Duration::from_secs(2), worker_ctx.stopped())
            .await
            .unwrap();
        assert_eq!(worker_ctx.is_killed(), mode == "kill");
        released_before_owner =
            match tokio::time::timeout(Duration::from_secs(1), &mut pending_rx).await {
                Ok(result) => {
                    result.unwrap();
                    true
                }
                Err(_) => false,
            };
        // Clean up even on the unfixed implementation before asserting the result.
        drop(held.lock().unwrap().take());
        if !released_before_owner {
            tokio::time::timeout(Duration::from_secs(2), pending_rx)
                .await
                .unwrap()
                .unwrap();
        }
    }
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(2), response_rx)
        .await
        .unwrap()
        .unwrap();
    drop(response);
    println!("{mode}: pending item released before consumer drop = {released_before_owner}");
    released_before_owner
}

#[tokio::test]
async fn cancelled_bidirectional_input_releases_pending_item() {
    // Configure the real request and callback sockets before their process-wide caches initialize.
    unsafe {
        std::env::set_var("DYN_TCP_RPC_HOST", "127.0.0.1");
        std::env::remove_var("DYN_TCP_RPC_PORT");
        std::env::set_var("DYN_TCP_RESPONSE_STREAM_HOST", "127.0.0.1");
        std::env::set_var("DYN_TCP_RESPONSE_STREAM_PORT", "0");
    }
    tokio::time::timeout(Duration::from_secs(30), async {
        let rt = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(rt.clone(), DistributedConfig::process_local()).await.unwrap();
        assert!(run_case(&drt, "health").await);
        let kill = run_case(&drt, "kill").await;
        let stop = run_case(&drt, "stop").await;
        rt.shutdown();
        assert!(kill && stop, "pending decoded input must be released without the engine dropping its input: kill={kill}, stop={stop}");
    }).await.expect("bidirectional cancellation test exceeded its total budget");
}
