// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How a worker's `Ingress` treats two requests with the same id.
//!
//! A migration retry keeps its request's id. If the retry reaches the worker
//! that is still inside the engine call for the first attempt, the worker
//! refuses it with `DuplicateRequestId`, which the frontend reads as a
//! `Conflict`: it retries on another worker and does not mark this one down.
//! Once the engine has returned its stream, the id is free again.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Error, Result};
use async_trait::async_trait;
use tokio::sync::{Semaphore, mpsc};

use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{
        ManyOut, PipelineError, ResponseStream, SingleIn,
        network::{Ingress, PushWorkHandler},
    },
    protocols::annotated::Annotated,
};

mod common;
use common::ingress::{request_payload, start_response_peer};

const PARK_IN_CALL: &str = "park-in-call";
const PARK_IN_STREAM: &str = "park-in-stream";
const ANSWER: &str = "answer";

/// Reports each call, then parks inside the call or inside its stream until
/// released, or answers at once, as its request text says.
struct GatedEngine {
    arrivals: mpsc::UnboundedSender<String>,
    call_gate: Arc<Semaphore>,
    stream_gate: Arc<Semaphore>,
}

#[async_trait]
impl AsyncEngine<SingleIn<String>, ManyOut<Annotated<String>>, Error> for GatedEngine {
    async fn generate(&self, request: SingleIn<String>) -> Result<ManyOut<Annotated<String>>> {
        let (text, context) = request.into_parts();
        let _ = self.arrivals.send(text.clone());
        if text == PARK_IN_CALL {
            self.call_gate.acquire().await?.forget();
        }
        let stream_gate = (text == PARK_IN_STREAM).then(|| self.stream_gate.clone());
        let item = async move {
            if let Some(gate) = stream_gate {
                gate.acquire().await.unwrap().forget();
            }
            Annotated::from_data(text)
        };
        Ok(ResponseStream::new(
            Box::pin(futures::stream::once(item)),
            context.context(),
        ))
    }
}

struct Worker {
    ingress: Arc<dyn PushWorkHandler>,
    address: String,
    arrivals: mpsc::UnboundedReceiver<String>,
    call_gate: Arc<Semaphore>,
    stream_gate: Arc<Semaphore>,
}

impl Worker {
    fn start() -> Self {
        let (tx, arrivals) = mpsc::unbounded_channel();
        let call_gate = Arc::new(Semaphore::new(0));
        let stream_gate = Arc::new(Semaphore::new(0));
        let ingress = Ingress::for_engine(Arc::new(GatedEngine {
            arrivals: tx,
            call_gate: call_gate.clone(),
            stream_gate: stream_gate.clone(),
        }))
        .unwrap();
        Self {
            ingress,
            address: start_response_peer(),
            arrivals,
            call_gate,
            stream_gate,
        }
    }

    fn spawn(&self, id: &str, text: &str) -> tokio::task::JoinHandle<Result<(), PipelineError>> {
        let ingress = self.ingress.clone();
        let payload = request_payload(&self.address, id, &text);
        let id = id.to_string();
        tokio::spawn(async move { ingress.handle_payload(payload, Some(id)).await })
    }

    async fn send(&self, id: &str, text: &str) -> Result<(), PipelineError> {
        tokio::time::timeout(Duration::from_secs(10), self.spawn(id, text))
            .await
            .expect("request did not finish")
            .expect("request task panicked")
    }

    async fn arrival(&mut self) -> String {
        tokio::time::timeout(Duration::from_secs(10), self.arrivals.recv())
            .await
            .expect("request did not reach the engine")
            .unwrap()
    }
}

fn is_duplicate_id(error: &PipelineError, id: &str) -> bool {
    let PipelineError::GenerateError(inner) = error else {
        return false;
    };
    matches!(
        inner.downcast_ref::<PipelineError>(),
        Some(PipelineError::DuplicateRequestId(duplicate)) if duplicate == id
    )
}

#[tokio::test]
async fn same_id_inside_the_engine_call_is_refused() {
    let mut worker = Worker::start();
    let first = worker.spawn("same-id", PARK_IN_CALL);
    assert_eq!(worker.arrival().await, PARK_IN_CALL);

    let error = worker
        .send("same-id", ANSWER)
        .await
        .expect_err("a second request with an id inside the engine call must be refused");
    assert!(
        is_duplicate_id(&error, "same-id"),
        "unexpected error: {error:?}"
    );
    assert!(
        worker.arrivals.try_recv().is_err(),
        "the refused request must not reach the engine"
    );

    worker.call_gate.add_permits(1);
    tokio::time::timeout(Duration::from_secs(10), first)
        .await
        .expect("first request did not finish")
        .unwrap()
        .expect("the first request must still be served");

    worker
        .send("same-id", ANSWER)
        .await
        .expect("the id is free once the first request is done");
}

#[tokio::test]
async fn same_id_after_the_engine_call_returned_is_served() {
    let mut worker = Worker::start();
    let first = worker.spawn("same-id", PARK_IN_STREAM);
    assert_eq!(worker.arrival().await, PARK_IN_STREAM);

    worker
        .send("same-id", ANSWER)
        .await
        .expect("the id is free once the engine has returned its stream");
    assert_eq!(worker.arrival().await, ANSWER);

    worker.stream_gate.add_permits(1);
    tokio::time::timeout(Duration::from_secs(10), first)
        .await
        .expect("first request did not finish")
        .unwrap()
        .expect("the first request must still be served");
}

#[tokio::test]
async fn dropped_request_frees_its_id() {
    let mut worker = Worker::start();
    let first = worker.spawn("same-id", PARK_IN_CALL);
    assert_eq!(worker.arrival().await, PARK_IN_CALL);

    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());

    worker
        .send("same-id", ANSWER)
        .await
        .expect("a dropped request must free its id");
    assert_eq!(worker.arrival().await, ANSWER);
}
