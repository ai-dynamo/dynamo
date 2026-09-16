// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_backend_common::{BackendError, LLMEngine};
use futures::{StreamExt, stream::BoxStream};
use tokio::sync::{Notify, watch};
use tonic::Status;

mod scenarios;

pub trait SidecarFixture {
    type Engine: LLMEngine;

    async fn start(control: Control) -> Self;
    async fn engine(&self) -> Self::Engine;
    fn eof_error() -> BackendError;
    fn active_request_count(&self) -> usize;
}

pub trait NativeResponse: Clone + Send + 'static {
    fn record_tokens(&self, tokens: &mut Vec<u32>) -> bool;
    fn is_terminal(&self) -> bool;
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    ExtraAfterTerminal,
    OpenError,
    EarlyEof,
    ReadError,
    PendingOpen,
    PendingRead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Idle,
    Open,
    Read,
    Dropped,
}

#[derive(Default)]
struct Observed {
    request_ids: Vec<String>,
    tokens: Vec<u32>,
}

#[derive(Clone)]
pub struct Control {
    fault: Fault,
    observed: Arc<Mutex<Observed>>,
    gate: watch::Sender<Gate>,
    release: Arc<Notify>,
}

pub struct OpenGuard(watch::Sender<Gate>);

impl Drop for OpenGuard {
    fn drop(&mut self) {
        self.0.send_replace(Gate::Dropped);
    }
}

impl Control {
    fn new(fault: Fault) -> Self {
        Self {
            fault,
            observed: Arc::default(),
            gate: watch::channel(Gate::Idle).0,
            release: Arc::default(),
        }
    }

    pub async fn open(&self, request_id: &str) -> Result<OpenGuard, Status> {
        self.observed
            .lock()
            .unwrap()
            .request_ids
            .push(request_id.to_owned());
        let guard = OpenGuard(self.gate.clone());
        self.gate.send_replace(Gate::Open);
        match self.fault {
            Fault::PendingOpen => futures::future::pending().await,
            Fault::OpenError => Err(Status::unavailable("injected open failure")),
            _ => Ok(guard),
        }
    }

    pub fn stream<T: NativeResponse>(
        &self,
        mut source: BoxStream<'static, Result<T, Status>>,
        guard: OpenGuard,
    ) -> BoxStream<'static, Result<T, Status>> {
        let control = self.clone();
        Box::pin(async_stream::try_stream! {
            let _guard = guard;
            let mut first = None;
            while let Some(response) = source.next().await {
                let response = response?;
                let has_tokens = response.record_tokens(
                    &mut control.observed.lock().unwrap().tokens,
                );
                if first.is_none() && has_tokens {
                    first = Some(response.clone());
                }
                let terminal = response.is_terminal();
                yield response;
                if terminal && control.fault == Fault::ExtraAfterTerminal {
                    yield first.take().expect("Mocker generated tokens");
                    futures::future::pending::<()>().await;
                }
                if has_tokens {
                    match control.fault {
                        Fault::EarlyEof | Fault::ReadError | Fault::PendingRead => {
                            control.gate.send_replace(Gate::Read);
                            // Let the client receive its first token before injecting a failure.
                            control.release.notified().await;
                            if control.fault == Fault::ReadError {
                                Err(Status::unavailable("injected read failure"))?;
                            }
                            return;
                        }
                        _ => {}
                    }
                }
            }
        })
    }

    async fn wait(&self, gate: Gate) {
        let mut receiver = self.gate.subscribe();
        bounded(async {
            receiver.wait_for(|value| *value == gate).await.unwrap();
        })
        .await;
    }

    fn tokens(&self) -> Vec<u32> {
        self.observed.lock().unwrap().tokens.clone()
    }

    fn request_ids(&self) -> Vec<String> {
        self.observed.lock().unwrap().request_ids.clone()
    }
}

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("sidecar or Mocker stalled")
}
