// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bidirectional realtime transcription adapter, independent of LLM backends.

mod session;
mod turn;
mod validation;

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_runtime::{
    engine::{AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, ResponseStream},
    pipeline::{Error, ManyIn, ManyOut},
    protocols::annotated::Annotated,
};
use dynamo_sidecar_common::GrpcChannelPool;
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::config::AsrConfig;

#[derive(Clone)]
pub struct AsrAdapter {
    config: Arc<AsrConfig>,
    channels: Arc<GrpcChannelPool>,
    slots: Arc<Semaphore>,
    shutdown: CancellationToken,
}

impl AsrAdapter {
    pub fn new(config: AsrConfig, channels: Arc<GrpcChannelPool>) -> Self {
        let slots = Arc::new(Semaphore::new(config.max_concurrent_turns.get()));
        Self {
            config: Arc::new(config),
            channels,
            slots,
            shutdown: CancellationToken::new(),
        }
    }

    pub fn close(&self) {
        self.shutdown.cancel();
    }

    pub fn session(
        &self,
        input: BoxStream<'static, Value>,
        context: Arc<dyn AsyncEngineContext>,
    ) -> BoxStream<'static, Value> {
        session::serve(self.clone(), input, context)
    }
}

#[async_trait]
impl AsyncEngine<ManyIn<Value>, ManyOut<Annotated<Value>>, Error> for AsrAdapter {
    async fn generate(&self, request: ManyIn<Value>) -> Result<ManyOut<Annotated<Value>>, Error> {
        let context = request.context();
        let (input, _) = request.into_parts();
        let input = input
            .take()
            .ok_or_else(|| anyhow::anyhow!("realtime input stream already consumed"))?;
        let output = self
            .session(input, context.clone())
            .map(Annotated::from_data);
        Ok(ResponseStream::new(Box::pin(output), context))
    }
}
