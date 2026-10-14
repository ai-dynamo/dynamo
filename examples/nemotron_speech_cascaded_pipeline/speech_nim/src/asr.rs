// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
use futures::{StreamExt, stream::BoxStream};
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::{client::SpeechNimConnection, config::AsrConfig};

#[derive(Clone)]
pub struct AsrAdapter {
    config: Arc<AsrConfig>,
    connection: SpeechNimConnection,
    slots: Arc<Semaphore>,
    shutdown: CancellationToken,
}

impl AsrAdapter {
    pub fn new(config: AsrConfig, connection: SpeechNimConnection) -> Self {
        let slots = Arc::new(Semaphore::new(config.max_concurrent_turns.get()));
        Self {
            config: Arc::new(config),
            connection,
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
